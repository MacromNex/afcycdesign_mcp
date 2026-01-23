#!/usr/bin/env python3
"""
Use Case 2: De Novo Cyclic Peptide Hallucination

This script simultaneously generates a new cyclic peptide structure and its
corresponding sequence from scratch (de novo design).

Based on: af_cyc_design.ipynb - cyclic peptide structure prediction and design using AlphaFold
Reference: Stephen Rettie et al., doi: https://doi.org/10.1101/2023.02.25.529956

Key Parameters (from paper):
- Input: Desired peptide length only (paper tested 7-16 residues successfully)
- Initialization: Random Gumbel distribution
- Loss Function: 1 - pLDDT + PAE/31 + con/2
- Contact (con) Loss: binary=True, cutoff=21.6875, num=length, seqsep=0
- Activation: Softmax with temperature scaling from 1.0 down to 0.01
- Optimization: 3-stage protocol (50, 50, 10 steps)
- Quality Filter: pLDDT > 0.9 for high-confidence scaffolds

Workflow (from paper):
1. Run the 3-stage optimization protocol (50, 50, 10 steps)
2. Enable cyclic offset specifically for sequence optimization
3. Filter resulting scaffolds for pLDDT > 0.9

Note: The authors generated ~24,000 high-confidence scaffolds for their library.

Usage:
    # Basic hallucination
    python use_case_2_cyclic_hallucination.py --length 13 --output hallucinated_cyclic.pdb

    # Paper-recommended settings for scaffold library generation
    python use_case_2_cyclic_hallucination.py --length 13 --rm_aa "C" --plddt_threshold 0.9 --output scaffold.pdb

    # Compact structure with Rg constraint
    python use_case_2_cyclic_hallucination.py --length 15 --rm_aa "C,M" --add_rg --output compact_cyclic.pdb

    # Use GPU
    python use_case_2_cyclic_hallucination.py --length 12 --gpu 0 --output gpu_hallucination.pdb
"""

# ==============================================================================
# GPU Configuration (MUST be set before importing JAX)
# ==============================================================================
import os
import sys

# Add examples directory to path for gpu_utils import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_utils import auto_setup_gpu
auto_setup_gpu()

# ==============================================================================
# Now safe to import JAX and other libraries
# ==============================================================================
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import jax.numpy as jnp
import jax
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.af.alphafold.common import residue_constants


def add_cyclic_offset(model, offset_type=2):
    """
    Add cyclic offset to connect N and C termini for head-to-tail cyclization.

    Args:
        model: AfDesign model instance
        offset_type: Type of offset (1, 2, or 3)
    """
    def cyclic_offset(L):
        i = np.arange(L)
        ij = np.stack([i, i+L], -1)
        offset = i[:,None] - i[None,:]
        c_offset = np.abs(ij[:,None,:,None] - ij[None,:,None,:]).min((2,3))

        if offset_type == 1:
            c_offset = c_offset
        elif offset_type >= 2:
            a = c_offset < np.abs(offset)
            c_offset[a] = -c_offset[a]
        if offset_type == 3:
            idx = np.abs(c_offset) > 2
            c_offset[idx] = (32 * c_offset[idx]) / abs(c_offset[idx])
        return c_offset * np.sign(offset)

    idx = model._inputs["residue_index"]
    offset = np.array(idx[:,None] - idx[None,:])

    if model.protocol == "binder":
        c_offset = cyclic_offset(model._binder_len)
        offset[model._target_len:, model._target_len:] = c_offset

    if model.protocol in ["fixbb", "partial", "hallucination"]:
        Ln = 0
        for L in model._lengths:
            offset[Ln:Ln+L, Ln:Ln+L] = cyclic_offset(L)
            Ln += L

    model._inputs["offset"] = offset


def add_rg_loss(model, weight=0.1):
    """
    Add radius of gyration loss to maintain compact structure.

    Args:
        model: AfDesign model instance
        weight: Weight for the RG loss term
    """
    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:,residue_constants.atom_order["CA"]]
        rg = jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
        rg_th = 2.38 * ca.shape[0] ** 0.365
        rg = jax.nn.elu(rg - rg_th)
        return {"rg": rg}

    model._callbacks["model"]["loss"].append(loss_fn)
    model.opt["weights"]["rg"] = weight


def hallucinate_cyclic_peptide(length=13, rm_aa="C", add_rg=False, rg_weight=0.1,
                              offset_type=2, output_file="cyclic_hallucinated.pdb",
                              num_recycles=0, soft_iters=50, stage_iters=(50, 50, 10),
                              plddt_threshold=0.9, verbose=True):
    """
    Hallucinate a cyclic peptide structure from scratch (de novo design).

    This is Use Case 3 from the AfCycDesign paper: De Novo Hallucination Workflow.

    The workflow simultaneously generates a new structure AND its corresponding
    sequence, using AlphaFold to predict well-structured peptides with high
    confidence scores.

    Paper-specified parameters:
    - Initialization: Random Gumbel distribution
    - Loss Function: 1 - pLDDT + PAE/31 + con/2
    - Contact Loss: binary=True, cutoff=21.6875, num=length, seqsep=0
    - Activation: Softmax with temperature scaling (1.0 -> 0.01)
    - Quality Filter: pLDDT > 0.9 for high-confidence scaffolds

    Args:
        length: Length of peptide to generate (7-16 residues tested in paper)
        rm_aa: Amino acids to remove (e.g., "C" or "C,M")
        add_rg: Whether to add radius of gyration loss for compactness
        rg_weight: Weight for RG loss
        offset_type: Type of cyclic offset (2 recommended)
        output_file: Output PDB file path
        num_recycles: Number of recycles for AlphaFold
        soft_iters: Iterations for soft pre-design (default: 50)
        stage_iters: Iterations for 3-stage design (default: 50, 50, 10)
        plddt_threshold: Quality threshold for filtering (default: 0.9 as per paper)
        verbose: Whether to print progress

    Returns:
        dict: Results including sequences, metrics, and quality assessment
    """
    # Clear previous models
    clear_mem()

    # Initialize model for hallucination
    af_model = mk_afdesign_model(protocol="hallucination", num_recycles=num_recycles)

    if verbose:
        print("=" * 60)
        print("AfCycDesign: De Novo Hallucination (Use Case 2)")
        print("=" * 60)
        print(f"Target length: {length} residues")
        print(f"Excluded amino acids: {rm_aa}")
        print(f"Stage iterations: {stage_iters}")
        print("Loss function: 1 - pLDDT + PAE/31 + con/2")
        print(f"Quality threshold: pLDDT > {plddt_threshold}")
        print("=" * 60)

    # Prepare inputs
    af_model.prep_inputs(length=length, rm_aa=rm_aa)

    # Add cyclic offset for head-to-tail cyclization
    # This is enabled specifically for sequence optimization as per paper
    add_cyclic_offset(af_model, offset_type=offset_type)

    # Optionally add radius of gyration loss for compact structures
    if add_rg:
        add_rg_loss(af_model, weight=rg_weight)

    if verbose:
        print(f"\nPeptide length: {af_model._len}")
        print(f"Loss weights: {af_model.opt['weights']}")

    # ==========================================================================
    # Pre-design with Gumbel initialization (from paper)
    # ==========================================================================
    if verbose:
        print("\nWorkflow:")
        print("  1. Initializing with random Gumbel distribution...")

    af_model.restart()
    af_model.set_seq(mode="gumbel")  # Random Gumbel initialization

    # ==========================================================================
    # Configure contact loss (from paper):
    # binary=True, cutoff=21.6875, num=length, seqsep=0
    # ==========================================================================
    af_model.set_opt("con", binary=True, cutoff=21.6875, num=af_model._len, seqsep=0)

    # ==========================================================================
    # Set loss weights (from paper): 1 - pLDDT + PAE/31 + con/2
    # The weights pae=1, plddt=1, con=0.5 correspond to this loss function
    # ==========================================================================
    af_model.set_weights(pae=1, plddt=1, con=0.5)

    if verbose:
        print("  2. Configured contact loss: binary=True, cutoff=21.6875")
        print("  3. Running soft pre-optimization...")

    # Run soft optimization with temperature scaling
    af_model.design_soft(soft_iters)

    if verbose:
        print(f"  4. Running 3-stage design optimization ({sum(stage_iters)} steps)...")
        print(f"     - Stage 1 (logits): {stage_iters[0]} iterations")
        print(f"     - Stage 2 (soft):   {stage_iters[1]} iterations")
        print(f"     - Stage 3 (hard):   {stage_iters[2]} iterations")

    # ==========================================================================
    # Three-stage design: logits -> soft -> hard
    # Uses softmax activation with temperature scaling from 1.0 down to 0.01
    # ==========================================================================
    af_model.set_seq(seq=af_model.aux["seq"]["pseudo"])
    af_model.design_3stage(*stage_iters)

    # Save results
    af_model.save_pdb(output_file)

    # Get metrics
    metrics = af_model.aux.get('log', {})
    plddt = metrics.get('plddt', 0)
    pae = metrics.get('pae', 1)
    con = metrics.get('con', 0)

    # Quality assessment (from paper: filter for pLDDT > 0.9)
    high_confidence = plddt > plddt_threshold

    if verbose:
        print(f"\n  5. Hallucination complete!")
        print(f"\n{'=' * 60}")
        print("RESULTS")
        print("=" * 60)
        print(f"Output file: {output_file}")
        print(f"pLDDT: {plddt:.3f}")
        print(f"PAE: {pae:.3f}")
        print(f"Contacts: {con:.3f}")
        print(f"\nQuality Assessment:")
        if high_confidence:
            print(f"  HIGH CONFIDENCE scaffold (pLDDT > {plddt_threshold})")
            print("  Suitable for scaffold library / binder design")
        else:
            print(f"  LOW CONFIDENCE scaffold (pLDDT <= {plddt_threshold})")
            print("  Consider re-running or adjusting parameters")
        print("=" * 60)

    # Get designed sequences
    sequences = af_model.get_seqs()

    results = {
        "sequences": sequences,
        "pdb_file": output_file,
        "metrics": {
            "plddt": float(plddt),
            "pae": float(pae),
            "con": float(con),
        },
        "length": af_model._len,
        "high_confidence": high_confidence,
        "plddt_threshold": plddt_threshold,
        "model": af_model
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Hallucinate cyclic peptide structures from scratch"
    )
    parser.add_argument("--length", type=int, required=True,
                       help="Length of peptide to generate")
    parser.add_argument("--rm_aa", type=str, default="C",
                       help="Amino acids to exclude (comma-separated, default: C)")
    parser.add_argument("--output", type=str, default="cyclic_hallucinated.pdb",
                       help="Output PDB file (default: cyclic_hallucinated.pdb)")
    parser.add_argument("--offset_type", type=int, choices=[1, 2, 3], default=2,
                       help="Cyclic offset type (default: 2)")
    parser.add_argument("--add_rg", action="store_true",
                       help="Add radius of gyration loss for compact structures")
    parser.add_argument("--rg_weight", type=float, default=0.1,
                       help="Weight for RG loss (default: 0.1)")
    parser.add_argument("--num_recycles", type=int, default=0,
                       help="Number of AF2 recycles (default: 0)")
    parser.add_argument("--soft_iters", type=int, default=50,
                       help="Iterations for soft pre-design (default: 50)")
    parser.add_argument("--stage_iters", type=int, nargs=3, default=[50, 50, 10],
                       help="Iterations for 3-stage design: logits soft hard (default: 50 50 10)")
    parser.add_argument("--plddt_threshold", type=float, default=0.9,
                       help="Quality threshold for high-confidence scaffolds (default: 0.9 as per paper)")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose output")

    # GPU arguments (parsed early by gpu_utils, included here for --help)
    parser.add_argument("--gpu", type=int, metavar="ID",
                       help="GPU device ID to use (0, 1, etc.)")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU mode")

    args = parser.parse_args()

    # Validate inputs
    if args.length < 5:
        parser.error("Peptide length must be at least 5 residues")
    if args.length > 50:
        print("Warning: Very long peptides (>50) may be challenging to design", file=sys.stderr)

    try:
        if not args.quiet:
            print("=== AfCycDesign: Cyclic Peptide Hallucination ===")
            print(f"Length: {args.length}")
            print(f"Excluded AAs: {args.rm_aa}")
            print(f"Output: {args.output}")
            if args.add_rg:
                print(f"Radius of gyration constraint: weight={args.rg_weight}")

        results = hallucinate_cyclic_peptide(
            length=args.length,
            rm_aa=args.rm_aa,
            add_rg=args.add_rg,
            rg_weight=args.rg_weight,
            offset_type=args.offset_type,
            output_file=args.output,
            num_recycles=args.num_recycles,
            soft_iters=args.soft_iters,
            stage_iters=tuple(args.stage_iters),
            plddt_threshold=args.plddt_threshold,
            verbose=not args.quiet
        )

        if not args.quiet:
            print("\n=== Designed Sequences ===")
            for i, seq in enumerate(results["sequences"]):
                print(f"Sequence {i+1}: {seq}")

            # Quality summary
            if results["high_confidence"]:
                print(f"\nScaffold PASSED quality filter (pLDDT > {results['plddt_threshold']})")
            else:
                print(f"\nScaffold FAILED quality filter (pLDDT <= {results['plddt_threshold']})")

        return 0

    except Exception as e:
        import traceback
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())