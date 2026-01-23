#!/usr/bin/env python3
"""
Use Case 1: Cyclic Peptide Fixed Backbone Sequence Redesign (Fixbb)

This script finds a sequence that maximizes the folding propensity for a specific
pre-defined cyclic peptide backbone structure.

Based on: af_cyc_design.ipynb - cyclic peptide structure prediction and design using AlphaFold
Reference: Stephen Rettie et al., doi: https://doi.org/10.1101/2023.02.25.529956

Key Parameters (from paper):
- Input: Target backbone structure in PDB format (e.g., Rosetta-generated backbones)
- Optimization: 110-step schedule - Stage 1 (50), Stage 2 (50), Stage 3 (10)
- Loss Function: Categorical Cross-Entropy (CCE) between target and predicted distograms
- Optimizer: Start from continuous logits, transition to one-hot using straight-through estimator

Workflow (from paper):
1. Extract the distogram from the target PDB
2. Initialize with a random sequence
3. Iteratively optimize the sequence to minimize CCE loss

Usage:
    # Basic fixed backbone design
    python use_case_1_cyclic_fixbb_design.py --pdb input.pdb --chain A --output designed_cyclic.pdb

    # Download and design from PDB code
    python use_case_1_cyclic_fixbb_design.py --pdb_code 7m28 --chain A --output designed_cyclic.pdb

    # With compactness constraint
    python use_case_1_cyclic_fixbb_design.py --pdb input.pdb --add_rg --rg_weight 0.1 --output compact.pdb

    # Paper-recommended 110-step schedule (50, 50, 10)
    python use_case_1_cyclic_fixbb_design.py --pdb input.pdb --stage_iters 50 50 10 --output optimized.pdb

    # Use GPU
    python use_case_1_cyclic_fixbb_design.py --pdb input.pdb --gpu 0 --output gpu_design.pdb
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
            1 - Basic cyclic offset
            2 - Signed cyclic offset (default)
            3 - Enhanced cyclic offset with scaling
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


def get_pdb_file(pdb_input):
    """
    Get PDB file from various sources.

    Args:
        pdb_input: PDB code (4 characters), file path, or None for upload

    Returns:
        str: Path to PDB file
    """
    if pdb_input is None or pdb_input == "":
        raise ValueError("PDB input is required. Provide --pdb or --pdb_code")
    elif os.path.isfile(pdb_input):
        return pdb_input
    elif len(pdb_input) == 4:
        # Download from PDB
        pdb_file = f"{pdb_input}.pdb"
        if not os.path.exists(pdb_file):
            os.system(f"wget -qnc https://files.rcsb.org/view/{pdb_input}.pdb")
        return pdb_file
    else:
        # Try AlphaFold DB
        af_file = f"AF-{pdb_input}-F1-model_v3.pdb"
        if not os.path.exists(af_file):
            os.system(f"wget -qnc https://alphafold.ebi.ac.uk/files/AF-{pdb_input}-F1-model_v3.pdb")
        return af_file


def design_cyclic_peptide_fixbb(pdb_file, chain="A", offset_type=2, add_rg=False,
                               rg_weight=0.1, output_file="cyclic_fixbb_designed.pdb",
                               num_recycles=0, stage_iters=(50, 50, 10), verbose=True):
    """
    Design a new sequence for a cyclic peptide backbone structure.

    This is Use Case 2 from the AfCycDesign paper: Fixed Backbone Sequence Redesign.

    The optimization uses a 3-stage schedule (default 110 steps total):
    - Stage 1 (logits): Optimize continuous representation
    - Stage 2 (soft): Softmax activation with temperature scaling
    - Stage 3 (hard): One-hot encoding with straight-through estimator

    Loss Function: Categorical Cross-Entropy (CCE) between target distogram
    (extracted from input PDB) and predicted distogram.

    Args:
        pdb_file: Path to input PDB file (target backbone)
        chain: Chain ID to use
        offset_type: Type of cyclic offset (1, 2, or 3)
        add_rg: Whether to add radius of gyration loss
        rg_weight: Weight for RG loss
        output_file: Output PDB file path
        num_recycles: Number of recycles for AlphaFold
        stage_iters: Iterations for 3-stage design (logits, soft, hard)
                    Default: (50, 50, 10) = 110 steps total as per paper
        verbose: Whether to print progress

    Returns:
        dict: Results including sequences and metrics
    """
    # Clear previous models
    clear_mem()

    # Initialize model for fixed backbone design
    # The fixbb protocol uses CCE loss between target and predicted distograms
    af_model = mk_afdesign_model(protocol="fixbb", num_recycles=num_recycles)

    if verbose:
        print("=" * 60)
        print("AfCycDesign: Fixed Backbone Sequence Design (Use Case 1)")
        print("=" * 60)
        print(f"Input PDB: {pdb_file}")
        print(f"Chain: {chain}")
        print(f"Stage iterations: {stage_iters} (total: {sum(stage_iters)} steps)")
        print("Loss function: CCE (target vs predicted distogram)")
        print("=" * 60)

    # Prepare inputs - this extracts the distogram from the target PDB
    af_model.prep_inputs(pdb_filename=pdb_file, chain=chain)

    # Add cyclic offset for head-to-tail cyclization
    add_cyclic_offset(af_model, offset_type=offset_type)

    # Optionally add radius of gyration loss
    if add_rg:
        add_rg_loss(af_model, weight=rg_weight)

    if verbose:
        print(f"\nPeptide length: {af_model._len}")
        print(f"Loss weights: {af_model.opt['weights']}")

    # Initialize with random sequence (as per paper workflow)
    af_model.restart()

    if verbose:
        print("\nWorkflow:")
        print("  1. Distogram extracted from target PDB")
        print("  2. Initialized with random sequence")
        print("  3. Running 3-stage optimization to minimize CCE loss...")
        print(f"     - Stage 1 (logits): {stage_iters[0]} iterations")
        print(f"     - Stage 2 (soft):   {stage_iters[1]} iterations")
        print(f"     - Stage 3 (hard):   {stage_iters[2]} iterations")
        print("       (uses straight-through estimator for one-hot transition)")

    # Run 3-stage design with specified iterations
    # Stage 1: Optimize continuous logits representation
    # Stage 2: Soft optimization with softmax activation
    # Stage 3: Hard optimization with straight-through estimator (logits -> one-hot)
    af_model.design_3stage(*stage_iters)

    # Save results
    af_model.save_pdb(output_file)

    if verbose:
        print(f"Design complete! Saved to: {output_file}")
        print(f"Final metrics: {af_model.aux['log']}")

    # Get designed sequences
    sequences = af_model.get_seqs()

    results = {
        "sequences": sequences,
        "pdb_file": output_file,
        "metrics": af_model.aux['log'],
        "length": af_model._len,
        "model": af_model
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Design cyclic peptide sequences for fixed backbone structures"
    )
    parser.add_argument("--pdb", type=str, help="Path to input PDB file")
    parser.add_argument("--pdb_code", type=str, help="4-letter PDB code to download")
    parser.add_argument("--chain", type=str, default="A", help="Chain ID to use (default: A)")
    parser.add_argument("--output", type=str, default="cyclic_fixbb_designed.pdb",
                       help="Output PDB file (default: cyclic_fixbb_designed.pdb)")
    parser.add_argument("--offset_type", type=int, choices=[1, 2, 3], default=2,
                       help="Cyclic offset type (default: 2)")
    parser.add_argument("--add_rg", action="store_true",
                       help="Add radius of gyration loss")
    parser.add_argument("--rg_weight", type=float, default=0.1,
                       help="Weight for RG loss (default: 0.1)")
    parser.add_argument("--num_recycles", type=int, default=0,
                       help="Number of AF2 recycles (default: 0)")
    parser.add_argument("--stage_iters", type=int, nargs=3, default=[50, 50, 10],
                       help="Iterations for 3-stage design: logits soft hard (default: 50 50 10, total 110 steps as per paper)")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose output")

    # GPU arguments (parsed early by gpu_utils, included here for --help)
    parser.add_argument("--gpu", type=int, metavar="ID",
                       help="GPU device ID to use (0, 1, etc.)")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU mode")

    args = parser.parse_args()

    # Get PDB file
    pdb_input = args.pdb or args.pdb_code
    if not pdb_input:
        parser.error("Either --pdb or --pdb_code is required")

    try:
        pdb_file = get_pdb_file(pdb_input)

        if not args.quiet:
            print("=== AfCycDesign: Cyclic Peptide Fixed Backbone Design ===")
            print(f"Input: {pdb_file}")
            print(f"Chain: {args.chain}")
            print(f"Output: {args.output}")

        results = design_cyclic_peptide_fixbb(
            pdb_file=pdb_file,
            chain=args.chain,
            offset_type=args.offset_type,
            add_rg=args.add_rg,
            rg_weight=args.rg_weight,
            output_file=args.output,
            num_recycles=args.num_recycles,
            stage_iters=tuple(args.stage_iters),
            verbose=not args.quiet
        )

        if not args.quiet:
            print("\n=== Results ===")
            for i, seq in enumerate(results["sequences"]):
                print(f"Sequence {i+1}: {seq}")

        return 0

    except Exception as e:
        import traceback
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())