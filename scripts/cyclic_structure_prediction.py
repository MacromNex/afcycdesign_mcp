#!/usr/bin/env python3
"""
Use Case 0: Cyclic Peptide Structure Prediction

This script predicts the 3D structure of a known cyclic peptide sequence using
AlphaFold with cyclic offset constraints for head-to-tail cyclization.

Based on: af_cyc_design.ipynb - cyclic peptide structure prediction and design using AlphaFold
Reference: Stephen Rettie et al., doi: https://doi.org/10.1101/2023.02.25.529956

Key Parameters (from paper):
- Recycles: 6 recycles for accurate structure prediction
- Cyclic Offset: Custom N x N cyclic offset matrix for relative positional encoding
- Terminal Separation: Define sequence separation between terminal residues as 1 or -1
- Quality Filter: pLDDT > 0.7 indicates high confidence

Benchmark Sequences (from paper):
- 1JBL: GFNYGPFGSC (10 residues)
- 2MW0: FRLLNYYA (8 residues)
- 2LWV: WTYTYDWFC (9 residues)
- 5KX1: CWLPCFGDAC (10 residues)

Usage:
    # Predict structure from sequence
    python use_case_0_structure_prediction.py --sequence "GFNYGPFGSC" --output predicted.pdb

    # Predict with 6 recycles (recommended for accuracy)
    python use_case_0_structure_prediction.py --sequence "FRLLNYYA" --num_recycles 6 --output benchmark.pdb

    # Use benchmark sequence directly
    python use_case_0_structure_prediction.py --benchmark 1JBL --output 1JBL_predicted.pdb

    # Compact structure with Rg constraint
    python use_case_0_structure_prediction.py --sequence "RVKDGYPF" --add_rg --rg_weight 0.1 --output compact.pdb

    # Structure refinement with soft optimization
    python use_case_0_structure_prediction.py --sequence "GFNYGPFGSC" --soft_iters 50 --output refined.pdb

    # Use GPU
    python use_case_0_structure_prediction.py --sequence "FRLLNYYA" --gpu 0 --output gpu_prediction.pdb
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
import json
from datetime import datetime

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import jax.numpy as jnp
import jax
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.af.alphafold.common import residue_constants


# Benchmark sequences from the AfCycDesign paper
BENCHMARK_SEQUENCES = {
    "1JBL": "GFNYGPFGSC",
    "2MW0": "FRLLNYYA",
    "2LWV": "WTYTYDWFC",
    "5KX1": "CWLPCFGDAC",
}


def add_cyclic_offset(model, offset_type=2):
    """
    Add cyclic offset to connect N and C termini for head-to-tail cyclization.

    This applies a custom N x N cyclic offset matrix to the relative positional
    encoding in AlphaFold, defining the sequence separation between terminal
    residues as 1 or -1.

    Args:
        model: AfDesign model instance
        offset_type: Type of offset (1, 2, or 3)
            1 - Basic cyclic offset
            2 - Signed cyclic offset (default, recommended)
            3 - Enhanced cyclic offset with scaling for long peptides
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


def predict_cyclic_structure(
    sequence: str,
    num_recycles: int = 6,
    offset_type: int = 2,
    add_rg: bool = False,
    rg_weight: float = 0.1,
    soft_iters: int = 0,
    output_file: str = "predicted_cyclic.pdb",
    verbose: bool = True
) -> dict:
    """
    Predict 3D structure of a cyclic peptide from its amino acid sequence.

    This is Use Case 0 from the AfCycDesign paper: Structure Prediction Workflow.

    Workflow:
    1. Apply a custom N x N cyclic offset matrix to the relative positional encoding
    2. Define the sequence separation between terminal residues as 1 or -1
    3. Run AlphaFold prediction with cyclic constraints
    4. Optionally refine with soft optimization iterations
    5. Filter results using pLDDT score (> 0.7 indicates high confidence)

    Args:
        sequence: Amino acid sequence of the cyclic peptide (1-letter codes)
        num_recycles: Number of AlphaFold recycles (default: 6 as per paper)
        offset_type: Type of cyclic offset (1, 2, or 3)
        add_rg: Whether to add radius of gyration loss for compactness
        rg_weight: Weight for Rg loss (default: 0.1)
        soft_iters: Number of soft optimization iterations (0 = just predict, >0 = refine structure)
        output_file: Output PDB file path
        verbose: Whether to print progress

    Returns:
        dict: Results including structure and metrics
    """
    # Validate sequence
    sequence = sequence.upper().strip()
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    invalid = set(sequence) - valid_aa
    if invalid:
        raise ValueError(f"Invalid amino acids in sequence: {invalid}")

    if len(sequence) < 5:
        raise ValueError("Sequence must be at least 5 residues long")

    if verbose:
        print("=" * 60)
        print("AfCycDesign: Cyclic Peptide Structure Prediction")
        print("=" * 60)
        print(f"Sequence: {sequence}")
        print(f"Length: {len(sequence)} residues")
        print(f"Recycles: {num_recycles}")
        print(f"Cyclic offset type: {offset_type}")
        if add_rg:
            print(f"Rg constraint: weight={rg_weight}")
        if soft_iters > 0:
            print(f"Soft optimization: {soft_iters} iterations")
        print("=" * 60)

    # Clear previous models
    clear_mem()

    # Initialize model for hallucination protocol (used for structure prediction)
    af_model = mk_afdesign_model(
        protocol="hallucination",
        num_recycles=num_recycles,
        use_templates=False
    )

    # Prepare inputs with the given sequence length
    af_model.prep_inputs(length=len(sequence))

    # Add cyclic offset for head-to-tail cyclization
    add_cyclic_offset(af_model, offset_type=offset_type)

    # Optionally add radius of gyration loss for compact structures
    if add_rg:
        add_rg_loss(af_model, weight=rg_weight)

    if verbose:
        print(f"\nInitialized model with cyclic constraints")
        print(f"Peptide length: {af_model._len}")
        print(f"Loss weights: {af_model.opt['weights']}")
        if soft_iters > 0:
            print(f"\nRunning structure prediction with {soft_iters} soft iterations...")
        else:
            print(f"\nRunning structure prediction...")

    # Initialize model
    af_model.restart()

    # Set the exact sequence (fixed for prediction, not design)
    af_model.set_seq(seq=sequence)

    # Configure contact loss for cyclic peptide
    af_model.set_opt("con", binary=True, cutoff=21.6875, num=af_model._len, seqsep=0)

    # Set loss weights
    af_model.set_weights(pae=1, plddt=1, con=0.5)

    # Run structure prediction
    if soft_iters > 0:
        # Use soft optimization for structure refinement
        af_model.design_soft(soft_iters)
    else:
        # Just run prediction without optimization
        af_model.predict(verbose=verbose)

    # Get metrics
    metrics = af_model.aux.get('log', {})
    plddt = metrics.get('plddt', 0)
    pae = metrics.get('pae', 1)

    if verbose:
        confidence = "HIGH" if plddt > 0.7 else "MEDIUM" if plddt > 0.5 else "LOW"
        print(f"\npLDDT: {plddt:.3f} ({confidence})")
        print(f"PAE: {pae:.3f}")

    # Save structure
    af_model.save_pdb(output_file)

    if verbose:
        print(f"\nSaved structure to: {output_file}")

    # Build results
    results = {
        "sequence": sequence,
        "length": len(sequence),
        "pdb_file": output_file,
        "metrics": {
            "plddt": float(plddt),
            "pae": float(pae),
        },
        "high_confidence": plddt > 0.7,
        "parameters": {
            "num_recycles": num_recycles,
            "offset_type": offset_type,
            "add_rg": add_rg,
            "rg_weight": rg_weight,
            "soft_iters": soft_iters
        }
    }

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Sequence: {sequence}")
        print(f"pLDDT: {results['metrics']['plddt']:.3f}")
        print(f"PAE: {results['metrics']['pae']:.3f}")
        print(f"High Confidence (pLDDT > 0.7): {'Yes' if results['high_confidence'] else 'No'}")
        print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Predict 3D structure of cyclic peptides from sequence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic prediction with paper-recommended settings
    python use_case_0_structure_prediction.py --sequence "GFNYGPFGSC" --output 1JBL_pred.pdb

    # Benchmark sequence from paper (2MW0)
    python use_case_0_structure_prediction.py --sequence "FRLLNYYA" --num_recycles 6

    # Use a benchmark PDB code directly
    python use_case_0_structure_prediction.py --benchmark 1JBL --output benchmark.pdb

    # Compact structure with Rg constraint
    python use_case_0_structure_prediction.py --sequence "RVKDGYPF" --add_rg --output compact.pdb

    # Structure refinement with soft optimization
    python use_case_0_structure_prediction.py --sequence "GFNYGPFGSC" --soft_iters 50 --output refined.pdb

Benchmark Sequences (from paper):
    1JBL: GFNYGPFGSC (10 residues)
    2MW0: FRLLNYYA (8 residues)
    2LWV: WTYTYDWFC (9 residues)
    5KX1: CWLPCFGDAC (10 residues)
        """
    )

    # Sequence input
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sequence", "-s", type=str,
                       help="Amino acid sequence (1-letter codes)")
    group.add_argument("--benchmark", "-b", type=str, choices=list(BENCHMARK_SEQUENCES.keys()),
                       help="Use benchmark sequence from paper (1JBL, 2MW0, 2LWV, 5KX1)")

    # Model parameters (paper recommendations)
    parser.add_argument("--num_recycles", "-r", type=int, default=6,
                       help="Number of AlphaFold recycles (default: 6, paper recommendation)")

    # Cyclic offset
    parser.add_argument("--offset_type", type=int, choices=[1, 2, 3], default=2,
                       help="Cyclic offset type (default: 2, signed offset)")

    # Compactness constraint
    parser.add_argument("--add_rg", action="store_true",
                       help="Add radius of gyration loss for compact structures")
    parser.add_argument("--rg_weight", type=float, default=0.1,
                       help="Weight for Rg loss (default: 0.1)")

    # Optimization
    parser.add_argument("--soft_iters", type=int, default=0,
                       help="Soft optimization iterations (default: 0, just predict; >0 for refinement)")

    # Output
    parser.add_argument("--output", "-o", type=str, default="predicted_cyclic.pdb",
                       help="Output PDB file (default: predicted_cyclic.pdb)")
    parser.add_argument("--save_json", action="store_true",
                       help="Save results as JSON alongside PDB")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress verbose output")

    # GPU arguments (parsed early by gpu_utils, included here for --help)
    parser.add_argument("--gpu", type=int, metavar="ID",
                       help="GPU device ID to use (0, 1, etc.)")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU mode")

    args = parser.parse_args()

    # Get sequence
    if args.benchmark:
        sequence = BENCHMARK_SEQUENCES[args.benchmark]
        if not args.quiet:
            print(f"Using benchmark sequence {args.benchmark}: {sequence}")
    else:
        sequence = args.sequence

    try:
        results = predict_cyclic_structure(
            sequence=sequence,
            num_recycles=args.num_recycles,
            offset_type=args.offset_type,
            add_rg=args.add_rg,
            rg_weight=args.rg_weight,
            soft_iters=args.soft_iters,
            output_file=args.output,
            verbose=not args.quiet
        )

        # Save JSON if requested
        if args.save_json:
            json_file = args.output.rsplit('.', 1)[0] + '.json'
            json_results = dict(results)
            json_results['timestamp'] = datetime.now().isoformat()

            with open(json_file, 'w') as f:
                json.dump(json_results, f, indent=2)

            if not args.quiet:
                print(f"Saved results to: {json_file}")

        # Final status
        if results['high_confidence']:
            if not args.quiet:
                print("\nPrediction completed with HIGH confidence (pLDDT > 0.7)")
            return 0
        else:
            if not args.quiet:
                print("\nWarning: Prediction has LOW/MEDIUM confidence (pLDDT <= 0.7)")
                print("Consider trying different parameters or checking the sequence.")
            return 0

    except Exception as e:
        import traceback
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
