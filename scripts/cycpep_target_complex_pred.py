#!/usr/bin/env python3
"""
Cyclic Peptide + Target Complex Structure Prediction

Predicts the 3D structure of a cyclic peptide bound to a target protein using
AlphaFold with cyclic offset constraints. Unlike binder *design* which optimizes
a new sequence, this tool runs AlphaFold forward pass for a given peptide sequence
to predict how it binds to the target.

Defaults follow the RFpeptides paper validation settings (AfCycDesign v1.1.1):
- use_multimer=True: Enables simultaneous peptide+target prediction
- use_initial_guess=True: Uses target coords as structural starting point
- target_flexible=True: AF predicts entire complex conformation
- Recycles: 6 recycles for accurate structure prediction

Based on: peptide_binder_design.ipynb with cyclic modifications (prediction only)
Reference: Stephen Rettie et al., doi: https://doi.org/10.1101/2023.02.25.529956

Key Parameters:
- Recycles: 6 recycles for accurate structure prediction (default)
- Cyclic Offset: Applied ONLY to peptide chain (not target)
- Dropout: False for deterministic prediction
- No optimization: Just forward pass prediction

Usage:
    # Predict complex structure (uses RFpeptides defaults: multimer, initial_guess, flexible)
    python cycpep_target_complex_pred.py --pdb 4HFZ --target_chain A --peptide_seq "FSDLWKLLPEN" --output complex.pdb

    # With hotspot residues
    python cycpep_target_complex_pred.py --pdb 2FLU --target_chain A --peptide_seq "DEETGE" --hotspot "20-30" --output keap1_complex.pdb

    # Disable multimer and flexible target (non-default)
    python cycpep_target_complex_pred.py --pdb 4HFZ --target_chain A --peptide_seq "FSDLWKLLPEN" --no_multimer --no_target_flexible --output complex.pdb

    # Batch prediction for multiple sequences
    python cycpep_target_complex_pred.py --pdb 4HFZ --target_chain A --peptide_seqs "FSDLWKLLPEN,FSDLWKLLPEA,FSDLWKLLPES" --output_dir results/

    # Use GPU
    python cycpep_target_complex_pred.py --pdb 4HFZ --target_chain A --peptide_seq "FSDLWKLLPEN" --gpu 0 --output complex.pdb
"""

# ==============================================================================
# GPU Configuration (MUST be set before importing JAX)
# ==============================================================================
import os
import sys

# Add scripts directory to path for gpu_utils import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_utils import auto_setup_gpu
auto_setup_gpu()

# ==============================================================================
# Now safe to import JAX and other libraries
# ==============================================================================
import argparse
import warnings
import re
import json
from datetime import datetime
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


def predict_complex(pdb_file, peptide_seq, target_chain="A",
                    target_hotspot=None, target_flexible=True,
                    use_multimer=True, num_recycles=6, num_models=2,
                    output_file="complex_prediction.pdb",
                    data_dir=None, verbose=True):
    """
    Predict the 3D structure of a cyclic peptide bound to a target protein.

    Runs AlphaFold forward pass (no optimization) with cyclic offset applied
    only to the peptide chain. This predicts how a known peptide sequence
    binds to the target, rather than designing a new sequence.

    Defaults follow the RFpeptides paper validation settings:
    - use_multimer=True for simultaneous peptide+target prediction
    - use_initial_guess=True (always on) to use target coords as starting point
    - target_flexible=True to let AF predict entire complex conformation

    Args:
        pdb_file: Path to target protein PDB file
        peptide_seq: Amino acid sequence of the cyclic peptide (1-letter codes)
        target_chain: Chain ID of target protein (default: "A")
        target_hotspot: Restrict binding to specific positions (e.g., "1-10,12,15")
        target_flexible: Allow target backbone flexibility (default: True, paper setting)
        use_multimer: Use AlphaFold-multimer (default: True, paper setting)
        num_recycles: Number of AF2 recycles (default: 6 for prediction accuracy)
        num_models: Number of models to use (default: 2)
        output_file: Output PDB file path
        data_dir: Directory containing AlphaFold model parameters (default: auto-detect)
        verbose: Whether to print progress

    Returns:
        dict: Results including metrics and quality assessment
    """
    # Clear previous models
    clear_mem()

    # Set default data_dir to project root (parent of scripts/)
    if data_dir is None:
        data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Validate peptide sequence
    peptide_seq = re.sub("[^A-Z]", "", peptide_seq.upper())
    if len(peptide_seq) < 5:
        raise ValueError(f"Peptide sequence must be at least 5 residues (got {len(peptide_seq)})")
    if len(peptide_seq) > 50:
        raise ValueError(f"Peptide sequence must be at most 50 residues (got {len(peptide_seq)})")

    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    invalid = set(peptide_seq) - valid_aa
    if invalid:
        raise ValueError(f"Invalid amino acids in sequence: {invalid}")

    if verbose:
        print("=" * 60)
        print("AfCycDesign: Complex Structure Prediction")
        print("=" * 60)
        print(f"Target PDB: {pdb_file}")
        print(f"Target chain: {target_chain}")
        print(f"Peptide sequence: {peptide_seq}")
        print(f"Peptide length: {len(peptide_seq)} residues")
        print(f"Hotspot: {target_hotspot or 'None (full interface)'}")
        print(f"Recycles: {num_recycles}")
        print(f"Models: {num_models}")
        print(f"Target flexible: {target_flexible}")
        print(f"Use multimer: {use_multimer}")
        print(f"Use initial guess: True")
        print("")
        print("NOTE: Cyclic offset applied to peptide chain ONLY (not target)")
        print("NOTE: Prediction mode (no sequence optimization)")
        print("NOTE: Defaults follow RFpeptides paper validation settings")
        print("=" * 60)

    # Initialize model for binder protocol (used for complex prediction)
    model = mk_afdesign_model(
        protocol="binder",
        use_multimer=use_multimer,
        use_initial_guess=True,
        num_recycles=num_recycles,
        recycle_mode="sample",
        data_dir=data_dir
    )

    # Prepare inputs
    model.prep_inputs(
        pdb_filename=pdb_file,
        chain=target_chain,
        binder_len=len(peptide_seq),
        hotspot=target_hotspot,
        use_multimer=use_multimer,
        rm_target_seq=target_flexible,
        ignore_missing=False
    )

    # Add cyclic constraint for peptide ONLY (not target)
    add_cyclic_offset(model, offset_type=2)

    if verbose:
        print(f"\nTarget length: {model._target_len}")
        print(f"Peptide length: {model._binder_len}")

    # Set the peptide sequence (no optimization, just prediction)
    model.restart(seq=peptide_seq)

    # Select models
    models = model._model_names[:num_models]

    if verbose:
        print(f"\nRunning forward pass prediction (no optimization)...")
        print(f"Using models: {models}")

    # Run prediction - forward pass only, no optimization
    model.predict(
        num_recycles=num_recycles,
        num_models=num_models,
        models=models,
        verbose=verbose
    )

    # Save predicted complex structure
    model.save_pdb(output_file)

    # Extract metrics from prediction
    metrics = model.aux.get("log", {})
    plddt = metrics.get("plddt", 0)
    pae = metrics.get("pae", 1)
    i_pae = metrics.get("i_pae", metrics.get("pae", 1))
    i_con = metrics.get("i_con", 0)
    ptm = metrics.get("pTM", metrics.get("ptm", 0))
    i_ptm = metrics.get("i_pTM", metrics.get("iptm", 0))

    # Quality assessment
    high_confidence = plddt > 0.7
    good_interface = i_pae < 0.15
    strong_interface = i_pae < 0.11

    if verbose:
        print(f"\n{'=' * 60}")
        print("RESULTS")
        print("=" * 60)
        print(f"Output file: {output_file}")
        print(f"\nMetrics:")
        print(f"  pLDDT: {plddt:.4f}")
        print(f"  PAE: {pae:.4f}")
        print(f"  iPAE (Interface PAE): {i_pae:.4f}")
        print(f"  Interface contacts: {i_con:.4f}")
        print(f"  pTM: {ptm:.4f}")
        print(f"  ipTM: {i_ptm:.4f}")
        print(f"\nQuality Assessment:")
        print(f"  pLDDT > 0.7 (high confidence): {'PASS' if high_confidence else 'FAIL'}")
        print(f"  iPAE < 0.15 (good interface):  {'PASS' if good_interface else 'FAIL'}")
        print(f"  iPAE < 0.11 (strong interface): {'PASS' if strong_interface else 'FAIL'}")
        print("=" * 60)

    results = {
        "peptide_sequence": peptide_seq,
        "peptide_length": len(peptide_seq),
        "target_pdb": pdb_file,
        "target_chain": target_chain,
        "target_len": model._target_len,
        "pdb_file": output_file,
        "metrics": {
            "plddt": float(plddt),
            "pae": float(pae),
            "i_pae": float(i_pae),
            "i_con": float(i_con),
            "ptm": float(ptm),
            "i_ptm": float(i_ptm),
        },
        "quality": {
            "high_confidence": high_confidence,
            "good_interface": good_interface,
            "strong_interface": strong_interface,
        },
        "parameters": {
            "num_recycles": num_recycles,
            "num_models": num_models,
            "target_flexible": target_flexible,
            "use_multimer": use_multimer,
            "use_initial_guess": True,
        }
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Predict 3D structure of cyclic peptide + target protein complex"
    )
    parser.add_argument("--pdb", type=str, help="Path to target protein PDB file")
    parser.add_argument("--pdb_code", type=str, help="4-letter PDB code to download")
    parser.add_argument("--target_chain", type=str, default="A",
                       help="Target protein chain ID (default: A)")

    # Peptide sequence input
    seq_group = parser.add_mutually_exclusive_group(required=True)
    seq_group.add_argument("--peptide_seq", type=str,
                          help="Cyclic peptide amino acid sequence")
    seq_group.add_argument("--peptide_seqs", type=str,
                          help="Comma-separated peptide sequences for batch prediction")

    # Binding options
    parser.add_argument("--hotspot", type=str,
                       help="Target hotspot residues (e.g., '1-10,12,15')")
    parser.add_argument("--target_flexible", action="store_true", default=True,
                       help="Allow target backbone flexibility (default: True, paper setting)")
    parser.add_argument("--no_target_flexible", dest="target_flexible", action="store_false",
                       help="Disable target backbone flexibility")
    parser.add_argument("--use_multimer", action="store_true", default=True,
                       help="Use AlphaFold-multimer (default: True, paper setting)")
    parser.add_argument("--no_multimer", dest="use_multimer", action="store_false",
                       help="Disable AlphaFold-multimer")

    # Prediction parameters
    parser.add_argument("--num_recycles", type=int, default=6,
                       help="Number of AF2 recycles (default: 6 for prediction accuracy)")
    parser.add_argument("--num_models", type=int, default=2,
                       help="Number of models to use (default: 2)")

    # Output
    parser.add_argument("--output", "-o", type=str, default="complex_prediction.pdb",
                       help="Output PDB file (default: complex_prediction.pdb)")
    parser.add_argument("--output_dir", type=str,
                       help="Output directory for batch mode")
    parser.add_argument("--save_json", action="store_true",
                       help="Save metrics as JSON alongside PDB")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress verbose output")

    # Model parameters directory
    parser.add_argument("--data_dir", type=str, default=None,
                       help="Directory containing AlphaFold params/ folder (default: auto-detect)")

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

        # Determine sequences to predict
        if args.peptide_seqs:
            sequences = [s.strip() for s in args.peptide_seqs.split(",") if s.strip()]
        else:
            sequences = [args.peptide_seq]

        # Batch mode
        if len(sequences) > 1:
            output_dir = args.output_dir or "complex_predictions"
            os.makedirs(output_dir, exist_ok=True)

            if not args.quiet:
                print(f"=== AfCycDesign: Batch Complex Prediction ===")
                print(f"Target: {pdb_file}")
                print(f"Sequences: {len(sequences)}")
                print(f"Output dir: {output_dir}")
                print()

            all_results = []
            for i, seq in enumerate(sequences):
                seq = re.sub("[^A-Z]", "", seq.upper())
                out_pdb = os.path.join(output_dir, f"complex_{i+1}_{seq[:8]}.pdb")

                if not args.quiet:
                    print(f"\n--- Sequence {i+1}/{len(sequences)}: {seq} ---")

                result = predict_complex(
                    pdb_file=pdb_file,
                    peptide_seq=seq,
                    target_chain=args.target_chain,
                    target_hotspot=args.hotspot,
                    target_flexible=args.target_flexible,
                    use_multimer=args.use_multimer,
                    num_recycles=args.num_recycles,
                    num_models=args.num_models,
                    output_file=out_pdb,
                    data_dir=args.data_dir,
                    verbose=not args.quiet
                )
                all_results.append(result)

            # Save batch summary
            if args.save_json:
                summary_file = os.path.join(output_dir, "batch_summary.json")
                summary = {
                    "target_pdb": pdb_file,
                    "target_chain": args.target_chain,
                    "num_sequences": len(sequences),
                    "timestamp": datetime.now().isoformat(),
                    "results": [
                        {k: v for k, v in r.items() if k != "model"}
                        for r in all_results
                    ]
                }
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                if not args.quiet:
                    print(f"\nSaved batch summary to: {summary_file}")

            if not args.quiet:
                print(f"\n=== Batch Complete: {len(all_results)} predictions ===")
                for r in all_results:
                    q = r["quality"]
                    m = r["metrics"]
                    tag = "STRONG" if q["strong_interface"] else "GOOD" if q["good_interface"] else "WEAK"
                    print(f"  {r['peptide_sequence']}: pLDDT={m['plddt']:.3f} iPAE={m['i_pae']:.3f} [{tag}]")

        else:
            # Single sequence mode
            seq = sequences[0]

            if not args.quiet:
                print("=== AfCycDesign: Complex Structure Prediction ===")
                print(f"Target: {pdb_file}")
                print(f"Peptide: {seq}")
                print(f"Output: {args.output}")

            result = predict_complex(
                pdb_file=pdb_file,
                peptide_seq=seq,
                target_chain=args.target_chain,
                target_hotspot=args.hotspot,
                target_flexible=args.target_flexible,
                use_multimer=args.use_multimer,
                num_recycles=args.num_recycles,
                num_models=args.num_models,
                output_file=args.output,
                data_dir=args.data_dir,
                verbose=not args.quiet
            )

            # Save JSON if requested
            if args.save_json:
                json_file = args.output.rsplit('.', 1)[0] + '.json'
                json_result = dict(result)
                json_result['timestamp'] = datetime.now().isoformat()
                with open(json_file, 'w') as f:
                    json.dump(json_result, f, indent=2)
                if not args.quiet:
                    print(f"Saved metrics to: {json_file}")

        return 0

    except Exception as e:
        import traceback
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
