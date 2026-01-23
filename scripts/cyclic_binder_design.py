#!/usr/bin/env python3
"""
Use Case 3: Cyclic Peptide Binder Design (Motif Grafting)

This is the most complex workflow, used to design cyclic peptide binders for
protein targets like MDM2 or Keap1.

Based on: peptide_binder_design.ipynb with cyclic modifications
Reference: Stephen Rettie et al., doi: https://doi.org/10.1101/2023.02.25.529956

Key Parameters (from paper):
- Input Files:
  - Target Protein PDB: e.g., MDM2 (4HFZ), Keap1 (2FLU)
  - Binding Motif: Short functional segments (p53 motif FSDLW, Nrf2 hot loop DEETGE)
  - Scaffold Library: ~24,000 hallucinated cyclic peptides with pLDDT > 0.9

- Grafting: Rosetta MotifGraft with 1.0 A RMSD tolerance

- Sequence Design: 3-4 iterative rounds of:
  - ProteinMPNN for sequence optimization
  - Rosetta energy minimization (REF2015 score function)

- Filtering Metrics:
  - iPAE (Interface PAE): < 0.11 or 0.15
  - ddG: < -30 kcal/mol
  - SAP score: < 30
  - CMS (Contact Molecular Surface): > 300
  - Binding mode RMSD: < 1.5 A

Workflow (from paper):
1. Graft the functional motif onto stable hallucinated scaffolds
2. Redesign non-motif residues using ProteinMPNN
3. Use AfCycDesign to predict target-peptide complex
   (apply cyclic offset ONLY to binder chain)
4. Verify binding mode via iPAE and RMSD < 1.5 A

Usage:
    # Basic binder design
    python use_case_3_cyclic_binder_design.py --pdb 4HFZ --target_chain A --binder_len 14 --output mdm2_binder.pdb

    # Target specific hotspot residues
    python use_case_3_cyclic_binder_design.py --pdb 2FLU --target_chain A --binder_len 12 --hotspot "20-30" --output keap1_binder.pdb

    # With initial motif sequence
    python use_case_3_cyclic_binder_design.py --pdb 4HFZ --target_chain A --binder_seq "FSDLWKLLPEN" --output motif_binder.pdb

    # Use GPU
    python use_case_3_cyclic_binder_design.py --pdb 4HFZ --target_chain A --binder_len 12 --gpu 0 --output gpu_binder.pdb
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
import re
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import jax.numpy as jnp
import jax
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.shared.utils import copy_dict
from scipy.special import softmax


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


# Paper-recommended filtering thresholds
FILTERING_THRESHOLDS = {
    "ipae": 0.15,         # Interface PAE: < 0.15 (or 0.11 for strict)
    "ipae_strict": 0.11,  # Strict iPAE threshold
    "ddg": -30.0,         # ddG: < -30 kcal/mol
    "sap": 30.0,          # SAP score: < 30
    "cms": 300.0,         # CMS: > 300
    "rmsd": 1.5,          # Binding mode RMSD: < 1.5 A
}


def design_cyclic_peptide_binder(pdb_file, target_chain="A", binder_len=14,
                                binder_seq=None, target_hotspot=None,
                                target_flexible=False, use_multimer=False,
                                optimizer="pssm_semigreedy", num_recycles=0,
                                num_models=2, output_file="cyclic_binder.pdb",
                                ipae_threshold=0.15, verbose=True):
    """
    Design a cyclic peptide binder for a target protein structure.

    This is Use Case 4 from the AfCycDesign paper: Peptide Binder Design (Motif Grafting).

    The workflow designs cyclic peptides that bind to protein targets with the
    cyclic offset applied ONLY to the binder chain (not the target).

    Paper-recommended workflow:
    1. Graft functional motif onto stable hallucinated scaffolds
    2. Redesign non-motif residues using ProteinMPNN
    3. Predict target-peptide complex with AfCycDesign
    4. Verify binding mode via iPAE and RMSD < 1.5 A

    Filtering thresholds (from paper):
    - iPAE < 0.15 (or 0.11 for strict)
    - ddG < -30 kcal/mol
    - SAP < 30
    - CMS > 300
    - RMSD < 1.5 A

    Args:
        pdb_file: Path to target protein PDB file
        target_chain: Chain ID of target protein
        binder_len: Length of binder peptide to design
        binder_seq: Initial binder sequence (e.g., motif like "FSDLW")
        target_hotspot: Restrict binding to specific positions (e.g., "1-10,12,15")
        target_flexible: Allow target backbone flexibility
        use_multimer: Use AlphaFold-multimer
        optimizer: Optimization method
        num_recycles: Number of AF2 recycles
        num_models: Number of models to use
        output_file: Output PDB file path
        ipae_threshold: Interface PAE threshold for quality filtering (default: 0.15)
        verbose: Whether to print progress

    Returns:
        dict: Results including sequences, metrics, and quality assessment
    """
    # Clear previous models
    clear_mem()

    # Process binder sequence if provided
    if binder_seq:
        binder_seq = re.sub("[^A-Z]", "", binder_seq.upper())
        if len(binder_seq) > 0:
            binder_len = len(binder_seq)
        else:
            binder_seq = None

    # Prepare inputs
    x = {
        "pdb_filename": pdb_file,
        "chain": target_chain,
        "binder_len": binder_len,
        "hotspot": target_hotspot,
        "use_multimer": use_multimer,
        "rm_target_seq": target_flexible
    }

    # Initialize model for binder design
    model = mk_afdesign_model(
        protocol="binder",
        use_multimer=use_multimer,
        num_recycles=num_recycles,
        recycle_mode="sample"
    )

    model.prep_inputs(**x, ignore_missing=False)

    # Add cyclic constraint for binder ONLY (not target)
    # This is a key distinction from the paper - cyclic offset applies only to binder chain
    add_cyclic_offset(model, offset_type=2)

    if verbose:
        print("=" * 60)
        print("AfCycDesign: Binder Design (Use Case 3)")
        print("=" * 60)
        print(f"Target PDB: {pdb_file}")
        print(f"Target chain: {target_chain}")
        print(f"Target length: {model._target_len}")
        print(f"Binder length: {model._binder_len}")
        print(f"Hotspot: {target_hotspot or 'None (full interface)'}")
        print(f"Optimizer: {optimizer}")
        print(f"iPAE threshold: < {ipae_threshold}")
        print("")
        print("NOTE: Cyclic offset applied to binder chain ONLY (not target)")
        print("=" * 60)

    # Set optimizer and models
    models = model._model_names[:num_models]
    flags = {
        "num_recycles": num_recycles,
        "models": models,
        "dropout": True
    }

    # Restart model with initial sequence
    model.restart(seq=binder_seq)

    if verbose:
        print(f"Running {optimizer} optimization...")

    # Run optimization based on method
    if optimizer == "3stage":
        model.design_3stage(120, 60, 10, **flags)
        pssm = softmax(model._tmp["seq_logits"], -1)

    elif optimizer == "pssm_semigreedy":
        model.design_pssm_semigreedy(120, 32, **flags)
        pssm = softmax(model._tmp["seq_logits"], 1)

    elif optimizer == "semigreedy":
        model.design_pssm_semigreedy(0, 32, **flags)
        pssm = None

    elif optimizer == "pssm":
        model.design_logits(120, e_soft=1.0, num_models=1, ramp_recycles=True, **flags)
        model.design_soft(32, num_models=1, **flags)
        flags.update({"dropout": False, "save_best": True})
        model.design_soft(10, num_models=num_models, **flags)
        pssm = softmax(model.aux["seq"]["logits"], -1)

    else:
        # logits, soft, or hard optimization
        optimization_methods = {
            "logits": model.design_logits,
            "soft": model.design_soft,
            "hard": model.design_hard
        }

        if optimizer in optimization_methods:
            opt_fn = optimization_methods[optimizer]
            opt_fn(120, num_models=1, ramp_recycles=True, **flags)
            flags.update({"dropout": False, "save_best": True})
            opt_fn(10, num_models=num_models, **flags)
            pssm = softmax(model.aux["seq"]["logits"], -1)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    # Save results
    model.save_pdb(output_file)

    # Get metrics
    best_aux = model._tmp.get("best", {}).get("aux", {})
    metrics = best_aux.get("log", {})

    # Extract key metrics for quality assessment
    plddt = metrics.get("plddt", 0)
    pae = metrics.get("pae", 1)
    i_pae = metrics.get("i_pae", metrics.get("pae", 1))  # Interface PAE
    i_con = metrics.get("i_con", 0)  # Interface contacts

    # Quality assessment based on paper thresholds
    # iPAE < 0.15 (or 0.11 for strict)
    passes_ipae = i_pae < ipae_threshold

    if verbose:
        print(f"\n{'=' * 60}")
        print("RESULTS")
        print("=" * 60)
        print(f"Output file: {output_file}")
        print(f"\nMetrics:")
        print(f"  pLDDT: {plddt:.3f}")
        print(f"  PAE: {pae:.3f}")
        print(f"  iPAE (Interface PAE): {i_pae:.3f}")
        print(f"  Interface contacts: {i_con:.3f}")
        print(f"\nQuality Assessment (Paper Thresholds):")
        print(f"  iPAE < {ipae_threshold}: {'PASS' if passes_ipae else 'FAIL'}")
        print(f"\nNote: For full validation, also check:")
        print(f"  - ddG < -30 kcal/mol (Rosetta)")
        print(f"  - SAP < 30 (Rosetta)")
        print(f"  - CMS > 300 (Rosetta)")
        print(f"  - Binding mode RMSD < 1.5 A")
        print("=" * 60)

    # Get designed sequences
    sequences = model.get_seqs()

    results = {
        "sequences": sequences,
        "pdb_file": output_file,
        "target_len": model._target_len,
        "binder_len": model._binder_len,
        "pssm": pssm,
        "metrics": {
            "plddt": float(plddt),
            "pae": float(pae),
            "i_pae": float(i_pae),
            "i_con": float(i_con),
        },
        "quality": {
            "passes_ipae": passes_ipae,
            "ipae_threshold": ipae_threshold,
        },
        "filtering_thresholds": FILTERING_THRESHOLDS,
        "model": model
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Design cyclic peptide binders for target proteins"
    )
    parser.add_argument("--pdb", type=str, help="Path to target protein PDB file")
    parser.add_argument("--pdb_code", type=str, help="4-letter PDB code to download")
    parser.add_argument("--target_chain", type=str, default="A",
                       help="Target protein chain ID (default: A)")
    parser.add_argument("--binder_len", type=int, default=14,
                       help="Length of binder peptide (default: 14)")
    parser.add_argument("--binder_seq", type=str,
                       help="Initial binder sequence (optional)")
    parser.add_argument("--hotspot", type=str,
                       help="Target hotspot residues (e.g., '1-10,12,15')")
    parser.add_argument("--target_flexible", action="store_true",
                       help="Allow target backbone flexibility")
    parser.add_argument("--use_multimer", action="store_true",
                       help="Use AlphaFold-multimer")
    parser.add_argument("--optimizer", type=str,
                       choices=["pssm_semigreedy", "3stage", "semigreedy", "pssm",
                               "logits", "soft", "hard"],
                       default="pssm_semigreedy",
                       help="Optimization method (default: pssm_semigreedy)")
    parser.add_argument("--num_recycles", type=int, default=0,
                       help="Number of AF2 recycles (default: 0)")
    parser.add_argument("--num_models", type=int, default=2,
                       help="Number of models to use (default: 2)")
    parser.add_argument("--output", type=str, default="cyclic_binder.pdb",
                       help="Output PDB file (default: cyclic_binder.pdb)")
    parser.add_argument("--ipae_threshold", type=float, default=0.15,
                       help="Interface PAE threshold for quality filtering (default: 0.15, use 0.11 for strict)")
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
            print("=== AfCycDesign: Cyclic Peptide Binder Design ===")
            print(f"Target: {pdb_file}")
            print(f"Target chain: {args.target_chain}")
            print(f"Binder length: {args.binder_len}")
            print(f"Optimizer: {args.optimizer}")
            print(f"Output: {args.output}")

        results = design_cyclic_peptide_binder(
            pdb_file=pdb_file,
            target_chain=args.target_chain,
            binder_len=args.binder_len,
            binder_seq=args.binder_seq,
            target_hotspot=args.hotspot,
            target_flexible=args.target_flexible,
            use_multimer=args.use_multimer,
            optimizer=args.optimizer,
            num_recycles=args.num_recycles,
            num_models=args.num_models,
            output_file=args.output,
            ipae_threshold=args.ipae_threshold,
            verbose=not args.quiet
        )

        if not args.quiet:
            print("\n=== Designed Binder Sequences ===")
            for i, seq in enumerate(results["sequences"]):
                print(f"Binder sequence {i+1}: {seq}")

            # Quality summary
            if results["quality"]["passes_ipae"]:
                print(f"\nBinder PASSED iPAE filter (< {results['quality']['ipae_threshold']})")
            else:
                print(f"\nBinder FAILED iPAE filter (>= {results['quality']['ipae_threshold']})")

        return 0

    except Exception as e:
        import traceback
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())