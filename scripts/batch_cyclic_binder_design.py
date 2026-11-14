#!/usr/bin/env python3
"""
Batched cyclic peptide binder design.

Identical protocol to cyclic_binder_design.py, but builds the AlphaFold model
once and reuses it across many designs. Everything from clear_mem() through
add_cyclic_offset() depends only on (target, binder_len), so running N designs
in one process pays the setup cost once instead of N times: AF2 parameters are
loaded once and XLA compiles the kernels once.

Per design only model.restart(seed=...) and the optimizer run, which is exactly
what cyclic_binder_design.py does after its own setup - so for a given seed the
output must match the single-design script.

Nothing here is specific to a target: the target PDB, chain, hotspots and binder
length are all arguments. One process handles one (target, binder_len) pair
because the model is built for those shapes; switching either one needs a new
process, where the persistent XLA cache (see gpu_utils) then supplies the
already-compiled kernels. So batching amortises within a target, and the cache
amortises across targets.

Usage:
    python batch_cyclic_binder_design.py --pdb <target>.pdb --target_chain A \\
        --binder_len 14 --hotspot 1,2,3 --seeds 1000-1009 \\
        --output_dir designs/ --prefix <target>_L14 --gpu 0
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_utils import auto_setup_gpu
auto_setup_gpu()

import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from colabdesign import mk_afdesign_model, clear_mem
from scipy.special import softmax

from cyclic_binder_design import add_cyclic_offset


def parse_seeds(spec):
    """'34000-34009' or '1,5,9' -> list of ints."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def build_model(pdb_file, target_chain, binder_len, hotspot, num_recycles,
                use_multimer, target_flexible, data_dir):
    """Setup that is invariant across designs - done once per process."""
    clear_mem()
    model = mk_afdesign_model(
        protocol="binder",
        use_multimer=use_multimer,
        num_recycles=num_recycles,
        recycle_mode="sample",
        data_dir=data_dir,
    )
    model.prep_inputs(
        pdb_filename=pdb_file,
        chain=target_chain,
        binder_len=binder_len,
        hotspot=hotspot,
        use_multimer=use_multimer,
        rm_target_seq=target_flexible,
        ignore_missing=False,
    )
    # Cyclic offset on the binder chain ONLY, never the target.
    add_cyclic_offset(model, offset_type=2)
    return model


def run_design(model, seed, optimizer, flags, out_pdb, ipae_threshold):
    """One design on an already-built model. Mirrors cyclic_binder_design.py."""
    model.restart(seq=None, seed=seed)

    if optimizer == "pssm_semigreedy":
        model.design_pssm_semigreedy(120, 32, **flags)
    elif optimizer == "3stage":
        model.design_3stage(120, 60, 10, **flags)
    elif optimizer == "semigreedy":
        model.design_pssm_semigreedy(0, 32, **flags)
    else:
        raise ValueError(f"Unsupported optimizer for batch mode: {optimizer}")

    model.save_pdb(out_pdb)

    metrics = model._tmp.get("best", {}).get("aux", {}).get("log", {})
    i_pae = metrics.get("i_pae", metrics.get("pae", 1))
    payload = {
        "plddt": float(metrics.get("plddt", 0)),
        "pae": float(metrics.get("pae", 1)),
        "i_pae": float(i_pae),
        "i_con": float(metrics.get("i_con", 0)),
        "ptm": float(metrics.get("ptm", 0)),
        "i_ptm": float(metrics.get("i_ptm", 0)),
        "loss": float(metrics.get("loss", 0)),
        "passes_ipae": bool(i_pae < ipae_threshold),
        "ipae_threshold": float(ipae_threshold),
    }
    with open(os.path.splitext(out_pdb)[0] + "_metrics.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    seqs = model.get_seqs()
    return seqs[0] if seqs else None, payload


def main():
    p = argparse.ArgumentParser(description="Batched cyclic binder design")
    p.add_argument("--pdb", required=True)
    p.add_argument("--target_chain", default="A")
    p.add_argument("--binder_len", type=int, default=14)
    p.add_argument("--hotspot", default=None)
    p.add_argument("--seeds", required=True, help="e.g. 34000-34009 or 1,5,9")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--prefix", required=True, help="output stem; _<seed> appended")
    p.add_argument("--names", default=None,
                   help="comma-separated output stems, one per seed, used "
                        "instead of <prefix>_<seed> (lets a caller keep its "
                        "own design-id convention)")
    p.add_argument("--optimizer", default="pssm_semigreedy")
    p.add_argument("--num_models", type=int, default=2)
    p.add_argument("--num_recycles", type=int, default=0)
    p.add_argument("--ipae_threshold", type=float, default=0.15)
    p.add_argument("--use_multimer", action="store_true")
    p.add_argument("--target_flexible", action="store_true")
    p.add_argument("--data_dir", default=None)
    # consumed by gpu_utils before JAX import; declared for --help
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--gpu_mem_fraction", type=float, default=None)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    data_dir = args.data_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    seeds = parse_seeds(args.seeds)
    if args.names:
        names = [n.strip() for n in args.names.split(",")]
        if len(names) != len(seeds):
            p.error(f"--names has {len(names)} entries but --seeds has {len(seeds)}")
    else:
        names = [f"{args.prefix}_{s}" for s in seeds]
    os.makedirs(args.output_dir, exist_ok=True)

    t_setup = time.time()
    model = build_model(args.pdb, args.target_chain, args.binder_len,
                        args.hotspot, args.num_recycles, args.use_multimer,
                        args.target_flexible, data_dir)
    setup_s = time.time() - t_setup
    print(f"[batch] model built once in {setup_s:.1f}s for {len(seeds)} designs",
          flush=True)

    flags = {"num_recycles": args.num_recycles,
             "models": model._model_names[:args.num_models],
             "dropout": True}

    summary = []
    for n, (seed, stem) in enumerate(zip(seeds, names), 1):
        out_pdb = os.path.join(args.output_dir, f"{stem}.pdb")
        t0 = time.time()
        seq, metrics = run_design(model, seed, args.optimizer, flags, out_pdb,
                                  args.ipae_threshold)
        dt = time.time() - t0
        summary.append({"seed": seed, "sequence": seq, "elapsed_s": round(dt, 1),
                        "pdb": out_pdb, "metrics": metrics})
        print(f"[batch] {n}/{len(seeds)} seed={seed} {dt:.0f}s {seq}", flush=True)

    total = time.time() - t_setup
    with open(os.path.join(args.output_dir, f"{args.prefix}_batch.json"), "w") as fh:
        json.dump({"setup_s": round(setup_s, 1), "total_s": round(total, 1),
                   "n_designs": len(seeds), "designs": summary}, fh, indent=2)
    print(f"[batch] {len(seeds)} designs in {total:.0f}s "
          f"({total / len(seeds):.0f}s/design, setup {setup_s:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
