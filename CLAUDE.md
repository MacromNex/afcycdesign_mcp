# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AfCycDesign MCP — a FastMCP server exposing cyclic peptide computational design tools based on AlphaFold and ColabDesign. It provides 13 MCP tools for structure prediction, sequence design, de novo hallucination, binder design, and complex structure prediction of cyclic peptides, with GPU-intensive tasks managed through a FIFO job queue.

## Commands

### Setup
```bash
./quick_setup.sh                    # Full 7-step environment setup (Python 3.10, ColabDesign, JAX+CUDA, RDKit, fastmcp)
```

### Running the Server
```bash
./env/bin/python src/server.py      # Direct execution
mamba run -p ./env python src/server.py  # Via mamba
fastmcp dev src/server.py           # Development mode with inspector
```

### Running Scripts Directly (without MCP)
```bash
./env/bin/python scripts/cyclic_structure_prediction.py --sequence GFNYGPFGSC --output results/
./env/bin/python scripts/cyclic_hallucination.py --length 10 --output results/
./env/bin/python scripts/cyclic_fixbb_design.py --pdb input.pdb --output results/
./env/bin/python scripts/cyclic_binder_design.py --target_pdb target.pdb --binder_len 10 --output results/
./env/bin/python scripts/cycpep_target_complex_pred.py --pdb target.pdb --target_chain A --peptide_seq FSDLWKLLPEN --output results/
# ^ Defaults: --use_multimer, --target_flexible, use_initial_guess=True (RFpeptides paper settings)
# To disable: --no_multimer --no_target_flexible
```

### Tests
```bash
python tests/run_integration_tests.py src/server.py ./env
# Outputs JSON report to reports/step7_integration_tests.json
```

### MCP Registration
```bash
fastmcp install claude-code src/server.py --name afcycdesign_mcp
```

## Architecture

### Three-Layer Design

1. **MCP Layer** (`src/server.py`): FastMCP server with 13 `@mcp.tool()` decorated functions. Design tools (`submit_*`) are async job submissions; job management tools query/control the queue; utility tools provide validation and server info.

2. **Job Queue** (`src/jobs/manager.py`): Thread-safe FIFO queue executing one GPU job at a time. Jobs persist as JSON metadata in `jobs/[job_id]/` with `metadata.json`, `job.log`, and `result.json`. Recovers pending/running jobs on server restart.

3. **Computation Scripts** (`scripts/cyclic_*.py`, `scripts/cycpep_*.py`): Five standalone Python scripts using JAX + ColabDesign for the actual AlphaFold-based computations. Each script is invoked as a subprocess by the job manager.

### GPU Handling

`scripts/gpu_utils.py` **must be imported before JAX** in all computation scripts. It configures `LD_LIBRARY_PATH` for CUDA/cuDNN and may re-execute the process if the GPU environment isn't set. Scripts accept `--gpu`, `--gpu_mem_fraction`, and `--cpu` flags.

### Cyclic Offset System

All scripts implement cyclic head-to-tail cyclization by modifying AlphaFold's relative positional encoding. Three offset types exist; **type 2** (signed cyclic offset) is the default and most balanced.

### Key Dependencies (version-pinned)

- JAX 0.4.28 + jaxlib 0.4.28+cuda12.cudnn89
- ColabDesign v1.1.1 (vendored in `repo/ColabDesign/`)
- chex 0.1.86, optax 0.2.2
- nvidia-cudnn-cu12 8.9.7.29
- fastmcp, loguru, click, tqdm, RDKit

### Quality Thresholds

| Metric | Excellent | Good | Acceptable |
|--------|-----------|------|------------|
| pLDDT  | >0.90     | >0.70| >0.50      |
| PAE    | <0.10     | <0.30| <0.50      |

Binder design uses iPAE (interface PAE) with a strict threshold of 0.11-0.15.

### Complex Prediction Defaults (RFpeptides Paper)

`scripts/cycpep_target_complex_pred.py` and `submit_complex_prediction` use settings from the RFpeptides paper validation:
- `use_multimer=True` — simultaneous peptide+target prediction
- `use_initial_guess=True` — target coords as structural starting point (always on, not exposed as parameter)
- `target_flexible=True` — AF predicts entire complex conformation
- Disable with `--no_multimer` / `--no_target_flexible` CLI flags, or `use_multimer=False` / `target_flexible=False` in MCP

### Peptide Constraints

- Length range: 5-50 residues (recommended 6-20)
- Cysteine (C) excluded by default to avoid disulfide interference
