# AfCycDesign MCP

> MCP tools for cyclic peptide computational analysis and design using AlphaFold

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Local Usage (Scripts)](#local-usage-scripts)
- [MCP Server Installation](#mcp-server-installation)
- [Using with Claude Code](#using-with-claude-code)
- [Using with Gemini CLI](#using-with-gemini-cli)
- [Available Tools](#available-tools)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

This MCP server provides computational tools for cyclic peptide structure prediction, sequence design, and binder development using the ColabDesign framework. The server offers both fast synchronous operations and long-running asynchronous jobs for comprehensive cyclic peptide research workflows.

### Features
- **3D Structure Prediction**: Predict cyclic peptide using AlphaFold2
- **Sequence Design**: Redesign amino acid sequences for given cyclic backbone structures
- **Binder Development**: Design cyclic peptide binders that bind to target protein structures
- **Batch Processing**: Generate multiple peptides with different parameters in parallel
- **Job Management**: Track long-running computations with status monitoring and log access
- **Quality Assessment**: Comprehensive structural quality metrics (pLDDT, PAE, contacts)


## Installation

### Quick Setup

Run the automated setup script:

```bash
./quick_setup.sh
```

This will create the environment and install all dependencies automatically.

Please view [`quick_setup.sh`](./quick_setup.sh) for manuall install.

## Local Usage (Scripts)

You can use the scripts directly without MCP for local processing.

### Script Examples

#### Predict 3D Structure

```bash
mamba activate ./env
# Basic prediction 
python scripts/cyclic_structure_prediction.py --benchmark 1JBL --output results/struct_pred_basic_1jbl.pdb
 
# Compact structure with Rg constraint 
python scripts/cyclic_structure_prediction.py --sequence "RVKDGYPF" --add_rg --rg_weight 0.1 --output results/struct_pred_compact.pdb
 
# Structure refinement with soft optimization
python scripts/cyclic_structure_prediction.py --sequence "GFNYGPFGSC" --soft_iters 50 --output results/struct_pred_refined.pdb 
```

**Parameters:**
- `--output, -o`: Output PDB file pat
- `--add_rg`: Add radius of gyration constraint for compact structures
- `--soft_iters`: Number of optimization iterations (20 for testing, 50+ for production)

#### Fixed Backbone Design (Fixbb)

```shell
python scripts/cyclic_fixbb_design.py \
    --pdb_code 1JBL \
    --chain A \
    --gpu 0 \
    --output results/fixbb_redesign_1JBL_from_pdb.pdb

python scripts/cyclic_fixbb_design.py \
    --pdb examples/structures/1JBL_chainA.pdb \
    --chain A \
    --add_rg --rg_weight 0.1 \
    --stage_iters 100 100 20 \
    --gpu 0 \
    --output results/fixbb_1JBL_redesigned.pdb
```

**Parameters:**
--`pdb`: Path to input PDB file
--`pdb_code`: 4-letter PDB code to download from RCSB
--`chain`: Chain ID to use (Default: A)
--`output, -o`: Output PDB file path (Default: cyclic_fixbb_designed.pdb)
--`offset_type`: Cyclic offset type (1, 2, or 3) (Default: 2)
--`add_rg`: Add radius of gyration constraint for compact structures (Default: False)
--`rg_weight`: Weight for the Radius of Gyration (Rg) loss function (Default: 0.1)
--`num_recycles`: Number of AlphaFold2 recycles (Default: 0)
--`stage_iters`: 3-stage optimization iterations for logits, soft, and hard stages (Default: 50 50 10)
--`gpu`: GPU device ID to use for acceleration
--`quiet`: Suppress verbose output (minimal output mode)

#### De Novo Hallucination
```shell
 ./env/bin/python scripts/cyclic_hallucination.py \
 --length 12 --rm_aa "C" --plddt_threshold 0.9 \
 --output examples/outputs/hallucinated.pdb

# Basic hallucination - generate 13-residue cyclic peptide 
python scripts/cyclic_hallucination.py \
 --length 13 \
 --gpu 0 \
 --output results/hallucinated_13mer.pdb 
  
# Paper-recommended settings for scaffold library generation 
python scripts/cyclic_hallucination.py \
 --length 13 \
 --rm_aa "C" \
 --plddt_threshold 0.9 \
 --gpu 0 \
 --output results/scaffold_13mer.pdb
  
# Compact structure with Rg constraint and extended optimization 
python scripts/cyclic_hallucination.py \
 --length 15 \
 --rm_aa "C,M" \
 --add_rg --rg_weight 0.15 \
 --soft_iters 100 \
 --stage_iters 100 100 20 \
 --gpu 0 \
 --output results/compact_15mer.pdb 
```
Parameters`:  
 --`length`: Length of peptide to generate (5-50 residues, paper tested 7-16) 
 --`rm_aa`: Amino acids to exclude from design, comma-separated (Default: C) 
 --`output`: Output PDB file path (Default: cyclic_hallucinated.pdb) 
 --`offset_type`: Cyclic offset type (1, 2, or 3) (Default: 2) 
 --`add_rg`: Add radius of gyration constraint for compact structures (Default: False) 
 --`rg_weight`: Weight for the Radius of Gyration (Rg) loss function (Default: 0.1) 
 --`num_recycles`: Number of AlphaFold2 recycles (Default: 0) 
 --`soft_iters`: Iterations for soft pre-design stage (Default: 50)  
 --`stage_iters`: 3-stage optimization iterations for logits, soft, and hard stages (Default: 50 50 10) 
 --`plddt_threshold`: Quality threshold for high-confidence scaffolds (Default: 0.9 as per paper) 
 --`gpu`: GPU device ID to use for acceleration 
 --`cpu`: Force CPU mode  
 --`quiet`: Suppress verbose output (minimal output mode) 

 #### Cyclic Peptide Binder Design
```shell
 # Basic binder design for MDM2 (p53 binding site) 
 python scripts/cyclic_binder_design.py \
 --pdb examples/structures/4HFZ_MDM2.pdb \
 --target_chain A \
 --binder_len 14 \
 --gpu 0 \
 --output results/binder_MDM2.pdb 
 
 # Download target from PDB and design binder 
 python scripts/cyclic_binder_design.py \
 --pdb_code 4HFZ \
 --target_chain A \
 --binder_len 12 \
 --gpu 0 \
 --output results/binder_4HFZ.pdb 
 
 # Design with hotspot specification and strict filtering 
 python scripts/cyclic_binder_design.py \
 --pdb examples/structures/4HFZ_MDM2.pdb \
 --target_chain A \
 --binder_len 14 \
 --hotspot "50-60,70-75"\
 --ipae_threshold 0.11 \
 --gpu 1 \
 --output results/binder_MDM2_hotspot.pdb 
 
 # Design with initial sequence and 3stage optimizer 
 python scripts/cyclic_binder_design.py \
 --pdb examples/structures/2FLU_Keap1.pdb \
 --target_chain A \
 --binder_len 10 \
 --binder_seq "RVKDGYPFAA" \
 --optimizer 3stage \
 --gpu 1 \
 --output results/binder_Keap1.pdb 
```
## MCP Server Installation


## Available Tools
### Quality Thresholds
- **pLDDT**: >0.90 (excellent), >0.70 (good), >0.50 (acceptable)
- **PAE**: <0.10 (excellent), <0.30 (good), <0.50 (acceptable)
- **Typical Runtime**: 2-5 min (small), 15-30 min (large/complex)

### Recommended Starting Points
- **Testing**: length=8, soft_iters=20 (~2-3 min)
- **Standard**: length=12, soft_iters=50 (~5-8 min)
- **Production**: length=15+, soft_iters=100+ (submit API)