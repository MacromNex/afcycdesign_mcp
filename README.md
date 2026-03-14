# AfCycDesign MCP Server

**AlphaFold-based cyclic peptide design and structure prediction via Docker**

An MCP (Model Context Protocol) server providing 13 tools for cyclic peptide computational design:
- Predict 3D structures of cyclic peptides from sequence
- Design sequences for given cyclic backbone structures (fixed backbone)
- De novo hallucination of cyclic peptide structure and sequence
- Design cyclic peptide binders for target proteins
- Predict cyclic peptide-target protein complex structures
- Monitor and manage long-running GPU jobs

## Quick Start with Docker

### Approach 1: Pull Pre-built Image from GitHub

The fastest way to get started. A pre-built Docker image is automatically published to GitHub Container Registry on every release.

```bash
# Pull the latest image
docker pull ghcr.io/macromnex/afcycdesign_mcp:latest

# Register with Claude Code (runs as current user to avoid permission issues)
claude mcp add afcycdesign -- docker run -i --rm --user `id -u`:`id -g` --gpus all --ipc=host -v `pwd`:`pwd` -v ~/.macromnex/cache/model/alphafold2/params:/app/params:ro ghcr.io/macromnex/afcycdesign_mcp:latest
```

**Note:** Run from your project directory. `` `pwd` `` expands to the current working directory. The AF2 model parameters (~5.3 GB) must be present at `~/.macromnex/cache/model/alphafold2/params/`.

**Requirements:**
- Docker with GPU support (`nvidia-docker` or Docker with NVIDIA runtime)
- Claude Code installed
- AlphaFold2 model parameters downloaded to `~/.macromnex/cache/model/alphafold2/params/`

That's it! The AfCycDesign MCP server is now available in Claude Code.

---

### Approach 2: Build Docker Image Locally

Build the image yourself and install it into Claude Code. Useful for customization or offline environments.

```bash
# Clone the repository
git clone https://github.com/MacromNex/afcycdesign_mcp.git
cd afcycdesign_mcp

# Build the Docker image
docker build -t afcycdesign_mcp:latest .

# Register with Claude Code (runs as current user to avoid permission issues)
claude mcp add afcycdesign -- docker run -i --rm --user `id -u`:`id -g` --gpus all --ipc=host -v `pwd`:`pwd` -v ~/.macromnex/cache/model/alphafold2/params:/app/params:ro afcycdesign_mcp:latest
```

**Note:** Run from your project directory. `` `pwd` `` expands to the current working directory.

**Requirements:**
- Docker with GPU support
- Claude Code installed
- Git (to clone the repository)
- AlphaFold2 model parameters downloaded to `~/.macromnex/cache/model/alphafold2/params/`

**About the Docker Flags:**
- `-i` -- Interactive mode for Claude Code
- `--rm` -- Automatically remove container after exit
- `` --user `id -u`:`id -g` `` -- Runs the container as your current user, so output files are owned by you (not root)
- `--gpus all` -- Grants access to all available GPUs
- `--ipc=host` -- Uses host IPC namespace for better performance
- `-v ~/.macromnex/cache/model/alphafold2/params:/app/params:ro` -- Mounts AlphaFold2 model parameters (read-only)
- `` -v `pwd`:`pwd` `` -- Mounts your project directory so the container can access your data

---

## Verify Installation

After adding the MCP server, you can verify it's working:

```bash
# List registered MCP servers
claude mcp list

# You should see 'afcycdesign' in the output
```

In Claude Code, you can now use all 13 AfCycDesign tools:

**Design Tools:**
- `submit_structure_prediction` -- Predict 3D structure from sequence
- `submit_fixbb_design` -- Fixed backbone sequence design
- `submit_hallucination` -- De novo cyclic peptide generation
- `submit_binder_design` -- Design binders for target proteins
- `submit_complex_prediction` -- Predict peptide-target complex structure

**Job Management:**
- `get_job_status`, `get_job_result`, `get_job_log`, `list_jobs`, `get_queue_info`, `cancel_job`, `resubmit_job`

**Utilities:**
- `validate_pdb_file`, `get_server_info`

---

## Next Steps

- **Detailed documentation**: See [detail.md](detail.md) for comprehensive guides on:
  - Available MCP tools and parameters
  - Local Python environment setup (alternative to Docker)
  - Script usage examples with full parameter references
  - Example workflows and use cases

---

## Usage Examples

Once registered, you can use the AfCycDesign tools directly in Claude Code. Here are some common workflows:

### Example 1: Structure Prediction

```
I have a cyclic peptide with sequence GFNYGPFGSC. Can you predict its 3D structure using submit_structure_prediction and save the results to /path/to/results/?
```

### Example 2: De Novo Hallucination

```
Generate a novel 12-residue cyclic peptide using submit_hallucination, excluding cysteine. Save the output to /path/to/output/.
```

### Example 3: Binder Design

```
I have a target protein PDB at /path/to/target.pdb. Can you design a 14-residue cyclic peptide binder using submit_binder_design targeting chain A?
```

### Example 4: Complex Structure Prediction

```
Predict how the cyclic peptide FSDLWKLLPEN binds to the target protein in /path/to/4HFZ.pdb chain A using submit_complex_prediction.
```

---

## Troubleshooting

**Docker not found?**
```bash
docker --version  # Install Docker if missing
```

**GPU not accessible?**
- Ensure NVIDIA Docker runtime is installed
- Check with `docker run --gpus all ubuntu nvidia-smi`

**Claude Code not found?**
```bash
# Install Claude Code
npm install -g @anthropic-ai/claude-code
```

**AF2 parameters missing?**
```bash
# Verify params are downloaded
ls ~/.macromnex/cache/model/alphafold2/params/
# Should contain params_model_*_ptm.npz and params_model_*_multimer_v3.npz
```

---

## License

MIT
