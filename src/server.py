"""MCP Server for Cyclic Peptide Design Tools (AfCycDesign)

Provides tools for cyclic peptide design based on AlphaFold:
1. Structure Prediction - Predict 3D structure from sequence
2. Fixed Backbone Design - Design sequence for given backbone structure
3. Hallucination - De novo generation of structure and sequence
4. Binder Design - Design cyclic peptides that bind to target proteins

All long-running tasks use async job submission with FIFO queue management.
"""

from fastmcp import FastMCP
from pathlib import Path
from typing import Optional, List
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
MCP_ROOT = SCRIPT_DIR.parent
SCRIPTS_DIR = MCP_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from jobs.manager import job_manager
from loguru import logger

# Create MCP server
mcp = FastMCP("afcycdesign")

# ==============================================================================
# Job Management Tools
# ==============================================================================

@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """
    Get the status of a submitted cyclic peptide design job.

    Args:
        job_id: The job ID returned from a submit_* function

    Returns:
        Dictionary with job status, timestamps, queue position (if pending), and any errors
    """
    return job_manager.get_job_status(job_id)


@mcp.tool()
def get_job_result(job_id: str) -> dict:
    """
    Get the results of a completed cyclic peptide design job.

    Args:
        job_id: The job ID of a completed job

    Returns:
        Dictionary with the output file path and job results
    """
    return job_manager.get_job_result(job_id)


@mcp.tool()
def get_job_log(job_id: str, tail: int = 50) -> dict:
    """
    Get log output from a running or completed job.

    Args:
        job_id: The job ID to get logs for
        tail: Number of lines from end (default: 50, use 0 for all)

    Returns:
        Dictionary with log lines and total line count
    """
    return job_manager.get_job_log(job_id, tail)


@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """
    Cancel a pending or running cyclic peptide design job.

    Args:
        job_id: The job ID to cancel

    Returns:
        Success or error message
    """
    return job_manager.cancel_job(job_id)


@mcp.tool()
def list_jobs(status: Optional[str] = None) -> dict:
    """
    List all submitted cyclic peptide design jobs.

    Args:
        status: Filter by status (pending, running, completed, failed, cancelled)

    Returns:
        List of jobs with their status
    """
    return job_manager.list_jobs(status)


@mcp.tool()
def get_queue_info() -> dict:
    """
    Get current job queue status.

    Returns:
        Dictionary with queue size, current running job, and job counts by status
    """
    return job_manager.get_queue_info()


# ==============================================================================
# Tool 1: Cyclic Peptide Structure Prediction
# ==============================================================================

@mcp.tool()
def submit_structure_prediction(
    sequence: str,
    output_file: Optional[str] = None,
    num_recycles: int = 6,
    add_rg: bool = False,
    rg_weight: float = 0.1,
    soft_iters: int = 0,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit a cyclic peptide structure prediction job.

    Predicts the 3D structure of a cyclic peptide from its amino acid sequence
    using AlphaFold with cyclic offset constraints for head-to-tail cyclization.

    Args:
        sequence: Amino acid sequence (1-letter codes, e.g., "GFNYGPFGSC")
        output_file: Optional output PDB file path (auto-generated if not provided)
        num_recycles: Number of AlphaFold recycles (default: 6, paper recommendation)
        add_rg: Add radius of gyration constraint for compact structures
        rg_weight: Weight for Rg loss if add_rg is True (default: 0.1)
        soft_iters: Number of soft optimization iterations for refinement (default: 0)
        job_name: Optional name for job tracking

    Returns:
        Dictionary with job_id for tracking. Use:
        - get_job_status(job_id) to check progress
        - get_job_result(job_id) to get output PDB when completed
        - get_job_log(job_id) to see execution logs

    Example sequences (from paper):
        - "GFNYGPFGSC" (1JBL, 10 residues)
        - "FRLLNYYA" (2MW0, 8 residues)
        - "WTYTYDWFC" (2LWV, 9 residues)
    """
    script_path = str(SCRIPTS_DIR / "cyclic_structure_prediction.py")

    args = {
        "sequence": sequence.upper().strip(),
        "num_recycles": num_recycles,
        "quiet": True
    }

    if output_file:
        args["output"] = output_file
    if add_rg:
        args["add_rg"] = True
        args["rg_weight"] = rg_weight
    if soft_iters > 0:
        args["soft_iters"] = soft_iters

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or f"structure_prediction_{len(sequence)}mer"
    )


# ==============================================================================
# Tool 2: Cyclic Peptide Fixed Backbone Sequence Design
# ==============================================================================

@mcp.tool()
def submit_fixbb_design(
    pdb_file: str,
    output_file: Optional[str] = None,
    chain: str = "A",
    add_rg: bool = False,
    rg_weight: float = 0.1,
    num_recycles: int = 0,
    stage_iters: Optional[str] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit a fixed backbone sequence design job for a cyclic peptide.

    Redesigns the amino acid sequence to maximize folding propensity for a
    given cyclic peptide backbone structure using categorical cross-entropy
    loss between target and predicted distograms.

    Args:
        pdb_file: Path to input PDB file with target backbone structure
        output_file: Optional output PDB file path for designed structure
        chain: Chain ID to design (default: "A")
        add_rg: Add radius of gyration loss for compactness
        rg_weight: Weight for Rg loss if add_rg is True (default: 0.1)
        num_recycles: Number of AlphaFold recycles (default: 0)
        stage_iters: 3-stage iteration counts as "logits,soft,hard" (default: "50,50,10")
        job_name: Optional name for job tracking

    Returns:
        Dictionary with job_id for tracking

    Note:
        Uses a 3-stage optimization schedule (default 110 steps total):
        - Stage 1 (logits): Optimize continuous representation
        - Stage 2 (soft): Softmax activation with temperature scaling
        - Stage 3 (hard): One-hot encoding with straight-through estimator
    """
    script_path = str(SCRIPTS_DIR / "cyclic_fixbb_design.py")

    args = {
        "pdb": pdb_file,
        "chain": chain,
        "num_recycles": num_recycles,
        "quiet": True
    }

    if output_file:
        args["output"] = output_file
    if add_rg:
        args["add_rg"] = True
        args["rg_weight"] = rg_weight
    if stage_iters:
        # Parse "50,50,10" format to "50 50 10" for argparse nargs
        args["stage_iters"] = stage_iters.replace(",", " ")

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or f"fixbb_design_{chain}"
    )


# ==============================================================================
# Tool 3: De Novo Cyclic Peptide Hallucination
# ==============================================================================

@mcp.tool()
def submit_hallucination(
    length: int,
    output_file: Optional[str] = None,
    rm_aa: str = "C",
    add_rg: bool = False,
    rg_weight: float = 0.1,
    soft_iters: int = 50,
    stage_iters: Optional[str] = None,
    plddt_threshold: float = 0.9,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit a de novo cyclic peptide hallucination job.

    Simultaneously generates a new cyclic peptide structure AND its corresponding
    sequence from scratch. Uses AlphaFold to predict well-structured peptides
    with high confidence scores.

    Args:
        length: Length of peptide to generate (7-16 residues tested in paper)
        output_file: Optional output PDB file path
        rm_aa: Amino acids to exclude (comma-separated, default: "C" for cysteine)
        add_rg: Add radius of gyration loss for compact structures
        rg_weight: Weight for Rg loss if add_rg is True (default: 0.1)
        soft_iters: Iterations for soft pre-design (default: 50)
        stage_iters: 3-stage iteration counts as "logits,soft,hard" (default: "50,50,10")
        plddt_threshold: Quality threshold for high-confidence scaffolds (default: 0.9)
        job_name: Optional name for job tracking

    Returns:
        Dictionary with job_id for tracking

    Note:
        Paper parameters:
        - Loss Function: 1 - pLDDT + PAE/31 + con/2
        - Contact Loss: binary=True, cutoff=21.6875
        - Quality Filter: pLDDT > 0.9 for high-confidence scaffolds
    """
    script_path = str(SCRIPTS_DIR / "cyclic_hallucination.py")

    args = {
        "length": length,
        "rm_aa": rm_aa,
        "soft_iters": soft_iters,
        "plddt_threshold": plddt_threshold,
        "quiet": True
    }

    if output_file:
        args["output"] = output_file
    if add_rg:
        args["add_rg"] = True
        args["rg_weight"] = rg_weight
    if stage_iters:
        args["stage_iters"] = stage_iters.replace(",", " ")

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or f"hallucination_{length}mer"
    )


# ==============================================================================
# Tool 4: Cyclic Peptide Binder Design
# ==============================================================================

@mcp.tool()
def submit_binder_design(
    target_pdb: str,
    binder_len: int,
    output_file: Optional[str] = None,
    target_chain: str = "A",
    binder_seq: Optional[str] = None,
    hotspot: Optional[str] = None,
    target_flexible: bool = False,
    use_multimer: bool = False,
    optimizer: str = "pssm_semigreedy",
    num_recycles: int = 0,
    num_models: int = 2,
    ipae_threshold: float = 0.15,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit a cyclic peptide binder design job for a target protein.

    Designs cyclic peptides that bind to protein targets with the cyclic offset
    applied only to the binder chain. Based on the motif grafting workflow.

    Args:
        target_pdb: Path to target protein PDB file (or 4-letter PDB code to download)
        binder_len: Length of binder peptide to design (6-20 residues typical)
        output_file: Optional output PDB file path for designed complex
        target_chain: Target protein chain ID (default: "A")
        binder_seq: Initial binder sequence if known (e.g., motif like "FSDLW")
        hotspot: Restrict binding to specific positions (e.g., "1-10,12,15")
        target_flexible: Allow target backbone flexibility
        use_multimer: Use AlphaFold-multimer
        optimizer: Optimization method (default: "pssm_semigreedy")
            Options: "pssm_semigreedy", "3stage", "semigreedy", "pssm", "logits", "soft", "hard"
        num_recycles: Number of AlphaFold recycles (default: 0)
        num_models: Number of models to use (default: 2)
        ipae_threshold: Interface PAE threshold for quality (default: 0.15, use 0.11 for strict)
        job_name: Optional name for job tracking

    Returns:
        Dictionary with job_id for tracking

    Note:
        Paper filtering thresholds:
        - iPAE (Interface PAE): < 0.15 (or 0.11 for strict)
        - ddG: < -30 kcal/mol (requires Rosetta)
        - SAP score: < 30 (requires Rosetta)
        - CMS (Contact Molecular Surface): > 300 (requires Rosetta)
        - Binding mode RMSD: < 1.5 A
    """
    script_path = str(SCRIPTS_DIR / "cyclic_binder_design.py")

    args = {
        "pdb": target_pdb,
        "target_chain": target_chain,
        "binder_len": binder_len,
        "optimizer": optimizer,
        "num_recycles": num_recycles,
        "num_models": num_models,
        "ipae_threshold": ipae_threshold,
        "quiet": True
    }

    if output_file:
        args["output"] = output_file
    if binder_seq:
        args["binder_seq"] = binder_seq
    if hotspot:
        args["hotspot"] = hotspot
    if target_flexible:
        args["target_flexible"] = True
    if use_multimer:
        args["use_multimer"] = True

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or f"binder_design_{binder_len}mer"
    )


# ==============================================================================
# Utility Tools
# ==============================================================================

@mcp.tool()
def validate_pdb_file(file_path: str) -> dict:
    """
    Validate a PDB file structure.

    Args:
        file_path: Path to PDB file to validate

    Returns:
        Dictionary with validation results and structural information
    """
    try:
        pdb_path = Path(file_path)
        if not pdb_path.exists():
            return {"status": "error", "error": f"File not found: {file_path}"}

        if not pdb_path.is_file():
            return {"status": "error", "error": f"Not a file: {file_path}"}

        if pdb_path.stat().st_size == 0:
            return {"status": "error", "error": f"Empty file: {file_path}"}

        # Basic PDB validation
        chains = set()
        residues = {}
        atom_count = 0

        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    atom_count += 1
                    chain = line[21:22].strip() or "_"
                    res_num = line[22:26].strip()
                    chains.add(chain)
                    if chain not in residues:
                        residues[chain] = set()
                    residues[chain].add(res_num)

        chain_info = {ch: len(res) for ch, res in residues.items()}

        return {
            "status": "success",
            "file_path": str(pdb_path),
            "file_size_bytes": pdb_path.stat().st_size,
            "chains": list(chains),
            "residues_per_chain": chain_info,
            "total_residues": sum(chain_info.values()),
            "atom_count": atom_count,
            "valid_pdb": atom_count > 0
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


@mcp.tool()
def get_server_info() -> dict:
    """
    Get information about the AfCycDesign MCP server.

    Returns:
        Dictionary with server information and available tools
    """
    return {
        "status": "success",
        "server_name": "afcycdesign",
        "version": "1.0.0",
        "description": "Cyclic peptide design tools based on AlphaFold",
        "reference": "Stephen Rettie et al., doi: 10.1101/2023.02.25.529956",
        "scripts_directory": str(SCRIPTS_DIR),
        "jobs_directory": str(job_manager.jobs_dir),
        "available_tools": {
            "design_tools": [
                {
                    "name": "submit_structure_prediction",
                    "description": "Predict 3D structure from amino acid sequence"
                },
                {
                    "name": "submit_fixbb_design",
                    "description": "Design sequence for given backbone structure"
                },
                {
                    "name": "submit_hallucination",
                    "description": "De novo generation of structure and sequence"
                },
                {
                    "name": "submit_binder_design",
                    "description": "Design cyclic peptides that bind to target proteins"
                }
            ],
            "job_management": [
                "get_job_status",
                "get_job_result",
                "get_job_log",
                "cancel_job",
                "list_jobs",
                "get_queue_info"
            ],
            "utilities": [
                "validate_pdb_file",
                "get_server_info"
            ]
        },
        "job_queue": "FIFO (one job at a time due to GPU constraints)"
    }


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    mcp.run()
