"""Job management for long-running cyclic peptide tasks with FIFO queue support."""

import sys
import uuid
import json
import subprocess
import threading
import queue
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from loguru import logger


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobManager:
    """Manages asynchronous job execution with FIFO queue for cyclic peptide computations.

    Only one job runs at a time (GPU-intensive tasks), others wait in queue.
    """

    def __init__(self, jobs_dir: Path = None):
        self.jobs_dir = jobs_dir or Path(__file__).parent.parent.parent / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

        # Job queue for FIFO execution
        self._job_queue: queue.Queue = queue.Queue()
        self._current_job: Optional[str] = None
        self._running_process: Optional[subprocess.Popen] = None

        # Lock for thread-safe operations
        self._lock = threading.Lock()

        # Start the queue worker thread
        self._worker_thread = threading.Thread(target=self._queue_worker, daemon=True)
        self._worker_thread.start()

        # Recover any pending jobs on startup
        self._recover_pending_jobs()

    def _recover_pending_jobs(self):
        """Recover and re-queue pending/running jobs from previous sessions."""
        try:
            # Collect jobs to recover, sorted by submission time (FIFO)
            jobs_to_recover = []
            for job_dir in self.jobs_dir.iterdir():
                if job_dir.is_dir():
                    metadata = self._load_metadata(job_dir.name)
                    if metadata and metadata["status"] in [JobStatus.PENDING.value, JobStatus.RUNNING.value]:
                        jobs_to_recover.append(metadata)

            # Sort by submission time to maintain FIFO order
            jobs_to_recover.sort(key=lambda x: x.get("submitted_at", ""))

            for metadata in jobs_to_recover:
                job_id = metadata["job_id"]
                job_dir = self.jobs_dir / job_id

                # Reset status to pending and clear timing info
                was_running = metadata["status"] == JobStatus.RUNNING.value
                metadata["status"] = JobStatus.PENDING.value
                metadata["started_at"] = None
                metadata["completed_at"] = None
                metadata["error"] = None
                if was_running:
                    metadata["recovered_from"] = "running"
                self._save_metadata(job_id, metadata)

                # Re-queue the job
                self._job_queue.put((
                    job_id,
                    metadata["script"],
                    metadata["args"],
                    job_dir
                ))
                logger.info(f"Recovered job {job_id} ({metadata.get('job_name', 'unnamed')})")

            if jobs_to_recover:
                logger.info(f"Recovered {len(jobs_to_recover)} job(s) from previous session")

        except Exception as e:
            logger.warning(f"Error recovering jobs: {e}")

    def _queue_worker(self):
        """Worker thread that processes jobs from the queue one at a time (FIFO)."""
        while True:
            try:
                # Block until a job is available
                job_id, script_path, args, job_dir = self._job_queue.get(block=True)

                with self._lock:
                    self._current_job = job_id

                # Execute the job
                self._execute_job(job_id, script_path, args, job_dir)

                with self._lock:
                    self._current_job = None
                    self._running_process = None

                self._job_queue.task_done()

            except Exception as e:
                logger.error(f"Queue worker error: {e}")

    def submit_job(
        self,
        script_path: str,
        args: Dict[str, Any],
        job_name: str = None
    ) -> Dict[str, Any]:
        """Submit a new job to the FIFO queue.

        Args:
            script_path: Path to the script to run
            args: Arguments to pass to the script
            job_name: Optional name for the job

        Returns:
            Dict with job_id, status, and queue position
        """
        job_id = str(uuid.uuid4())[:8]
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Save job metadata
        metadata = {
            "job_id": job_id,
            "job_name": job_name or f"job_{job_id}",
            "script": script_path,
            "args": args,
            "status": JobStatus.PENDING.value,
            "submitted_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "error": None
        }

        self._save_metadata(job_id, metadata)

        # Add job to queue
        self._job_queue.put((job_id, script_path, args, job_dir))

        # Get queue position
        queue_position = self._job_queue.qsize()

        return {
            "status": "submitted",
            "job_id": job_id,
            "queue_position": queue_position,
            "message": f"Job queued (position {queue_position}). Use get_job_status('{job_id}') to check progress."
        }

    def _execute_job(self, job_id: str, script_path: str, args: Dict, job_dir: Path):
        """Execute a job (called by queue worker)."""
        metadata = self._load_metadata(job_id)

        # Check if job was cancelled while waiting
        if metadata["status"] == JobStatus.CANCELLED.value:
            logger.info(f"Skipping cancelled job {job_id}")
            return

        metadata["status"] = JobStatus.RUNNING.value
        metadata["started_at"] = datetime.now().isoformat()
        self._save_metadata(job_id, metadata)

        try:
            # Build command - use env Python which has JAX/ColabDesign installed
            env_python = Path(__file__).resolve().parent.parent.parent / "env" / "bin" / "python"
            python_exe = str(env_python) if env_python.exists() else sys.executable
            cmd = [python_exe, script_path]
            for key, value in args.items():
                if value is True:
                    cmd.append(f"--{key}")
                elif value is not None and value is not False:
                    # Handle space-separated values (e.g., "50 50 10" for nargs=3)
                    str_value = str(value)
                    if " " in str_value:
                        cmd.append(f"--{key}")
                        cmd.extend(str_value.split())
                    else:
                        cmd.extend([f"--{key}", str_value])

            # Set output paths if output is specified in args.
            # Batch scripts take --output_dir instead and have no --output flag,
            # so never inject one when the caller already chose a directory.
            output_file = (args.get("output") or args.get("output_dir")
                           or str(job_dir / "output.pdb"))
            if "output" not in args and "output_dir" not in args:
                cmd.extend(["--output", output_file])

            logger.info(f"Running job {job_id}: {' '.join(cmd)}")

            # Run script
            log_file = job_dir / "job.log"
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=str(Path(script_path).parent)
                )

                with self._lock:
                    self._running_process = process

                process.wait()

            # Update status
            if process.returncode == 0:
                metadata["status"] = JobStatus.COMPLETED.value
                # Save result info
                result = {
                    "output_file": output_file,
                    "success": True
                }
                result_file = job_dir / "result.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
            else:
                metadata["status"] = JobStatus.FAILED.value
                metadata["error"] = f"Process exited with code {process.returncode}"

        except Exception as e:
            metadata["status"] = JobStatus.FAILED.value
            metadata["error"] = str(e)
            logger.error(f"Job {job_id} failed: {e}")

        finally:
            metadata["completed_at"] = datetime.now().isoformat()
            self._save_metadata(job_id, metadata)

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a submitted job."""
        metadata = self._load_metadata(job_id)
        if not metadata:
            return {"status": "error", "error": f"Job {job_id} not found"}

        result = {
            "job_id": job_id,
            "job_name": metadata.get("job_name"),
            "status": metadata["status"],
            "submitted_at": metadata.get("submitted_at"),
            "started_at": metadata.get("started_at"),
            "completed_at": metadata.get("completed_at")
        }

        # Add queue info for pending jobs
        if metadata["status"] == JobStatus.PENDING.value:
            result["queue_position"] = self._get_queue_position(job_id)

        if metadata["status"] == JobStatus.FAILED.value:
            result["error"] = metadata.get("error")

        return result

    def _get_queue_position(self, job_id: str) -> int:
        """Get position of a job in the queue."""
        position = 1
        pending_jobs = self.list_jobs(status=JobStatus.PENDING.value)
        for job in pending_jobs.get("jobs", []):
            if job["job_id"] == job_id:
                return position
            position += 1
        return 0

    def get_job_result(self, job_id: str) -> Dict[str, Any]:
        """Get results of a completed job."""
        metadata = self._load_metadata(job_id)
        if not metadata:
            return {"status": "error", "error": f"Job {job_id} not found"}

        if metadata["status"] != JobStatus.COMPLETED.value:
            return {
                "status": "error",
                "error": f"Job not completed. Current status: {metadata['status']}"
            }

        # Load result
        job_dir = self.jobs_dir / job_id
        result_file = job_dir / "result.json"

        if result_file.exists():
            with open(result_file) as f:
                result = json.load(f)
            return {"status": "success", "job_id": job_id, **result}
        else:
            # Return basic info if result file not found
            return {
                "status": "success",
                "job_id": job_id,
                "output_file": str(job_dir / "output.pdb")
            }

    def get_job_log(self, job_id: str, tail: int = 50) -> Dict[str, Any]:
        """Get log output from a job."""
        job_dir = self.jobs_dir / job_id
        log_file = job_dir / "job.log"

        if not log_file.exists():
            return {"status": "error", "error": f"Log not found for job {job_id}"}

        with open(log_file) as f:
            lines = f.readlines()

        return {
            "status": "success",
            "job_id": job_id,
            "log_lines": lines[-tail:] if tail > 0 else lines,
            "total_lines": len(lines)
        }

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel a pending or running job."""
        metadata = self._load_metadata(job_id)
        if not metadata:
            return {"status": "error", "error": f"Job {job_id} not found"}

        if metadata["status"] == JobStatus.COMPLETED.value:
            return {"status": "error", "error": "Cannot cancel completed job"}

        if metadata["status"] == JobStatus.CANCELLED.value:
            return {"status": "error", "error": "Job already cancelled"}

        # If running, terminate the process
        with self._lock:
            if self._current_job == job_id and self._running_process:
                self._running_process.terminate()

        # Update metadata
        metadata["status"] = JobStatus.CANCELLED.value
        metadata["completed_at"] = datetime.now().isoformat()
        self._save_metadata(job_id, metadata)

        return {"status": "success", "message": f"Job {job_id} cancelled"}

    def resubmit_job(self, job_id: str) -> Dict[str, Any]:
        """Resubmit a failed or cancelled job.

        Args:
            job_id: The job ID to resubmit

        Returns:
            Dict with new job submission info or error
        """
        metadata = self._load_metadata(job_id)
        if not metadata:
            return {"status": "error", "error": f"Job {job_id} not found"}

        # Only allow resubmit for failed or cancelled jobs
        if metadata["status"] not in [JobStatus.FAILED.value, JobStatus.CANCELLED.value]:
            return {
                "status": "error",
                "error": f"Can only resubmit failed or cancelled jobs. Current status: {metadata['status']}"
            }

        # Get the original script and args
        script_path = metadata.get("script")
        args = metadata.get("args")
        job_name = metadata.get("job_name")

        if not script_path or args is None:
            return {
                "status": "error",
                "error": "Cannot resubmit: missing script or args in job metadata"
            }

        # Submit as a new job with reference to original
        new_job_name = f"{job_name}_retry" if job_name else None
        result = self.submit_job(
            script_path=script_path,
            args=args,
            job_name=new_job_name
        )

        # Add reference to original job
        if result.get("status") == "submitted":
            new_job_id = result["job_id"]
            new_metadata = self._load_metadata(new_job_id)
            if new_metadata:
                new_metadata["resubmitted_from"] = job_id
                self._save_metadata(new_job_id, new_metadata)
            result["original_job_id"] = job_id

        return result

    def list_jobs(self, status: Optional[str] = None) -> Dict[str, Any]:
        """List all jobs, optionally filtered by status."""
        jobs = []
        for job_dir in sorted(self.jobs_dir.iterdir(), key=lambda x: x.stat().st_mtime):
            if job_dir.is_dir():
                metadata = self._load_metadata(job_dir.name)
                if metadata:
                    if status is None or metadata["status"] == status:
                        jobs.append({
                            "job_id": metadata["job_id"],
                            "job_name": metadata.get("job_name"),
                            "status": metadata["status"],
                            "submitted_at": metadata.get("submitted_at"),
                            "started_at": metadata.get("started_at"),
                            "completed_at": metadata.get("completed_at")
                        })

        return {"status": "success", "jobs": jobs, "total": len(jobs)}

    def get_queue_info(self) -> Dict[str, Any]:
        """Get current queue status."""
        pending = self.list_jobs(status=JobStatus.PENDING.value)
        running = self.list_jobs(status=JobStatus.RUNNING.value)

        return {
            "status": "success",
            "queue_size": self._job_queue.qsize(),
            "current_job": self._current_job,
            "pending_jobs": pending.get("total", 0),
            "running_jobs": running.get("total", 0)
        }

    def _save_metadata(self, job_id: str, metadata: Dict):
        """Save job metadata to disk atomically."""
        meta_file = self.jobs_dir / job_id / "metadata.json"
        meta_file.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file first, then rename (atomic on POSIX)
        temp_file = meta_file.with_suffix('.json.tmp')
        with open(temp_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        temp_file.rename(meta_file)

    def _load_metadata(self, job_id: str) -> Optional[Dict]:
        """Load job metadata from disk."""
        meta_file = self.jobs_dir / job_id / "metadata.json"
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    content = f.read()
                    if content.strip():
                        return json.loads(content)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load metadata for {job_id}: {e}")
        return None


# Global job manager instance
job_manager = JobManager()
