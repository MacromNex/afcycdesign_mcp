#!/usr/bin/env python3
"""
GPU Utilities for AfCycDesign Use Case Scripts

This module provides GPU environment setup that must be called BEFORE importing JAX.
It handles:
1. Finding NVIDIA library paths for CUDA
2. Re-executing the script with correct LD_LIBRARY_PATH
3. Configuring JAX GPU settings

Usage:
    # At the TOP of your script, before any other imports except os, sys, argparse:
    from gpu_utils import setup_gpu_environment
    setup_gpu_environment()

    # Now you can import JAX and other libraries
    import jax
    from colabdesign import mk_afdesign_model
"""

import os
import sys
from pathlib import Path


def get_nvidia_lib_paths() -> list:
    """
    Get nvidia pip package library paths for LD_LIBRARY_PATH.
    This is needed for JAX to find CUDA libraries like cuSPARSE.

    Returns:
        List of library paths.
    """
    lib_paths = []

    # Find nvidia packages in site-packages
    possible_nvidia_dirs = []

    # Method 1: Check relative to script location (for conda/mamba envs)
    script_dir = Path(__file__).parent.resolve()

    # Check various possible env locations
    for env_path in [
        script_dir.parent / "env" / "lib" / "python3.10" / "site-packages" / "nvidia",
        script_dir.parent.parent / "env" / "lib" / "python3.10" / "site-packages" / "nvidia",
    ]:
        if env_path.exists():
            possible_nvidia_dirs.append(env_path)

    # Method 2: Find nvidia package location via Python import
    try:
        import nvidia.cusparse as _cusparse
        if hasattr(_cusparse, '__file__') and _cusparse.__file__:
            nvidia_pkg_dir = Path(_cusparse.__file__).parent.parent
            if nvidia_pkg_dir.exists() and nvidia_pkg_dir not in possible_nvidia_dirs:
                possible_nvidia_dirs.append(nvidia_pkg_dir)
        elif hasattr(_cusparse, '__path__'):
            # Namespace package - get path from __path__
            for p in _cusparse.__path__:
                nvidia_pkg_dir = Path(p).parent
                if nvidia_pkg_dir.exists() and nvidia_pkg_dir not in possible_nvidia_dirs:
                    possible_nvidia_dirs.append(nvidia_pkg_dir)
    except (ImportError, AttributeError):
        pass

    # Method 3: Check in sys.path for site-packages
    for p in sys.path:
        nvidia_dir = Path(p) / "nvidia"
        if nvidia_dir.exists() and nvidia_dir not in possible_nvidia_dirs:
            possible_nvidia_dirs.append(nvidia_dir)

    # Libraries subdirectories to add
    nvidia_lib_subdirs = [
        "cuda_runtime/lib", "nvjitlink/lib", "cublas/lib",
        "cufft/lib", "cusparse/lib", "cusolver/lib", "cudnn/lib", "nccl/lib"
    ]

    # Collect all lib paths
    for nvidia_dir in possible_nvidia_dirs:
        for lib_subdir in nvidia_lib_subdirs:
            lib_path = nvidia_dir / lib_subdir
            if lib_path.exists() and str(lib_path) not in lib_paths:
                lib_paths.append(str(lib_path))

    return lib_paths


def check_gpu_env_and_reexec():
    """
    Check if LD_LIBRARY_PATH is set for GPU. If not, re-exec with correct env.
    This ensures CUDA libraries are available before JAX loads.
    """
    # Skip if explicitly using CPU
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return

    # Check if we've already re-exec'd
    if os.environ.get("_CYCPEP_GPU_ENV_SET") == "1":
        return

    # Get required nvidia lib paths
    nvidia_paths = get_nvidia_lib_paths()
    if not nvidia_paths:
        return  # No nvidia libs found, skip

    # Check if LD_LIBRARY_PATH already has these paths
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    needs_update = False
    for p in nvidia_paths:
        if p not in current_ld:
            needs_update = True
            break

    if needs_update:
        # Re-exec with updated LD_LIBRARY_PATH
        new_ld = ":".join(nvidia_paths)
        if current_ld:
            new_ld = f"{new_ld}:{current_ld}"
        os.environ["LD_LIBRARY_PATH"] = new_ld
        os.environ["_CYCPEP_GPU_ENV_SET"] = "1"

        # Re-exec this script
        os.execv(sys.executable, [sys.executable] + sys.argv)


def configure_gpu(gpu_id: int = None, mem_fraction: float = None,
                  preallocate: bool = True) -> dict:
    """
    Configure GPU settings for JAX. Must be called BEFORE importing JAX.

    Args:
        gpu_id: GPU device ID to use (0, 1, etc.). None for CPU or auto-select.
        mem_fraction: Fraction of GPU memory to use (0.0-1.0). None for default.
        preallocate: Whether to preallocate GPU memory (default: True).
                    Set to False for more flexible memory usage.

    Returns:
        Dict with configuration status and device info.
    """
    config_info = {
        "gpu_requested": gpu_id,
        "mem_fraction": mem_fraction,
        "preallocate": preallocate,
        "env_vars_set": []
    }

    if gpu_id is not None:
        if gpu_id < 0:
            # CPU mode
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            config_info["env_vars_set"].append("CUDA_VISIBLE_DEVICES='' (CPU mode)")
        else:
            # Set specific GPU device
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            config_info["env_vars_set"].append(f"CUDA_VISIBLE_DEVICES={gpu_id}")

    if mem_fraction is not None:
        # Limit GPU memory fraction
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)
        config_info["env_vars_set"].append(f"XLA_PYTHON_CLIENT_MEM_FRACTION={mem_fraction}")

    if not preallocate:
        # Disable memory preallocation for more flexible usage
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        config_info["env_vars_set"].append("XLA_PYTHON_CLIENT_PREALLOCATE=false")

    return config_info


def get_device_info() -> dict:
    """
    Get information about available compute devices.
    Must be called AFTER importing JAX.

    Returns:
        Dict with device information.
    """
    import jax

    devices = jax.devices()
    device_info = {
        "num_devices": len(devices),
        "devices": [],
        "default_backend": jax.default_backend(),
        "using_gpu": jax.default_backend() == "gpu"
    }

    for d in devices:
        device_info["devices"].append({
            "id": d.id,
            "platform": d.platform,
            "device_kind": d.device_kind if hasattr(d, 'device_kind') else str(d),
        })

    return device_info


def setup_gpu_environment(gpu_id: int = None, mem_fraction: float = None,
                          preallocate: bool = True, verbose: bool = False):
    """
    Complete GPU environment setup. Call this at the TOP of your script,
    BEFORE importing JAX or any JAX-dependent libraries.

    Args:
        gpu_id: GPU device ID to use (0, 1, etc.). None for auto, -1 for CPU.
        mem_fraction: Fraction of GPU memory to use (0.0-1.0).
        preallocate: Whether to preallocate GPU memory.
        verbose: Print configuration info.

    Example:
        # At the very top of your script:
        from gpu_utils import setup_gpu_environment
        setup_gpu_environment(gpu_id=0)

        # Now import JAX
        import jax
    """
    # First, ensure LD_LIBRARY_PATH is set correctly
    # This may re-exec the script
    check_gpu_env_and_reexec()

    # Configure GPU settings
    config = configure_gpu(gpu_id, mem_fraction, preallocate)

    if verbose:
        print("GPU Configuration:")
        for env_var in config["env_vars_set"]:
            print(f"  {env_var}")


def parse_gpu_args_early():
    """
    Pre-parse GPU-related arguments before full argument parsing.
    This allows GPU configuration before importing JAX.

    Returns:
        Tuple of (gpu_id, mem_fraction, use_cpu)
    """
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=int, default=None,
                       help="GPU device ID")
    parser.add_argument("--gpu_mem_fraction", type=float, default=None,
                       help="GPU memory fraction")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU mode")

    args, _ = parser.parse_known_args()

    gpu_id = None
    if args.cpu:
        gpu_id = -1  # CPU mode
    elif args.gpu is not None:
        gpu_id = args.gpu

    return gpu_id, args.gpu_mem_fraction, args.cpu


def auto_setup_gpu():
    """
    Automatically parse GPU arguments and setup environment.
    Call this at the TOP of your script before other imports.

    Example:
        from gpu_utils import auto_setup_gpu
        auto_setup_gpu()

        # Now import JAX
        import jax
    """
    gpu_id, mem_fraction, _ = parse_gpu_args_early()
    setup_gpu_environment(gpu_id=gpu_id, mem_fraction=mem_fraction)
