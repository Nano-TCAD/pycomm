from warnings import warn

from pycomm.core.ocomm import OCOMM
from pycomm.obare.obare import OBARE
from pycomm.ompi.ompi import OMPI
from pycomm.onccl.onccl import ONCCL

backend_flags = {
    "cupy_avail": False,
    "nccl_avail": False,
    "mpi_avail": False,
    "mpi_cuda_aware": False,
}

try:
    import cupy as cp
    import cupyx.scipy.linalg as cu_la

    # Check if cupy is actually working. This could still raise
    # a cudaErrorInsufficientDriver error or something.
    cp.abs(1)
    backend_flags["cupy_avail"] = True

    from cupy.cuda import nccl
    if nccl.available:
        backend_flags["nccl_avail"] = True
    else:
        warn("NCCL is not available.")
except (ImportError, ImportWarning, ModuleNotFoundError) as w:
    warn(f"'CuPy' is unavailable. ({w})")

try:
    # Check if mpi4py is available
    from mpi4py import MPI

    backend_flags["mpi_avail"] = True

    if backend_flags["cupy_avail"]:
        # Check if MPI is CUDA-aware
        try:
            comm = MPI.COMM_WORLD
            comm_rank = comm.Get_rank()
            comm_size = comm.Get_size()

            # Create a small GPU array
            gpu_array = cp.array([comm_rank], dtype=cp.float32)

            # Perform an MPI operation on the GPU array
            if comm_size > 1:
                if comm_rank == 0:
                    comm.Send([gpu_array, MPI.FLOAT], dest=1)
                elif comm_rank == 1:
                    comm.Recv([gpu_array, MPI.FLOAT], source=0)

            backend_flags["mpi_cuda_aware"] = True
        except Exception as e:
            warn(f"MPI is not CUDA-aware. ({e})")

except (ImportError, ImportWarning, ModuleNotFoundError) as w:
    warn(f"'mpi4py' is unavailable. ({w})")

__all__ = [
    "OCOMM",
    "OBARE",
    "OMPI",
    "ONCCL",
    "backend_flags",
]