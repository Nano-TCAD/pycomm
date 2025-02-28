from numpy.typing import ArrayLike

from pycomm import OCOMM, backend_flags

if backend_flags["mpi_avail"]:
    from mpi4py import MPI

    if backend_flags["nccl_avail"]:
        import cupy as cp
        from cupy.cuda import nccl
        from cupyx.distarray import NCCLBackend

class ONCCL(OCOMM):
    """Oblivious Device Aware MPI communicator."""

    def __init__(
        self,
        comm_mpi: MPI.Comm = MPI.COMM_WORLD,
    ):
        self.comm_mpi = comm_mpi
        self.comm_nccl = NCCLBackend(
            comm_mpi.Get_size(), comm_mpi.Get_rank(), use_mpi=True
        )

    # Point-to-point communication (blocking) --------------------------------
    def send(self, data: ArrayLike, dest: int, tag: int = 0) -> None:
        """
        Send data from one process to another.

        Parameters:
        data (ArrayLike): The data to send.
        dest (int): The rank of the destination process.
        tag (int, optional): The message tag. Defaults to 0.
        """
        pass

    def recv(self, buf: ArrayLike, source: int, tag: int = 0) -> None:
        """
        Receive data from another process.

        Parameters:
        buf (ArrayLike): The buffer to receive the data.
        source (int): The rank of the source process.
        tag (int, optional): The message tag. Defaults to 0.
        """
        pass

    # Point-to-point communication (non-blocking) -----------------------------
    ...

    # Collective communication (blocking) -------------------------------------
    def bcast(self, data: ArrayLike, root: int = 0) -> ArrayLike:
        """
        Broadcast data from one process to all others.

        Parameters:
        data (ArrayLike): The data to broadcast.
        root (int, optional): The rank of the root process. Defaults to 0.

        Returns:
        ArrayLike: The broadcasted data.
        """
        pass

    def scatter(
        self, send_data: ArrayLike, recv_data: ArrayLike, root: int = 0
    ) -> None:
        """
        Scatter data from one process to all others.

        Parameters:
        send_data (ArrayLike): The data to scatter.
        recv_data (ArrayLike): The buffer to receive the scattered data.
        root (int, optional): The rank of the root process. Defaults to 0.
        """
        pass

    def gather(self, send_data: ArrayLike, recv_data: ArrayLike, root: int = 0) -> None:
        """
        Gather data from all processes to one.

        Parameters:
        send_data (ArrayLike): The data to send.
        recv_data (ArrayLike): The buffer to receive the gathered data.
        root (int, optional): The rank of the root process. Defaults to 0.
        """
        pass

    def allgather(self, send_data: ArrayLike, recv_data: ArrayLike) -> None:
        """
        Gather data from all processes to all.

        Parameters:
        send_data (ArrayLike): The data to send.
        recv_data (ArrayLike): The buffer to receive the gathered data.
        """
        pass

    def reduce(
        self, send_data: ArrayLike, recv_data: ArrayLike, op: str, root: int = 0
    ) -> None:
        """
        Reduce data from all processes to one.

        Parameters:
        send_data (ArrayLike): The data to send.
        recv_data (ArrayLike): The buffer to receive the reduced data.
        op (str): The reduction operation (e.g., 'sum', 'max').
        root (int, optional): The rank of the root process. Defaults to 0.
        """
        pass

    def allreduce(self, send_data: ArrayLike, recv_data: ArrayLike, op: str) -> None:
        """
        Reduce data from all processes to all.

        Parameters:
        send_data (ArrayLike): The data to send.
        recv_data (ArrayLike): The buffer to receive the reduced data.
        op (str): The reduction operation (e.g., 'sum', 'max').
        """
        pass

    def alltoall(self, send_data: ArrayLike, recv_data: ArrayLike) -> None:
        """
        Send data from all processes to all.

        Parameters:
        send_data (ArrayLike): The data to send.
        recv_data (ArrayLike): The buffer to receive the data.
        """
        pass

        # Collective communication (non-blocking) ---------------------------------
        ...

    # Synchronization ---------------------------------------------------------
    def barrier(self) -> None:
        """
        Synchronize all processes.
        """
        pass

    # Communicators -----------------------------------------------------------
    def split(self, color: int, key: int) -> "OCOMM":
        """
        Split the communicator into subgroups.

        Parameters:
        color (int): Control of subset assignment.
        key (int): Control of rank assignment.

        Returns:
        Communicator: A new communicator.
        """
        pass

    def dup(self) -> "OCOMM":
        """
        Duplicate the communicator.

        Returns:
        Communicator: A new communicator.
        """
        pass

    # Process management -------------------------------------------------------
    def rank(self) -> int:
        """
        Get the rank of the process.

        Returns:
        int: The rank of the process.
        """
        pass

    def size(self) -> int:
        """
        Get the size of the communicator.

        Returns:
        int: The size of the communicator.
        """
        pass
