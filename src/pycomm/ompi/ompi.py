
from numpy.typing import ArrayLike

from pycomm import OCOMM, backend_flags

if backend_flags["mpi_avail"]:
    from mpi4py import MPI

class OMPI(OCOMM):
    """Oblivious MPI communicator."""
    def __init__(
            self,
            comm : MPI.Comm = MPI.COMM_WORLD,
        ):
        
        self.comm = comm

    # Point-to-point communication (blocking) --------------------------------
    def send(self, data: ArrayLike, dest: int, tag: int = 0) -> None:
        """
        Send data from one process to another.

        Parameters:
        data (ArrayLike): The data to send.
        dest (int): The rank of the destination process.
        tag (int, optional): The message tag. Defaults to 0.
        """
        self.comm.send(data, dest=dest, tag=tag)

    def recv(self, buf: ArrayLike, source: int, tag: int = 0) -> None:
        """
        Receive data from another process.

        Parameters:
        buf (ArrayLike): The buffer to receive the data.
        source (int): The rank of the source process.
        tag (int, optional): The message tag. Defaults to 0.
        """
        self.comm.recv(buf, source=source, tag=tag)

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
        self.comm.bcast(data, root=root)
        return data

    def scatter(self, send_data: ArrayLike, recv_data: ArrayLike, root: int = 0) -> None:
        """
        Scatter data from one process to all others.

        Parameters:
        send_data (ArrayLike): The data to scatter.
        recv_data (ArrayLike): The buffer to receive the scattered data.
        root (int, optional): The rank of the root process. Defaults to 0.
        """
        self.comm.scatter(send_data, recv_data, root=root)

    def gather(self, send_data: ArrayLike, recv_data: ArrayLike, root: int = 0) -> None:
        """
        Gather data from all processes to one.

        Parameters:
        send_data (ArrayLike): The data to send.
        recv_data (ArrayLike): The buffer to receive the gathered data.
        root (int, optional): The rank of the root process. Defaults to 0.
        """
        self.comm.gather(send_data, recv_data, root=root)

    def allgather(self, send_data: ArrayLike, recv_data: ArrayLike) -> None:
        """
        Gather data from all processes to all.

        Parameters:
        send_data (ArrayLike): The data to send.
        recv_data (ArrayLike): The buffer to receive the gathered data.
        """
        self.comm.Allgather(send_data, recv_data)

    def reduce(self, send_data: ArrayLike, recv_data: ArrayLike, op: str, root: int = 0) -> None:
        """
        Reduce data from all processes to one.

        Parameters:
        send_data (ArrayLike): The data to send.
        recv_data (ArrayLike): The buffer to receive the reduced data.
        op (str): The reduction operation (e.g., 'sum', 'max').
        root (int, optional): The rank of the root process. Defaults to 0.
        """
        self.comm.Reduce(send_data, recv_data, op=MPI.__dict__[op.upper()], root=root)

    def allreduce(self, send_data: ArrayLike, recv_data: ArrayLike, op: str) -> None:
        """
        Reduce data from all processes to all.

        Parameters:
        send_data (ArrayLike): The data to send.
        recv_data (ArrayLike): The buffer to receive the reduced data.
        op (str): The reduction operation (e.g., 'sum', 'max').
        """
        self.comm.Allreduce(send_data, recv_data, op=MPI.__dict__[op.upper()])

    def alltoall(self, send_data: ArrayLike, recv_data: ArrayLike) -> None:
        """
        Send data from all processes to all.

        Parameters:
        send_data (ArrayLike): The data to send.
        recv_data (ArrayLike): The buffer to receive the data.
        """
        self.comm.Alltoall(send_data, recv_data)

    # Collective communication (non-blocking) ---------------------------------
        ...

    # Synchronization ---------------------------------------------------------
    def barrier(self) -> None:
        """
        Synchronize all processes.
        """
        self.comm.Barrier()

    # Communicators -----------------------------------------------------------
    def split(self, color: int, key: int) -> 'OCOMM':
        """
        Split the communicator into subgroups.

        Parameters:
        color (int): Control of subset assignment.
        key (int): Control of rank assignment.

        Returns:
        Communicator: A new communicator.
        """
        return OMPI(self.comm.Split(color, key))

    def dup(self) -> 'OCOMM':
        """
        Duplicate the communicator.

        Returns:
        Communicator: A new communicator.
        """
        return OMPI(self.comm.Dup())

    # Process management -------------------------------------------------------
    def rank(self) -> int:
        """
        Get the rank of the process.

        Returns:
        int: The rank of the process.
        """
        return self.comm.Get_rank()

    def size(self) -> int:
        """
        Get the size of the communicator.

        Returns:
        int: The size of the communicator.
        """
        return self.comm.Get_size()