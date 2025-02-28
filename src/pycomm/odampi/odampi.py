from numpy.typing import ArrayLike

from pycomm import OCOMM, backend_flags
from .utils import _get_module_from_array

import numpy as np

if backend_flags["mpi_avail"]:
    from mpi4py import MPI

if backend_flags["cuda_avail"]:
    import cupy as cp


class ODAMPI(OCOMM):
    """Oblivious Device Aware MPI communicator."""
    def __init__(
            self,
            comm : MPI.Comm = MPI.COMM_WORLD,
        ):

        self.comm = comm

    # Point-to-point communication (blocking) --------------------------------
    def send(
        self,
        data: ArrayLike,
        dest: int,
        send_buffer: None | ArrayLike = None,
        tag: int = 0,
    ) -> None:
        """
        Send data from one process to another.

        Parameters:
        data (ArrayLike): The data to send.
        dest (int): The rank of the destination process.
        send_buffer (ArrayLike, optional): The host buffer to store the data. Defaults to None.
        tag (int, optional): The message tag. Defaults to 0.
        """
        if _get_module_from_array(data) == np:
            self.comm.send(data, dest=dest, tag=tag)
        else:
            if send_buffer is None:
                send_buffer = np.empty_like(data)
            if _get_module_from_array(send_buffer) != np:
                raise ValueError("Host buffer must be on a host array.")

            data.get(out=send_buffer)
            self.comm.send(send_buffer, dest=dest, tag=tag)

    def recv(
        self,
        buf: ArrayLike,
        source: int,
        recv_buffer: None | ArrayLike = None,
        tag: int = 0,
    ) -> None:
        """
        Receive data from another process.

        Parameters:
        buf (ArrayLike): The buffer to receive the data.
        source (int): The rank of the source process.
        tag (int, optional): The message tag. Defaults to 0.
        """
        if _get_module_from_array(buf) == np:
            self.comm.recv(buf, source=source, tag=tag)
        else:
            if recv_buffer is None:
                recv_buffer = np.empty_like(buf)
            if _get_module_from_array(recv_buffer) != np:
                raise ValueError("Host buffer must be on a host array.")

            self.comm.recv(recv_buffer, source=source, tag=tag)
            buf.set(arr=recv_buffer)

    # Point-to-point communication (non-blocking) -----------------------------
    ...

    # Collective communication (blocking) -------------------------------------
    def bcast(
        self,
        data: ArrayLike,
        root: int = 0,
        comm_buffer: None | ArrayLike = None,
    ) -> ArrayLike:
        """
        Broadcast data from one process to all others.

        Parameters:
        data (ArrayLike): The data to broadcast.
        root (int, optional): The rank of the root process. Defaults to 0.

        Returns:
        ArrayLike: The broadcasted data.
        """
        if _get_module_from_array(data) == np:
            self.comm.bcast(data, root=root)
        else:
            if comm_buffer is None:
                comm_buffer = np.empty_like(data)
            if _get_module_from_array(comm_buffer) != np:
                raise ValueError("Host buffer must be on a host array.")

            if root == self.rank():
                data.get(out=comm_buffer)

            self.comm.bcast(comm_buffer, root=root)

            return data.set(arr=comm_buffer)

    def scatter(
        self,
        send_data: ArrayLike,
        recv_data: ArrayLike,
        root: int = 0,
        send_buffer: None | ArrayLike = None,
        recv_buffer: None | ArrayLike = None,
    ) -> None:
        """
        Scatter data from one process to all others.

        Parameters:
        send_data (ArrayLike): The data to scatter.
        recv_data (ArrayLike): The buffer to receive the scattered data.
        root (int, optional): The rank of the root process. Defaults to 0.
        """
        if _get_module_from_array(send_data) != _get_module_from_array(recv_data):
            raise ValueError("Send and receive data must be on the same array module.")

        if _get_module_from_array(send_data) == np and _get_module_from_array(recv_data) == np:
            self.comm.scatter(send_data, recv_data, root=root)
        else:
            if send_buffer is None:
                send_buffer = np.empty_like(send_data)
            if _get_module_from_array(send_buffer) != np:
                raise ValueError("Host buffer must be on a host array.")
            if recv_buffer is None:
                recv_buffer = np.empty_like(recv_data)
            if _get_module_from_array(recv_buffer) != np:
                raise ValueError("Host buffer must be on a host array.")

            if root == self.rank():
                send_data.get(out=send_buffer)

            self.comm.scatter(send_data, recv_data, root=root)

            recv_data.set(arr=recv_buffer)

    def gather(
        self,
        send_data: ArrayLike,
        recv_data: ArrayLike,
        root: int = 0,
        send_buffer: None | ArrayLike = None,
        recv_buffer: None | ArrayLike = None,
    ) -> None:
        """
        Gather data from all processes to one.

        Parameters:
        send_data (ArrayLike): The data to send.
        recv_data (ArrayLike): The buffer to receive the gathered data.
        root (int, optional): The rank of the root process. Defaults to 0.
        """
        if _get_module_from_array(send_data) != _get_module_from_array(recv_data):
            raise ValueError("Send and receive data must be on the same array module.")

        if (
            _get_module_from_array(send_data) == np
            and _get_module_from_array(recv_data) == np
        ):
            self.comm.gather(send_data, recv_data, root=root)
        else:
            if send_buffer is None:
                send_buffer = np.empty_like(send_data)
            if _get_module_from_array(send_buffer) != np:
                raise ValueError("Host buffer must be on a host array.")
            if recv_buffer is None:
                recv_buffer = np.empty_like(recv_data)
            if _get_module_from_array(recv_buffer) != np:
                raise ValueError("Host buffer must be on a host array.")

            if root != self.rank():
                send_data.get(out=send_buffer)

            self.comm.gather(send_data, recv_data, root=root)

            recv_data.set(arr=recv_buffer)

    def allgather(
        self,
        send_data: ArrayLike,
        recv_data: ArrayLike,
        send_buffer: None | ArrayLike = None,
        recv_buffer: None | ArrayLike = None,
    ) -> None:
        """
        Gather data from all processes to all.

        Parameters:
        send_data (ArrayLike): The data to send.
        recv_data (ArrayLike): The buffer to receive the gathered data.
        """
        if _get_module_from_array(send_data) != _get_module_from_array(recv_data):
            raise ValueError("Send and receive data must be on the same array module.")

        if (
            _get_module_from_array(send_data) == np
            and _get_module_from_array(recv_data) == np
        ):
            self.comm.Allgather(send_data, recv_data)
        else:
            if send_buffer is None:
                send_buffer = np.empty_like(send_data)
            if _get_module_from_array(send_buffer) != np:
                raise ValueError("Host buffer must be on a host array.")
            if recv_buffer is None:
                recv_buffer = np.empty_like(recv_data)
            if _get_module_from_array(recv_buffer) != np:
                raise ValueError("Host buffer must be on a host array.")

            send_data.get(out=send_buffer)

            self.comm.Allgather(send_data, recv_data)

            recv_data.set(arr=recv_buffer)

    def reduce(
        self,
        send_data: ArrayLike,
        recv_data: ArrayLike,
        op: str,
        root: int = 0,
        send_buffer: None | ArrayLike = None,
        recv_buffer: None | ArrayLike = None,
    ) -> None:
        """
        Reduce data from all processes to one.

        Parameters:
        send_data (ArrayLike): The data to send.
        recv_data (ArrayLike): The buffer to receive the reduced data.
        op (str): The reduction operation (e.g., 'sum', 'max').
        root (int, optional): The rank of the root process. Defaults to 0.
        """
        if _get_module_from_array(send_data) != _get_module_from_array(recv_data):
            raise ValueError("Send and receive data must be on the same array module.")

        if (
            _get_module_from_array(send_data) == np
            and _get_module_from_array(recv_data) == np
        ):
            self.comm.Reduce(
                send_data, recv_data, op=MPI.__dict__[op.upper()], root=root
            )
        else:
            if send_buffer is None:
                send_buffer = np.empty_like(send_data)
            if _get_module_from_array(send_buffer) != np:
                raise ValueError("Host buffer must be on a host array.")
            if recv_buffer is None:
                recv_buffer = np.empty_like(recv_data)
            if _get_module_from_array(recv_buffer) != np:
                raise ValueError("Host buffer must be on a host array.")

            send_data.get(out=send_buffer)

            self.comm.Reduce(
                send_data, recv_data, op=MPI.__dict__[op.upper()], root=root
            )

            if root == self.rank():
                recv_data.set(arr=recv_buffer)

    def allreduce(
        self,
        send_data: ArrayLike,
        recv_data: ArrayLike,
        op: str,
        send_buffer: None | ArrayLike = None,
        recv_buffer: None | ArrayLike = None,
    ) -> None:
        """
        Reduce data from all processes to all.

        Parameters:
        send_data (ArrayLike): The data to send.
        recv_data (ArrayLike): The buffer to receive the reduced data.
        op (str): The reduction operation (e.g., 'sum', 'max').
        """
        if _get_module_from_array(send_data) != _get_module_from_array(recv_data):
            raise ValueError("Send and receive data must be on the same array module.")

        if (
            _get_module_from_array(send_data) == np
            and _get_module_from_array(recv_data) == np
        ):
            self.comm.Allreduce(send_data, recv_data, op=MPI.__dict__[op.upper()])
        else:
            if send_buffer is None:
                send_buffer = np.empty_like(send_data)
            if _get_module_from_array(send_buffer) != np:
                raise ValueError("Host buffer must be on a host array.")
            if recv_buffer is None:
                recv_buffer = np.empty_like(recv_data)
            if _get_module_from_array(recv_buffer) != np:
                raise ValueError("Host buffer must be on a host array.")

            send_data.get(out=send_buffer)

            self.comm.Allreduce(send_data, recv_data, op=MPI.__dict__[op.upper()])

            recv_data.set(arr=recv_buffer)

    def alltoall(
        self,
        send_data: ArrayLike,
        recv_data: ArrayLike,
        send_buffer: None | ArrayLike = None,
        recv_buffer: None | ArrayLike = None,
    ) -> None:
        """
        Send data from all processes to all.

        Parameters:
        send_data (ArrayLike): The data to send.
        recv_data (ArrayLike): The buffer to receive the data.
        """
        if _get_module_from_array(send_data) != _get_module_from_array(recv_data):
            raise ValueError("Send and receive data must be on the same array module.")

        if (
            _get_module_from_array(send_data) == np
            and _get_module_from_array(recv_data) == np
        ):
            self.comm.Alltoall(send_data, recv_data)
        else:
            if send_buffer is None:
                send_buffer = np.empty_like(send_data)
            if _get_module_from_array(send_buffer) != np:
                raise ValueError("Host buffer must be on a host array.")
            if recv_buffer is None:
                recv_buffer = np.empty_like(recv_data)
            if _get_module_from_array(recv_buffer) != np:
                raise ValueError("Host buffer must be on a host array.")

            send_data.get(out=send_buffer)

            self.comm.Alltoall(send_data, recv_data)

            recv_data.set(arr=recv_buffer)

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
        return ODAMPI(self.comm.Split(color, key))

    def dup(self) -> 'OCOMM':
        """
        Duplicate the communicator.

        Returns:
        Communicator: A new communicator.
        """
        return ODAMPI(self.comm.Dup())

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
