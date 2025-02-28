# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

from abc import ABC, abstractmethod
from numpy.typing import ArrayLike

class OCOMM(ABC):
    """Base class for communication backends."""
    def __init__(self):
        ...

    # Point-to-point communication (blocking) --------------------------------
    @abstractmethod
    def send(self, data: ArrayLike, dest: int, tag: int = 0) -> None:
        """
        Send data from one process to another.

        Parameters:
        data (ArrayLike): The data to send.
        dest (int): The rank of the destination process.
        tag (int, optional): The message tag. Defaults to 0.
        """
        pass

    @abstractmethod
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
    @abstractmethod
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

    @abstractmethod
    def scatter(self, send_data: ArrayLike, recv_data: ArrayLike, root: int = 0) -> None:
        """
        Scatter data from one process to all others.

        Parameters:
        send_data (ArrayLike): The data to scatter.
        recv_data (ArrayLike): The buffer to receive the scattered data.
        root (int, optional): The rank of the root process. Defaults to 0.
        """
        pass

    @abstractmethod
    def gather(self, send_data: ArrayLike, recv_data: ArrayLike, root: int = 0) -> None:
        """
        Gather data from all processes to one.

        Parameters:
        send_data (ArrayLike): The data to send.
        recv_data (ArrayLike): The buffer to receive the gathered data.
        root (int, optional): The rank of the root process. Defaults to 0.
        """
        pass

    @abstractmethod
    def allgather(self, send_data: ArrayLike, recv_data: ArrayLike) -> None:
        """
        Gather data from all processes to all.

        Parameters:
        send_data (ArrayLike): The data to send.
        recv_data (ArrayLike): The buffer to receive the gathered data.
        """
        pass

    @abstractmethod
    def reduce(self, send_data: ArrayLike, recv_data: ArrayLike, op: str, root: int = 0) -> None:
        """
        Reduce data from all processes to one.

        Parameters:
        send_data (ArrayLike): The data to send.
        recv_data (ArrayLike): The buffer to receive the reduced data.
        op (str): The reduction operation (e.g., 'sum', 'max').
        root (int, optional): The rank of the root process. Defaults to 0.
        """
        pass

    @abstractmethod
    def allreduce(self, send_data: ArrayLike, recv_data: ArrayLike, op: str) -> None:
        """
        Reduce data from all processes to all.

        Parameters:
        send_data (ArrayLike): The data to send.
        recv_data (ArrayLike): The buffer to receive the reduced data.
        op (str): The reduction operation (e.g., 'sum', 'max').
        """
        pass

    @abstractmethod
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
    @abstractmethod
    def barrier(self) -> None:
        """
        Synchronize all processes.
        """
        pass

    # Communicators -----------------------------------------------------------
    @abstractmethod    
    def split(self, color: int, key: int) -> 'OCOMM':
        """
        Split the communicator into subgroups.

        Parameters:
        color (int): Control of subset assignment.
        key (int): Control of rank assignment.

        Returns:
        Communicator: A new communicator.
        """
        pass

    @abstractmethod
    def dup(self) -> 'OCOMM':
        """
        Duplicate the communicator.

        Returns:
        Communicator: A new communicator.
        """
        pass

    # Process management -------------------------------------------------------
    @abstractmethod
    def rank(self) -> int:
        """
        Get the rank of the process.

        Returns:
        int: The rank of the process.
        """
        pass

    @abstractmethod
    def size(self) -> int:
        """
        Get the size of the communicator.

        Returns:
        int: The size of the communicator.
        """
        pass

