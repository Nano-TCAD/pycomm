from numpy.typing import ArrayLike

from pycomm import OCOMM


class OBARE(OCOMM):
    """Oblivious Bare communicator."""
    def __init__(self):
        
        self.send_buff: dict = {}

    # Point-to-point communication (blocking) --------------------------------
    def send(self, data: ArrayLike, dest: int, tag: int = 0) -> None:
        """
        Send data from one process to another.

        Parameters:
        data (ArrayLike): The data to send.
        dest (int): The rank of the destination process.
        tag (int, optional): The message tag. Defaults to 0.
        """
        if dest != self.rank():
            raise ValueError("Destination rank must be the same as the source rank.")

        if tag not in self.send_buff:
            self.send_buff[tag] = []

        # We purposedly copy the data as send return imediatly and we cannot 
        # ensure otherwise that the data will not be modified.
        self.send_buff[tag].append(data.copy()) 


    def recv(self, buf: ArrayLike, source: int, tag: int = 0) -> None:
        """
        Receive data from another process.

        Parameters:
        buf (ArrayLike): The buffer to receive the data.
        source (int): The rank of the source process.
        tag (int, optional): The message tag. Defaults to 0.
        """
        if source != self.rank():
            raise ValueError("Source rank must be the same as the source rank.")
        
        if tag not in self.send_buff:
            raise ValueError("No data to receive.")
        
        buf[:] = self.send_buff[tag].pop(0)
        

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
        if root != self.rank():
            raise ValueError("Root rank must be the same as the source rank.")
        
        return data

    def scatter(self, send_data: ArrayLike, recv_data: ArrayLike, root: int = 0) -> None:
        """
        Scatter data from one process to all others.

        Parameters:
        send_data (ArrayLike): The data to scatter.
        recv_data (ArrayLike): The buffer to receive the scattered data.
        root (int, optional): The rank of the root process. Defaults to 0.
        """
        if root != self.rank():
            raise ValueError("Root rank must be the same as the source rank.")
        if send_data.shape != recv_data.shape:
            raise ValueError("Send and receive data shapes must match.")

        recv_data[:] = send_data

    def gather(self, send_data: ArrayLike, recv_data: ArrayLike, root: int = 0) -> None:
        """
        Gather data from all processes to one.

        Parameters:
        send_data (ArrayLike): The data to send.
        recv_data (ArrayLike): The buffer to receive the gathered data.
        root (int, optional): The rank of the root process. Defaults to 0.
        """
        if root != self.rank():
            raise ValueError("Root rank must be the same as the source rank.")
        if send_data.shape != recv_data.shape:
            raise ValueError("Send and receive data shapes must match.")

        recv_data[:] = send_data

    def allgather(self, send_data: ArrayLike, recv_data: ArrayLike) -> None:
        """
        Gather data from all processes to all.

        Parameters:
        send_data (ArrayLike): The data to send.
        recv_data (ArrayLike): The buffer to receive the gathered data.
        """
        if send_data.shape != recv_data.shape:
            raise ValueError("Send and receive data shapes must match.")
        
        recv_data[:] = send_data

    def reduce(self, send_data: ArrayLike, recv_data: ArrayLike, op: str, root: int = 0) -> None:
        """
        Reduce data from all processes to one.

        Parameters:
        send_data (ArrayLike): The data to send.
        recv_data (ArrayLike): The buffer to receive the reduced data.
        op (str): The reduction operation (e.g., 'sum', 'max').
        root (int, optional): The rank of the root process. Defaults to 0.
        """
        if root != self.rank():
            raise ValueError("Root rank must be the same as the source rank.")
        if send_data.shape != recv_data.shape:
            raise ValueError("Send and receive data shapes must match.")
        
        if op not in ['sum', 'max', 'min']:
            raise ValueError("Invalid reduction operation.")


        recv_data[:] = send_data


    def allreduce(self, send_data: ArrayLike, recv_data: ArrayLike, op: str) -> None:
        """
        Reduce data from all processes to all.

        Parameters:
        send_data (ArrayLike): The data to send.
        recv_data (ArrayLike): The buffer to receive the reduced data.
        op (str): The reduction operation (e.g., 'sum', 'max').
        """
        if send_data.shape != recv_data.shape:
            raise ValueError("Send and receive data shapes must match.")
        
        if op not in ['sum', 'max', 'min']:
            raise ValueError("Invalid reduction operation.")

        recv_data[:] = send_data

    def alltoall(self, send_data: ArrayLike, recv_data: ArrayLike) -> None:
        """
        Send data from all processes to all.

        Parameters:
        send_data (ArrayLike): The data to send.
        recv_data (ArrayLike): The buffer to receive the data.
        """
        if send_data.shape != recv_data.shape:
            raise ValueError("Send and receive data shapes must match.")

        recv_data[:] = send_data

    # Collective communication (non-blocking) ---------------------------------
        ...

    # Synchronization ---------------------------------------------------------
    def barrier(self) -> None:
        """
        Synchronize all processes.
        """
        pass

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
        if color != 0:
            raise ValueError("Color must be 0.")
        return self.copy()

    def dup(self) -> 'OCOMM':
        """
        Duplicate the communicator.

        Returns:
        Communicator: A new communicator.
        """
        return self.copy()

    # Process management -------------------------------------------------------
    def rank(self) -> int:
        """
        Get the rank of the process.

        Returns:
        int: The rank of the process.
        """
        return 0

    def size(self) -> int:
        """
        Get the size of the communicator.

        Returns:
        int: The size of the communicator.
        """
        return 1
