import pytest

from pycomm import OCOMM

@pytest.mark.usefixtures("ocomm_type")
class TestCreation:
    
    def test_init_comm(
        self,
        ocomm_type: OCOMM,
    ):
        try:
            comm = ocomm_type()
        except Exception as e:
            assert False, f"Error: {e}"
        assert True

@pytest.mark.usefixtures("ocomm_type")
@pytest.mark.usefixtures("xp")
@pytest.mark.usefixtures("input_shape")
class TestPointToPoint:
    
    def test_ring_ptp(
            self,
            ocomm_type: OCOMM,
            xp,
            input_shape: tuple,
    ):
        comm = ocomm_type()
        size = comm.size()
        rank = comm.rank()
        
        data = rank * xp.ones(input_shape)
        comm.send(data, (rank + 1) % size)
        data = comm.recv(data, (rank - 1) % size)
        
        assert xp.allclose(data, (rank - 1) % size * xp.ones(input_shape))


@pytest.mark.usefixtures("ocomm_type")
@pytest.mark.usefixtures("xp")
@pytest.mark.usefixtures("input_shape")
class TestCollectives:
    
    def test_bcast(
        self,
        ocomm_type: OCOMM,
        xp,
        input_shape: tuple,
    ):
        comm = ocomm_type()
        rank = comm.rank()

        if rank == 0:
            data_send = xp.ones(input_shape)
        else:
            data_send = xp.empty(input_shape)

        data_recv = comm.bcast(data_send, root=0)

        assert xp.allclose(xp.ones(input_shape), data_recv)

    def test_scatter(
        self,
        ocomm_type: OCOMM,
        xp,
        input_shape: tuple,
    ):
        comm = ocomm_type()
        size = comm.size()
        rank = comm.rank()

        if input_shape[0] % size != 0:
            return
        
        recv_shape = (input_shape[0] // size,) + input_shape[1:]

        if rank == 0:
            data_send = xp.ones(input_shape)
        else:
            data_send = xp.empty(input_shape)

        data_recv = xp.empty(recv_shape)
        comm.scatter(data_send, data_recv, root=0)

        assert xp.allclose(data_recv, xp.ones(recv_shape))


@pytest.mark.usefixtures("ocomm_type")
class TestSynchronization:
    ...



@pytest.mark.usefixtures("ocomm_type")
class TestCommunicators:
    ...


@pytest.mark.usefixtures("ocomm_type")
class TestProcessManagment:
    ...