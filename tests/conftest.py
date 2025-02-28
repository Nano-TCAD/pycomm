
import pytest
import numpy as np

from pycomm import backend_flags

if backend_flags["cupy_avail"]:
    import cupy as cp

from pycomm import OCOMM, OBARE, OMPI, ODAMPI, ONCCL

OCOMM_TYPES = [OBARE]

if backend_flags["mpi_avail"]:
    OCOMM_TYPES.extend([OMPI])

    if backend_flags["cuda_avail"]:
        OCOMM_TYPES.extend([ODAMPI])

    if backend_flags["nccl_avail"]:
        OCOMM_TYPES.extend([ONCCL])


ARRAY_MODULE = [
    pytest.param(np, id="host"),
]
if backend_flags["cupy_avail"]:
    ARRAY_MODULE.extend(
        [
            pytest.param(cp, id="device"),
        ]
    )

INPUT_SHAPES = [
    pytest.param((10,), id="1D-stack"),
    pytest.param((7, 2), id="2D-stack"),
    pytest.param((9, 2, 4), id="3D-stack"),
]


@pytest.fixture(params=OCOMM_TYPES, autouse=True)
def ocomm_type(request: pytest.FixtureRequest) -> OCOMM:
    return request.param


@pytest.fixture(params=ARRAY_MODULE, autouse=True)
def xp(request: pytest.FixtureRequest) -> str:
    return request.param

@pytest.fixture(params=INPUT_SHAPES, autouse=True)
def input_shape(request: pytest.FixtureRequest) -> str:
    return request.param