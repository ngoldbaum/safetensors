import sys
import tempfile
import threading

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from safetensors.numpy import load_file as load_file_np
from safetensors.numpy import save_file as save_file_np
from safetensors.torch import load_file as load_file_pt
from safetensors.torch import save_file as save_file_pt
from safetensors.flax import load_file as load_file_jax
from safetensors.flax import save_file as save_file_jax

CREATE_TENSOR = {
    "numpy": np.array,
    "pytorch": torch.tensor,
    "jax": jnp.array,
}


def random_shim(*args):
    jax.random.normal(args)


CREATE_RANDOM = {
    "numpy": np.random.randn,
    "pytorch": torch.randn,
    "jax": random_shim,
}

CREATE_ONES = {
    "numpy": np.ones,
    "pytorch": torch.ones,
    "jax": jnp.ones,
}

ALL = {
    "numpy": np.all,
    "pytorch": torch.all,
    "jax": jnp.all,
}

SAVE_FILE = {
    "numpy": save_file_np,
    "pytorch": save_file_pt,
    "jax": save_file_jax,
}

LOAD_FILE = {
    "numpy": load_file_np,
    "pytorch": load_file_pt,
    "jax": load_file_jax,
}

INT32 = {
    "numpy": np.int32,
    "pytorch": torch.int32,
    "jax": jnp.int32,
}


@pytest.mark.parametrize("backend", ["numpy", "pytorch", "jax"])
def test_multithreaded_roundtripping(backend):
    b = threading.Barrier(4)
    done = 0

    def save_worker(tensors):
        b.wait()
        try:
            for _ in range(10):
                with tempfile.NamedTemporaryFile() as fp:
                    SAVE_FILE[backend](tensors, fp.name)
                    loaded_tensors = LOAD_FILE[backend](fp.name)
                    for name, tensor in tensors.items():
                        assert ALL[backend](loaded_tensors[name] == tensor)
        finally:
            nonlocal done
            done += 1

    tensors = {
        "1": CREATE_RANDOM[backend](5, 25),
        "2": CREATE_RANDOM[backend](876, 768, 2),
        "3": CREATE_ONES[backend](5000),
        "4": CREATE_TENSOR[backend](5000.0),
        "5": CREATE_TENSOR[backend](768, dtype=INT32[backend]),
    }

    try:
        # the default thread switch interval is 5 milliseconds
        orig_switch = sys.getswitchinterval()
        sys.setswitchinterval(0.000001)  # in seconds

        tasks = [threading.Thread(target=save_worker, args=(tensors,)) for _ in range(4)]
        [t.start() for t in tasks]
        [t.join() for t in tasks]
    finally:
        # just in case one of the threads never started
        b.abort()
        sys.setswitchinterval(orig_switch)
