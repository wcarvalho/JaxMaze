import os
import jax.numpy as jnp
import jax.tree_util as jtu

import numpy as np

from jaxmaze import levels
from jaxmaze.utils import *
from jaxmaze.human_dyna import multitask_env as maze


def make_int_array(x):
  return jnp.asarray(x, dtype=jnp.int32)


def load_groups(file: str = None):
  if file is None or file == "":
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    file = f"{current_directory}/list_of_groups.npy"
    print(f"No file specified for groups.\nUsing: {file}")
  return np.load(file)


def make_reset_params(
  map_init, train_objects, test_objects, max_objects: int = 3, **kwargs
):
  train_objects_ = np.ones(max_objects) * -1
  train_objects_[: len(train_objects)] = train_objects
  test_objects_ = np.ones(max_objects) * -1
  test_objects_[: len(test_objects)] = test_objects
  map_init = map_init.replace(
    grid=make_int_array(map_init.grid),
    agent_pos=make_int_array(map_init.agent_pos),
    agent_dir=make_int_array(map_init.agent_dir),
  )
  return maze.ResetParams(
    map_init=map_init,
    train_objects=make_int_array(train_objects_),
    test_objects=make_int_array(test_objects_),
    **kwargs,
  )
