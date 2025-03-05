import os

import jax
import jax.numpy as jnp
import numpy as np

from housemaze import levels
from housemaze.human_dyna import multitask_env as maze
from housemaze.utils import *


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


# def load_env_params(
#      num_groups: int,
#      max_objects: int = 3,
#      file: str = 'list_of_groups.npy',
#      large_only: bool = False,
#    ):
#    # load groups
#    if os.path.exists(file):
#      list_of_groups = np.load(file)
#    else:
#      raise RuntimeError(f"Missing file specifying groups for maze: {file}")

#    group_set = list_of_groups[0]
#    assert num_groups <= 3
#    group_set = group_set[:num_groups]

#    # load levels
#    pretrain_level = levels.two_objects
#    train_level = levels.three_pairs_maze1

#    ##################
#    # create reset parameters
#    ##################

#    list_of_reset_params = []
#    num_starting_locs = 4
#    max_starting_locs = 10
#    # -------------
#    # pretraining levels
#    # -------------
#    for group in group_set:
#      grid, agent_pos, agent_dir = from_str(
#         pretrain_level, char_to_key=dict(A=group[0], B=group[1]))
#      spawn_locs = from_str_spawning(pretrain_level)
#      list_of_reset_params.append(
#          make_reset_params(
#              map_init=maze.MapInit(
#                  grid, agent_pos, agent_dir, spawn_locs),
#              train_objects=group[:1],
#              test_objects=group[1:],
#              max_objects=max_objects,
#              starting_locs=make_int_array(
#                  np.ones((len(group_set), max_starting_locs, 2))*-1)
#          )
#      )

#    # -------------
#    # MAIN training level
#    # -------------
#    train_objects = group_set[:, 0]
#    test_objects = group_set[:, 1]
#    spawn_locs = from_str_spawning(train_level)
#    map_init = maze.MapInit(*from_str(
#        train_level,
#        char_to_key=dict(
#            A=group_set[0, 0],
#            B=group_set[0, 1],
#            C=group_set[1, 0],
#            D=group_set[1, 1],
#            E=group_set[2, 0],
#            F=group_set[2, 1],
#        )),
#        spawn_locs=spawn_locs)

#    all_starting_locs = np.ones((len(group_set), max_starting_locs, 2))*-1
#    for idx, goal in enumerate(train_objects):
#        path = find_optimal_path(
#            map_init.grid, map_init.agent_pos, np.array([goal]))
#        width = len(path)//num_starting_locs
#        starting_locs = np.array([path[i] for i in range(0, len(path), width)])
#        all_starting_locs[idx, :len(starting_locs)] = starting_locs

#    if large_only:
#       list_of_reset_params = []
#    list_of_reset_params.append(
#        make_reset_params(
#            map_init=map_init,
#            train_objects=train_objects,
#            test_objects=test_objects,
#            max_objects=3,
#            starting_locs=make_int_array(all_starting_locs),
#            curriculum=jnp.array(True),
#        )
#    )

#    return group_set, maze.EnvParams(
#        reset_params=jax.tree.map(
#            lambda *v: jnp.stack(v), *list_of_reset_params),
#    )
