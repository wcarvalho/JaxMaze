import jax.numpy as jnp
import jax.tree_util as jtu


from jaxmaze.human_dyna import multitask_env
from jaxmaze.human_dyna import mazes
from jaxmaze.utils import reverse
import numpy as np


def maze1_all(config):
  """Maze 1: testing offtaskness for all 3 spaces."""
  env_kwargs = config.get("rlenv", {}).get("ENV_KWARGS", {})
  num_groups = env_kwargs.pop("NUM_GROUPS", 3)
  char2key, group_set, task_objects = mazes.get_group_set(num_groups)

  pretrain_params = mazes.get_pretraining_reset_params(
    groups=group_set,
  )
  main_params = mazes.get_maze_reset_params(
    groups=group_set,
    char2key=char2key,
    maze_str=mazes.maze1,
    label=jnp.array(0),
    curriculum=True,
  )
  train_params = multitask_env.EnvParams(
    reset_params=jtu.tree_map(
      lambda *v: jnp.stack(v), *(pretrain_params + main_params)
    ),
  )

  test_params = multitask_env.EnvParams(
    reset_params=jtu.tree_map(lambda *v: jnp.stack(v), *main_params),
  ).replace(training=False)

  return train_params, test_params, task_objects


def maze3_open(config):
  """Maze 3: testing if open space is skipped. should be."""
  env_kwargs = config.get("rlenv", {}).get("ENV_KWARGS", {})
  num_groups = env_kwargs.pop("NUM_GROUPS", 1)
  char2key, group_set, task_objects = mazes.get_group_set(num_groups)

  pretrain_params = mazes.get_pretraining_reset_params(
    groups=group_set,
  )
  main_params = mazes.get_maze_reset_params(
    groups=group_set,
    char2key=char2key,
    maze_str=mazes.maze3,
    # label=jnp.array(Labels.large),
    curriculum=True,
  )
  main_open_params = mazes.get_maze_reset_params(
    groups=group_set,
    char2key=char2key,
    maze_str=mazes.maze3_open,
    # label=jnp.array(Labels.shortcut),
  )

  train_params = pretrain_params + main_params
  train_params = multitask_env.EnvParams(
    reset_params=jtu.tree_map(lambda *v: jnp.stack(v), *train_params),
  )

  test_params = main_params + main_open_params
  test_params = multitask_env.EnvParams(
    reset_params=jtu.tree_map(lambda *v: jnp.stack(v), *test_params),
  ).replace(training=False)

  return train_params, test_params, task_objects


def maze3_randomize(config):
  """Maze 3: testing if open space is skipped. should be."""
  env_kwargs = config.get("rlenv", {}).get("ENV_KWARGS", {})
  num_groups = env_kwargs.pop("NUM_GROUPS", 1)
  char2key, group_set, task_objects = mazes.get_group_set(num_groups)

  pretrain_params = mazes.get_pretraining_reset_params(
    groups=group_set,
  )
  main_params = mazes.get_maze_reset_params(
    groups=group_set,
    char2key=char2key,
    maze_str=mazes.maze3,
    randomize_agent=True,
  )
  main_open_params = mazes.get_maze_reset_params(
    groups=group_set,
    char2key=char2key,
    maze_str=mazes.maze3_open,
    randomize_agent=True,
  )

  train_params = pretrain_params + main_params
  train_params = multitask_env.EnvParams(
    randomize_agent=True,
    reset_params=jtu.tree_map(lambda *v: jnp.stack(v), *train_params),
  )

  test_params = main_params + main_open_params
  test_params = multitask_env.EnvParams(
    training=False,
    randomize_agent=False,
    reset_params=jtu.tree_map(lambda *v: jnp.stack(v), *test_params),
  )

  return train_params, test_params, task_objects


def maze5_two_paths(config):
  """Maze 3: testing if open space is skipped. should be."""
  env_kwargs = config.get("rlenv", {}).get("ENV_KWARGS", {})
  num_groups = env_kwargs.pop("NUM_GROUPS", 1)
  char2key, group_set, task_objects = mazes.get_group_set(num_groups)

  pretrain_params = mazes.get_pretraining_reset_params(
    groups=group_set,
  )
  main_params = mazes.get_maze_reset_params(
    groups=group_set,
    char2key=char2key,
    maze_str=mazes.maze5,
    curriculum=True,
  )

  train_params = pretrain_params + main_params
  train_params = multitask_env.EnvParams(
    reset_params=jtu.tree_map(lambda *v: jnp.stack(v), *train_params),
  )

  test_params = main_params
  test_params = multitask_env.EnvParams(
    training=False,
    reset_params=jtu.tree_map(lambda *v: jnp.stack(v), *test_params),
  )

  return train_params, test_params, task_objects


def permute_groups(groups):
  # Flatten the groups
  flattened = groups.flatten()

  # Create a random permutation
  permutation = np.random.permutation(len(flattened))

  # Apply the permutation
  permuted_flat = flattened[permutation]

  # Reshape back to the original shape
  new_groups = permuted_flat.reshape(groups.shape)

  # Create a new char2idx mapping
  new_char2idx = mazes.groups_to_char2key(new_groups)

  return new_groups, new_char2idx


def rotate_group(groups):
  # Flatten the groups
  flattened = groups.flatten()

  # Apply the permutation
  permuted_flat = np.concatenate([flattened[1:], [flattened[0]]])
  # Reshape back to the original shape
  new_groups = permuted_flat.reshape(groups.shape)

  # Create a new char2idx mapping
  new_char2idx = mazes.groups_to_char2key(new_groups)

  return new_groups, new_char2idx


def basic_make_exp_block(
  config,
  train_mazes,
  eval_mazes,
  train_kwargs=None,
  eval_kwargs=None,
  pretrain_level=None,
  max_starting_locs=10,
  include_rotations=False,
  train_curriculum: bool = True,
  pass_through_objects: bool = False,
):
  def make_int(i):
    return jnp.array(i, dtype=jnp.int32)

  train_kwargs = train_kwargs or dict()
  eval_kwargs = eval_kwargs or dict()

  # setup
  env_kwargs = config.get("rlenv", {}).get("ENV_KWARGS", {})
  num_groups = env_kwargs.pop("NUM_GROUPS", 2)
  char2key, group_set, task_objects = mazes.get_group_set(num_groups)

  all_train_params = []
  all_eval_params = []

  all_mazes = list(set(train_mazes + eval_mazes))
  maze2idx = {
    maze_name: idx for idx, maze_name in enumerate(all_mazes + [pretrain_level])
  }

  #################################################
  # make function for creating params
  #################################################
  def make_params(maze_str, label, **kwargs):
    return mazes.get_maze_reset_params(
      groups=group_set,
      char2key=char2key,
      maze_str=maze_str,
      label=label,
      max_starting_locs=max_starting_locs,
      pass_through_objects=pass_through_objects,
      **kwargs,
    )

  def get_all_rotations(maze_str, label, **kwargs):
    params = []
    idx = 0
    for h in [False, True]:
      for w in [False, True]:
        params += mazes.get_maze_reset_params(
          groups=group_set,
          char2key=char2key,
          maze_str=reverse(maze_str, horizontal=h, vertical=w),
          label=label,
          max_starting_locs=max_starting_locs,
          pass_through_objects=pass_through_objects,
          rotation=jnp.array([int(h), int(w)]),
          **kwargs,
        )
        idx += 1
    return params

  make_fn = get_all_rotations if include_rotations else make_params

  if pretrain_level:
    all_train_params += make_fn(
      maze_str=getattr(mazes, pretrain_level),
      label=make_int(maze2idx[pretrain_level]),
      swap_train_test=True,
      curriculum=train_curriculum,
    )

  for maze_name in train_mazes:
    params = make_fn(
      maze_str=getattr(mazes, maze_name),
      label=make_int(maze2idx[maze_name]),
      curriculum=train_curriculum,
    )
    all_train_params += params
    all_eval_params += params

  for maze_name in eval_mazes:
    params = make_fn(
      maze_str=getattr(mazes, maze_name),
      label=make_int(maze2idx[maze_name]),
    )
    all_eval_params += params

  train_params = multitask_env.EnvParams(
    **train_kwargs,
    reset_params=jtu.tree_map(lambda *v: jnp.stack(v), *all_train_params),
  )

  test_params = multitask_env.EnvParams(
    **eval_kwargs,
    training=False,
    reset_params=jtu.tree_map(lambda *v: jnp.stack(v), *all_eval_params),
  )

  label2name = {idx: name for idx, name in enumerate(all_mazes)}
  return train_params, test_params, task_objects, label2name


def make_human_experiments_block(
  config,
  train_test_pairs,
  train_kwargs=None,
  eval_kwargs=None,
  pretrain_level=None,
  max_starting_locs=10,
  include_rotations=True,
  train_curriculum: bool = True,
):
  def make_int(i):
    return jnp.array(i, dtype=jnp.int32)

  train_kwargs = train_kwargs or dict()
  eval_kwargs = eval_kwargs or dict()

  # setup
  env_kwargs = config.get("rlenv", {}).get("ENV_KWARGS", {})
  num_groups = env_kwargs.pop("NUM_GROUPS", 2)
  char2key, group_set, task_objects = mazes.get_group_set(num_groups)

  all_train_params = []
  all_eval_params = []

  all_mazes = []
  for train_maze, eval_maze in train_test_pairs:
    all_mazes.append(train_maze)
    all_mazes.append(eval_maze)
  all_mazes = list(set(all_mazes))
  maze2idx = {
    maze_name: idx for idx, maze_name in enumerate(all_mazes + [pretrain_level])
  }
  import itertools

  rotations = list(itertools.product([False, True], repeat=2))
  # idx = 0
  original_group_set = group_set
  original_char2key = char2key

  if not include_rotations:
    rotations = [(False, False)]

  for train_maze, eval_maze in train_test_pairs:
    train_maze_str = getattr(mazes, train_maze)
    eval_maze_str = getattr(mazes, eval_maze)
    for rotation in rotations:
      if include_rotations:
        group_set, char2key = rotate_group(group_set)
      train_params = mazes.get_maze_reset_params(
        groups=group_set,
        char2key=char2key,
        maze_str=reverse(train_maze_str, horizontal=rotation[0], vertical=rotation[1]),
        label=make_int(maze2idx[train_maze]),
        max_starting_locs=max_starting_locs,
        curriculum=True,
        rotation=jnp.asarray(rotation),
      )
      all_train_params += train_params
      all_eval_params += train_params

      eval_params = mazes.get_maze_reset_params(
        groups=group_set,
        char2key=char2key,
        maze_str=reverse(eval_maze_str, horizontal=rotation[0], vertical=rotation[1]),
        label=make_int(maze2idx[eval_maze]),
        max_starting_locs=max_starting_locs,
        rotation=jnp.asarray(rotation),
        curriculum=False,
      )
      all_eval_params += eval_params

  if pretrain_level:
    pretrain_level_str = getattr(mazes, pretrain_level)
    for rotation in rotations:
      if include_rotations:
        group_set, char2key = rotate_group(group_set)
      all_train_params += mazes.get_maze_reset_params(
        groups=group_set,
        char2key=char2key,
        maze_str=reverse(
          pretrain_level_str, horizontal=rotation[0], vertical=rotation[1]
        ),
        label=make_int(maze2idx[pretrain_level]),
        max_starting_locs=max_starting_locs,
        swap_train_test=True,
        curriculum=True,
        rotation=jnp.asarray(rotation),
      )

  train_params = multitask_env.EnvParams(
    **train_kwargs,
    reset_params=jtu.tree_map(lambda *v: jnp.stack(v), *all_train_params),
  )

  test_params = multitask_env.EnvParams(
    **eval_kwargs,
    training=False,
    reset_params=jtu.tree_map(lambda *v: jnp.stack(v), *all_eval_params),
  )

  label2name = {idx: name for idx, name in enumerate(all_mazes)}
  return train_params, test_params, task_objects, label2name


def make_(
  config,
  train_mazes,
  eval_mazes,
  train_kwargs=None,
  eval_kwargs=None,
  pretrain_level=None,
  max_starting_locs=10,
  include_rotations=False,
):
  def make_int(i):
    return jnp.array(i, dtype=jnp.int32)

  train_kwargs = train_kwargs or dict()
  eval_kwargs = eval_kwargs or dict()

  # setup
  env_kwargs = config.get("rlenv", {}).get("ENV_KWARGS", {})
  num_groups = env_kwargs.pop("NUM_GROUPS", 2)
  char2key, group_set, task_objects = mazes.get_group_set(num_groups)

  all_train_params = []
  all_eval_params = []

  all_mazes = list(set(train_mazes + eval_mazes))
  maze2idx = {
    maze_name: idx for idx, maze_name in enumerate(all_mazes + [pretrain_level])
  }

  #################################################
  # make function for creating params
  #################################################
  def make_params(maze_str, label, **kwargs):
    return mazes.get_maze_reset_params(
      groups=group_set,
      char2key=char2key,
      maze_str=maze_str,
      label=label,
      max_starting_locs=max_starting_locs,
      **kwargs,
    )

  def get_all_rotations(maze_str, label, **kwargs):
    params = []
    idx = 0
    for h in [False, True]:
      for w in [False, True]:
        params += mazes.get_maze_reset_params(
          groups=group_set,
          char2key=char2key,
          maze_str=reverse(maze_str, horizontal=h, vertical=w),
          label=label,
          max_starting_locs=max_starting_locs,
          rotation=jnp.array([int(h), int(w)]),
          **kwargs,
        )
        idx += 1
    return params

  make_fn = get_all_rotations if include_rotations else make_params

  if pretrain_level:
    all_train_params += make_fn(
      maze_str=getattr(mazes, pretrain_level),
      label=make_int(maze2idx[pretrain_level]),
      swap_train_test=True,
      curriculum=True,
    )

  for maze_name in train_mazes:
    params = make_fn(
      maze_str=getattr(mazes, maze_name),
      label=make_int(maze2idx[maze_name]),
      curriculum=True,
    )
    all_train_params += params
    all_eval_params += params

  for maze_name in eval_mazes:
    params = make_fn(
      maze_str=getattr(mazes, maze_name),
      label=make_int(maze2idx[maze_name]),
    )
    all_eval_params += params

  train_params = multitask_env.EnvParams(
    **train_kwargs,
    reset_params=jtu.tree_map(lambda *v: jnp.stack(v), *all_train_params),
  )

  test_params = multitask_env.EnvParams(
    **eval_kwargs,
    training=False,
    reset_params=jtu.tree_map(lambda *v: jnp.stack(v), *all_eval_params),
  )

  label2name = {idx: name for idx, name in enumerate(all_mazes)}
  return train_params, test_params, task_objects, label2name


def exp1_block1(config):
  train_mazes = ["maze3"]
  eval_mazes = ["maze3_open2"]
  return basic_make_exp_block(config, train_mazes, eval_mazes)


def exp1_block2(config):
  train_mazes = ["maze3_r"]
  eval_mazes = ["maze3_onpath_shortcut_r", "maze3_offpath_shortcut_r"]
  return basic_make_exp_block(config, train_mazes, eval_mazes)


def exp1_block3(config):
  train_mazes = ["maze5"]
  eval_mazes = ["maze5"]
  return basic_make_exp_block(config, train_mazes, eval_mazes)


def exp1_block4(config):
  train_mazes = ["maze6"]
  eval_mazes = ["maze6_flipped_offtask"]
  return basic_make_exp_block(config, train_mazes, eval_mazes)


def exp1(config, analysis_eval: bool = False):
  train_mazes = ["maze3", "maze3_r", "maze5", "maze6"]
  if analysis_eval:
    eval_mazes = [
      "maze3_open2",
      "maze3_onpath_shortcut_r",
      "maze3_offpath_shortcut_r",
      "maze5",
      "maze6_flipped_offtask",
    ]
  else:
    eval_mazes = train_mazes
  return basic_make_exp_block(config, train_mazes, eval_mazes, max_starting_locs=10)


def exp2(config, analysis_eval: bool = False):
  train_mazes = [
    "big_m1_maze3",
    # "big_m2_maze2",
    "big_m3_maze1",
  ]
  if analysis_eval:
    eval_mazes = [
      "big_m1_maze3",
      "big_m1_maze3_shortcut",
      # "big_m2_maze2",
      # "big_m2_maze2_onpath",
      # "big_m2_maze2_offpath",
      "big_m3_maze1",
    ]
  else:
    eval_mazes = train_mazes
  return basic_make_exp_block(
    config,
    train_mazes,
    eval_mazes,
    pretrain_level="big_practice_maze",
    max_starting_locs=20,
  )


def exp3(config, analysis_eval: bool = False):
  train_mazes = ["big_m1_maze3", "big_m3_maze1"]
  if analysis_eval:
    eval_mazes = [
      "big_m1_maze3",
      "big_m1_maze3_shortcut",
      "big_m3_maze1",
    ]
  else:
    eval_mazes = train_mazes
  return basic_make_exp_block(
    config,
    train_mazes,
    eval_mazes,
    pretrain_level="big_practice_maze",
    max_starting_locs=20,
    include_rotations=True,
  )


def exp4(config, analysis_eval: bool = False):
  """
  This is blocked so each block has the same group rotation.

  A block is 1 rotation of a maze.
  It has both train and test on that maze.

  Blocks are:
    - (big_m3_maze1, big_m3_maze1) (0, 0), group_rotation = 0
    - (big_m1_maze3, big_m1_maze3_shortcut) (0, 0), group_rotation = 1
    - (big_m3_maze1, big_m3_maze1) (0, 1), group_rotation = 2
    - (big_m1_maze3, big_m1_maze3_shortcut) (0, 1), group_rotation = 3
    - etc.
  """
  if analysis_eval:
    train_test_pairs = [
      ("big_m3_maze1", "big_m3_maze1"),
      # ("big_m1_maze3", "big_m1_maze3"),
      ("big_m1_maze3", "big_m1_maze3_shortcut"),
    ]
  else:
    train_test_pairs = [
      ("big_m3_maze1", "big_m3_maze1"),
      ("big_m1_maze3", "big_m1_maze3"),
    ]

  return make_human_experiments_block(
    config,
    train_test_pairs,
    pretrain_level="big_practice_maze",
    max_starting_locs=config.get("NUM_STARTING_LOCS", 100),
    include_rotations=config.get("INCLUDE_ROTATIONS", True),
  )


def exp_test(config, analysis_eval: bool = False):
  del analysis_eval
  train_mazes = ["big_test_level"]
  eval_mazes = train_mazes
  return basic_make_exp_block(
    config,
    train_mazes,
    eval_mazes,
    pretrain_level="big_practice_maze",
    max_starting_locs=20,
  )


def her_test(config, analysis_eval: bool = False):
  del analysis_eval
  train_mazes = ["her_test"]
  eval_mazes = train_mazes
  return basic_make_exp_block(
    config,
    train_mazes,
    eval_mazes,
    pretrain_level=None,
    include_rotations=False,
    train_curriculum=False,
  )


def her_test_big(config, analysis_eval: bool = False):
  del analysis_eval
  train_mazes = ["her_test_big"]
  eval_mazes = train_mazes
  return basic_make_exp_block(
    config,
    train_mazes,
    eval_mazes,
    pretrain_level=None,
    include_rotations=False,
    train_curriculum=True,
    pass_through_objects=True,
  )

def preplay_test(config, analysis_eval: bool = False):
  del analysis_eval
  train_mazes = ["preplay_test"]
  eval_mazes = train_mazes
  return basic_make_exp_block(
    config,
    train_mazes,
    eval_mazes,
    pretrain_level=None,
    include_rotations=False,
    train_curriculum=False,
    pass_through_objects=False,
  )

def preplay_test_big(config, analysis_eval: bool = False):
  del analysis_eval
  train_mazes = ["preplay_test_big"]
  eval_mazes = train_mazes
  return basic_make_exp_block(
    config,
    train_mazes,
    eval_mazes,
    pretrain_level=None,
    include_rotations=False,
    train_curriculum=True,
    pass_through_objects=False,
  )
