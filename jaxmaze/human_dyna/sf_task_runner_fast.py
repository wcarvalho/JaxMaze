"""Optimized version of sf_task_runner.py.

Changes from sf_task_runner.py:
1. TaskState no longer stores grid (removed duplicate data)
2. TaskRunner.step() uses O(D) single-cell lookup instead of 2x full grid scan
3. compute_nearby_objects() uses dynamic_slice window instead of full-grid masking
"""

from typing import Callable, Tuple

from flax import struct
import jax
import jax.numpy as jnp


Grid = jax.Array
AgentPos = jax.Array
AgentDir = jax.Array
ActionOutput = Tuple[Grid, AgentPos, AgentDir]


# OPTIMIZATION: TaskState no longer stores grid
@struct.dataclass
class TaskState:
  features: jax.Array


test_level1 = """
.......................A..
..........................
..........................
..........................
............B.............
..........................
..........................
...........>..............
........C.................
......D...................
..........................
..........................
..........................
..........................
..........................
..........................
..........................
..........................
""".strip()


class TaskRunner(struct.PyTreeNode):
  """_summary_

  members:
      task_objects (jax.Array): [task_objects]

  Returns:
      _type_: _description_
  """

  task_objects: jax.Array
  convert_type: Callable[[jax.Array], jax.Array] = lambda x: x.astype(jnp.float32)
  radius: int = 5
  vis_coeff: float = 0.1

  def task_vector(self, object):
    """once for obtained. once for visible."""
    w = self.convert_type((object[None] == self.task_objects))
    # only get reward for getting object, not seeing it
    return jnp.concatenate([w, w * self.vis_coeff])

  def check_terminated(self, features, task_w):
    del task_w
    half = features.shape[-1] // 2
    return (features[:half]).sum(-1) > 0

  # OPTIMIZATION: dynamic_slice window instead of full-grid masking
  def compute_nearby_objects(self, grid, agent_pos):
    y, x = agent_pos
    H, W, C = grid.shape
    window_size = 2 * self.radius + 1

    # Pad grid so we never go out of bounds
    pad_grid = jnp.pad(
      grid,
      ((self.radius, self.radius), (self.radius, self.radius), (0, 0)),
      constant_values=0,
    )
    # Slice the window (shifted by padding offset)
    sub_grid = jax.lax.dynamic_slice(pad_grid, (y, x, 0), (window_size, window_size, C))

    # Check each object
    def check_object(obj):
      return (sub_grid == obj).any()

    is_nearby = jax.vmap(check_object)(self.task_objects)
    return 0.05 * self.convert_type(is_nearby)

  def reset(self, grid: jax.Array, agent_pos: jax.Array):
    obtained_features = self.convert_type(jnp.zeros_like(self.task_objects))

    # Compute which objects are nearby
    nearby_objects = self.compute_nearby_objects(grid, agent_pos)

    # Concatenate obtained_features and nearby_objects
    features = jnp.concatenate([obtained_features, nearby_objects])
    features = self.convert_type(features)

    return TaskState(features=features)

  # OPTIMIZATION: O(D) single-cell lookup instead of 2x full grid scan
  # Takes prior_grid as argument instead of reading from TaskState
  def step(self, prior_grid: jax.Array, grid: jax.Array, agent_pos: jax.Array):
    old_cell = prior_grid[agent_pos[0], agent_pos[1]]
    decrease = self.convert_type(old_cell == self.task_objects)

    # Compute which objects are nearby
    nearby_objects = self.compute_nearby_objects(grid, agent_pos)

    # Concatenate decrease and nearby_objects
    features = jnp.concatenate([decrease, nearby_objects])

    return TaskState(features=self.convert_type(features))
