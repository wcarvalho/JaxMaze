from typing import Callable, Tuple

from flax import struct
import jax
import jax.numpy as jnp


Grid = jax.Array
AgentPos = jax.Array
AgentDir = jax.Array
ActionOutput = Tuple[Grid, AgentPos, AgentDir]


@struct.dataclass
class TaskState:
  features: jax.Array
  grid: jax.Array


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

  def compute_nearby_objects(self, grid, agent_pos):
    # Create a window around the agent's position
    y, x = agent_pos
    H, W, _ = grid.shape
    y_min, y_max = (
      jnp.maximum(0, y - self.radius),
      jnp.minimum(H, y + self.radius + 1),
    )
    x_min, x_max = (
      jnp.maximum(0, x - self.radius),
      jnp.minimum(W, x + self.radius + 1),
    )

    # Create a mask for the window
    y_indices = jnp.arange(H)
    x_indices = jnp.arange(W)
    y_mask = (y_indices >= y_min) & (y_indices < y_max)
    x_mask = (x_indices >= x_min) & (x_indices < x_max)
    window_mask = y_mask[:, jnp.newaxis] & x_mask[jnp.newaxis, :]

    # Use the mask to check for task objects
    def check_object(obj):
      object_present = grid == obj
      nearby = (object_present[:, :, 0] & window_mask).any()
      return nearby

    is_nearby = jax.vmap(check_object)(self.task_objects)
    return 0.05 * self.convert_type(is_nearby)

  def reset(self, grid: jax.Array, agent_pos: jax.Array):
    """Get initial features.

    Args:
        visible_grid (GridState): _description_
        agent (AgentState): _description_

    Returns:
        _type_: _description_
    """
    obtained_features = self.convert_type(jnp.zeros_like(self.task_objects))

    # Compute which objects are nearby
    nearby_objects = self.compute_nearby_objects(grid, agent_pos)

    # Concatenate obtained_features and nearby_objects
    features = jnp.concatenate([obtained_features, nearby_objects])
    features = self.convert_type(features)

    return TaskState(
      features=features,
      grid=grid,
    )

  def step(self, prior_state: TaskState, grid: jax.Array, agent_pos: jax.Array):
    # was an object removed?
    def count(g):
      present = g == self.task_objects[None, None]
      return present.sum(axis=(0, 1))

    count_prev = count(prior_state.grid)
    count_now = count(grid)

    decrease = count_prev - count_now

    # Compute which objects are nearby
    nearby_objects = self.compute_nearby_objects(grid, agent_pos)

    # Concatenate decrease and nearby_objects
    features = jnp.concatenate([decrease, nearby_objects])

    return TaskState(grid=grid, features=self.convert_type(features))
