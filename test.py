"""Test script to verify jaxmaze installation works correctly.

Runs the core functionality from example.ipynb.
"""

import jax
import jax.tree_util as jtu
import jax.numpy as jnp
import numpy as np

from jaxmaze import levels
from jaxmaze import env as maze
from jaxmaze import renderer
from jaxmaze import utils


def test_find_optimal_path():
  """Test the find_optimal_path function using a predefined map."""
  # Load images to get object_to_index mapping
  image_dict = utils.load_image_dict()
  object_to_index = {key: idx for idx, key in enumerate(image_dict["keys"])}

  # Character to object mapping for the two_objects level
  char_to_key = dict(A="knife", B="fork")

  # Parse the predefined level
  map_init = utils.from_str(
    levels.two_objects, char_to_key=char_to_key, object_to_index=object_to_index
  )

  grid = np.array(map_init.grid)
  agent_pos = map_init.agent_pos  # bfs handles JAX array conversion internally
  goal = object_to_index["knife"]  # Find path to object A (knife)

  path = utils.find_optimal_path(grid, agent_pos, goal)

  assert path is not None, "Should find a path"
  assert len(path) > 0, "Path should not be empty"
  assert tuple(path[0]) == tuple(agent_pos), "Path should start at agent position"

  # Verify path ends at a cell containing the goal
  end_pos = path[-1]
  assert grid[end_pos[0], end_pos[1], 0] == goal, "Path should end at goal"

  # Verify path is valid (each step is adjacent)
  for i in range(1, len(path)):
    prev = path[i - 1]
    curr = path[i]
    dist = abs(curr[0] - prev[0]) + abs(curr[1] - prev[1])
    assert dist == 1, f"Path step {i} is not adjacent: {prev} -> {curr}"

  print(f"Found optimal path with {len(path)} steps from {agent_pos} to knife")
  print("find_optimal_path test passed!")


def test_basic_functionality():
  """Test basic maze creation, environment setup, and stepping."""

  # Load images
  image_dict = utils.load_image_dict()
  print(f"Loaded {len(image_dict['keys'])} image categories")
  assert len(image_dict["keys"]) > 0, "No image categories loaded"

  # Define test mazes
  maze1 = """
.............
.............
.............
.............
...#######...
...#.....#...
...#..>..#...
...#.A...#...
...#...B.#...
...#######...
.............
.............
.............
""".strip()

  maze2 = """
.#.C...##....
.#..D...####.
.######......
......######.
.#.#..#......
.#.#.##..#...
##.#.#>.###.#
A..#.##..#...
.B.#.........
#####.#..####
......####.#.
.######E.#.#.
........F#...
""".strip()

  # Character to object mapping
  char_to_key = dict(
    A="knife",
    B="fork",
    C="pan",
    D="pot",
    E="bowl",
    F="plates",
  )

  object_to_index = {key: idx for idx, key in enumerate(image_dict["keys"])}
  objects = np.array([object_to_index[v] for v in char_to_key.values()])

  # Parse mazes
  map1_init = utils.from_str(
    maze1, char_to_key=char_to_key, object_to_index=object_to_index
  )
  print("Parsed maze1 successfully")

  map2_init = utils.from_str(
    maze2, char_to_key=char_to_key, object_to_index=object_to_index
  )
  print("Parsed maze2 successfully")

  # Test rendering
  image1 = renderer.create_image_from_grid(
    map1_init.grid, map1_init.agent_pos, map1_init.agent_dir, image_dict
  )
  assert image1.shape[0] > 0 and image1.shape[1] > 0, "Invalid image shape"
  print(f"Rendered maze1: shape={image1.shape}")

  image2 = renderer.create_image_from_grid(
    map2_init.grid, map2_init.agent_pos, map2_init.agent_dir, image_dict
  )
  assert image2.shape[0] > 0 and image2.shape[1] > 0, "Invalid image shape"
  print(f"Rendered maze2: shape={image2.shape}")

  # Combine map inits (tree_map already returns a MapInit with stacked leaves)
  map_init = jtu.tree_map(lambda *v: jnp.stack(v), map1_init, map2_init)
  print(f"Combined map_init grid shape: {map_init.grid.shape}")

  # Create env params
  env_params = maze.EnvParams(
    map_init=jax.tree_util.tree_map(jnp.asarray, map_init),
    time_limit=jnp.array(50),
    objects=jnp.asarray(objects),
  )

  # Initialize environment
  seed = 6
  rng = jax.random.PRNGKey(seed)

  task_runner = maze.TaskRunner(task_objects=env_params.objects)
  env = maze.HouseMaze(
    task_runner=task_runner,
    num_categories=len(image_dict["keys"]),
  )
  env = utils.AutoResetWrapper(env)

  # Test reset
  reset_timestep = env.reset(rng, env_params)
  print(f"Reset successful, agent_pos: {reset_timestep.state.agent_pos}")
  assert reset_timestep.state.grid is not None, "Grid not initialized"

  # Test step
  rng, step_rng = jax.random.split(rng)
  action = jnp.array(0)  # right
  next_timestep = env.step(step_rng, reset_timestep, action, env_params)
  print(f"Step successful, new agent_pos: {next_timestep.state.agent_pos}")

  # Render final state
  final_image = renderer.create_image_from_grid(
    next_timestep.state.grid,
    next_timestep.state.agent_pos,
    next_timestep.state.agent_dir,
    image_dict,
  )
  print(f"Final render shape: {final_image.shape}")

  print("\nAll tests passed!")


if __name__ == "__main__":
  test_find_optimal_path()
  test_basic_functionality()
