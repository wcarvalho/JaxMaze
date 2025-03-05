import importlib.resources
from typing import Optional
from collections import deque

import jax
import jax.numpy as jnp
import numpy as np
import pickle
from housemaze.env import KeyboardActions


def replace_color(image, old_color, new_color):
  # Convert the image and colors to JAX arrays if they aren't already
  image = np.asarray(image)
  old_color = np.asarray(old_color)
  new_color = np.asarray(new_color)

  # Create a mask where all pixels match the old_color
  mask = np.all(image == old_color, axis=-1)

  # Replace the color
  image[mask] = new_color

  return image


def sample_groups(
  categories, ordered_keys, rng, num_groups: int = 4, elem_per_group: int = 2
):
  potential_groups = list(categories.values())
  potential_groups = [[ordered_keys.index(i) for i in g] for g in potential_groups]
  assert num_groups >= len(potential_groups)

  # Splitting the key for sampling group indices
  key1, key2 = jax.random.split(rng)
  groups_idxs = jax.random.choice(
    key1, jnp.arange(len(potential_groups)), shape=(num_groups,), replace=False
  )

  groups = []
  for idx in groups_idxs:
    # Split the key for each iteration to ensure different randomness
    key2, subkey = jax.random.split(key2)
    groups.append(
      jax.random.choice(
        subkey,
        jnp.array(potential_groups[idx]),
        shape=(elem_per_group,),
        replace=False,
      )
    )

  return jnp.array(groups)


def sample_n_groups(
  categories, ordered_keys, n: int, num_groups: int = 4, elem_per_group: int = 2
):
  list_of_groups = []
  for seed in range(n):
    rng = jax.random.PRNGKey(seed)
    groups = sample_groups(
      categories=categories,
      ordered_keys=ordered_keys,
      rng=rng,
      num_groups=num_groups,
      elem_per_group=elem_per_group,
    )
    list_of_groups.append(groups)
  return jnp.array(list_of_groups)


def find_optimal_path(grid, agent_pos, goal, rng=None):
  if rng is None:
    rng = jax.random.PRNGKey(42)
  return bfs(grid, agent_pos, goal, rng)[0]


def bfs(grid, agent_pos, goal, key, budget=1e8):
  rows, cols, _ = grid.shape
  queue = deque([(agent_pos, [agent_pos])])
  visited = set()
  iterations = 0

  while queue:
    key, subkey = jax.random.split(key)
    iterations += 1
    if iterations >= budget:
      return None, iterations

    current_pos, path = queue.popleft()
    if grid[current_pos[0], current_pos[1], 0] == goal:
      return jnp.array([p for p in path]), iterations
    visited.add(current_pos)

    # Shuffle the order of directions
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    key, subkey = jax.random.split(key)
    directions = jax.random.permutation(subkey, jnp.array(directions))

    for dx, dy in directions:
      new_x, new_y = int(current_pos[0] + dx), int(current_pos[1] + dy)
      if (
        0 <= new_x < rows
        and 0 <= new_y < cols
        and (new_x, new_y) not in visited
        and (grid[new_x, new_y, 0] == 0 or grid[new_x, new_y, 0] == goal)
      ):
        new_path = path + [(new_x, new_y)]
        iterations += 1
        queue.append(((new_x, new_y), new_path))

  return None, iterations


def dfs(grid, agent_pos, goal, key, budget=1e8):
  rows, cols, _ = grid.shape
  stack = deque([(agent_pos, [agent_pos])])
  visited = set()
  iterations = 0

  while stack:
    key, subkey = jax.random.split(key)
    iterations += 1
    if iterations >= budget:
      return None, iterations

    current_pos, path = stack.pop()  # Use pop to simulate stack (LIFO)
    if grid[current_pos[0], current_pos[1], 0] == goal:
      return jnp.array([p for p in path]), iterations
    visited.add(current_pos)

    # Shuffle the order of directions (optional)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    directions = jax.random.permutation(subkey, jnp.array(directions))

    for dx, dy in directions:
      new_x, new_y = int(current_pos[0] + dx), int(current_pos[1] + dy)
      if (
        0 <= new_x < rows
        and 0 <= new_y < cols
        and (new_x, new_y) not in visited
        and (grid[new_x, new_y, 0] == 0 or grid[new_x, new_y, 0] == goal)
      ):
        new_path = path + [(new_x, new_y)]
        iterations += 1
        stack.append(((new_x, new_y), new_path))

  return None, iterations


def actions_from_path(path):
  if path is None or len(path) < 2:
    return np.array([KeyboardActions.done])

  actions = []
  for i in range(1, len(path)):
    prev_pos = path[i - 1]
    curr_pos = path[i]

    dx = curr_pos[0] - prev_pos[0]
    dy = curr_pos[1] - prev_pos[1]

    if dx == 1:
      actions.append(KeyboardActions.down)
    elif dx == -1:
      actions.append(KeyboardActions.up)
    elif dy == 1:
      actions.append(KeyboardActions.right)
    elif dy == -1:
      actions.append(KeyboardActions.left)

  actions.append(KeyboardActions.done)
  return np.array(actions)


def count_action_changes(actions):
  if len(actions) < 2:
    return np.zeros(len(actions), dtype=int)

  changes = np.zeros(len(actions), dtype=int)
  for i in range(1, len(actions)):
    if actions[i] != actions[i - 1]:
      changes[i] = 1

  return changes


def load_image_dict(file: str = None, add_borders: bool = False):
  if file is None or file == "":
    file = importlib.resources.files(__package__) / 'assets' / 'image_data.pkl'
    print(f"No file specified for image dict.\nUsing: {file}")

  if not add_borders:

    def add_border(x):
      return x

  with open(file, "rb") as f:
    image_dict = pickle.load(f)

  tile_size = image_dict["images"].shape[-2]

  images = image_dict["images"]

  new_images = []
  for image in images:
    image = replace_color(image, (255, 255, 255), (0, 0, 0))
    image = add_border(image)
  new_images = np.array(new_images)

  extra_keys = [
    ("wall", np.tile([100, 100, 100], (tile_size, tile_size, 1))),
    ("empty", add_border(np.tile([0, 0, 0], (tile_size, tile_size, 1)))),
  ]

  for key, img in extra_keys:
    assert not key in image_dict["keys"]
    image_dict["keys"] = [key] + image_dict["keys"]
    image_dict["images"] = jnp.concatenate((img[None], image_dict["images"]))

  return image_dict


def from_str(
  level_str: str,
  char_to_key: dict,
  object_to_index: Optional[dict] = None,
  check_grid_letters: bool = True,
):
  level_str = level_str.strip()
  rows = level_str.split("\n")
  nrows = len(rows)
  assert all(len(row) == len(rows[0]) for row in rows), "All rows must have same length"
  ncols = len(rows[0])

  def make_element(k: str):
    return np.array([object_to_index[k]], dtype=np.uint8)

  grid = np.zeros((nrows, ncols, 1), dtype=np.uint8)

  agent_pos = None
  agent_dir = None

  for y, row in enumerate(rows):
    for x, char in enumerate(row):
      if char == ".":  # EMPTY
        continue
      elif char == "#":  # WALL
        grid[y, x] = np.array(1)
      elif char == ">":
        #  assert agent_pos is None, "Agent position can only be set once."
        agent_pos, agent_dir = (y, x), 0
      elif char == "v":
        #  assert agent_pos is None, "Agent position can only be set once."
        agent_pos, agent_dir = (y, x), 1
      elif char == "<":
        #  assert agent_pos is None, "Agent position can only be set once."
        agent_pos, agent_dir = (y, x), 2
      elif char == "^":
        #  assert agent_pos is None, "Agent position can only be set once."
        agent_pos, agent_dir = (y, x), 3
      elif char == "@":
        pass
      elif char in char_to_key:
        key = char_to_key[char]
        if isinstance(key, str):
          object = make_element(key)
        elif isinstance(key, np.ndarray):
          object = key
        elif isinstance(key, np.int32):
          object = key
        else:
          raise NotImplementedError(f"type: {type(key)}")
        grid[y, x] = object
      else:
        if check_grid_letters:
          raise RuntimeError(f"Unknown char: {char}")
        else:
          print(f"Skipping '{char}' at ({y}, {x})")

  assert agent_pos is not None
  assert agent_dir is not None

  return grid, agent_pos, agent_dir


def from_str_spawning(level_str: str):
  level_str = level_str.strip()
  rows = level_str.split("\n")
  nrows = len(rows)
  assert all(len(row) == len(rows[0]) for row in rows), "All rows must have same length"
  ncols = len(rows[0])

  grid = np.zeros((nrows, ncols, 1), dtype=np.uint8)

  for y, row in enumerate(rows):
    for x, char in enumerate(row):
      if char in [">", "v", "<", "^", "@"]:
        grid[y, x] = np.array(1)
  return grid


def reverse(maze, horizontal=True, vertical=True):
  # Reverse each line
  if horizontal:
    reversed_lines = [line[::-1] for line in maze.splitlines()]
  else:
    reversed_lines = [line for line in maze.splitlines()]

  # Reverse the order of the lines
  if vertical:
    return "\n".join(reversed(reversed_lines))
  else:
    return "\n".join(reversed_lines)


def multiply_maze(maze, n=2, horizontal=True):
  if horizontal:
    return multiply_horizontally(maze, n)
  else:
    return multiply_vertically(maze, n)


def multiply_horizontally(maze, n=2):
  lines = maze.strip().split("\n")
  multiplied_lines = [line * n for line in lines]
  return "\n".join(multiplied_lines)


def multiply_vertically(maze, n=2):
  lines = maze.strip().split("\n")
  multiplied_lines = lines * n
  return "\n".join(multiplied_lines)


def cut(maze, n=3, h=True):
  if n == 0:
    return maze
  lines = maze.strip().split("\n")
  if h:
    cut_lines = [line[:-n] for line in lines]
  else:
    cut_lines = lines[:n]
  return "\n".join(cut_lines)


def combine_horizontally(*mazes):
  """
  Combines multiple mazes horizontally.

  Args:
  mazes (list): A list of mazes, where each maze is a string.

  Returns:
  str: The combined maze as a string.

  Raises:
  AssertionError: If the mazes don't have the same number of rows.
  ValueError: If the input list is empty.
  """
  if not mazes:
    raise ValueError("The list of mazes is empty")

  # Split all mazes into lines
  maze_lines = [maze.strip().split("\n") for maze in mazes]

  # Assert that all mazes have the same number of rows
  num_rows = len(maze_lines[0])
  assert all(len(lines) == num_rows for lines in maze_lines), (
    "All mazes must have the same number of rows"
  )

  # Combine the lines horizontally
  combined_lines = ["".join(lines) for lines in zip(*maze_lines)]

  # Join the lines back into a single string
  return "\n".join(combined_lines)


def combine_vertically(*mazes):
  """
  Combines multiple mazes vertically.

  Args:
  mazes (list): A list of mazes, where each maze is a string.

  Returns:
  str: The combined maze as a string.

  Raises:
  ValueError: If the input list is empty.
  AssertionError: If the mazes don't have the same number of columns.
  """
  if not mazes:
    raise ValueError("The list of mazes is empty")

  # Split all mazes into lines
  maze_lines = [maze.strip().split("\n") for maze in mazes]

  # Assert that all mazes have the same number of columns
  num_cols = len(maze_lines[0][0])
  assert all(len(line) == num_cols for maze in maze_lines for line in maze), (
    "All mazes must have the same number of columns"
  )

  # Combine the mazes vertically
  combined_lines = [line for maze in maze_lines for line in maze]

  # Join the lines back into a single string
  return "\n".join(combined_lines)


def insert(maze, column, char="#"):
  lines = maze.strip().split("\n")
  new_lines = []
  for line in lines:
    if column >= len(line):
      new_line = line + char
    else:
      new_line = line[:column] + char + line[column:]
    new_lines.append(new_line)
  return "\n".join(new_lines)


def compare_mazes(maze1, maze2):
  maze1_lines = maze1.strip().split("\n")
  maze2_lines = maze2.strip().split("\n")

  if len(maze1_lines) != len(maze2_lines):
    return (
      False,
      f"Mazes have different number of rows: {len(maze1_lines)} vs {len(maze2_lines)}",
    )

  differences = []

  for i, (line1, line2) in enumerate(zip(maze1_lines, maze2_lines)):
    if len(line1) != len(line2):
      return False, f"Row {i} has different length: {len(line1)} vs {len(line2)}"

    for j, (char1, char2) in enumerate(zip(line1, line2)):
      if char1 in ".#" and char2 in ".#" and char1 != char2:
        differences.append((i, j, char1, char2))

  return differences


class AutoResetWrapper:
  def __init__(self, env):
    self._env = env

  # provide proxy access to regular attributes of wrapped object
  def __getattr__(self, name):
    return getattr(self._env, name)

  def __auto_reset(self, key, params, timestep):
    key, key_ = jax.random.split(key)
    return self._env.reset(key_, params)

  def step(self, key: jax.Array, prior_timestep, action, params):
    return jax.lax.cond(
      prior_timestep.last(),
      lambda: self.__auto_reset(key, params, prior_timestep),
      lambda: self._env.step(key, prior_timestep, action, params),
    )
