from functools import partial
from typing import List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from housemaze.env import KeyboardActions


def point_in_rect(xmin, xmax, ymin, ymax):
  def fn(x, y):
    return x >= xmin and x <= xmax and y >= ymin and y <= ymax

  return fn


def fill_coords(img, fn, color):
  new_img = img.copy()
  for y in range(img.shape[0]):
    for x in range(img.shape[1]):
      yf = (y + 0.5) / img.shape[0]
      xf = (x + 0.5) / img.shape[1]
      if fn(xf, yf):
        new_img[y, x] = color
  return new_img


def point_in_triangle(a, b, c):
  a = np.array(a)
  b = np.array(b)
  c = np.array(c)

  def fn(x, y):
    v0 = c - a
    v1 = b - a
    v2 = np.array((x, y)) - a

    # Compute dot products
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    # Compute barycentric coordinates
    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    # Check if point is in triangle
    return (u >= 0) and (v >= 0) and (u + v) < 1

  return fn


def add_border(tile):
  new_tile = fill_coords(tile, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
  return fill_coords(new_tile, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))


def create_image_grid_from_image_tensor(images, max_cols: int = 10):
  num_images = images.shape[0]
  img_height, img_width, channels = images[0].shape

  # Calculate the number of rows and columns
  num_cols = min(max_cols, num_images)
  num_rows = (num_images + num_cols - 1) // num_cols  # Ceiling division

  # Create a blank canvas
  grid_height = num_rows * img_height
  grid_width = num_cols * img_width
  canvas = jnp.zeros((grid_height, grid_width, channels), dtype=images.dtype)

  # Place each image in the canvas
  for idx, image in enumerate(images):
    row = idx // num_cols
    col = idx % num_cols
    row_start = row * img_height
    row_end = row_start + img_height
    col_start = col * img_width
    col_end = col_start + img_width

    canvas = canvas.at[row_start:row_end, col_start:col_end, :].set(image)

  return canvas


def make_agent_tile(direction: int, tile_size: int):
  TRI_COORDS = np.array(
    [
      [0.12, 0.19],
      [0.87, 0.50],
      [0.12, 0.81],
    ]
  )

  agent_tile = np.tile([0, 0, 0], (tile_size, tile_size, 1))
  agent_tile = fill_coords(agent_tile, point_in_triangle(*TRI_COORDS), [255, 0, 0])
  add_border = lambda x: x
  return jax.lax.switch(
    direction,
    (
      lambda: add_border(agent_tile),  # right
      lambda: add_border(np.rot90(agent_tile, k=3)),  # down
      lambda: add_border(np.rot90(agent_tile, k=2)),  # left
      lambda: add_border(np.rot90(agent_tile, k=1)),  # up
    ),
  )


def create_image_from_grid(
  grid: jnp.array,
  agent_pos: Tuple[int, int],
  agent_dir: int,
  image_dict: dict,
  spawn_locs: Optional[jnp.array] = None,
  include_objects: bool = True,
  add_border: bool = True,
):
  # Assumes wall_index is the index for the wall image in image_dict['images']
  wall_index = image_dict["keys"].index("wall")

  if add_border:
    # Expand grid size by 2 in each direction (top, bottom, left, right)
    H, W = grid.shape[:2]
    new_H, new_W = H + 2, W + 2

    # Create a new grid with wall index
    new_grid = jnp.full((new_H, new_W, 1), wall_index, dtype=grid.dtype)

    # Place the original grid in the center of the new grid
    new_grid = new_grid.at[1 : H + 1, 1 : W + 1, :].set(grid)
  else:
    # Use the original grid without border
    H, W = grid.shape[:2]
    new_H, new_W = H, W
    new_grid = grid.reshape(H, W, 1)

  # Flatten the grid for easier indexing
  new_grid_flat = new_grid.squeeze()  # Removes the last dimension assuming it's 1

  # Retrieve the images tensor
  images = image_dict["images"]

  # Create a light blue tile and a yellow tile
  tile_size = images.shape[-2]
  light_blue_tile = jnp.full(
    (tile_size, tile_size, 3),
    jnp.asarray([173, 216, 230], dtype=jnp.uint8),
    dtype=jnp.uint8,
  )

  # Use advanced indexing to map the grid indices to actual images
  object_mask = new_grid_flat > 1
  if not include_objects:
    new_grid_flat = jnp.where(object_mask, 0, new_grid_flat)
    new_grid_flat = jax.vmap(lambda x: images[x])(new_grid_flat)
    new_grid_flat = jnp.where(
      object_mask[:, :, jnp.newaxis, jnp.newaxis, jnp.newaxis],
      light_blue_tile,
      new_grid_flat,
    )
  else:
    new_grid_flat = jax.vmap(lambda x: images[x])(new_grid_flat)

  # Add pretty green color to spawn locations if spawn_locs is provided
  if spawn_locs is not None:
    # Expand spawn_locs to match the new grid size
    expanded_spawn_locs = jnp.zeros((new_H, new_W, 1), dtype=spawn_locs.dtype)
    expanded_spawn_locs = expanded_spawn_locs.at[1 : H + 1, 1 : W + 1, :].set(
      spawn_locs
    )

    # Create a pretty green color
    pretty = jnp.array([228, 130, 83], dtype=jnp.uint8)  # Light green color

    # Create a mask for spawn locations
    spawn_mask = expanded_spawn_locs[:, :, 0, jnp.newaxis, jnp.newaxis, jnp.newaxis]

    # Blend the pretty green color with the existing tiles
    alpha = 1.0  # Adjust this value to change the intensity of the green tint
    blended_color = (1 - alpha) * new_grid_flat + alpha * pretty

    # Apply the blended color to spawn locations
    new_grid_flat = jnp.where(
      spawn_mask, blended_color.astype(jnp.uint8), new_grid_flat
    )

  # Create the agent tile with the specified direction
  agent_tile = make_agent_tile(agent_dir, tile_size)

  # Dimensions of the new grid
  # Assuming all images are of the same shape and channel number
  img_H, img_W, C = images.shape[1:]

  # Reshape and transpose to form the single image
  reshaped_images = new_grid_flat.reshape(new_H, new_W, img_H, img_W, C)

  # Add all agents to the image
  agent_y, agent_x = agent_pos
  # Adjust agent position only if we're adding a border
  if add_border:
    agent_x += 1
    agent_y += 1
  reshaped_images = reshaped_images.at[agent_y, agent_x].set(agent_tile)

  # Then, transpose to (new_H, img_H, new_W, img_W, C)
  transposed_images = reshaped_images.transpose(0, 2, 1, 3, 4)
  # Finally, reshape to (new_H * img_H, new_W * img_W, C)
  final_image = transposed_images.reshape(new_H * img_H, new_W * img_W, C)

  return final_image


def agent_position_in_grid(
  grid: jnp.array,
  agent_pos: Union[Tuple[int, int], List[Tuple[int, int]]],
  agent_dir: int,
  image_dict: dict,
):
  # Get dimensions
  H, W = grid.shape[:2]
  new_H, new_W = H + 2, W + 2

  # Get floor image dimensions from image_dict
  images = image_dict["images"]
  tile_size = images.shape[-2]
  img_H, img_W, C = images.shape[1:]

  # Create floor tile (using index 0 which is typically floor)
  floor_tile = images[0]

  # Create grid filled with floor tiles
  new_grid_flat = jnp.tile(floor_tile, (new_H, new_W, 1, 1, 1))

  # Create agent tile
  agent_tile = make_agent_tile(agent_dir, tile_size)

  # Handle both single agent and multiple agents
  if isinstance(agent_pos, tuple) or isinstance(agent_pos[0], int):
    agent_positions = [agent_pos]
  else:
    agent_positions = agent_pos

  # Add agents to image
  for pos in agent_positions:
    agent_y, agent_x = pos
    # Adjust agent position to account for the expanded grid
    agent_x += 1
    agent_y += 1
    new_grid_flat = new_grid_flat.at[agent_y, agent_x].set(agent_tile)

  # Reshape to final image
  transposed_images = new_grid_flat.transpose(0, 2, 1, 3, 4)
  final_image = transposed_images.reshape(new_H * img_H, new_W * img_W, C)

  return final_image


def place_arrows_on_image(
  image,
  positions,
  actions,
  maze_height,
  maze_width,
  arrow_scale=5,
  arrow_color="g",
  ax=None,
  plot_image=True,
):
  # Get the dimensions of the image and the maze
  image_height, image_width, _ = image.shape

  # Calculate the scaling factors for mapping maze coordinates to image coordinates
  scale_y = image_height // (maze_height + 2)
  scale_x = image_width // (maze_width + 2)

  # Calculate the offset to account for the border of walls
  offset_y = (image_height - scale_y * maze_height) // 2
  offset_x = (image_width - scale_x * maze_width) // 2

  # Create a figure and axis
  if ax is None:
    fig, ax = plt.subplots(1, figsize=(5, 5))

  # Display the rendered image
  if plot_image:
    ax.imshow(image)

  # Iterate over each position and action
  for (y, x), action in zip(positions, actions):
    # Calculate the center coordinates of the cell in the image
    center_y = offset_y + (y + 0.5) * scale_y
    center_x = offset_x + (x + 0.5) * scale_x

    # Define the arrow directions based on the action
    if action == KeyboardActions.up:
      dx, dy = 0, -scale_y / 2
    elif action == KeyboardActions.down:
      dx, dy = 0, scale_y / 2
    elif action == KeyboardActions.left:
      dx, dy = -scale_x / 2, 0
    elif action == KeyboardActions.right:
      dx, dy = scale_x / 2, 0
    else:  # KeyboardActions.done
      continue  # Skip drawing an arrow for the 'done' action

    # Draw the arrow on the image with specified color
    ax.arrow(
      center_x,
      center_y,
      dx,
      dy,
      head_width=scale_x / (arrow_scale * 0.7),  # Increased head width by ~40%
      head_length=scale_y / (arrow_scale * 0.7),  # Increased head length by ~40%
      width=scale_x / (arrow_scale * 2),
      fc=arrow_color,
      ec=arrow_color,
    )

  # Remove the axis ticks and labels

  ax.set_xticks([])
  ax.set_yticks([])
  return ax
