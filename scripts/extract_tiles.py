#!/usr/bin/env python
"""Extract tile images from image_data.pkl and convert to base64 for web use."""

import base64
import io
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image


def numpy_to_base64(arr: np.ndarray) -> str:
  """Convert numpy RGB array to base64 PNG data URI."""
  img = Image.fromarray(arr.astype(np.uint8))
  buffer = io.BytesIO()
  img.save(buffer, format="PNG")
  b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
  return f"data:image/png;base64,{b64}"


def fill_coords(img, fn, color):
  """Fill pixels matching condition fn with color."""
  new_img = img.copy()
  for y in range(img.shape[0]):
    for x in range(img.shape[1]):
      yf = (y + 0.5) / img.shape[0]
      xf = (x + 0.5) / img.shape[1]
      if fn(xf, yf):
        new_img[y, x] = color
  return new_img


def point_in_triangle(a, b, c):
  """Return function that checks if point is in triangle."""
  a = np.array(a)
  b = np.array(b)
  c = np.array(c)

  def fn(x, y):
    v0 = c - a
    v1 = b - a
    v2 = np.array((x, y)) - a
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    return (u >= 0) and (v >= 0) and (u + v) < 1

  return fn


def make_agent_tile(direction: int, tile_size: int) -> np.ndarray:
  """Create agent triangle tile facing given direction."""
  TRI_COORDS = np.array(
    [
      [0.12, 0.19],
      [0.87, 0.50],
      [0.12, 0.81],
    ]
  )
  agent_tile = np.tile([0, 0, 0], (tile_size, tile_size, 1)).astype(np.uint8)
  agent_tile = fill_coords(agent_tile, point_in_triangle(*TRI_COORDS), [255, 0, 0])
  if direction == 0:  # right
    return agent_tile
  elif direction == 1:  # down
    return np.rot90(agent_tile, k=3)
  elif direction == 2:  # left
    return np.rot90(agent_tile, k=2)
  elif direction == 3:  # up
    return np.rot90(agent_tile, k=1)


def replace_color(image, old_color, new_color):
  """Replace old_color with new_color in image."""
  image = np.asarray(image).copy()
  old_color = np.asarray(old_color)
  new_color = np.asarray(new_color)
  mask = np.all(image == old_color, axis=-1)
  image[mask] = new_color
  return image


def main():
  # Paths
  script_dir = Path(__file__).parent
  repo_dir = script_dir.parent
  pkl_path = repo_dir / "jaxmaze" / "image_data.pkl"
  output_path = repo_dir / "tiles_base64.json"

  # Load pickle
  with open(pkl_path, "rb") as f:
    image_dict = pickle.load(f)

  keys = image_dict["keys"]
  images = image_dict["images"]
  tile_size = images.shape[-2]

  print(f"Loaded {len(keys)} object keys, tile size: {tile_size}x{tile_size}")

  result = {}

  # Process object images (replace white with black background)
  for i, key in enumerate(keys):
    img = images[i].copy()
    img = replace_color(img, (255, 255, 255), (0, 0, 0))
    result[key] = numpy_to_base64(img)
    print(f"  {key}")

  # Add empty (black) tile
  empty_tile = np.tile([0, 0, 0], (tile_size, tile_size, 1)).astype(np.uint8)
  result["empty"] = numpy_to_base64(empty_tile)
  print("  empty")

  # Add wall (gray) tile
  wall_tile = np.tile([100, 100, 100], (tile_size, tile_size, 1)).astype(np.uint8)
  result["wall"] = numpy_to_base64(wall_tile)
  print("  wall")

  # Add agent tiles for each direction
  direction_names = ["right", "down", "left", "up"]
  for direction, name in enumerate(direction_names):
    agent_tile = make_agent_tile(direction, tile_size)
    result[f"agent_{name}"] = numpy_to_base64(agent_tile)
    print(f"  agent_{name}")

  # Write output
  with open(output_path, "w") as f:
    json.dump(result, f)

  print(f"\nWrote {len(result)} tiles to {output_path}")


if __name__ == "__main__":
  main()
