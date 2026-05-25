# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JaxMaze (HouseMaze) is a high-performance JAX-based reinforcement learning environment for maze/grid-world navigation and object collection tasks. It supports parallel execution across thousands of environments simultaneously via JAX's vectorization.

## Development Commands

```bash
# Install (editable mode for development)
uv pip install -e .

# Format Python files (required after edits) - uses 2-space indent per ruff.toml
uv run ruff format

# Run tests
uv run python test.py
```

Use `uv` for package management.

## Architecture

### Core Modules (housemaze/)

- **env.py** - Main environment with `HouseMaze` class, `EnvState`, `TimeStep`, `TaskRunner`, and action handling
- **utils.py** - Maze parsing (`from_str()`), pathfinding (BFS/DFS), `AutoResetWrapper`, image loading
- **renderer.py** - Visual rendering via `create_image_from_grid()` using tile images from `image_data.pkl`
- **levels.py** - Pre-defined maze layouts

### Extended Research Module (housemaze/human_dyna/)

- **multitask_env.py** - Multi-task environment with train/test splits
- **sf_task_runner.py** - State-feature based task runner
- **experiments.py** - Experiment configurations
- **mazes.py** - Extended maze definitions

### Key Patterns

**State Management:** Immutable Flax dataclasses (`EnvState`, `Observation`, `MapInit`) for JAX PyTree compatibility.

**Functional Interface:** `env.reset(rng)` and `env.step(state, action)` return new states, never mutate.

**Action Spaces:**
- Keyboard: right(0), down(1), left(2), up(3), done(4)
- Minigrid: left(0), right(1), forward(2), done(3)

**Maze Definition:** String-based with character mapping - `.` empty, `#` wall, `>/<v/^` agent spawn with direction, letters for objects.

### JAX Conventions

- Use `jax.vmap()` for vectorization over batches
- Use `jax.lax.cond()`/`jax.lax.switch()` instead of Python if/else inside JIT
- Always split RNG keys with `jax.random.split()` for reproducibility
- Use `jnp` arrays, not NumPy, for JAX-traced functions
