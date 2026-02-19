"""Compare baseline vs optimized env outputs for correctness, then benchmark both."""

import time
import numpy as np
import jax
import jax.numpy as jnp

from jaxmaze.utils import from_str, from_str_spawning
from jaxmaze.human_dyna import mazes

# --- Baseline imports ---
from jaxmaze import env as env_orig
from jaxmaze.human_dyna.sf_task_runner import TaskRunner as SFTaskRunnerOrig
from jaxmaze.human_dyna import multitask_env as mt_orig

# --- Optimized imports ---
from jaxmaze import env_fast
from jaxmaze.human_dyna.sf_task_runner_fast import TaskRunner as SFTaskRunnerFast
from jaxmaze.human_dyna import multitask_env_fast as mt_fast

# ============================================================
# Setup
# ============================================================
maze_str = mazes.big_m3_maze1

num_groups = 2
char2key, group_set, task_objects_np = mazes.get_group_set(num_groups)
task_objects = jnp.array(task_objects_np, dtype=jnp.int32)

map_init = from_str(maze_str, char_to_key=char2key, check_grid_letters=False)
spawn_locs = from_str_spawning(maze_str)
map_init = map_init.replace(spawn_locs=spawn_locs)

NUM_ENVS = 64
NUM_STEPS = 200
NUM_REPEATS = 20


def tile_pytree(pytree, n):
  return jax.tree_util.tree_map(lambda x: jnp.stack([x] * n), pytree)


batched_map_init = tile_pytree(map_init, 2)

train_objects = jnp.array(
  [group_set[0].tolist(), group_set[0].tolist()], dtype=jnp.int32
)
test_objects = jnp.array(
  [group_set[1].tolist(), group_set[1].tolist()], dtype=jnp.int32
)

starting_locs = jnp.array(
  [[[[1, 1], [1, 2], [-1, -1]], [[1, 1], [1, 2], [-1, -1]]]] * 2,
  dtype=jnp.int32,
)


# ============================================================
# Build env params for both versions
# ============================================================
def make_reset_params(module):
  return module.ResetParams(
    map_init=batched_map_init,
    train_objects=train_objects,
    test_objects=test_objects,
    starting_locs=starting_locs,
    curriculum=jnp.array([True, True]),
    label=jnp.array([0, 1]),
    randomize_agent=jnp.array([False, False]),
    rotation=jnp.array([(0, 0), (0, 0)]),
  )


orig_params = mt_orig.EnvParams(
  reset_params=make_reset_params(mt_orig), time_limit=NUM_STEPS
)
fast_params = mt_fast.EnvParams(
  reset_params=make_reset_params(mt_fast), time_limit=NUM_STEPS
)


# ============================================================
# Build runner function
# ============================================================
def build_run_fn(env_module, mt_module, runner_cls, params):
  task_runner = runner_cls(task_objects=task_objects, radius=5)
  house_env = mt_module.HouseMaze(task_runner=task_runner, action_spec="keyboard")
  step_type_last = env_module.StepType.LAST

  def env_step(carry, _):
    rng, timestep = carry
    rng, rng_step = jax.random.split(rng)
    action = jax.random.randint(rng_step, shape=(), minval=0, maxval=4)
    next_timestep = house_env.step(rng, timestep, action, params)
    rng, rng_reset = jax.random.split(rng)
    next_timestep = jax.lax.cond(
      next_timestep.step_type == step_type_last,
      lambda: house_env.reset(rng_reset, params),
      lambda: next_timestep,
    )
    return (rng, next_timestep), (next_timestep.reward, next_timestep.observation.image)

  def run_episode(rng):
    timestep = house_env.reset(rng, params)
    (_, final_ts), (rewards, images) = jax.lax.scan(
      env_step, (rng, timestep), None, length=NUM_STEPS
    )
    return final_ts, rewards, images

  return jax.jit(jax.vmap(run_episode))


# ============================================================
# Correctness comparison
# ============================================================
print("=" * 60)
print("  CORRECTNESS CHECK")
print("=" * 60)

rngs = jax.random.split(jax.random.PRNGKey(0), NUM_ENVS)

# SFTaskRunner comparison
print("\n--- SFTaskRunner ---")
run_orig = build_run_fn(env_orig, mt_orig, SFTaskRunnerOrig, orig_params)
run_fast = build_run_fn(env_fast, mt_fast, SFTaskRunnerFast, fast_params)

ts_orig, rewards_orig, images_orig = run_orig(rngs)
ts_fast, rewards_fast, images_fast = run_fast(rngs)

rewards_orig.block_until_ready()
rewards_fast.block_until_ready()

rewards_match = jnp.allclose(rewards_orig, rewards_fast, atol=1e-5)
images_match = jnp.array_equal(images_orig, images_fast)
features_orig = ts_orig.state.task_state.features
features_fast = ts_fast.state.task_state.features
features_match = jnp.allclose(features_orig, features_fast, atol=1e-5)
grid_match = jnp.array_equal(ts_orig.state.grid, ts_fast.state.grid)

print(
  f"  rewards match:  {rewards_match}  (orig sum={rewards_orig.sum():.4f}, fast sum={rewards_fast.sum():.4f})"
)
print(f"  images match:   {images_match}")
print(f"  features match: {features_match}")
print(f"  grids match:    {grid_match}")

if not rewards_match:
  diff = jnp.abs(rewards_orig - rewards_fast)
  print(f"  max reward diff: {diff.max():.6f}")
  print(f"  nonzero diffs:   {(diff > 1e-5).sum()}")

# env.TaskRunner comparison
print("\n--- env.TaskRunner ---")
run_orig_base = build_run_fn(
  env_orig,
  mt_orig,
  lambda **kw: env_orig.TaskRunner(task_objects=kw["task_objects"]),
  orig_params,
)
run_fast_base = build_run_fn(
  env_fast,
  mt_fast,
  lambda **kw: env_fast.TaskRunner(task_objects=kw["task_objects"]),
  fast_params,
)

ts_orig2, rewards_orig2, images_orig2 = run_orig_base(rngs)
ts_fast2, rewards_fast2, images_fast2 = run_fast_base(rngs)

rewards_orig2.block_until_ready()
rewards_fast2.block_until_ready()

rewards_match2 = jnp.allclose(rewards_orig2, rewards_fast2, atol=1e-5)
images_match2 = jnp.array_equal(images_orig2, images_fast2)
features_match2 = jnp.allclose(
  ts_orig2.state.task_state.features, ts_fast2.state.task_state.features, atol=1e-5
)
grid_match2 = jnp.array_equal(ts_orig2.state.grid, ts_fast2.state.grid)

print(
  f"  rewards match:  {rewards_match2}  (orig sum={rewards_orig2.sum():.4f}, fast sum={rewards_fast2.sum():.4f})"
)
print(f"  images match:   {images_match2}")
print(f"  features match: {features_match2}")
print(f"  grids match:    {grid_match2}")

if not rewards_match2:
  diff2 = jnp.abs(rewards_orig2 - rewards_fast2)
  print(f"  max reward diff: {diff2.max():.6f}")
  print(f"  nonzero diffs:   {(diff2 > 1e-5).sum()}")

has_grid_orig = (
  hasattr(ts_orig.state.task_state, "grid")
  and ts_orig.state.task_state.grid is not None
)
has_grid_fast = (
  hasattr(ts_fast.state.task_state, "grid")
  and ts_fast.state.task_state.grid is not None
)
print(f"\n  task_state has grid (orig): {has_grid_orig}")
print(f"  task_state has grid (fast): {has_grid_fast}")

all_pass = all(
  [
    rewards_match,
    images_match,
    features_match,
    grid_match,
    rewards_match2,
    images_match2,
    features_match2,
    grid_match2,
  ]
)
print(f"\n{'=' * 60}")
print(f"  ALL CHECKS PASSED: {all_pass}")
print(f"{'=' * 60}")

if not all_pass:
  print("\n  WARNING: outputs differ! Investigate before trusting timing results.")


# ============================================================
# Timing benchmark
# ============================================================
def time_fn(run_fn, label):
  # warmup
  rngs = jax.random.split(jax.random.PRNGKey(0), NUM_ENVS)
  _, rewards, _ = run_fn(rngs)
  rewards.block_until_ready()

  times = []
  for i in range(NUM_REPEATS):
    rngs = jax.random.split(jax.random.PRNGKey(i + 1), NUM_ENVS)
    start = time.perf_counter()
    _, rewards, _ = run_fn(rngs)
    rewards.block_until_ready()
    elapsed = time.perf_counter() - start
    times.append(elapsed)

  times = np.array(times)
  total_steps = NUM_ENVS * NUM_STEPS
  print(f"\n  {label}")
  print(f"    mean: {times.mean() * 1000:.2f} ms  std: {times.std() * 1000:.2f} ms")
  print(f"    steps/sec: {total_steps / times.mean():.0f}")
  print(f"    per-step: {times.mean() / total_steps * 1e6:.2f} us")
  return times.mean()


print(f"\n{'=' * 60}")
print(f"  TIMING BENCHMARK")
print(f"  {NUM_ENVS} envs x {NUM_STEPS} steps x {NUM_REPEATS} repeats")
print(f"  maze: big_m3_maze1 ({map_init.grid.shape[0]}x{map_init.grid.shape[1]})")
print(f"{'=' * 60}")

t_sf_orig = time_fn(run_orig, "SFTaskRunner (baseline)")
t_sf_fast = time_fn(run_fast, "SFTaskRunner (optimized)")
print(f"\n  SFTaskRunner speedup: {t_sf_orig / t_sf_fast:.2f}x")

t_base_orig = time_fn(run_orig_base, "env.TaskRunner (baseline)")
t_base_fast = time_fn(run_fast_base, "env.TaskRunner (optimized)")
print(f"\n  env.TaskRunner speedup: {t_base_orig / t_base_fast:.2f}x")
