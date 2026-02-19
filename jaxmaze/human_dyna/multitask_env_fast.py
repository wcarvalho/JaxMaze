"""Optimized version of multitask_env.py.

Changes from multitask_env.py:
1. Imports from env_fast and sf_task_runner_fast instead of env and sf_task_runner
2. mask_sample uses jax.random.categorical instead of distrax.Categorical
3. TaskRunner.step() receives prior_grid instead of prior TaskState.grid
4. TaskState no longer stores grid
"""

from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from flax import struct
from flax.struct import field

from jaxmaze import env_fast as env

TaskRunner = env.TaskRunner
TimeStep = env.TimeStep
StepType = env.StepType

MapInit = env.MapInit


@struct.dataclass
class ResetParams:
  map_init: env.MapInit
  train_objects: jax.Array
  test_objects: jax.Array
  starting_locs: Optional[jax.Array] = None
  curriculum: jax.Array = field(default_factory=lambda: jnp.array(False))
  label: jax.Array = field(default_factory=lambda: jnp.array(0))
  randomize_agent: bool = field(default_factory=lambda: jnp.array(False))
  rotation: Tuple[int, int] = (0, 0)


@struct.dataclass
class EnvParams:
  reset_params: ResetParams
  time_limit: int = 200
  p_test_sample_train: float = 0.5
  force_room: bool = field(default_factory=lambda: jnp.array(False))
  default_room: bool = field(default_factory=lambda: jnp.array(0))
  training: bool = True
  terminate_with_done: int = 0  # more relevant for web app
  randomize_agent: bool = False
  randomization_radius: int = 0  # New parameter
  task_probs: jax.Array = None


class FlatObservation(struct.PyTreeNode):
  image: jax.Array
  task_w: jax.Array
  state_features: jax.Array


@struct.dataclass
class EnvState:
  # episode information
  key: jax.Array
  step_num: jax.Array

  # map info
  grid: jax.Array
  agent_pos: jax.Array
  agent_dir: int

  # task info
  map_idx: jax.Array
  task_w: jax.Array
  is_train_task: jax.Array
  task_object: jax.Array
  current_label: jax.Array
  offtask_w: jax.Array
  objects: jax.Array = None
  task_state: Optional[env.TaskState] = None
  successes: Optional[jax.Array] = None
  rotation: Optional[jax.Array] = None


class TimeStep(struct.PyTreeNode):
  state: EnvState

  step_type: StepType
  reward: jax.Array
  discount: jax.Array
  observation: env.Observation
  finished: jax.Array = field(default_factory=lambda: jnp.array(False))

  def first(self):
    return self.step_type == StepType.FIRST

  def mid(self):
    return self.step_type == StepType.MID

  def last(self):
    return self.step_type == StepType.LAST


# OPTIMIZATION: jax.random.categorical instead of distrax.Categorical
def mask_sample(mask, rng):
  logits = jnp.where(mask == 1, 0.0, -1e8).astype(jnp.float32)
  rng, rng_ = jax.random.split(rng)
  return jax.random.categorical(rng_, logits)


def sample_spawn_locs(rng, spawn_locs):
  H, W, C = spawn_locs.shape

  spawn_locs = spawn_locs / spawn_locs.sum()
  inner_coords = jax.random.choice(
    key=rng,
    shape=(1,),
    a=jnp.arange(H * W),
    replace=False,
    # Flatten the empty_spaces mask and use it
    # as probability distribution
    p=spawn_locs.flatten(),
  )

  # Convert the flattened index to y, x coordinates
  y, x = jnp.divmod(inner_coords[0], W)
  return jnp.array([y, x])


class HouseMaze(env.HouseMaze):
  def total_categories(self, params: EnvParams):
    grid = params.reset_params.map_init.grid
    H, W = grid.shape[-3:-1]
    num_object_categories = self.num_categories
    num_directions = len(env.DIR_TO_VEC)
    num_spatial_positions = H * W
    num_actions = self.num_actions(params) + 1  # including reset action
    return num_object_categories + num_directions + num_spatial_positions + num_actions

  def reset(self, rng: jax.Array, params: EnvParams) -> TimeStep:
    """

    1. Sample level.
    """
    ##################
    # sample level
    ##################
    nlevels = len(params.reset_params.curriculum)
    rng, rng_ = jax.random.split(rng)
    reset_params_idx = jax.random.randint(rng_, shape=(), minval=0, maxval=nlevels)

    def index(p):
      return jax.lax.dynamic_index_in_dim(p, reset_params_idx, keepdims=False)

    reset_params = jax.tree_util.tree_map(index, params.reset_params)

    grid = reset_params.map_init.grid
    agent_dir = reset_params.map_init.agent_dir

    ##################
    # sample pair
    ##################
    pair_idx = mask_sample(mask=reset_params.train_objects >= 0, rng=rng)

    ##################
    # sample position (function of which pair has been choice)
    ##################
    def sample_pos_from_curriculum(rng_):
      locs = jax.lax.dynamic_index_in_dim(
        reset_params.starting_locs, pair_idx, keepdims=False
      )
      loc_idx = mask_sample(mask=(locs >= 0).all(-1), rng=rng_)
      loc = jax.lax.dynamic_index_in_dim(locs, loc_idx, keepdims=False)
      return loc

    def sample_normal(rng_, reset_params, params):
      return jax.lax.cond(
        jnp.logical_and(reset_params.curriculum, params.training),
        lambda: sample_pos_from_curriculum(rng_),
        lambda: reset_params.map_init.agent_pos,
      )

    rng, rng_ = jax.random.split(rng)
    agent_pos = jax.lax.cond(
      jnp.logical_and(params.randomize_agent, reset_params.randomize_agent),
      lambda: sample_spawn_locs(rng_, reset_params.map_init.spawn_locs),
      lambda: sample_normal(rng_, reset_params, params),
    )

    ##################
    # sample either train or test object as task object
    ##################
    def index(v, i):
      return jax.lax.dynamic_index_in_dim(v, i, keepdims=False)

    train_object = index(reset_params.train_objects, pair_idx)
    test_object = index(reset_params.test_objects, pair_idx)

    train_object, test_object = jax.lax.cond(
      params.force_room,
      lambda: (
        index(reset_params.train_objects, params.default_room),
        index(reset_params.test_objects, params.default_room),
      ),
      lambda: (train_object, test_object),
    )

    def train_sample(rng):
      is_train_task = jnp.array(True)
      return train_object, test_object, is_train_task

    def test_sample(rng):
      is_train_task = jnp.array(False)
      return test_object, train_object, is_train_task

    def train_or_test_sample(rng):
      return jax.lax.cond(
        jax.random.bernoulli(rng, p=params.p_test_sample_train),
        train_sample,
        test_sample,
        rng,
      )

    rng, rng_ = jax.random.split(rng)
    task_object, offtask_object, is_train_task = jax.lax.cond(
      params.training, train_sample, train_or_test_sample, rng_
    )

    ##################
    # create task vectors
    ##################
    task_w = self.task_runner.task_vector(task_object)
    offtask_w = self.task_runner.task_vector(offtask_object)
    task_state = self.task_runner.reset(grid, agent_pos)

    ##################
    # create ouputs
    ##################
    state = EnvState(
      key=rng,
      step_num=jnp.asarray(0),
      grid=grid,
      agent_pos=agent_pos,
      agent_dir=agent_dir,
      is_train_task=is_train_task,
      map_idx=reset_params_idx,
      current_label=reset_params.label,
      task_w=task_w,
      task_object=task_object,
      offtask_w=offtask_w,
      objects=self.task_runner.task_objects,
      task_state=task_state,
      rotation=reset_params.rotation,
    )

    reset_action = jnp.array(self.num_actions() + 1, dtype=jnp.int32)
    observation = self.make_observation(state, prev_action=reset_action)
    timestep = TimeStep(
      state=state,
      step_type=StepType.FIRST,
      reward=jnp.asarray(0.0),
      discount=jnp.asarray(1.0),
      observation=observation,
    )
    timestep = jax.tree_util.tree_map(jax.lax.stop_gradient, timestep)
    return timestep

  def step(
    self, rng: jax.Array, timestep: TimeStep, action: jax.Array, params: EnvParams
  ) -> TimeStep:
    del rng  # deterministic function

    if self.action_spec == "keyboard":
      grid, agent_pos, agent_dir = env.take_action(
        timestep.state.replace(agent_dir=action),
        action=env.MinigridActions.forward,
      )
    elif self.action_spec == "minigrid":
      grid, agent_pos, agent_dir = env.take_action(timestep.state, action)
    else:
      raise NotImplementedError(self.action_spec)

    # OPTIMIZATION: pass prior grid directly instead of from TaskState
    task_state = self.task_runner.step(timestep.state.grid, grid, agent_pos)

    state = timestep.state.replace(
      grid=grid,
      agent_pos=agent_pos,
      agent_dir=agent_dir,
      task_state=task_state,
      step_num=timestep.state.step_num + 1,
    )

    terminated_done = action == self.action_enum().done
    # any object picked up
    terminated_features = self.task_runner.check_terminated(
      task_state.features, timestep.state.task_w
    )
    terminated = jax.lax.switch(
      params.terminate_with_done,
      (
        lambda: terminated_features,
        lambda: terminated_done,
        lambda: terminated_features + terminated_done,
      ),
    )
    terminated = terminated >= 1
    task_w = timestep.state.task_w.astype(jnp.float32)
    features = task_state.features.astype(jnp.float32)
    reward = (task_w * features).sum(-1)
    truncated = jnp.equal(state.step_num, params.time_limit)

    step_type = jax.lax.select(terminated | truncated, StepType.LAST, StepType.MID)
    discount = jax.lax.select(terminated, jnp.asarray(0.0), jnp.asarray(1.0))

    observation = self.make_observation(state, prev_action=action)
    timestep = TimeStep(
      state=state,
      step_type=step_type,
      reward=reward,
      discount=discount,
      observation=observation,
    )

    timestep = jax.tree_util.tree_map(jax.lax.stop_gradient, timestep)
    return timestep


if __name__ == "__main__":
  import time
  import numpy as np
  from jaxmaze.utils import from_str, from_str_spawning
  from jaxmaze.human_dyna.sf_task_runner_fast import TaskRunner as SFTaskRunner
  from jaxmaze.human_dyna import mazes

  maze_str = mazes.big_m3_maze1

  num_groups = 2
  char2key, group_set, task_objects = mazes.get_group_set(num_groups)
  task_objects = jnp.array(task_objects, dtype=jnp.int32)

  map_init = from_str(maze_str, char_to_key=char2key, check_grid_letters=False)
  spawn_locs = from_str_spawning(maze_str)
  map_init = map_init.replace(spawn_locs=spawn_locs)

  NUM_ENVS = 64
  NUM_STEPS = 200
  NUM_REPEATS = 20

  # Stack map_init for batch
  def tile_pytree(pytree, n):
    return jax.tree_util.tree_map(lambda x: jnp.stack([x] * n), pytree)

  batched_map_init = tile_pytree(map_init, 2)  # 2 levels

  train_objects = jnp.array(
    [group_set[0].tolist(), group_set[0].tolist()], dtype=jnp.int32
  )
  test_objects = jnp.array(
    [group_set[1].tolist(), group_set[1].tolist()], dtype=jnp.int32
  )

  H, W = map_init.grid.shape[:2]
  starting_locs = jnp.array(
    [[[[1, 1], [1, 2], [-1, -1]], [[1, 1], [1, 2], [-1, -1]]]] * 2,
    dtype=jnp.int32,
  )

  reset_params = ResetParams(
    map_init=batched_map_init,
    train_objects=train_objects,
    test_objects=test_objects,
    starting_locs=starting_locs,
    curriculum=jnp.array([True, True]),
    label=jnp.array([0, 1]),
    randomize_agent=jnp.array([False, False]),
    rotation=jnp.array([(0, 0), (0, 0)]),
  )

  env_params = EnvParams(
    reset_params=reset_params,
    time_limit=NUM_STEPS,
  )

  def run_benchmark(runner_cls, label):
    if runner_cls == SFTaskRunner:
      task_runner = runner_cls(task_objects=task_objects, radius=5)
    else:
      task_runner = runner_cls(task_objects=task_objects)

    house_env = HouseMaze(task_runner=task_runner, action_spec="keyboard")

    def env_step(carry, _):
      rng, timestep = carry
      rng, rng_step = jax.random.split(rng)
      action = jax.random.randint(rng_step, shape=(), minval=0, maxval=4)
      next_timestep = house_env.step(rng, timestep, action, env_params)
      # auto-reset on last
      rng, rng_reset = jax.random.split(rng)
      next_timestep = jax.lax.cond(
        next_timestep.step_type == StepType.LAST,
        lambda: house_env.reset(rng_reset, env_params),
        lambda: next_timestep,
      )
      return (rng, next_timestep), next_timestep.reward

    def run_episode(rng):
      timestep = house_env.reset(rng, env_params)
      (_, final_ts), rewards = jax.lax.scan(
        env_step, (rng, timestep), None, length=NUM_STEPS
      )
      return final_ts, rewards

    run_vmapped = jax.jit(jax.vmap(run_episode))

    # Warmup
    rngs = jax.random.split(jax.random.PRNGKey(0), NUM_ENVS)
    final_ts, rewards = run_vmapped(rngs)
    rewards.block_until_ready()

    # Correctness checks
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  rewards shape: {rewards.shape}")
    print(f"  total reward: {rewards.sum():.4f}")
    print(f"  obs image shape: {final_ts.observation.image.shape}")
    print(f"  obs state_features shape: {final_ts.observation.state_features.shape}")
    print(f"  task_state features shape: {final_ts.state.task_state.features.shape}")
    has_grid = (
      hasattr(final_ts.state.task_state, "grid")
      and final_ts.state.task_state.grid is not None
    )
    print(f"  task_state has grid: {has_grid}")

    # Timing
    times = []
    for i in range(NUM_REPEATS):
      rngs = jax.random.split(jax.random.PRNGKey(i + 1), NUM_ENVS)
      start = time.perf_counter()
      _, rewards = run_vmapped(rngs)
      rewards.block_until_ready()
      elapsed = time.perf_counter() - start
      times.append(elapsed)

    times = np.array(times)
    total_steps = NUM_ENVS * NUM_STEPS
    print(f"\n  Timing ({NUM_REPEATS} repeats, {NUM_ENVS} envs x {NUM_STEPS} steps):")
    print(f"    mean: {times.mean() * 1000:.2f} ms  std: {times.std() * 1000:.2f} ms")
    print(f"    steps/sec: {total_steps / times.mean():.0f}")
    print(f"    per-step: {times.mean() / total_steps * 1e6:.2f} us")
    return rewards

  print("Running OPTIMIZED benchmarks...")
  print(f"Config: {NUM_ENVS} envs x {NUM_STEPS} steps x {NUM_REPEATS} repeats")

  rewards_base = run_benchmark(env.TaskRunner, "env_fast.TaskRunner (optimized)")
  rewards_sf = run_benchmark(SFTaskRunner, "SFTaskRunner_fast (optimized)")
