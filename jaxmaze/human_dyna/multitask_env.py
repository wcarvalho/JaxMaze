from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from flax import struct
from flax.struct import field
import distrax

from jaxmaze import env

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
  time_limit: int = 300
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


def mask_sample(mask, rng):
  # Creating logits based on the mask: -1e8 where mask is 0, 1 where mask is 1
  logits = jnp.where(mask == 1, mask.astype(jnp.float32), -1e8).astype(jnp.float32)

  # Creating the Categorical distribution with the specified logits
  sampler = distrax.Categorical(logits=logits)

  # Splitting the RNG
  rng, rng_ = jax.random.split(rng)

  # Sampling from the distribution
  return sampler.sample(seed=rng_)


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

    task_state = self.task_runner.step(timestep.state.task_state, grid, agent_pos)

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
