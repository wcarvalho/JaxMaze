"""Things to add from regular env:

1. sample goal objects k times each. keep track of the number of times an object has been sampled.
"""

from typing import Optional

import distrax
import jax
import jax.numpy as jnp
from flax import struct

from housemaze import env
from housemaze.human_dyna import multitask_env
from housemaze.human_dyna.multitask_env import (
  EnvParams,
  ResetParams,
  StepType,
  TimeStep,
  sample_spawn_locs,
)


def index(v, i):
  return jax.lax.dynamic_index_in_dim(v, i, keepdims=False)


class HouseMaze(multitask_env.HouseMaze):
  def reset(self, rng: jax.Array, params: EnvParams) -> TimeStep:
    """

    1. Sample level.
    """
    ##################
    # sample level: assume only 1 level
    ##################
    reset_params_idx = 0
    reset_params = jax.tree_map(
      lambda p: index(p, reset_params_idx), params.reset_params
    )
    grid = reset_params.map_init.grid
    agent_dir = reset_params.map_init.agent_dir

    if params.task_probs is None:
      nobjects = len(self.task_runner.task_objects)
      task_probs = jnp.ones(nobjects) / nobjects
    else:
      task_probs = params.task_probs

    task_sampler = distrax.Categorical(probs=task_probs)
    rng, rng_ = jax.random.split(rng)
    task_idx = task_sampler.sample(seed=rng_)
    train_object = index(self.task_runner.task_objects, task_idx)
    test_object = train_object

    ##################
    # sample position (function of which pair has been choice)
    ##################

    rng, rng_ = jax.random.split(rng)
    agent_pos = jax.lax.cond(
      jnp.logical_and(params.randomize_agent, reset_params.randomize_agent),
      lambda: sample_spawn_locs(rng_, reset_params.map_init.spawn_locs),
      lambda: reset_params.map_init.agent_pos,
    )

    ##################
    # sample train or test
    ##################

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

    rng, rng_ = jax.random.split(rng)
    task_object, offtask_object, is_train_task = jax.lax.cond(
      params.training, train_sample, test_sample, rng_
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
    state = multitask_env.EnvState(
      key=rng,
      step_num=jnp.asarray(0),
      grid=grid,
      is_train_task=is_train_task,
      agent_pos=agent_pos,
      agent_dir=agent_dir,
      map_idx=reset_params_idx,
      current_label=reset_params.label,
      task_w=task_w,
      task_object=task_object,
      offtask_w=offtask_w,
      task_state=task_state,
    )

    reset_action = jnp.array(self.num_actions() + 1, dtype=jnp.int32)
    timestep = TimeStep(
      state=state,
      step_type=StepType.FIRST,
      reward=jnp.asarray(0.0),
      discount=jnp.asarray(1.0),
      observation=self.make_observation(state, prev_action=reset_action),
    )
    return timestep
