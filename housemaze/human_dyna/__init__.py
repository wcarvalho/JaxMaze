"""Backward-compatible shim for housemaze.human_dyna -> jaxmaze.human_dyna rename."""

from jaxmaze.human_dyna import *
from jaxmaze.human_dyna import (
  experiments,
  mazes,
  multitask_env,
  sf_task_runner,
  utils,
  web_env,
)
