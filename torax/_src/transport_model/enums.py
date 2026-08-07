# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Enums for the transport model."""

import enum


class MergeMode(enum.StrEnum):
  """Defines how a transport model's output is combined with previous models.

  Only impacts models used in the `combined` transport model.

  Attributes:
    ADD: The model's output will be added to all other transport models for the
      regions and channels it is enabled for, unless another model uses the
      `OVERWRITE` mode in which case it will be ignored in that region.
    OVERWRITE: This model's output will be used instead of any other models,
      within the domain and over the channels that it is enabled for.
  """
  ADD = 'add'
  OVERWRITE = 'overwrite'


class ClipMode(enum.StrEnum):
  """Defines how transport coefficients are constrained to their bounds.

  The min/max bounds on the transport coefficients exist for PDE stability.
  However, the way the bounds are imposed matters for gradient-based workflows
  (optimisation, sensitivity analysis, surrogate training), because it
  determines the derivative of the transport model output with respect to its
  inputs once a bound is active.

  Attributes:
    HARD: Bounds are imposed with `jnp.clip`. Saturated points are exactly
      constant, so their derivative with respect to any upstream quantity is
      exactly zero. This is the default and reproduces historical TORAX
      behaviour.
    SOFT: Bounds are imposed with a smooth (softplus-based) saturation. The
      bounds are still respected, but the output is a strictly monotonic,
      infinitely differentiable function of the unclipped value, so saturated
      points retain a small non-zero derivative instead of a zero one. The
      width of the transition region is set by `clip_softness`.
  """

  HARD = 'hard'
  SOFT = 'soft'
