# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .tasks.stack_cube import IsaaclabStackCubeEnv
from .tasks.ma_vla_lab_configurable import IsaaclabConfigurableEnv

REGISTER_ISAACLAB_ENVS = {
    "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0": IsaaclabStackCubeEnv,
    # Generic adapter — works with ANY ma-vla-lab YAML config
    "Isaac-MA-VLA-Lab-Configurable-v0": IsaaclabConfigurableEnv,
}

__all__ = [list(REGISTER_ISAACLAB_ENVS.keys())]
