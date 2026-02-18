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

import types

import torch

from rlinf.envs.isaaclab.isaaclab_env import IsaaclabBaseEnv


def test_chunk_step_flushes_actions_after_done():
    env = object.__new__(IsaaclabBaseEnv)
    env.num_envs = 2
    env.device = torch.device("cpu")
    env.auto_reset = False
    env.ignore_terminations = False

    action_history = []
    step_idx = {"i": 0}

    step_outputs = [
        (
            {"obs": torch.zeros((2, 1))},
            torch.tensor([1.0, 2.0]),
            torch.tensor([True, False]),
            torch.tensor([False, False]),
            {},
        ),
        (
            {"obs": torch.zeros((2, 1))},
            torch.tensor([10.0, 20.0]),
            torch.tensor([False, False]),
            torch.tensor([False, False]),
            {},
        ),
        (
            {"obs": torch.zeros((2, 1))},
            torch.tensor([30.0, 40.0]),
            torch.tensor([False, False]),
            torch.tensor([False, False]),
            {},
        ),
    ]

    def _step(self, actions, auto_reset=False):
        action_history.append(actions.clone())
        out = step_outputs[step_idx["i"]]
        step_idx["i"] += 1
        return out

    env.step = types.MethodType(_step, env)

    chunk_actions = torch.tensor(
        [
            [[0.5, 1.0], [2.0, 3.0], [4.0, 5.0]],
            [[0.6, 1.1], [2.1, 3.1], [4.1, 5.1]],
        ],
        dtype=torch.float32,
    )

    _, chunk_rewards, chunk_terminations, chunk_truncations, _ = env.chunk_step(
        chunk_actions
    )

    # Env 0 terminated at step 0, so actions in later chunk steps should be zeroed.
    assert torch.allclose(action_history[1][0], torch.zeros_like(action_history[1][0]))
    assert torch.allclose(action_history[2][0], torch.zeros_like(action_history[2][0]))
    # Env 1 should still receive the planned actions.
    assert torch.allclose(action_history[1][1], chunk_actions[1, 1])
    assert torch.allclose(action_history[2][1], chunk_actions[1, 2])

    # Rewards after done are ignored for env 0.
    assert torch.allclose(chunk_rewards[0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(chunk_rewards[1], torch.tensor([2.0, 20.0, 40.0]))

    # Done signal remains at the first step only for non-auto-reset mode.
    assert torch.equal(chunk_terminations[0], torch.tensor([True, False, False]))
    assert torch.equal(chunk_truncations[0], torch.tensor([False, False, False]))
