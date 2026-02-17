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

"""Generic RLinf adapter for ALL ma-vla-lab configurable environments.

One class, parameterized by config_path.  Switching tasks = changing
init_params.config_path in the RLinf YAML.  No per-task registration.

Observation mapping:
    - main_images  ← first table/scene camera (overhead or corner)
    - wrist_images ← wrist camera for the controlled robot
    - states       ← EE pos(3) + axis_angle(3) + finger_joint1(1) + finger_joint2(1) = 8D
                     (both fingers are from the SAME gripper, not two different robots)
    - task_descriptions ← generated from YAML prompt_templates at each reset

Action mapping:
    - RLinf sends 7D [dx, dy, dz, droll, dpitch, dyaw, gripper]
    - Adapter routes to the correct action slice in the full multi-robot tensor
"""

import torch

from rlinf.envs.isaaclab.utils import quat2axisangle_torch

from ..isaaclab_env import IsaaclabBaseEnv


class IsaaclabConfigurableEnv(IsaaclabBaseEnv):
    """Generic RLinf env for any ma-vla-lab YAML configuration.

    Required init_params:
        config_path: str  — path to any ma-vla-lab YAML config

    Optional init_params:
        robot_id: str       — which robot to control (auto-detects first robot if omitted)
        table_cam / wrist_cam — camera resolution overrides (height, width)
    """

    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        self.config_path = cfg.init_params.config_path
        self.robot_id = cfg.init_params.get("robot_id", None)

        # Populated during _make_env_function by pre-parsing the YAML
        self._wrist_obs_key = None   # obs key for wrist camera image
        self._table_obs_key = None   # obs key for table/scene camera image
        self._ee_obs_key = None      # obs key for ee_pose (7D: pos + quat_wxyz)
        self._gripper_joint_ids = None

        # Action routing (multi-robot → single-robot)
        self._action_dim_full = None
        self._arm_slice = None       # (start, end) into full action tensor
        self._gripper_idx = None     # index of gripper action

        # Prompt state — updated on each reset from subprocess info dict
        self._current_prompts = None  # list[str] of length num_envs

        # Pre-parse config BEFORE super().__init__ (which calls _make_env_function)
        # This must bypass ma_vla_lab.__init__ which imports .tasks → Isaac Sim.
        self._pre_parse_config(cfg)

        super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)

    @staticmethod
    def _import_config_module():
        """Import ma_vla_lab.config without triggering the top-level __init__.

        ma_vla_lab.__init__ does ``from .tasks import *`` which triggers
        isaaclab.envs → omni.kit.app.  The config subpackage is pure Python,
        so we import it directly using importlib to skip the parent init.
        """
        import importlib
        import sys

        pkg_name = "ma_vla_lab"
        config_name = f"{pkg_name}.config"

        # If someone already imported ma_vla_lab successfully, just use it
        if config_name in sys.modules:
            return sys.modules[config_name]

        # Otherwise, register a stub parent package so Python resolves
        # the sub-package without executing ma_vla_lab/__init__.py
        if pkg_name not in sys.modules:
            import types

            stub = types.ModuleType(pkg_name)
            stub.__path__ = []
            # Discover the real path from installed metadata
            import importlib.util

            spec = importlib.util.find_spec(pkg_name)
            if spec is not None and spec.submodule_search_locations:
                stub.__path__ = list(spec.submodule_search_locations)
            sys.modules[pkg_name] = stub

        return importlib.import_module(config_name)

    def _pre_parse_config(self, cfg):
        """Pre-parse the YAML config to discover cameras, robot, and action layout.

        This runs in the main process — no Isaac Sim available.
        """
        config_mod = self._import_config_module()
        load_environment_config = config_mod.load_environment_config
        build_scene_entities = config_mod.build_scene_entities
        build_managers = config_mod.build_managers

        env_config = load_environment_config(self.config_path)
        entities = build_scene_entities(env_config)

        robot_id = self.robot_id

        # Discover camera names by semantic type
        wrist_cam_name = None
        table_cam_name = None
        for e in entities:
            if e.entity_type == "wrist_camera":
                if robot_id is not None and e.properties.get("robot_id") == robot_id:
                    wrist_cam_name = e.name
                elif wrist_cam_name is None:
                    wrist_cam_name = e.name
            elif e.entity_type == "table_camera" and table_cam_name is None:
                table_cam_name = e.name
            elif e.entity_type == "scene_camera" and table_cam_name is None:
                table_cam_name = e.name

        # Auto-detect robot if not specified
        if robot_id is None:
            for e in entities:
                if e.entity_type == "robot":
                    robot_id = e.name
                    break
        self.robot_id = robot_id

        # Map scene camera names → observation term keys
        self._wrist_obs_key = "wrist_rgb"
        if table_cam_name is not None:
            self._table_obs_key = f"{table_cam_name}_rgb"
        self._ee_obs_key = "ee_pose"

        # Store scene camera names for resolution overrides in subprocess
        self._wrist_cam_scene_name = wrist_cam_name
        self._table_cam_scene_name = table_cam_name

        # Determine the observation group name (policy_<robot_id>)
        self._obs_group_name = f"policy_{robot_id}"

        # Store object IDs for random target selection in subprocess
        self._object_ids = [obj.id for obj in env_config.objects]

        # Discover action layout from managers
        managers = build_managers(env_config)
        offset = 0
        for ad in managers.actions:
            dim = {
                "arm_ik": 6,
                "joint_velocity": 7,
                "joint_position": 7,
                "gripper": 1,
            }.get(ad.action_type, 0)
            if ad.robot_id == robot_id:
                if ad.action_type != "gripper":
                    self._arm_slice = (offset, offset + dim)
                else:
                    self._gripper_idx = offset
            offset += dim
        self._action_dim_full = offset

    def _make_env_function(self):
        """Factory that boots Isaac Sim + builds env from any YAML config.

        Returns a closure that runs inside a subprocess.  The returned env
        is wrapped with ``_PromptTargetWrapper`` which picks a random target
        object on each reset and generates a prompt from YAML templates.
        """
        config_path = self.config_path
        num_envs = self.cfg.init_params.num_envs
        robot_id = self.robot_id
        object_ids = self._object_ids

        # Camera resolution overrides from init_params
        cam_h = None
        cam_w = None
        if hasattr(self.cfg.init_params, "table_cam"):
            cam_h = self.cfg.init_params.table_cam.height
            cam_w = self.cfg.init_params.table_cam.width

        # Names discovered during pre-parse
        wrist_cam_name = self._wrist_cam_scene_name
        table_cam_name = self._table_cam_scene_name

        def make_env_isaaclab():
            import os

            # Force headless (avoid GLX errors in subprocess)
            os.environ.pop("DISPLAY", None)

            # ── Fix stale ISAAC_PATH env vars ───────────────────────────
            # run_embodiment.sh may export ISAAC_PATH=/path/to/isaac-sim
            # (a placeholder) which poisons isaacsim.expose_api().  Clear
            # stale values so bootstrap_kernel() auto-detects the correct
            # paths from the pip-installed isaacsim package.
            for _var in ("ISAAC_PATH", "EXP_PATH", "CARB_APP_PATH"):
                _val = os.environ.get(_var, "")
                if "/path/to/" in _val or (
                    _val and not os.path.isdir(_val)
                ):
                    os.environ.pop(_var, None)

            from isaaclab.app import AppLauncher

            sim_app = AppLauncher(headless=True, enable_cameras=True).app

            from isaaclab.envs import ManagerBasedRLEnv
            from ma_vla_lab.config import load_environment_config
            from ma_vla_lab.tasks.manager_based.ma_vla_lab.configurable_env import (
                build_env_cfg,
            )

            env_cfg = build_env_cfg(config_path, num_envs=num_envs)

            # Override camera resolution if requested
            if cam_h is not None and cam_w is not None:
                for scene_cam_name in [wrist_cam_name, table_cam_name]:
                    if scene_cam_name and hasattr(env_cfg.scene, scene_cam_name):
                        scene_cam_cfg = getattr(env_cfg.scene, scene_cam_name)
                        scene_cam_cfg.height = cam_h
                        scene_cam_cfg.width = cam_w

            inner_env = ManagerBasedRLEnv(cfg=env_cfg)

            # Load env_config for prompt resolution (pure Python, no extra cost)
            env_config = load_environment_config(config_path)

            # Wrap to handle target selection + prompt generation
            env = _PromptTargetWrapper(
                inner_env, env_config, robot_id, object_ids
            )
            return env, sim_app

        return make_env_isaaclab

    def reset(self, seed=None, env_ids=None):
        """Override to capture prompts from the subprocess info dict."""
        obs, infos = super().reset(seed=seed, env_ids=env_ids)
        # _last_info is set by _wrap_obs_and_info from the subprocess
        if self._current_prompts is None:
            # First reset — initialize with num_envs copies
            self._current_prompts = [""] * self.num_envs
        return obs, infos

    def step(self, actions=None, auto_reset=True):
        """Route 7D VLA actions → full multi-robot action tensor.

        RLinf sends 7D: [dx, dy, dz, droll, dpitch, dyaw, gripper].
        We map this into the correct slice of the full action space
        (which may include multiple robots, each with arm + gripper).
        """
        if actions is not None and self._action_dim_full is not None:
            if actions.shape[-1] != self._action_dim_full:
                # Remap: 7D → full action tensor
                full = torch.zeros(
                    actions.shape[0], self._action_dim_full, device=actions.device
                )
                if self._arm_slice is not None:
                    arm_start, arm_end = self._arm_slice
                    arm_dim = arm_end - arm_start
                    full[:, arm_start:arm_end] = actions[:, :arm_dim]
                if self._gripper_idx is not None:
                    full[:, self._gripper_idx] = actions[:, -1]
                actions = full
        return super().step(actions, auto_reset=auto_reset)

    def _wrap_obs(self, obs):
        """Convert configurable env obs → RLinf standard dict.

        RLinf expects:
            main_images:      [B, H, W, 3] uint8 — overhead/table camera
            wrist_images:     [B, H, W, 3] uint8 — wrist camera
            states:           [B, 8] float — EE pos(3) + axis_angle(3) + finger1(1) + finger2(1)
                              (both fingers from same gripper, absolute joint_pos [0..0.04])
            task_descriptions: list[str] of length B
        """
        policy_obs = obs[self._obs_group_name]

        # Prompts come from the subprocess wrapper via the info dict.
        # They are extracted in _wrap_obs_and_info and cached in
        # self._current_prompts.  On first call before any info is
        # available, we use a list of empty strings.
        prompts = obs.get("__prompts__")
        if prompts is not None:
            self._current_prompts = prompts
        instruction = self._current_prompts or [""] * self.num_envs

        # Camera images
        wrist_image = policy_obs.get(self._wrist_obs_key)
        table_image = (
            policy_obs.get(self._table_obs_key) if self._table_obs_key else None
        )

        # EE state: ee_pose returns 7D [pos(3), quat_wxyz(4)]
        ee_pose = policy_obs.get(self._ee_obs_key)
        if ee_pose is not None:
            eef_pos = ee_pose[:, :3]
            # Isaac Lab uses wxyz quaternion; convert to xyzw for quat2axisangle
            quat_wxyz = ee_pose[:, 3:]
            quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
            aa = quat2axisangle_torch(quat_xyzw)
        else:
            eef_pos = torch.zeros(self.num_envs, 3, device=self.device)
            aa = torch.zeros(self.num_envs, 3, device=self.device)

        # Gripper state: two finger joints of the SAME gripper (not two robots).
        # Franka Panda has 9 joints: 7 arm + panda_finger_joint1 + panda_finger_joint2.
        # Each finger ranges [0, 0.04] m (0.0 = closed, 0.04 = fully open).
        #
        # The configurable env now provides absolute joint_pos (not joint_pos_rel),
        # matching how the SFT training data was collected (via robot_asset.data.joint_pos
        # in IsaacLabVLAEnv._extract_ee_state).  Norm stats confirm: q99 ≈ 0.04.
        joint_pos = policy_obs.get("joint_pos_rel")
        if joint_pos is not None:
            gripper_pos = joint_pos[:, -2:]  # [B, 2] — absolute finger positions
        else:
            gripper_pos = torch.full(
                (self.num_envs, 2), 0.04, device=self.device
            )

        states = torch.cat([eef_pos, aa, gripper_pos], dim=1)  # [B, 8]

        env_obs = {
            "main_images": table_image,
            "wrist_images": wrist_image,
            "states": states,
            "task_descriptions": instruction,
        }
        return env_obs


class _PromptTargetWrapper:
    """Thin wrapper that runs INSIDE the Isaac Sim subprocess.

    On each reset:
    1. Picks a random target object from available objects
    2. Sets ``target_object_names`` on the inner env (used by terminations)
    3. Generates a prompt from YAML templates
    4. Injects prompts into the obs dict under ``__prompts__`` so the
       RLinf adapter can read them without modifying the subprocess protocol.
    """

    def __init__(self, env, env_config, robot_id, object_ids):
        self._env = env
        self._env_config = env_config
        self._robot_id = robot_id
        self._object_ids = object_ids
        self._num_envs = env.num_envs
        # Current prompt per env
        self._prompts = [""] * self._num_envs
        # Expose attributes that SubProcIsaacLabEnv / IsaaclabBaseEnv need
        self.device = env.device

    def reset(self, seed=None, env_ids=None):
        import random as _random

        from ma_vla_lab.config import resolve_prompt

        obs, info = self._env.reset(seed=seed, env_ids=env_ids)

        # Determine which envs are being reset
        if env_ids is None:
            reset_indices = range(self._num_envs)
        else:
            reset_indices = env_ids.tolist()

        # Pick random target + generate prompt for each resetting env
        for idx in reset_indices:
            if self._object_ids:
                target_id = _random.choice(self._object_ids)
            else:
                target_id = None
            # Set target on inner env so terminations can use it
            if not hasattr(self._env, "target_object_names"):
                self._env.target_object_names = [None] * self._num_envs
            self._env.target_object_names[idx] = target_id

            self._prompts[idx] = resolve_prompt(
                self._env_config, self._robot_id, target_id
            )

        # Log prompts for debugging
        for idx in reset_indices:
            target = self._env.target_object_names[idx]
            print(f"[PromptTargetWrapper] env={idx} target={target} "
                  f"prompt={self._prompts[idx]!r}")

        # Inject prompts into obs so they cross the subprocess boundary
        obs["__prompts__"] = list(self._prompts)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        # Include current prompts in obs for the adapter
        obs["__prompts__"] = list(self._prompts)
        return obs, reward, terminated, truncated, info

    def close(self):
        self._env.close()

    def __getattr__(self, name):
        return getattr(self._env, name)
