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
        self.sample_control_robot = bool(
            cfg.init_params.get("sample_control_robot", False)
        )

        # Populated during _make_env_function by pre-parsing the YAML
        self._wrist_obs_key = None   # obs key for wrist camera image
        self._table_obs_key = None   # obs key for table/scene camera image
        self._side_obs_key = None    # obs key for side/corner camera image
        self._ee_obs_key = None      # obs key for ee_pose (7D: pos + quat_wxyz)
        self._gripper_joint_ids = None

        # Action routing (multi-robot → single-robot)
        self._action_dim_full = None
        self._arm_slice = None       # (start, end) into full action tensor
        self._gripper_idx = None     # index of gripper action
        self._action_map_by_robot = {}  # robot_id -> {"arm_slice": tuple | None, "gripper_idx": int | None}
        self._robot_ids = []
        self._obs_group_name_by_robot = {}
        self._controlled_robot_ids = None  # list[str] length num_envs

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
        wrist_cam_names = []
        wrist_cam_name_by_robot = {}
        table_cam_name = None
        side_cam_name = None
        for e in entities:
            if e.entity_type == "wrist_camera":
                wrist_cam_names.append(e.name)
                _rid = e.properties.get("robot_id")
                if _rid is not None:
                    wrist_cam_name_by_robot[_rid] = e.name
            elif e.entity_type == "table_camera" and table_cam_name is None:
                table_cam_name = e.name
            elif e.entity_type == "scene_camera":
                if table_cam_name is None:
                    table_cam_name = e.name
                elif side_cam_name is None:
                    # Second scene camera becomes the side camera
                    side_cam_name = e.name

        robot_names = [e.name for e in entities if e.entity_type == "robot"]

        # Auto-detect robot if not specified
        if robot_id is None:
            for e in entities:
                if e.entity_type == "robot":
                    robot_id = e.name
                    break
            if len(robot_names) > 1:
                print(
                    "[IsaaclabConfigurableEnv] init_params.robot_id is unset; "
                    f"defaulting to first robot '{robot_id}' from {robot_names}."
                )
        self.robot_id = robot_id
        self._robot_ids = list(robot_names) if robot_names else ([robot_id] if robot_id else [])
        if self.sample_control_robot and len(self._robot_ids) <= 1:
            self.sample_control_robot = False
        if robot_id and robot_id not in self._robot_ids:
            self._robot_ids = [robot_id] + self._robot_ids

        # Map scene camera names → observation term keys
        self._wrist_obs_key = "wrist_rgb"
        if table_cam_name is not None:
            self._table_obs_key = f"{table_cam_name}_rgb"
        if side_cam_name is not None:
            self._side_obs_key = f"{side_cam_name}_rgb"
        self._ee_obs_key = "ee_pose"

        # Store scene camera names for resolution overrides in subprocess
        self._wrist_cam_scene_name = wrist_cam_name_by_robot.get(robot_id)
        self._wrist_cam_scene_names = wrist_cam_names
        self._table_cam_scene_name = table_cam_name
        self._side_cam_scene_name = side_cam_name

        # Determine observation group names (policy_<robot_id>)
        self._obs_group_name = f"policy_{robot_id}"
        self._obs_group_name_by_robot = {
            rid: f"policy_{rid}" for rid in self._robot_ids
        }

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
            if ad.robot_id is not None:
                m = self._action_map_by_robot.setdefault(
                    ad.robot_id, {"arm_slice": None, "gripper_idx": None}
                )
                if ad.action_type != "gripper":
                    m["arm_slice"] = (offset, offset + dim)
                else:
                    m["gripper_idx"] = offset
            offset += dim
        self._action_dim_full = offset
        robot_action_map = self._action_map_by_robot.get(robot_id, {})
        self._arm_slice = robot_action_map.get("arm_slice")
        self._gripper_idx = robot_action_map.get("gripper_idx")

    def _make_env_function(self):
        """Factory that boots Isaac Sim + builds env from any YAML config.

        Returns a closure that runs inside a subprocess.  The returned env
        is wrapped with ``_PromptTargetWrapper`` which picks a random target
        object on each reset and generates a prompt from YAML templates.
        """
        config_path = self.config_path
        num_envs = self.cfg.init_params.num_envs
        robot_id = self.robot_id
        robot_ids = self._robot_ids
        object_ids = self._object_ids
        env_seed = int(self.seed)

        # Camera resolution overrides from init_params
        cam_h = None
        cam_w = None
        if hasattr(self.cfg.init_params, "table_cam"):
            cam_h = self.cfg.init_params.table_cam.height
            cam_w = self.cfg.init_params.table_cam.width
        settle_seconds = float(getattr(self.cfg.init_params, "settle_seconds", 0.0))

        # Names discovered during pre-parse
        wrist_cam_name = self._wrist_cam_scene_name
        wrist_cam_names = self._wrist_cam_scene_names
        table_cam_name = self._table_cam_scene_name
        side_cam_name = self._side_cam_scene_name

        headless = getattr(self.cfg.init_params, "headless", True)

        def make_env_isaaclab():
            import os

            # Force headless (avoid GLX errors in subprocess)
            if headless:
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

            # Force single-GPU Omniverse mode in RLinf worker subprocesses.
            # This avoids unstable multi-GPU Vulkan paths on systems with mixed ICD setups.
            sim_app = AppLauncher(
                headless=headless,
                enable_cameras=True,
                device="cuda:0",
                multi_gpu=False,
            ).app

            from isaaclab.envs import ManagerBasedRLEnv
            from ma_vla_lab.config import load_environment_config
            from ma_vla_lab.tasks.manager_based.ma_vla_lab.configurable_env import (
                build_env_cfg,
            )

            env_cfg = build_env_cfg(config_path, num_envs=num_envs)
            # Set seed at env construction to avoid Isaac Lab's non-determinism warning.
            env_cfg.seed = env_seed

            # Override camera resolution if requested
            if cam_h is not None and cam_w is not None:
                scene_cam_names = []
                scene_cam_names.extend(wrist_cam_names)
                scene_cam_names.extend([wrist_cam_name, table_cam_name, side_cam_name])
                for scene_cam_name in scene_cam_names:
                    if scene_cam_name and hasattr(env_cfg.scene, scene_cam_name):
                        scene_cam_cfg = getattr(env_cfg.scene, scene_cam_name)
                        scene_cam_cfg.height = cam_h
                        scene_cam_cfg.width = cam_w

            inner_env = ManagerBasedRLEnv(cfg=env_cfg)

            # Load env_config for prompt resolution (pure Python, no extra cost)
            env_config = load_environment_config(config_path)

            # Wrap to handle target selection + prompt generation
            env = _PromptTargetWrapper(
                inner_env,
                env_config,
                robot_id,
                robot_ids,
                object_ids,
                settle_seconds,
                self.sample_control_robot,
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
        if self._controlled_robot_ids is None:
            self._controlled_robot_ids = [self.robot_id] * self.num_envs
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
                controlled_ids = self._controlled_robot_ids or [self.robot_id] * actions.shape[0]
                for rid in sorted(set(controlled_ids)):
                    mapping = self._action_map_by_robot.get(rid, {})
                    arm_slice = mapping.get("arm_slice")
                    gripper_idx = mapping.get("gripper_idx")
                    row_ids = [i for i, x in enumerate(controlled_ids) if x == rid]
                    if not row_ids:
                        continue
                    row_idx = torch.as_tensor(row_ids, device=actions.device, dtype=torch.long)
                    if arm_slice is not None:
                        arm_start, arm_end = arm_slice
                        arm_dim = arm_end - arm_start
                        full[row_idx, arm_start:arm_end] = actions[row_idx, :arm_dim]
                    if gripper_idx is not None:
                        full[row_idx, gripper_idx] = actions[row_idx, -1]
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
        controlled_ids = obs.get("__controlled_robot_ids")
        if controlled_ids is None:
            controlled_ids = self._controlled_robot_ids or [self.robot_id] * self.num_envs
        else:
            controlled_ids = list(controlled_ids)
            self._controlled_robot_ids = list(controlled_ids)

        def _select_policy_term(term_key):
            # Fast path: all envs use one robot this step.
            if len(set(controlled_ids)) == 1:
                rid = controlled_ids[0]
                group_name = self._obs_group_name_by_robot.get(rid, self._obs_group_name)
                return obs[group_name].get(term_key)

            out = None
            for rid in sorted(set(controlled_ids)):
                group_name = self._obs_group_name_by_robot.get(rid, self._obs_group_name)
                term = obs[group_name].get(term_key)
                if term is None:
                    continue
                if out is None:
                    out = torch.zeros_like(term)
                row_ids = [i for i, x in enumerate(controlled_ids) if x == rid]
                row_idx = torch.as_tensor(row_ids, device=term.device, dtype=torch.long)
                out[row_idx] = term[row_idx]
            return out

        # Prompts come from the subprocess wrapper via the info dict.
        # They are extracted in _wrap_obs_and_info and cached in
        # self._current_prompts.  On first call before any info is
        # available, we use a list of empty strings.
        prompts = obs.get("__prompts__")
        if prompts is not None:
            self._current_prompts = prompts
        instruction = self._current_prompts or [""] * self.num_envs

        # Camera images
        wrist_image = _select_policy_term(self._wrist_obs_key)
        table_image = (
            _select_policy_term(self._table_obs_key) if self._table_obs_key else None
        )
        side_image = (
            _select_policy_term(self._side_obs_key) if self._side_obs_key else None
        )

        # Prefer subprocess-computed VLA state that exactly matches eval_smolvla.py:
        # [ee_pos_b(3), axis_angle(3), gripper(2)].
        # Fallback to legacy ee_pose + joint_pos reconstruction if unavailable.
        states = obs.get("__vla_state__")
        if states is not None:
            if isinstance(states, torch.Tensor):
                states = states.to(device=self.device, dtype=torch.float32)
            else:
                states = torch.as_tensor(states, device=self.device, dtype=torch.float32)
        else:
            # EE state fallback: ee_pose returns 7D [pos(3), quat_wxyz(4)]
            ee_pose = _select_policy_term(self._ee_obs_key)
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
            joint_pos = _select_policy_term("joint_pos_rel")
            if joint_pos is not None:
                gripper_pos = joint_pos[:, -2:]  # [B, 2]
            else:
                gripper_pos = torch.full((self.num_envs, 2), 0.04, device=self.device)
            states = torch.cat([eef_pos, aa, gripper_pos], dim=1)  # [B, 8]

        env_obs = {
            "main_images": table_image,
            "wrist_images": wrist_image,
            "side_images": side_image,
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

    def __init__(
        self,
        env,
        env_config,
        robot_id,
        robot_ids,
        object_ids,
        settle_seconds: float = 0.0,
        sample_control_robot: bool = False,
    ):
        self._env = env
        self._env_config = env_config
        self._robot_id = robot_id
        self._robot_ids = list(robot_ids) if robot_ids else [robot_id]
        self._sample_control_robot_enabled = bool(
            sample_control_robot and len(self._robot_ids) > 1
        )
        self._object_ids = object_ids
        self._num_envs = env.num_envs
        self._settle_seconds = float(settle_seconds)
        # Current prompt per env
        self._prompts = [""] * self._num_envs
        self._controlled_robot_ids = [self._robot_id] * self._num_envs
        # Expose attributes that SubProcIsaacLabEnv / IsaaclabBaseEnv need
        self.device = env.device
        # Initialize per-env targets and cache robot state extraction indices.
        self._env.target_object_names = [None] * self._num_envs
        self._ee_body_idx_by_robot = {}
        self._gripper_joint_ids_by_robot = {}
        for rid in self._robot_ids:
            robot_asset = self._env.scene[rid]
            self._ee_body_idx_by_robot[rid] = robot_asset.find_bodies("panda_hand")[0][0]
            self._gripper_joint_ids_by_robot[rid] = robot_asset.find_joints("panda_finger_joint.*")[0]

    def _extract_vla_state(self):
        """Match eval_smolvla.py state extraction: [ee_pos_b, axis_angle, gripper]."""
        import isaaclab.utils.math as math_utils

        state = torch.zeros((self._num_envs, 8), device=self.device, dtype=torch.float32)
        for rid in sorted(set(self._controlled_robot_ids)):
            robot_asset = self._env.scene[rid]
            ee_body_idx = self._ee_body_idx_by_robot[rid]
            gripper_joint_ids = self._gripper_joint_ids_by_robot[rid]

            ee_pos_w = robot_asset.data.body_pos_w[:, ee_body_idx]
            ee_quat_w = robot_asset.data.body_quat_w[:, ee_body_idx]
            root_pos_w = robot_asset.data.root_pos_w
            root_quat_w = robot_asset.data.root_quat_w
            ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
                root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
            )
            ee_aa = math_utils.axis_angle_from_quat(ee_quat_b)
            gripper_pos = robot_asset.data.joint_pos[:, gripper_joint_ids]
            robot_state = torch.cat([ee_pos_b, ee_aa, gripper_pos], dim=-1).to(dtype=state.dtype)

            row_ids = [i for i, x in enumerate(self._controlled_robot_ids) if x == rid]
            row_idx = torch.as_tensor(row_ids, device=self.device, dtype=torch.long)
            state[row_idx] = robot_state[row_idx]
        return state

    def _settle_simulation(self, obs):
        """Advance sim with zero actions after reset to let objects settle."""
        if self._settle_seconds <= 0:
            return obs

        settle_steps = max(1, int(round(float(self._settle_seconds) / float(self._env.step_dt))))
        zero_actions = torch.zeros(
            (self._env.num_envs, self._env.action_space.shape[-1]), device=self.device
        )
        is_rendering = self._env.sim.has_gui() or self._env.sim.has_rtx_sensors()

        try:
            self._env.action_manager.process_action(zero_actions)
            for _ in range(settle_steps):
                for _ in range(self._env.cfg.decimation):
                    self._env._sim_step_counter += 1
                    self._env.action_manager.apply_action()
                    self._env.scene.write_data_to_sim()
                    self._env.sim.step(render=False)
                    if (
                        self._env._sim_step_counter % self._env.cfg.sim.render_interval == 0
                        and is_rendering
                    ):
                        self._env.sim.render()
                    self._env.scene.update(dt=self._env.physics_dt)
            obs = self._env.observation_manager.compute(update_history=True)
        except Exception:
            for _ in range(settle_steps):
                obs, _, _, _, _ = self._env.step(zero_actions)
        return obs

    def _sample_control_robot(self, idx: int):
        import random as _random

        if self._sample_control_robot_enabled and self._robot_ids:
            rid = _random.choice(self._robot_ids)
        else:
            rid = self._robot_id
        self._controlled_robot_ids[idx] = rid

    def _sample_target_prompt(self, idx: int):
        import random as _random
        from ma_vla_lab.config import resolve_prompt

        robot_id = self._controlled_robot_ids[idx]
        if self._object_ids:
            target_id = _random.choice(self._object_ids)
        else:
            target_id = None
        self._env.target_object_names[idx] = target_id
        self._prompts[idx] = resolve_prompt(self._env_config, robot_id, target_id)

    def _resample_episode(self, idx: int):
        self._sample_control_robot(idx)
        self._sample_target_prompt(idx)

    def _active_termination_keys_from_info(self, info: dict, env_idx: int) -> list[str]:
        term_info = info.get("termination_info", {}) if isinstance(info, dict) else {}
        active = []
        for key, value in term_info.items():
            try:
                term_tensor = torch.as_tensor(value)
                if term_tensor.ndim == 0:
                    hit = bool(term_tensor.item())
                elif env_idx < term_tensor.shape[0]:
                    hit = bool(torch.as_tensor(term_tensor[env_idx]).any().item())
                else:
                    hit = bool(term_tensor.any().item())
                if hit:
                    active.append(key)
            except Exception:
                continue
        return active

    def _active_termination_keys_from_manager(self, env_idx: int) -> list[str]:
        manager = getattr(self._env, "termination_manager", None)
        if manager is None and hasattr(self._env, "unwrapped"):
            manager = getattr(self._env.unwrapped, "termination_manager", None)
        if manager is None:
            return []

        active = []
        term_names = getattr(manager, "_term_names", None)
        term_dones = getattr(manager, "_term_dones", None)
        if term_names is not None and term_dones is not None:
            try:
                for term_name, term_done in zip(term_names, term_dones):
                    if bool(torch.as_tensor(term_done[env_idx]).any().item()):
                        active.append(str(term_name))
                if active:
                    return active
            except Exception:
                pass

        for term_name in getattr(manager, "active_terms", []):
            try:
                term_val = manager.get_term(term_name)
                if bool(torch.as_tensor(term_val[env_idx]).any().item()):
                    active.append(str(term_name))
            except Exception:
                continue
        return active

    def _classify_episode_result(self, info: dict, env_idx: int, terminated: bool, truncated: bool):
        active_terms = self._active_termination_keys_from_info(info, env_idx)
        source = "info"
        if not active_terms:
            active_terms = self._active_termination_keys_from_manager(env_idx)
            source = "manager"

        success_hit = any(k == "success" or k.startswith("success_") for k in active_terms)
        failure_hit = any(k == "failure" or k.startswith("failure_") for k in active_terms)
        unknown_hit = (terminated or truncated) and not (success_hit or failure_hit)
        classified_success = success_hit and not failure_hit
        return classified_success, success_hit, failure_hit, unknown_hit, active_terms, source

    def reset(self, seed=None, env_ids=None):
        obs, info = self._env.reset(seed=seed, env_ids=env_ids)

        # Determine which envs are being reset
        if env_ids is None:
            reset_indices = range(self._num_envs)
        else:
            reset_indices = env_ids.tolist()

        # Pick random target + generate prompt for each resetting env
        for idx in reset_indices:
            self._resample_episode(idx)

        # Settle only on explicit/full reset to avoid perturbing live envs.
        if env_ids is None:
            obs = self._settle_simulation(obs)

        # Log prompts for debugging
        for idx in reset_indices:
            robot_id = self._controlled_robot_ids[idx]
            target = self._env.target_object_names[idx]
            print(f"[PromptTargetWrapper] env={idx} robot={robot_id} target={target} "
                  f"prompt={self._prompts[idx]!r}")

        # Inject prompts into obs so they cross the subprocess boundary
        obs["__prompts__"] = list(self._prompts)
        obs["__controlled_robot_ids"] = list(self._controlled_robot_ids)
        obs["__vla_state__"] = self._extract_vla_state()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        # ManagerBasedRLEnv auto-resets done envs inside step(). Refresh target/prompt
        # for those new episodes immediately to keep rollout state synchronized.
        done_mask = torch.logical_or(terminated, truncated)
        reset_ids = done_mask.nonzero(as_tuple=True)[0].tolist()

        # Print termination cause for each env before we sample next-episode metadata.
        for idx in reset_ids:
            term_flag = bool(torch.as_tensor(terminated[idx]).item())
            trunc_flag = bool(torch.as_tensor(truncated[idx]).item())
            (
                classified_success,
                success_hit,
                failure_hit,
                unknown_hit,
                active_terms,
                term_source,
            ) = self._classify_episode_result(info, idx, term_flag, trunc_flag)

            robot_id = self._controlled_robot_ids[idx]
            target = self._env.target_object_names[idx]
            reward_val = float(torch.as_tensor(reward[idx]).item())
            print(
                f"[PromptTargetWrapper][TERM] env={idx} robot={robot_id} target={target} "
                f"reward={reward_val:.4f} terminated={term_flag} truncated={trunc_flag} "
                f"active={active_terms} source={term_source} "
                f"classified_success={classified_success} success_hit={success_hit} "
                f"failure_hit={failure_hit} unknown_hit={unknown_hit}"
            )

        for idx in reset_ids:
            self._resample_episode(idx)
        # Include current prompts in obs for the adapter
        obs["__prompts__"] = list(self._prompts)
        obs["__controlled_robot_ids"] = list(self._controlled_robot_ids)
        obs["__vla_state__"] = self._extract_vla_state()
        return obs, reward, terminated, truncated, info

    def close(self):
        self._env.close()

    def __getattr__(self, name):
        return getattr(self._env, name)
