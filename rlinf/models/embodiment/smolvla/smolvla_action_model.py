"""SmolVLA RL action prediction model for RLinf.

Wraps a pretrained SmolVLAPolicy (LeRobot format) and adds:
- Flow-matching denoising with chain recording for RL logprob computation
- SDE noise injection for exploration
- Value head for PPO
- obs_processor to convert RLinf env obs → SmolVLA input format
"""

import math
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.value_head import ValueHead


def get_logprob_norm(sample, mu, sigma):
    """Gaussian log probability: log N(sample | mu, sigma)."""
    mask = sigma == 0
    sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
    constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
        2 * torch.pi * torch.ones_like(sample)
    )
    exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
    log_prob = constant_term + exponent_term
    log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
    return log_prob


def gaussian_entropy(sigma):
    """Differential entropy of a Gaussian: 0.5 * log(2 * pi * e * sigma^2)."""
    mask = sigma == 0
    sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
    entropy = 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe**2))
    return entropy


class SmolVLAForRLActionPrediction(nn.Module, BasePolicy):
    """SmolVLA model wrapped for RL action prediction with PPO."""

    def __init__(self, policy, smolvla_cfg):
        nn.Module.__init__(self)

        # Store the full SmolVLAPolicy (includes model, tokenizer, norm stats)
        self.policy = policy
        self.flow_model = policy.model  # VLAFlowMatching

        # RL config from Hydra
        self.noise_level = getattr(smolvla_cfg, "noise_level", 0.5)
        self.noise_method = getattr(smolvla_cfg, "noise_method", "flow_sde")
        self.action_chunk = getattr(smolvla_cfg, "action_chunk", 10)
        self.action_env_dim = getattr(smolvla_cfg, "action_env_dim", 7)
        self.num_steps = getattr(smolvla_cfg, "num_steps", 10)
        self.joint_logprob = getattr(smolvla_cfg, "joint_logprob", False)
        self.num_images_in_input = getattr(smolvla_cfg, "num_images_in_input", 3)
        self._add_value_head = getattr(smolvla_cfg, "add_value_head", True)
        self._value_after_vlm = getattr(smolvla_cfg, "value_after_vlm", True)
        self._value_vlm_mode = getattr(smolvla_cfg, "value_vlm_mode", "mean_token")

        # Load normalization stats from the pretrained model
        self._load_norm_stats()

        # Build tokenizer reference
        self._tokenizer = self.flow_model.vlm_with_expert.processor.tokenizer
        self._tokenizer_max_length = policy.config.tokenizer_max_length

        # Value head
        if self._add_value_head:
            if self._value_after_vlm:
                # VLM hidden size
                proj_width = self.flow_model.vlm_with_expert.config.text_config.hidden_size
            else:
                # Expert hidden size
                proj_width = self.flow_model.vlm_with_expert.expert_hidden_size
            self.value_head = ValueHead(
                input_dim=proj_width,
                hidden_sizes=(512, 256, 128),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )

        self.use_vlm_value = self._value_after_vlm and self._add_value_head

    def _load_norm_stats(self):
        """Load MEAN_STD normalization stats from pretrained model."""
        from huggingface_hub import hf_hub_download
        import safetensors.torch as st

        model_path = self.policy.config._name_or_path
        if not model_path:
            # Fallback: try to get from the config
            model_path = getattr(self.policy.config, "model_path", None)

        # Try to load preprocessor stats (for state normalization)
        try:
            pre_path = hf_hub_download(
                model_path,
                "policy_preprocessor_step_5_normalizer_processor.safetensors",
            )
            pre_stats = st.load_file(pre_path)
            self.register_buffer(
                "state_mean", pre_stats["observation.state.mean"]
            )
            self.register_buffer(
                "state_std", pre_stats["observation.state.std"]
            )
        except Exception:
            # Fallback: no normalization
            self.register_buffer("state_mean", torch.zeros(8))
            self.register_buffer("state_std", torch.ones(8))

        # Try to load postprocessor stats (for action unnormalization)
        try:
            post_path = hf_hub_download(
                model_path,
                "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            )
            post_stats = st.load_file(post_path)
            self.register_buffer(
                "action_mean", post_stats["action.mean"]
            )
            self.register_buffer(
                "action_std", post_stats["action.std"]
            )
        except Exception:
            self.register_buffer("action_mean", torch.zeros(7))
            self.register_buffer("action_std", torch.ones(7))

    def _normalize_state(self, state):
        """Normalize state using MEAN_STD: (x - mean) / (std + eps)."""
        eps = 1e-8
        # state is [B, 8], stats may be [8] — match device
        mean = self.state_mean.to(state.device)
        std = self.state_std.to(state.device)
        # Only normalize the dims we have stats for
        dim = min(state.shape[-1], mean.shape[-1])
        out = state.clone()
        out[..., :dim] = (state[..., :dim] - mean[:dim]) / (std[:dim] + eps)
        return out

    def _unnormalize_actions(self, actions):
        """Unnormalize actions: x * (std + eps) + mean."""
        eps = 1e-8
        mean = self.action_mean.to(actions.device)
        std = self.action_std.to(actions.device)
        dim = min(actions.shape[-1], mean.shape[-1])
        out = actions.clone()
        out[..., :dim] = actions[..., :dim] * (std[:dim] + eps) + mean[:dim]
        return out

    @property
    def _no_split_modules(self):
        return ["LlamaDecoderLayer"]

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError

    def obs_processor(self, env_obs):
        """Convert RLinf env obs → SmolVLA model inputs.

        RLinf provides:
            main_images:  [B, H, W, 3] uint8
            wrist_images: [B, H, W, 3] uint8
            side_images:  [B, H, W, 3] uint8 (optional)
            states:       [B, 8] float32
            task_descriptions: list[str]

        SmolVLA expects:
            images: list of [B, C, H, W] float32 [0, 1]
            state:  [B, max_state_dim] float32 (normalized, padded to 32)
            lang_tokens: [B, max_length] long
            lang_masks:  [B, max_length] long
        """
        device = next(self.parameters()).device

        # Images: HWC uint8 → CHW float32 [0, 1]
        def process_image(img):
            if img is None:
                return None
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            img = img.to(device=device, dtype=torch.float32)
            if img.max() > 1.0:
                img = img / 255.0
            # HWC → CHW
            if img.ndim == 4 and img.shape[-1] == 3:
                img = img.permute(0, 3, 1, 2)
            return img

        images = []
        img_masks = []
        bsize = None

        for key in ["main_images", "wrist_images", "side_images"]:
            img = env_obs.get(key)
            processed = process_image(img)
            if processed is not None:
                bsize = processed.shape[0]
                # Resize with padding to 512x512, then normalize to [-1, 1] for SigLIP
                from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad
                processed = resize_with_pad(processed, 512, 512, pad_value=0)
                processed = processed * 2.0 - 1.0
                images.append(processed)
                img_masks.append(
                    torch.ones(bsize, dtype=torch.bool, device=device)
                )
            else:
                # Create a dummy masked-out image if we need 3 images
                if bsize is not None and len(images) < self.num_images_in_input:
                    dummy = torch.ones_like(images[0]) * -1
                    images.append(dummy)
                    img_masks.append(
                        torch.zeros(bsize, dtype=torch.bool, device=device)
                    )

        # State: normalize with MEAN_STD, pad to max_state_dim=32
        model_dtype = next(self.parameters()).dtype
        state = env_obs["states"]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        state = state.to(device=device, dtype=torch.float32)
        state = self._normalize_state(state)
        # Pad to 32D
        if state.shape[-1] < 32:
            pad = torch.zeros(
                state.shape[0], 32 - state.shape[-1],
                device=device, dtype=torch.float32,
            )
            state = torch.cat([state, pad], dim=-1)
        state = state.to(dtype=model_dtype)

        # Language: tokenize task descriptions
        prompts = env_obs["task_descriptions"]
        # Add newline (SmolVLA convention)
        prompts = [
            p + "\n" if not p.endswith("\n") else p for p in prompts
        ]
        tokens = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            max_length=self._tokenizer_max_length,
            truncation=True,
        )
        lang_tokens = tokens["input_ids"].to(device)
        lang_masks = tokens["attention_mask"].to(device, dtype=torch.bool)

        # Cast images to model dtype (bfloat16)
        images = [img.to(dtype=model_dtype) for img in images]

        return images, img_masks, lang_tokens, lang_masks, state

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        compute_values=True,
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Full inference pipeline: obs → actions + RL metadata."""
        images, img_masks, lang_tokens, lang_masks, state = self.obs_processor(
            env_obs
        )

        bsize = state.shape[0]
        device = state.device
        num_steps = self.num_steps
        chunk_size = self.flow_model.config.chunk_size
        max_action_dim = self.flow_model.config.max_action_dim

        # Sample initial noise (cast to model dtype)
        model_dtype = next(self.parameters()).dtype
        noise = self.flow_model.sample_noise(
            (bsize, chunk_size, max_action_dim), device
        ).to(dtype=model_dtype)

        # Embed prefix and compute KV cache
        prefix_embs, prefix_pad_masks, prefix_att_masks = (
            self.flow_model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
        )
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        outputs_embeds, past_key_values = self.flow_model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )

        # VLM output for value head (captured from the prefix forward)
        prefix_output = outputs_embeds[0] if self.use_vlm_value else None

        # Denoising loop with chain recording
        dt = -1.0 / num_steps
        x_t = noise
        chains = [x_t]
        log_probs_list = []
        values_list = []

        # Noise schedule for SDE
        noise_level = torch.tensor(self.noise_level, device=device, dtype=model_dtype)

        # Denoise index selection (same pattern as OpenPI)
        if mode == "train":
            if self.joint_logprob:
                denoise_inds = torch.arange(
                    num_steps, device=device, dtype=torch.long
                )[None].repeat(bsize, 1)
            else:
                # Sample one denoise index per environment (not one shared index for
                # the whole batch) to avoid synchronized exploration timing.
                denoise_inds = torch.randint(
                    low=0,
                    high=num_steps,
                    size=(bsize, 1),
                    device=device,
                    dtype=torch.long,
                ).repeat(1, num_steps)
        else:
            denoise_inds = torch.full(
                (bsize, num_steps), -1, device=device, dtype=torch.long
            )

        for step_idx in range(num_steps):
            time = 1.0 + step_idx * dt
            time_tensor = torch.tensor(
                time, dtype=model_dtype, device=device
            ).expand(bsize)

            # Get velocity prediction
            v_t = self.flow_model.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time_tensor,
            )

            # Compute value from expert output if not using VLM value
            if (
                self._add_value_head
                and compute_values
                and not self._value_after_vlm
            ):
                # Re-run suffix to get hidden states for value
                suffix_embs, suffix_pad_masks, suffix_att_masks = (
                    self.flow_model.embed_suffix(x_t, time_tensor)
                )
                suffix_len = suffix_pad_masks.shape[1]
                prefix_len = prefix_pad_masks.shape[1]
                prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
                    bsize, suffix_len, prefix_len
                )
                suffix_att_2d_masks = make_att_2d_masks(
                    suffix_pad_masks, suffix_att_masks
                )
                full_att_2d_masks = torch.cat(
                    [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2
                )
                prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
                position_ids = (
                    prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
                )
                outputs_embeds, _ = self.flow_model.vlm_with_expert.forward(
                    attention_mask=full_att_2d_masks,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=[None, suffix_embs],
                    use_cache=False,
                    fill_kv_cache=False,
                )
                suffix_out = outputs_embeds[1]
                suffix_out = suffix_out[:, -chunk_size:]
                suffix_out = suffix_out.to(dtype=torch.float32)
                suffix_out_mean = torch.mean(
                    suffix_out[:, : self.action_chunk], dim=1
                )
                value_t = self.value_head(suffix_out_mean)[:, 0]
            else:
                value_t = torch.zeros(bsize, device=device)
            values_list.append(value_t)

            if mode == "train" and self.noise_method == "flow_sde":
                sampled_step_mask = denoise_inds[:, step_idx] == step_idx
            else:
                sampled_step_mask = torch.zeros(
                    bsize, dtype=torch.bool, device=device
                )

            if sampled_step_mask.any():
                # SDE noise injection
                t_val = time
                t_next = time + dt  # = time - 1/num_steps
                # sigma_i = noise_level * sqrt(t / (1 - t))
                t_safe = max(t_val, 1e-6)
                sigma_i = noise_level * math.sqrt(t_safe / max(1.0 - t_safe, 1e-6))
                delta = abs(dt)

                # x_t_mean via ODE-SDE mix (same as OpenPI)
                x0_pred = x_t - v_t * t_val
                x1_pred = x_t + v_t * (1 - t_val)
                t_next_val = max(t_next, 0.0)
                x0_weight = 1.0 - t_next_val
                x1_weight = t_next_val - sigma_i**2 * delta / (2 * t_safe)
                x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
                x_t_std = math.sqrt(delta) * sigma_i

                sde_noise = self.flow_model.sample_noise(x_t.shape, device).to(dtype=model_dtype)
                x_t_sde = x_t_mean + sde_noise * x_t_std
                x_t_ode = x_t + dt * v_t
                sampled_step_mask = sampled_step_mask[:, None, None]
                x_t = torch.where(sampled_step_mask, x_t_sde, x_t_ode)

                sde_log_prob = get_logprob_norm(
                    x_t_sde,
                    x_t_mean,
                    torch.full_like(x_t_mean, x_t_std),
                )
                log_prob = torch.where(
                    sampled_step_mask, sde_log_prob, torch.zeros_like(sde_log_prob)
                )
            else:
                # Pure ODE step
                x_t = x_t + dt * v_t
                log_prob = torch.zeros_like(x_t)

            chains.append(x_t)
            log_probs_list.append(log_prob)

        x_0 = x_t
        chains = torch.stack(chains, dim=1)  # [B, num_steps+1, chunk, action_dim]

        # Process logprobs
        log_probs = torch.stack(log_probs_list, dim=1)  # [B, num_steps, chunk, action_dim]
        log_probs = log_probs[
            :, :, : self.action_chunk, : self.action_env_dim
        ]
        if self.joint_logprob:
            log_probs = log_probs.mean(dim=1)
        else:
            log_probs = log_probs[
                torch.arange(log_probs.shape[0], device=log_probs.device),
                denoise_inds[:, 0],
            ]

        # Process values
        if self.use_vlm_value and prefix_output is not None:
            values = self._get_value_from_vlm(prefix_output, prefix_pad_masks)[:, None]
        else:
            values = torch.stack(values_list, dim=1).mean(dim=-1, keepdim=True)

        # Unpad and unnormalize actions
        actions = x_0[:, : self.action_chunk, : self.action_env_dim]
        actions = self._unnormalize_actions(actions)
        actions_np = actions.cpu().float().numpy()

        # Cache forward inputs for training
        # Store pre-processed tensors (not raw strings) for serialization
        forward_inputs = {
            "chains": chains,
            "denoise_inds": denoise_inds,
            # Cache raw env obs for re-processing in default_forward
            "env_main_images": env_obs["main_images"],
            "env_wrist_images": env_obs.get("wrist_images"),
            "env_side_images": env_obs.get("side_images"),
            "env_states": env_obs["states"],
            "env_lang_tokens": lang_tokens,
            "env_lang_masks": lang_masks,
        }

        result = {
            "prev_logprobs": log_probs,
            "prev_values": values,
            "forward_inputs": forward_inputs,
        }
        return actions_np, result

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def default_forward(
        self,
        forward_inputs: dict[str, Any],
        **kwargs,
    ) -> dict[str, Any]:
        """Re-compute logprobs, values, entropy for RL loss (PPO actor update)."""
        compute_values = kwargs.get("compute_values", False)
        chains = forward_inputs["chains"]
        denoise_inds = forward_inputs["denoise_inds"]

        # Re-process observations (use cached tokens, not raw strings)
        env_obs = {
            "main_images": forward_inputs["env_main_images"],
            "wrist_images": forward_inputs.get("env_wrist_images"),
            "side_images": forward_inputs.get("env_side_images"),
            "states": forward_inputs["env_states"],
        }
        # obs_processor without task_descriptions — we use cached tokens
        images, img_masks, _, _, state = self.obs_processor(
            {**env_obs, "task_descriptions": ["_"] * forward_inputs["env_states"].shape[0]}
        )
        lang_tokens = forward_inputs["env_lang_tokens"]
        lang_masks = forward_inputs["env_lang_masks"]

        device = chains.device

        # Embed prefix and compute KV cache
        prefix_embs, prefix_pad_masks, prefix_att_masks = (
            self.flow_model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
        )
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        outputs_embeds, past_key_values = self.flow_model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        prefix_output = outputs_embeds[0] if self.use_vlm_value else None

        bsize = chains.shape[0]
        num_steps = self.num_steps
        chunk_size = self.flow_model.config.chunk_size
        model_dtype = next(self.parameters()).dtype
        noise_level = torch.tensor(self.noise_level, device=device, dtype=model_dtype)

        chains_log_probs = []
        chains_values = []
        chains_entropy = []

        # Determine how many steps to re-evaluate
        if self.joint_logprob:
            eval_steps = num_steps
        else:
            eval_steps = 1

        for idx in range(eval_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[torch.arange(bsize), denoise_ind]
            chains_next = chains[torch.arange(bsize), denoise_ind + 1]

            # Time for this denoising step
            dt = -1.0 / num_steps
            time_vals = 1.0 + denoise_ind.to(dtype=model_dtype) * dt
            time_tensor = time_vals.to(device)

            # Get velocity
            v_t = self.flow_model.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=chains_pre,
                timestep=time_tensor,
            )

            # Compute mean and std for this step (SDE)
            t_input = time_tensor[:, None, None].expand_as(chains_pre)
            delta = abs(dt)

            if self.noise_method == "flow_sde":
                timesteps = torch.linspace(
                    1, 1 / num_steps, num_steps, device=device, dtype=model_dtype
                )
                timesteps = torch.cat(
                    [timesteps, torch.tensor([0.0], device=device, dtype=model_dtype)]
                )
                sigmas = (
                    noise_level
                    * torch.sqrt(
                        timesteps
                        / (
                            1
                            - torch.where(
                                timesteps == 1, timesteps[1], timesteps
                            )
                        )
                    )[:-1]
                )
                sigma_i = sigmas[denoise_ind][:, None, None].expand_as(
                    chains_pre
                )
                x0_pred = chains_pre - v_t * t_input
                x1_pred = chains_pre + v_t * (1 - t_input)
                x0_weight = 1.0 - (t_input - delta)
                x1_weight = (
                    t_input - delta - sigma_i**2 * delta / (2 * t_input)
                )
                x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
                x_t_std = torch.sqrt(
                    torch.tensor(delta, device=device, dtype=model_dtype)
                ) * sigma_i
            else:
                # Pure ODE
                x_t_mean = chains_pre + dt * v_t
                x_t_std = torch.zeros_like(chains_pre)

            log_prob = get_logprob_norm(chains_next, x_t_mean, x_t_std)
            entropy = gaussian_entropy(x_t_std)
            chains_log_probs.append(log_prob)
            chains_entropy.append(entropy)

            # Value from expert (non-VLM mode)
            if self._add_value_head and compute_values and not self._value_after_vlm:
                suffix_embs, suffix_pad_masks, suffix_att_masks = (
                    self.flow_model.embed_suffix(chains_pre, time_tensor)
                )
                suffix_len = suffix_pad_masks.shape[1]
                prefix_len = prefix_pad_masks.shape[1]
                prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
                    bsize, suffix_len, prefix_len
                )
                suffix_att_2d_masks = make_att_2d_masks(
                    suffix_pad_masks, suffix_att_masks
                )
                full_att_2d_masks = torch.cat(
                    [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2
                )
                prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
                position_ids = (
                    prefix_offsets
                    + torch.cumsum(suffix_pad_masks, dim=1)
                    - 1
                )
                outputs_embeds, _ = self.flow_model.vlm_with_expert.forward(
                    attention_mask=full_att_2d_masks,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=[None, suffix_embs],
                    use_cache=False,
                    fill_kv_cache=False,
                )
                suffix_out = outputs_embeds[1]
                suffix_out = suffix_out[:, -chunk_size:]
                suffix_out = suffix_out.to(dtype=torch.float32)
                suffix_out_mean = torch.mean(
                    suffix_out[:, : self.action_chunk], dim=1
                )
                value_t = self.value_head(suffix_out_mean)[:, 0]
                chains_values.append(value_t)
            elif not self.use_vlm_value:
                chains_values.append(torch.zeros(bsize, device=device))

        if self.use_vlm_value and prefix_output is not None:
            chains_values.append(
                self._get_value_from_vlm(prefix_output, prefix_pad_masks)
            )

        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        chains_entropy = torch.stack(chains_entropy, dim=1)

        # Slice to action_chunk x action_env_dim
        chains_log_probs = chains_log_probs[
            :, :, : self.action_chunk, : self.action_env_dim
        ]
        chains_entropy = chains_entropy[
            :, :, : self.action_chunk, : self.action_env_dim
        ]

        # Average across denoising steps
        log_probs = chains_log_probs.mean(dim=1)
        entropy = chains_entropy.mean(dim=[1, 2, 3], keepdim=False)[:, None]

        chains_values = torch.stack(chains_values, dim=1)
        values = chains_values.mean(dim=-1, keepdim=False)

        return {
            "logprobs": log_probs,
            "values": values,
            "entropy": entropy,
        }

    def _get_value_from_vlm(self, prefix_output, prefix_pad_masks):
        """Compute value from VLM prefix output."""
        if self._value_vlm_mode == "mean_token":
            # Use all valid prefix tokens
            mask = prefix_pad_masks.unsqueeze(-1).float()
            prefix_out_value = (prefix_output * mask).sum(dim=1) / mask.sum(
                dim=1
            ).clamp(min=1)
        elif self._value_vlm_mode == "last_token":
            # Use the last valid token
            lengths = prefix_pad_masks.sum(dim=1).long() - 1
            prefix_out_value = prefix_output[
                torch.arange(prefix_output.shape[0]), lengths
            ]
        elif self._value_vlm_mode == "first_token":
            prefix_out_value = prefix_output[:, 0]
        else:
            raise ValueError(
                f"Unknown value_vlm_mode: {self._value_vlm_mode}"
            )
        prefix_out_value = prefix_out_value.to(dtype=torch.float32)
        return self.value_head(prefix_out_value)[:, 0]
