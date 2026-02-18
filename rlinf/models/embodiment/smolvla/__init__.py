"""SmolVLA model loader for RLinf."""

from omegaconf import DictConfig


def get_model(cfg: DictConfig, torch_dtype=None):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    from rlinf.models.embodiment.smolvla.smolvla_action_model import (
        SmolVLAForRLActionPrediction,
    )

    # Load pretrained SmolVLA policy (LeRobot format)
    base_policy = SmolVLAPolicy.from_pretrained(cfg.model_path)
    # Ensure model path is stored for norm stats loading
    base_policy.config._name_or_path = str(cfg.model_path)

    # SmolVLA's saved config freezes VLM by default (train_expert_only=True).
    # If our RLinf config says train_expert_only=False, unfreeze VLM.
    if not getattr(cfg.smolvla, "train_expert_only", True):
        for param in base_policy.model.vlm_with_expert.vlm.parameters():
            param.requires_grad = True

    # Wrap in RL action prediction model
    smolvla_cfg = cfg.smolvla
    model = SmolVLAForRLActionPrediction(base_policy, smolvla_cfg)

    return model
