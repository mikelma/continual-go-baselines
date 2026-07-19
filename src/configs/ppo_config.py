from pydantic import BaseModel


class PPOConfig(BaseModel):
    # env
    board_size: int = 9
    max_stones: int = 32

    # wandb
    wandb: bool = True
    # seeding
    seed: int = 42

    # network (alpha_zero resnetv2)
    num_channels: int = 128
    num_blocks: int = 6

    # ppo hyperparams
    gamma: float = 0.99
    gae_lambda: float = 0.95

    rollout_steps: int = 2048
    epochs: int = 4
    num_mini_batch: int = 32

    clip_eps: float = 0.2
    vf_coef: float = 0.5
    entropy_coef: float = 0.0
    gradient_clipping: bool = True
    max_grad_norm: float = 0.5

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    # training loop
    total_steps: int = 1_000_000
    save_interval: int = 50_000
