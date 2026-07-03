from pydantic import BaseModel


class DQNConfig(BaseModel):
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

    # dqn hyperparams
    gamma: float = 0.99
    tau: float = 1.0                # 1.0 = hard target update
    q_lr: float = 1e-4

    buffer_size: int = 1_000_000
    buffer_batch_size: int = 32
    learning_starts: int = 1_000
    training_interval: int = 4
    target_update_interval: int = 250

    # epsilon schedule
    start_epsilon: float = 1.0
    end_epsilon: float = 0.01
    explore_frac: float = 0.025

    # training loop
    total_steps: int = 1_000_000
    logging_freq: int = 1_000
    save_interval: int = 50_000
