import wandb

def init_wandb(config):
    if config.log_to_wandb:
        wandb.init(
            project=config.project_name,
            entity=config.entity,
            name=config.run_name,
            config=config
        )

def log_to_wandb(data, step=None):
    if wandb.run is not None:
        wandb.log(data, step=step)