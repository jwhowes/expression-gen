import click

from src.train.vae import VAETrainer


@click.group()
def cli(): ...


@cli.command()
@click.argument("config_path", type=str)
def train_vae(config_path: str):
    trainer = VAETrainer(config_path)

    trainer.train()


if __name__ == "__main__":
    cli()
