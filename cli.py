import click
from src.train import train_loop
from src.evaluate import evaluate
from src.infer import infer_folder

@click.group()
def cli():
    "Medical Microscopy - CLI"

@cli.command()
@click.option("--config", default="config/train.yaml")
def train(config):
    train_loop(config)

@cli.command(name="evaluate")
@click.option("--ckpt", "ckpt_path", required=True)
@click.option("--data-root", required=True)
@click.option("--img-size", default=224, type=int)
def evaluate_cmd(ckpt_path, data_root, img_size):
    evaluate(ckpt_path, data_root, img_size)

@cli.command()
@click.option("--ckpt", "ckpt_path", required=True)
@click.option("--input-dir", required=True)
@click.option("--out", "out_dir", default="reports/infer_json")
@click.option("--img-size", default=224, type=int)
@click.option("--mc", "mc_passes", default=20, type=int)
def infer(ckpt_path, input_dir, out_dir, img_size, mc_passes):
    infer_folder(ckpt_path, input_dir, out_dir, img_size, mc_passes)

if __name__ == "__main__":
    cli()
