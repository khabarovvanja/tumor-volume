import fire
from tumor_volume.training.train import train_entrypoint
from tumor_volume.inference.infer import infer

def main():
    fire.Fire(
        {
            "train": train_entrypoint,
            "infer": infer,
        }
    )

if __name__ == "__main__":
    train_entrypoint()