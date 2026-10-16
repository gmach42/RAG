"""Entry point for `python -m src`."""
import fire

from src.output import CLI

if __name__ == "__main__":
    fire.Fire(CLI)
