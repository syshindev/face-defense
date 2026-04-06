import argparse

from face_defense.core.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train a face defense model")
    parser.add_argument("--config", type=str, required=True, help="Path to model config YAML")
    parser.add_argument("overrides", nargs="*", help="Config overrides (e.g. training.lr=0.001)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config, overrides=args.overrides)

    print(f"Training model: {config.get('model', {}).get('name', 'unknown')}")
    print(f"Config: {args.config}")

    # TODO: Build model, dataset, and trainer from config
    # model = build_model(config)
    # dataset = build_dataset(config)
    # trainer = Trainer(config.training)
    # trainer.train(model, dataset)


if __name__ == "__main__":
    main()
