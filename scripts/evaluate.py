import argparse

from face_defense.core.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate face defense models")
    parser.add_argument("--config", type=str, required=True, help="Path to evaluation config YAML")
    parser.add_argument("--output", type=str, default="outputs/results/", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    print(f"Evaluating with config: {args.config}")
    print(f"Output directory: {args.output}")

    # TODO: Build pipeline, load dataset, run evaluation
    # pipeline = build_pipeline(config)
    # evaluator = Evaluator(config)
    # results = evaluator.run(pipeline, dataset)
    # results.save(args.output)


if __name__ == "__main__":
    main()
