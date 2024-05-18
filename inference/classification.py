import jax, argparse
from tokenizer.tokenizer import Tokenizer
from model.classification import Classification
from model.model import load_model
import numpy as np
from jax import numpy as jnp


def get_args():
    parser = argparse.ArgumentParser(
        description="Inference using a Classification model."
    )
    parser.add_argument(
        "--prefix", type=str, default="out/ag_news", help="Model prefix."
    )
    parser.add_argument("input", type=str)

    return parser.parse_args()


def main(args):
    """
    The main entry point for the classification task.

    Args:
        args: An object containing command-line arguments.

    The function creates a `Tokenizer` object with the prefix specified in `args` and describes its configuration.
    It then loads the model configuration and parameters from the prefix path. A `Classification` model is created
    with the loaded configuration and initialized with a seed and a zero array.

    The model is then used to predict the class of the input text. The input text is encoded with padding using the
    tokenizer and passed to the `predict` method of the model. The predicted class is printed to the console.

    """
    t = Tokenizer(args.prefix)
    t.describe()

    config, params = load_model(args.prefix)
    model = Classification(config)

    seed = jax.random.PRNGKey(0)
    model.init(seed, jnp.zeros((1, 255), dtype=jnp.int32))
    model.describe(True)

    predicts = model.predict(params, t.encode_with_padding([args.input]))
    print("")
    for i, pred in enumerate(predicts):
        print(f"input= {args.input}")
        print(f"predicted= {config.classes[pred]}")


if __name__ == "__main__":
    args = get_args()
    assert args.input != "", "Please provide the input."

    main(args)
