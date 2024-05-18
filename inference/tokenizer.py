import argparse
from tokenizer.tokenizer import Tokenizer


def get_args():
    parser = argparse.ArgumentParser(description="Inference using a Tokenizer model.")
    parser.add_argument(
        "--prefix", type=str, default="data/botchan", help="Model prefix."
    )
    parser.add_argument("--lower", type=bool, default=True, help="Lowercase the input.")
    parser.add_argument("input", type=str)

    return parser.parse_args()


def main(args):
    """
    The main entry point for inspecting a tokenizer.

    Args:
        args: An object containing command-line arguments.

    The function creates a `Tokenizer` object with the prefix specified in `args`. It calls the `describe` method of the `Tokenizer`
    object to print out information about the tokenizer configuration. It calls the `inspect` method of the `Tokenizer` object with
    the input specified in `args`. If the `lower` attribute of `args` is `True`, it converts the input to lowercase before passing
    it to `inspect`.
    """
    t = Tokenizer(args.prefix, lower=args.lower)
    t.describe()
    t.inspect(args.input)


if __name__ == "__main__":
    args = get_args()
    assert args.input != "", "Please provide the input."

    main(args)
