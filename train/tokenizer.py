import argparse
import sentencepiece as spm
import os, re


def get_args():
    parser = argparse.ArgumentParser(description="Trains a Tokenizer model.")
    parser.add_argument("--model_type", type=str, default="bpe", help="Model type.")
    parser.add_argument(
        "--vocab_size", type=int, default=16384, help="Vocabulary size."
    )
    parser.add_argument(
        "--unk_piece", type=str, default="<|unknown|>", help="Unknown token."
    )
    parser.add_argument(
        "--bos_piece",
        type=str,
        default="<|begin_of_text|>",
        help="Beginning of sentence token.",
    )
    parser.add_argument(
        "--eos_piece",
        type=str,
        default="<|end_of_text|>",
        help="End of sentence token.",
    )
    parser.add_argument(
        "--pad_piece",
        type=str,
        default="<|end_of_text|>",
        help="Padding token",
    )
    parser.add_argument(
        "--control_symbols",
        type=str,
        default="<|start_of_turn|>,<|end_of_turn|>",
        help="Control tokens.",
    )
    parser.add_argument(
        "--user_defined_symbols",
        type=str,
        default='▁,\t,\n,0,1,2,3,4,5,6,7,8,9,`,!,@,#,$,&,(,),_,+,-,*,/,%,^,\\,~,=,{,},[,],|,.,",",:,;,<,>,?,\',"""',
        help="User defined symbols.",
    )
    parser.add_argument("--lower", type=bool, default=True, help="Lowercase input.")
    parser.add_argument("input", type=str)

    return parser.parse_args()


def clean(input: str, lower: bool = True) -> str:
    """
    Cleans a given input string by converting it to lowercase,
    replacing sequences of whitespace characters with a single space, and removing leading and trailing spaces.

    Args:
        input (str): The input string to clean.
        lower (bool): Whether to convert the input string to lowercase.

    Returns:
        str: The cleaned input string.
    """
    input = input.lower() if lower else input

    return re.sub(r"\s+", " ", input).strip()


def main(args):
    """
    The main entry point for training a SentencePiece tokenizer.

    Args:
        args: An object containing command-line arguments.

    The function initializes an empty dictionary, `freq`, to store the frequency of each token in the input file. It opens the input
    file and reads it line by line. For each line, it splits it into tokens, increments the frequency of each token in `freq`, and
    writes the token and its frequency to a frequency file. It calls `SentencePieceTrainer.train` with the frequency file and the
    specified configuration to train a SentencePiece model.
    """
    freq = {}
    prefix = os.path.splitext(args.input)[0]
    filename = prefix.split("/")[-1]
    freq_file = f"out/{filename}.freq.tsv"
    with open(args.input, "r") as f:
        for line in f:
            line = clean(line, args.lower)
            for piece in line.split(" "):
                freq.setdefault(piece, 0)
                freq[piece] += 1

    with open(freq_file, "w") as f:
        for k, v in sorted(freq.items(), key=lambda x: x[1], reverse=True):
            f.write("%s\t%d\n" % (k, v))

    spm.SentencePieceTrainer.train(
        f"--input={args.input} --model_prefix=out/{filename} --model_type={args.model_type} --vocab_size={args.vocab_size} --unk_piece={args.unk_piece} --bos_piece={args.bos_piece} --eos_piece={args.eos_piece} --pad_piece={args.pad_piece} --control_symbols={args.control_symbols} --user_defined_symbols={args.user_defined_symbols} --add_dummy_prefix=false --remove_extra_whitespaces=true"
    )


if __name__ == "__main__":
    args = get_args()
    assert args.input != "", "Please provide the input file (corpus)."

    main(args)
