from tokenizer.tokenizer import Tokenizer
from model.classification import Config, Classification
from model.model import save_model
from dataset.dataset import Dataset
import sentencepiece as spm
from jax import value_and_grad, numpy as jnp
import argparse, os, jax, optax
from tqdm import tqdm
from typing import Dict


def get_args():
    parser = argparse.ArgumentParser(description="Trains a classification model.")
    parser.add_argument(
        "--vocab_size", type=int, default=32768, help="Vocabulary size."
    )
    parser.add_argument("--embed_size", type=int, default=128, help="Embedding size.")
    parser.add_argument("--hidden_size", type=int, default=32, help="Hidden size.")
    parser.add_argument(
        "--hidden_kernel_size", type=int, default=7, help="Hidden kernel size."
    )
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate."
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
        "--user_defined_symbols",
        type=str,
        default='▁,\t,\n,0,1,2,3,4,5,6,7,8,9,`,!,@,#,$,&,(,),_,+,-,*,/,%,^,\\,~,=,{,},[,],|,.,",",:,;,<,>,?,\',"""',
        help="User defined symbols.",
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.01, help="Test set ratio."
    )
    parser.add_argument("--lower", type=bool, default=True, help="Lowercase input.")
    parser.add_argument("--classes", type=str, default="", help="Class names.")
    parser.add_argument("input", type=str)

    return parser.parse_args()


def train_in_batches(
    model: Classification,
    params: Dict[str, jax.Array],
    learning_rate: float,
    batch_size: int,
    epochs: int,
    train_input_ids: jax.Array,
    train_classes: jax.Array,
    test_input_ids: jax.Array,
    test_classes: jax.Array,
) -> Dict[str, jax.Array]:
    optimizer = optax.adam(learning_rate=learning_rate)
    optimizer_state = optimizer.init(params)

    for epoch in range(1, epochs):
        batches = jnp.arange((len(train_input_ids) // batch_size) + 1)
        for batch in tqdm(batches):
            start = int(batch * batch_size)
            end = int(start + batch_size) if batch != batches[-1] else None

            loss, gradients = value_and_grad(model.loss)(
                params,
                train_input_ids[start:end],
                train_classes[start:end],
            )

            updates, optimizer_state = optimizer.update(gradients, optimizer_state)
            params = optax.apply_updates(params, updates)

        print(
            "  epoch= {} of {} loss= {:.3f} accuracy= {:.3f}%".format(
                epoch,
                args.epochs,
                loss,
                model.accuracy(params, test_input_ids, test_classes),
            )
        )

    return params


def main(args):
    """
    The main entry point for training a classification model.

    Args:
        args: An object containing command-line arguments.

    The function creates a `Dataset` object from the input file and splits it into training and testing sets. It creates a
    `Tokenizer` object to tokenize the text data. It tokenizes the training and testing data. It creates a `Classification` model
    with the specified configuration. It initializes the model parameters and describes the model. It sets up an Adam optimizer. It
    trains the model for a specified number of epochs, updating the model parameters with the gradients of the loss with respect to
    the parameters and printing the average loss after each epoch.
    """
    dataset = Dataset(args.input, test_ratio=args.test_ratio, lower=args.lower)
    dataset.describe()

    prefix = os.path.splitext(args.input)[0]
    filename = prefix.split("/")[-1]

    spm.SentencePieceTrainer.train(
        f"--input=out/{filename}.freq.tsv --input_format=tsv --model_prefix=out/{filename} --model_type=bpe --vocab_size={args.vocab_size} --unk_piece={args.unk_piece} --bos_piece={args.bos_piece} --eos_piece={args.eos_piece} --pad_piece={args.pad_piece} --user_defined_symbols={args.user_defined_symbols} --add_dummy_prefix=false --remove_extra_whitespaces=true"
    )

    t = Tokenizer(f"out/{filename}", lower=args.lower)
    t.describe(True)

    train_input_ids, train_classes = dataset.train_samples(t)
    test_input_ids, test_classes = dataset.test_samples(t)

    model = Classification(
        Config(
            vocab_size=t.vocab_size,
            embed_size=args.embed_size,
            hidden_size=args.hidden_size,
            hidden_kernel_size=args.hidden_kernel_size,
            classes=args.classes.split(","),
        )
    )

    seed = jax.random.PRNGKey(0)
    params = model.init(seed, jnp.zeros((1, 255), dtype=jnp.int32))
    model.describe(True)

    print("")
    print("train configuration")
    print(f"  batch_size= {args.batch_size}")
    print(f"  epochs= {args.epochs}")
    print(f"  learning_rate= {args.learning_rate}")
    print(f"  optimizer= {'adam'}")
    print(f"  train_input_ids_shape= {train_input_ids.shape}")
    print(f"  test_input_ids_shape= {test_input_ids.shape}")
    print("")

    final_params = train_in_batches(
        model,
        params,
        args.learning_rate,
        args.batch_size,
        args.epochs,
        train_input_ids,
        train_classes,
        test_input_ids,
        test_classes,
    )

    save_model(model.config, final_params, "out/" + filename)


if __name__ == "__main__":
    args = get_args()
    assert args.input != "", "Please provide the input file (corpus)."

    main(args)
