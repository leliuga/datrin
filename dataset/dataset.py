from tokenizer.tokenizer import Tokenizer
import numpy as np
import jax.numpy as jnp
from typing import List
import jax, csv, re, os

class Dataset(object):
    """
    Handles a dataset for a text classification task.

    The class is responsible for reading the dataset file, cleaning and tokenizing the text, counting the frequency of each token,
    splitting the dataset into training and test sets, and providing methods for accessing the samples.
    """

    def __init__(
        self,
        file: str,
        test_ratio: float,
        lower: bool = True,
        delimiter: str = ",",
        quotechar: str | None = '"',
        escapechar: str | None = None,
        doublequote: bool = True,
        skipinitialspace: bool = False,
        lineterminator: str = "\r\n",
        quoting: int = 0,
    ):
        texts: List[str] = []
        classes: List[int] = []
        frequencies = {}

        self.file = file
        self.lower = lower

        with open(file, "r") as f:
            reader = csv.reader(
                f,
                delimiter=delimiter,
                quotechar=quotechar,
                escapechar=escapechar,
                doublequote=doublequote,
                skipinitialspace=skipinitialspace,
                lineterminator=lineterminator,
                quoting=quoting,
            )
            for row in reader:
                (
                    text,
                    class_id,
                ) = row

                text = self.clean(text)
                texts.append(text)
                classes.append(int(class_id))

                for token in self.text_to_pieces(text):
                    frequencies[token] = frequencies.get(token, 0) + 1

        self.texts = texts
        self.classes = classes
        self.num_classes = len(set(classes))
        self.test_ratio = test_ratio
        self.split_idx = int(len(texts) * (1 - test_ratio))

        prefix = os.path.splitext(file)[0]
        filename = prefix.split("/")[-1]
        freq_file = f"out/{filename}.freq.tsv"
        with open(freq_file, "w") as f:
            for k, v in sorted(frequencies.items(), key=lambda x: x[1], reverse=True):
                f.write("%s\t%d\n" % (k, v))

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[int, int]:
        """
        Returns the text and class of a sample at a given index.

        Args:
            idx (int): The index of the sample to get.

        Returns:
            tuple[int, int]: A tuple containing the text and class of the sample at the given index.
        """
        return self.texts[idx], self.classes[idx]

    def describe(self) -> None:
        """
        Prints out information about the dataset configuration.

        The method prints out the `file` attribute, the size of the dataset, the `train_sample_size`, the `test_sample_size`, and the
        `num_classes` attribute.
        """
        print("dataset configuration")
        print(f"  file= {self.file}")
        print(f"  lower= {self.lower}")
        print(f"  num_classes= {self.num_classes}")
        print(f"  size= {len(self)}")
        print(f"  ratio= {self.test_ratio}")
        print(f"  train_sample_size= {self.split_idx}")
        print(f"  test_sample_size= {len(self) - self.split_idx}")

    def clean(self, input: str) -> str:
        """
        Cleans a given input string by converting it to lowercase,
        replacing sequences of whitespace characters with a single space, and removing leading and trailing spaces.

        Args:
            input (str): The input string to clean.

        Returns:
            str: The cleaned input string.
        """
        input = input.lower() if self.lower else input

        return re.sub(r"\s+", " ", input).strip()

    def text_to_pieces(self, input: str) -> List[str]:
        """
        Splits a given input string into a list of tokens. If a token is a digit, it is split into individual characters.

        Args:
            input (str): The input string to be split into tokens.

        Returns:
            List[str]: The list of tokens.
        """

        pieces = []
        for piece in input.split(" "):
            if piece == "":
                continue
            if piece.isdigit():
                for t in list(piece):
                    pieces.append(t)
            else:
                pieces.append(piece)

        return pieces

    def samples(
        self, tokenizer: Tokenizer, start: int, end: int
    ) -> tuple[jax.Array, jax.Array]:
        """
        Processes a range of text samples for a classification task.

        Args:
            tokenizer (Tokenizer): The tokenizer to use for converting text into numerical tokens.
            start (int): The start index of the range of text samples to process.
            end (int): The end index of the range of text samples to process.

        Returns:
            tuple[jax.Array, jax.Array]: A tuple of two arrays. The first array contains the padded token sequences
            (`input_ids`) and the second array contains the classes for each text sample.
        """
        inputs_ids = tokenizer.encode_with_padding(self.texts[start:end])
        classes = self.classes[start:end]

        return jnp.array(np.array(inputs_ids), dtype=jnp.int32), jnp.array(
            np.array(classes)
        )

    def train_samples(self, tokenizer: Tokenizer) -> tuple[jax.Array, jax.Array]:
        """
        Processes the training samples for a classification task.

        Args:
            tokenizer (Tokenizer): The tokenizer to use for converting text into numerical tokens.

        Returns:
            tuple[jax.Array, jax.Array]: A tuple of two arrays. The first array contains the padded token sequences
            (`input_ids`) and the second array contains the classes for each text sample.
        """
        return self.samples(tokenizer, 0, self.split_idx)

    def test_samples(self, tokenizer: Tokenizer) -> tuple[jax.Array, jax.Array]:
        """
        Processes the test samples for a classification task.

        Args:
            tokenizer (Tokenizer): The tokenizer to use for converting text into numerical tokens.

        Returns:
            tuple[jax.Array, jax.Array]: A tuple of two arrays. The first array contains the padded token sequences
            (`input_ids`) and the second array contains the classes for each text sample.
        """
        return self.samples(tokenizer, self.split_idx, len(self))