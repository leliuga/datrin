import sentencepiece as spm
import re
from typing import (
    List,
    Dict,
)


class Tokenizer:
    """
    A class used to tokenize and detokenize input using a SentencePiece model.

    The class initializes an instance of the `SentencePieceProcessor` class and loads a SentencePiece model from a file with a given
    prefix. It stores the vocab size, unk ID, bos ID, eos ID, and pad ID from the SentencePiece model as attributes. It initializes a
    dictionary, `special_tokens`, to store the special tokens and their IDs, and a set, `stop_tokens`, to store the IDs of the stop
    tokens. It provides methods to tokenize and detokenize input, check if a given token ID corresponds to a special token or a stop
    token, and print out information about the tokenizer configuration and the tokenized version of a given input string.
    """

    special_tokens: Dict[str, int]

    def __init__(self, model_prefix: str, lower: bool=True):
        """
        Initializes an instance of the Tokenizer class.

        Args:
            model_prefix (str): The prefix of the SentencePiece model file.
            lower (bool): Whether to lowercase the input before tokenization.

        The method initializes an empty dictionary, `special_tokens`, to store the special tokens and their IDs. It creates a
        `SentencePieceProcessor` object and loads the SentencePiece model from the file with the given `model_prefix`. It gets the
        vocab size, unk ID, bos ID, eos ID, and pad ID from the SentencePiece model and stores them as attributes of the `Tokenizer`
        object. It initializes a set, `stop_tokens`, to store the IDs of the stop tokens. The eos ID is added to this set. If the unk
        ID is not -1, it adds the unk token and its ID to the `special_tokens` dictionary. It iterates over all the IDs in the vocab.
        If an ID corresponds to a control token, it adds the token and its ID to the `special_tokens` dictionary.
        """
        self.file = model_prefix + ".model"
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.file)
        self.special_tokens = {}
        self.lower = lower
        self.vocab_size: int = self.sp.vocab_size()
        self.unk_id: int = self.sp.unk_id()
        self.bos_id: int = self.sp.bos_id()
        self.eos_id: int = self.sp.eos_id()
        self.pad_id: int = self.sp.pad_id()
        self.stop_tokens = {
            self.eos_id,
        }

        if self.unk_id != -1:
            self.special_tokens[self.sp.IdToPiece(self.unk_id)] = self.unk_id

        for id in range(self.vocab_size):
            if self.sp.is_control(id):
                self.special_tokens[self.sp.IdToPiece(id)] = id

    def describe(self, newLine: bool = False) -> None:
        """
        Prints out information about the tokenizer configuration.

    
        The method prints out the `vocab_size` attribute and the special tokens and their IDs. For each special token, it also prints
        out whether the token is the unk token, the bos token, the eos token, or the pad token.

        Args:
            newline: boolean. Whether to print a newline at the end.
        """
        if newLine:
            print()
        print("tokenizer configuration")
        print(f"  model= {self.file}")
        print(f"  vocab size= {self.vocab_size}")
        print(f"  lower= {self.lower}")
        print(f"  control tokens")
        for piece, id in self.special_tokens.items():
            print(f"    {id:<2} {piece:<17}", end=" ")

            if self.unk_id == id:
                print(f" unk", end=" ")
            if self.bos_id == id:
                print(f" bos", end=" ")
            if self.eos_id == id:
                print(f" eos", end=" ")
            if self.pad_id == id:
                print(f" pad", end=" ")
            print()

    def inspect(self, input: str, add_bos=False) -> None:
        """
        Prints out the tokenized version of a given input string.

        Args:
            input (str): The string to tokenize.
            add_bos (bool): Whether to add a beginning-of-sentence token.
            add_eos (bool): Whether to add an end-of-sentence token.

        The method calls the `encode` and `encode_as_pieces` methods on the `Tokenizer` object with the `input`, `add_bos`, and `add_eos`
        as arguments. It prints out the original input string, the number of tokens in the tokenized input, and the IDs and tokens in
        the tokenized input.
        """
        encoded_ids = self.encode(input, add_bos=add_bos)
        encoded_pieces = self.encode_as_pieces(input, add_bos=add_bos)
        print("\nprompt=", input)
        print("  {: <8}{} count={}".format("id", "piece", len(encoded_ids)))
        for id, piece in zip(encoded_ids, encoded_pieces):
            print("  {: <8}{}".format(id, piece))

    def encode(self, input: str, add_bos=False) -> List[int]:
        """
        Converts an input string into a list of token IDs.

        Args:
            input (str): The string to tokenize.
            add_bos (bool): Whether to add a beginning-of-sentence token.
            add_eos (bool): Whether to add an end-of-sentence token.

        Returns:
            List[int]: The list of IDs representing the tokenized input.

        The method calls the `EncodeAsIds` method on the `SentencePieceProcessor` object with the `input`, `add_bos`, and `add_eos`
        as arguments. It returns the list of IDs.
        """
        return self.sp.EncodeAsIds(self.clean(input), add_bos=add_bos)

    def encode_as_pieces(self, input: str, add_bos=False) -> List[str]:
        """
        Converts an input string into a list of token strings.

        Args:
            input (str): The string to tokenize.
            add_bos (bool): Whether to add a beginning-of-sentence token.
            add_eos (bool): Whether to add an end-of-sentence token.

        Returns:
            List[str]: The list of strings representing the tokenized input.

        The method calls the `EncodeAsPieces` method on the `SentencePieceProcessor` object with the `input`, `add_bos`, and `add_eos`
        as arguments. It returns the list of strings.
        """
        return self.sp.EncodeAsPieces(self.clean(input), add_bos=add_bos)

    def encode_with_padding(self, inputs: List[str], add_bos=False) -> List[List[int]]:
        """
        Encodes a list of strings into sequences of integers, with optional beginning of sentence token.
        Pads all sequences to the length of the longest sequence with the unknown token id.

        Args:
            inputs (List[str]): List of strings to encode.
            add_bos (bool, optional): If True, adds a beginning of sentence token to each sequence. Defaults to False.

        Returns:
            List[List[int]]: List of encoded sequences, all of the same length.
        """
        sequences = []
        max_length = 0

        for input in inputs:
            encoded = self.encode(input, add_bos=add_bos)
            sequences.append(encoded)
            max_length = max(max_length, len(encoded))

        return [seq + [self.pad_id] * (max_length - len(seq)) for seq in sequences]

    def decode(self, inputs: List[int], add_specials=True) -> str:
        """
        Converts a list of token IDs back into a string.

        Args:
            inputs (List[int]): The list of token IDs to decode.
            add_specials (bool): Whether to include special tokens in the output.

        Returns:
            str: The output string.

        If `add_specials` is `True`, the method calls the `IdToPiece` method on the `SentencePieceProcessor` object for each ID in
        the `input` and joins the resulting strings together. If `add_specials` is `False`, it calls the `DecodeIds` method on the
        `SentencePieceProcessor` object with the `input` as the argument. It replaces all instances of "▁" in the output string with a
        space and removes leading and trailing whitespace. It returns the output string.
        """
        out = (
            "".join([self.sp.IdToPiece(i) for i in inputs])
            if add_specials
            else self.sp.DecodeIds(inputs)
        )

        return out.replace("▁", " ").strip()

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
    
    def is_special_token(self, token_id: int) -> bool:
        """
        Checks if a given token ID corresponds to a special token.

        Args:
            token_id (int): The token ID to check.

        Returns:
            bool: `True` if `token_id` is in the values of the `special_tokens` dictionary and `False` otherwise.

        The method checks if `token_id` is in the values of the `special_tokens` dictionary, which stores the special tokens and their
        IDs. It returns `True` if `token_id` is in the values of the `special_tokens` dictionary and `False` otherwise.
        """
        return token_id in self.special_tokens.values()

    def is_stop_token(self, token_id: int) -> bool:
        """
        Checks if a given token ID corresponds to a stop token.

        Args:
            token_id (int): The token ID to check.

        Returns:
            bool: `True` if `token_id` is in the `stop_tokens` set and `False` otherwise.

        The method checks if `token_id` is in the `stop_tokens` set, which stores the IDs of the stop tokens. It returns `True` if
        `token_id` is in the `stop_tokens` set and `False` otherwise.
        """
        return token_id in self.stop_tokens
