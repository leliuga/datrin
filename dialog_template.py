from tokenizer.tokenizer import Tokenizer
from enum import Enum
from typing import (
    List,
    TypedDict,
)


class Role(Enum):
    System = "system"
    User = "user"
    Tool = "tool"
    Assistant = "assistant"


class Message(TypedDict):
    role: Role
    end_of_turn: bool
    content: str


Dialog = List[Message]


def create_message(role: Role, end_of_turn: bool, content: str) -> Message:
    """
    Creates a message dictionary with the provided role, end_of_turn, and content.

    Args:
        role: The role of the message sender. Must be an instance of the Role enum.
        end_of_turn: A flag indicating whether it's the end of the turn.
        content: The content of the message.

    Returns:
        A dictionary with the role's value, the end_of_turn flag, and the content.

    Raises:
        ValueError: If the role is not an instance of the Role enum.
    """
    if not isinstance(role, Role):
        raise ValueError(f"Invalid role: {role}. Expected a Role enum.")

    return {"role": role.value, "end_of_turn": end_of_turn, "content": content}


def system_message(content: str) -> Message:
    """
    Creates a system message with the provided content.

    Args:
        content: The content of the message.

    Returns:
        A dictionary representing a system message with the provided content and an end_of_turn flag set to True.

    The function calls the `create_message` function with the `Role.System` enum, a `True` end_of_turn flag, and the provided content.
    """
    return create_message(Role.System, True, content)


def user_message(content: str) -> Message:
    """
    Creates a user message with the provided content.

    Args:
        content: The content of the message.

    Returns:
        A dictionary representing a user message with the provided content and an end_of_turn flag set to True.

    The function calls the `create_message` function with the `Role.User` enum, a `True` end_of_turn flag, and the provided content.
    """
    return create_message(Role.User, True, content)


def tool_message(content: str) -> Message:
    """
    Creates a tool message with the provided content.

    Args:
        content: The content of the message.

    Returns:
        A dictionary representing a tool message with the provided content and an end_of_turn flag set to True.

    The function calls the `create_message` function with the `Role.Tool` enum, a `True` end_of_turn flag, and the provided content.
    """
    return create_message(Role.Tool, True, content)


def assistant_message(content: str) -> Message:
    """
    Creates an assistant message with the provided content.

    Args:
        content: The content of the message.

    Returns:
        A dictionary representing an assistant message with the provided content and an end_of_turn flag set to False.

    The function calls the `create_message` function with the `Role.Assistant` enum, a `False` end_of_turn flag, and the provided content.
    """
    return create_message(Role.Assistant, False, content)


class DialogFormat:
    """
    Class for encoding dialogues and messages into a list of tokens.

    This class uses a provided tokenizer to encode dialogues and messages. It has two methods: `encode_message` and 
    `encode_dialog`.

    The `encode_message` method takes a message as input and encodes it into a list of tokens. It appends special tokens at the 
    start and end of a turn, and encodes the role and content of the message.

    The `encode_dialog` method takes a dialog as input and encodes it into a list of tokens. It appends a special token at the 
    beginning of the sequence, encodes each message in the dialog, and finally encodes an empty assistant message.
    """
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def encode_message(self, message: Message) -> List[int]:
        """
        Encodes a message into a list of tokens.

        Args:
            message: The message to encode.

        Returns:
            A list of tokens representing the encoded message.

        The function first creates an empty list of tokens. Then, it appends the special token for the start of a turn. It encodes
        the role and content of the message and appends them to the list of tokens. If the message indicates the end of a turn, it
        appends the special token for the end of a turn and encodes a newline character.
        """
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_of_turn|>"])
        tokens.extend(self.tokenizer.encode(message["role"]))
        tokens.extend(self.tokenizer.encode("\n" + message["content"].strip()))
        if message["end_of_turn"]:
            tokens.append(self.tokenizer.special_tokens["<|end_of_turn|>"])
            tokens.extend(self.tokenizer.encode("\n"))

        return tokens

    def encode_dialog(self, dialog: Dialog) -> List[int]:
        """
        Encodes a dialog into a list of tokens.

        Args:
            dialog: The dialog to encode.

        Returns:
            A list of tokens representing the encoded dialog.

        The function first creates an empty list of tokens and appends the special token for the beginning of a sequence. Then, it
        iterates over the messages in the dialog, encoding each one and appending the result to the list of tokens. Finally, it
        encodes an empty assistant message and appends the result to the list of tokens.
        """
        tokens = []
        tokens.append(self.tokenizer.bos_id)
        for message in dialog:
            tokens.extend(self.encode_message(message))

        tokens.extend(self.encode_message(assistant_message("")))

        return tokens
