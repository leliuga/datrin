import jax.numpy as jnp
import yaml


class Config:
    """
    Stores the configuration parameters for a text classification model.

    The class is responsible for managing the model configuration, including the vocabulary size, embedding size, hidden layer size,
    hidden kernel size, classes, and data type. It also calculates the number of classes based on the classes list.
    """

    def __init__(
        self,
        vocab_size: int = 32768,
        embed_size: int = 128,
        hidden_size: int = 32,
        hidden_kernel_size: int = 7,
        classes: str = [],
        dtype: jnp.dtype = jnp.float32,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.hidden_kernel_size = hidden_kernel_size
        self.classes = classes
        self.num_classes = len(classes)
        self.dtype = dtype

        for key, value in kwargs.items():
            setattr(self, key, value)

    def save(self, prefix: str):
        """
        Saves the current configuration object to a YAML file.

        Args:
            prefix: The prefix for the filename to which the configuration will be saved.

        The function iterates over the instance variables of the configuration object and writes them to a file.
        The filename is created by appending ".yaml" to the provided prefix.
        """
        with open(prefix + ".yaml", "w") as f:
            yaml.dump(
                {k: v for k, v in vars(self).items()},
                f,
            )


def load(prefix: str) -> Config:
    """
    Loads a configuration object from a YAML file.

    Args:
        prefix: The prefix for the filename from which the configuration will be loaded.

    The function opens the file created by appending ".yaml" to the provided prefix, loads the YAML content into a dictionary,
    and then uses that dictionary to create a new `Config` object which is then returned.
    """
    with open(prefix + ".yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return Config(**config)
