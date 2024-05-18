from flax import linen as nn
import numpy as np
import jax.numpy as jnp
import jax, optax
from .config import Config


class Classification(nn.Module):
    """
    Defines a text classification model.

    The class is responsible for initializing the layers of the model, defining the forward pass, calculating the loss,
    calculating the accuracy, and printing out information about the model configuration.
    """

    config: Config

    def describe(self, newLine: bool = False):
        """
        Prints out information about the model configuration.

        The method prints out the `vocab_size`, `embed_size`, `hidden_size`, `hidden_kernel_size`, `classes`, and `dtype` attributes
        of the `config` attribute of the `Classification` object.
        """
        if newLine:
            print()
        print("classification model configuration")
        print(f"  vocab_size= {self.config.vocab_size}")
        print(f"  embed_size= {self.config.embed_size}")
        print(f"  hidden_size= {self.config.hidden_size}")
        print(f"  hidden_kernel_size= {self.config.hidden_kernel_size}")
        print(f"  classes= {self.config.classes}")
        print(f"  dtype= {self.config.dtype.__name__}")

    def setup(self):
        """
        Initializes the layers of the text classification model.

        The method sets up an embedding layer, a 1D convolutional layer, and a dense layer. The sizes of these layers are determined
        by the `config` attribute of the `Classification` object.
        """
        self.embeddings = nn.Embed(
            self.config.vocab_size,
            self.config.embed_size,
            dtype=self.config.dtype,
            name="embeddings",
        )
        self.hidden = nn.Conv(
            self.config.hidden_size,
            kernel_size=(self.config.hidden_kernel_size,),
            dtype=self.config.dtype,
            name="hidden",
        )
        self.classes = nn.Dense(
            self.config.num_classes, dtype=self.config.dtype, name="classes"
        )

    def __call__(self, input_ids: jax.Array):
        """
        Defines the forward pass of the text classification model.

        Args:
            input_ids (jax.Array): A batch of input IDs.

        Returns:
            The final outputs of the model.
        """
        x = self.embeddings(input_ids)
        x = nn.relu(self.hidden(x))
        x = x.max(axis=1)

        return self.classes(x)

    def logits(self, params: jax.Array, input_ids: jax.Array) -> jax.Array:
        """
        Returns the raw, unnormalized predictions (logits) of the text classification model on a batch of input data.

        Args:
            params: The parameters of the model.
            input_ids (jax.Array): A batch of input IDs.

        Returns:
            jax.Array: The logits of the model.
        """
        return self.apply(params, input_ids)

    def loss(self, params: jax.Array, input_ids: jax.Array, actual: jax.Array) -> float:
        """
        Calculates the softmax cross-entropy loss between the model's predictions and the actual classes.

        Args:
            params: The parameters of the model.
            input_ids (jax.Array): A batch of input IDs.
            actual (jax.Array): The actual classes.

        Returns:
            float: The total loss.
        """
        logits = self.logits(params, input_ids)
        one_hot_actual = jax.nn.one_hot(actual, self.config.num_classes)

        return jnp.sum(optax.softmax_cross_entropy(logits, one_hot_actual)) * 100

    def accuracy(
        self, params: jax.Array, input_ids: jax.Array, actual: jax.Array
    ) -> float:
        """
        Calculates the accuracy of the model's predictions compared to the actual classes.

        Args:
            params: The parameters of the model.
            input_ids (jax.Array): A batch of input IDs.
            actual (jax.Array): The actual classes.

        Returns:
            float: The accuracy.
        """
        logits = self.logits(params, input_ids)

        return jnp.mean(jnp.argmax(logits, 1) == actual) * 100
    
    def predict(self, params: jax.Array, input_ids: jax.Array) -> jax.Array:
        """
        Predicts the classes of a batch of input data.

        Args:
            params: The parameters of the model.
            input_ids (jax.Array): A batch of input IDs.

        Returns:
            jax.Array: The predicted classes.
        """
        logits = self.logits(params, jnp.array(np.array(input_ids), dtype=jnp.int32))

        return jnp.argmax(logits, 1)
