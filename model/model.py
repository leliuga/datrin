from typing import Dict, Union
from safetensors.flax import save_file, load_file
from .config import Config, load
from flax.serialization import to_state_dict, from_state_dict
from flax.traverse_util import flatten_dict, unflatten_dict

import jax


def save_model(config: Config, params: Dict[str, jax.Array], prefix: str):
    """
    Saves a model's configuration and parameters to files.

    Args:
        config: The configuration object of the model.
        params: The parameters of the model.
        prefix: The prefix for the filenames to which the configuration and parameters will be saved.

    The function calls the `save` method of the `Config` object with the provided prefix, and then calls the `save_file`
    function with the flattened parameters and a filename created by appending ".safetensors" to the prefix.
    """
    config.save(prefix)
    save_file(flatten(params["params"]), prefix + ".safetensors")


def load_model(prefix: str):
    """
    Loads a model's configuration and parameters from files.

    Args:
        prefix: The prefix for the filenames from which the configuration and parameters will be loaded.

    The function calls the `load` function with the provided prefix to load the configuration, and then calls the `load_file`
    function with a filename created by appending ".safetensors" to the prefix to load the parameters. The parameters are then
    unflattened before being returned along with the configuration.
    """
    return load(prefix), unflatten(load_file(prefix + ".safetensors"))


def flatten(
    params: jax.Array, key_prefix: Union[str, None] = None
) -> Dict[str, jax.Array]:
    """
    Flattens a dictionary of parameters.

    Args:
        params: The dictionary of parameters to flatten.
        key_prefix: An optional prefix to prepend to the keys in the flattened dictionary.

    The function iterates over the items in the dictionary. If the value of an item is a `jax.Array`, it is added to the
    flattened dictionary with its key. If the value is another dictionary, the function is called recursively with the value
    as the new parameters and the current key as the key prefix. The keys in the flattened dictionary are all converted to
    lowercase and, if a key prefix is provided, it is prepended to the key with a dot separator.
    """
    flattened = {}
    for key, value in params.items():
        key = (f"{key_prefix}.{key}" if key_prefix else key).lower()

        if isinstance(value, (jax.Array)):
            flattened[key] = value
            continue
        if isinstance(value, (Dict)):
            flattened.update(flatten(params=value, key_prefix=key))

    return flattened


def unflatten(params: Dict[str, jax.Array]) -> jax.Array:
    """
    Unflattens a dictionary of parameters.

    Args:
        params: The dictionary of parameters to unflatten.

    The function iterates over the items in the dictionary. For each item, it splits the key on the dot separator and uses the
    parts to navigate through the unflattened dictionary. If a part does not exist in the current dictionary, it is added with
    an empty dictionary as its value. The last part of the key is used to add the value to the current dictionary.
    """
    unflattened = {}
    for key, value in params.items():
        keys = key.split(".")
        current = unflattened
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        current[keys[-1]] = value

    return {"params": unflattened}
