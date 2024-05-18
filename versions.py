import sys, jax, flax, optax, sentencepiece, safetensors, tqdm

print("Versions of the libraries")
print(f"  python= {sys.version}")
print(f"  jax= {jax.__version__}")
print(f"  jaxlib= {jax.lib.__version__}")
print(f"  flax= {flax.__version__}")
print(f"  optax= {optax.__version__}")
print(f"  sentencepiece= {sentencepiece.__version__}")
print(f"  safetensors= {safetensors.__version__}")
print(f"  tqdm= {tqdm.__version__}")
print(f"\n  devices= {jax.devices()}")
