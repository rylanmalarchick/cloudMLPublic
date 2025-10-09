import os
import pickle


def save_cache(data, cache_path):
    """Saves data to a cache file."""
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)


def load_cache(cache_path):
    """Loads data from a cache file if it exists."""
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return None
