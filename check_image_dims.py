import sys
sys.path.insert(0, '.')
import yaml
from src.hdf5_dataset import HDF5CloudDataset

# Load config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load dataset with one sample
dataset = HDF5CloudDataset(
    flight_configs=[config["flights"][0]],  # Just first flight
    indices=[0],  # Just first sample
    augment=False,
    swath_slice=[40, 480],
)

img, sza, saa, y, gidx, lidx = dataset[0]
print(f"Image shape: {img.shape}")
print(f"Image tensor type: {type(img)}")
print(f"SZA shape: {sza.shape}")
print(f"SAA shape: {saa.shape}")
print(f"Y shape: {y.shape}")
