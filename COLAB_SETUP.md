# Google Colab Setup Guide

This guide provides step-by-step instructions for running the Cloud-ML project on Google Colab.

## Prerequisites

1. A Google account with access to Google Drive
2. Your NASA flight data (see main README for data sources)
3. (Optional) A GitHub Personal Access Token if the repository is private

## Initial Setup

### Step 1: Prepare Your Data

1. **Create the data directory in Google Drive:**
   - Open Google Drive in your browser
   - Navigate to `My Drive`
   - Create folder: `CloudML/data/`

2. **Upload your flight data:**
   - Each flight should have its own folder (e.g., `10Feb25`, `30Oct24`, etc.)
   - Each flight folder should contain three files:
     - `GLOVE2025_IRAI_L1B_Rev-_YYYYMMDD.h5` (or similar IRAI camera file)
     - `CPL_L2_V1-02_01kmLay_XXXXXX_DDmmmYY.hdf5` (CPL lidar file)
     - `CRS_YYYYMMDD_nav.hdf` (navigation file)

   Example structure:
   ```
   /content/drive/MyDrive/CloudML/
   ├── data/
   │   ├── 10Feb25/
   │   │   ├── GLOVE2025_IRAI_L1B_Rev-_20250210.h5
   │   │   ├── CPL_L2_V1-02_01kmLay_259015_10feb25.hdf5
   │   │   └── CRS_20250210_nav.hdf
   │   ├── 30Oct24/
   │   │   ├── WHYMSIE2024_IRAI_L1B_Rev-_20241030.h5
   │   │   ├── CPL_L2_V1-02_01kmLay_259006_30oct24.hdf5
   │   │   └── CRS_20241030_nav.hdf
   │   └── ...
   └── plots/  (will be created automatically)
   ```

### Step 2: Open the Notebook

1. **Upload to Colab:**
   - Open Google Colab: https://colab.research.google.com
   - Click `File > Upload notebook`
   - Upload `colab_training.ipynb` from this repository

   OR

2. **Open from GitHub:**
   - In Colab, click `File > Open notebook`
   - Select the `GitHub` tab
   - Enter repository URL: `rylanmalarchick/cloudMLPublic`
   - Select `colab_training.ipynb`

### Step 3: Set Runtime to GPU

1. Click `Runtime > Change runtime type`
2. Select `T4 GPU` under "Hardware accelerator"
3. Click `Save`

### Step 4: Run Setup Cells

Execute each cell in order:

1. **Mount Google Drive** - Authorize access when prompted
2. **Clone repository** - Downloads the latest code
3. **Install dependencies** - This may take 5-10 minutes
4. **Set up data paths** - Updates config to point to your Drive folder

## Running Experiments

### Single Experiment

To run a quick test experiment:

```python
!python main.py --config configs/bestComboConfig.yaml --epochs 5 --save_name test_run
```

### Ablation Studies

The notebook includes pre-configured ablation studies. You can run them all or modify the list:

```python
ablations = [
    ('--angles_mode both', 'All angles'),
    ('--angles_mode sza_only', 'Zenith only'),
    ('--no-use_spatial_attention --no-use_temporal_attention', 'No attention'),
    ('--loss_type mae', 'Plain MAE loss'),
    ('', 'Baseline'),
    ('--no-augment', 'No augmentation'),
    ('--architecture_name gnn', 'GNN architecture'),
    ('--architecture_name cnn', 'Simple CNN baseline')
]
```

Each ablation will save results to Google Drive with a unique identifier.

## Troubleshooting

### Problem: "No valid flights loaded" Error

**Symptoms:**
```
Error loading flight 10Feb25: [Errno 2] Unable to synchronously open file
ValueError: No valid flights loaded.
```

**Solutions:**

1. **Verify data paths:**
   - Check that `data_directory` in your config matches your Drive structure
   - Default is: `/content/drive/MyDrive/CloudML/data/`

2. **Check file names match config:**
   - Open `configs/bestComboConfig.yaml`
   - Verify flight names and file paths match your uploaded data exactly
   - File names are case-sensitive

3. **Verify files are accessible:**
   ```python
   import os
   data_path = '/content/drive/MyDrive/CloudML/data/10Feb25/'
   print(os.listdir(data_path))  # Should show your three files
   ```

4. **Update config if needed:**
   ```python
   import yaml
   with open('configs/bestComboConfig.yaml', 'r') as f:
       config = yaml.safe_load(f)
   
   # Check current flights
   print(config['flights'])
   
   # Update if necessary
   config['flights'][0]['iFileName'] = '10Feb25/YOUR_ACTUAL_FILENAME.h5'
   
   with open('configs/bestComboConfig.yaml', 'w') as f:
       yaml.dump(config, f)
   ```

### Problem: "unrecognized arguments" Error

**Symptoms:**
```
main.py: error: unrecognized arguments: --angles_mode both
```

**Solution:**
Make sure you've pulled the latest code from GitHub. The fixes were pushed in the most recent commit. Re-run the clone/pull cell:

```python
%cd /content/repo
!git pull origin main
```

### Problem: Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size:**
   ```bash
   !python main.py --config configs/bestComboConfig.yaml --batch_size 2
   ```

2. **Reduce temporal frames:**
   ```bash
   !python main.py --config configs/bestComboConfig.yaml --temporal_frames 1
   ```

3. **Use a simpler architecture:**
   ```bash
   !python main.py --config configs/bestComboConfig.yaml --architecture_name cnn
   ```

### Problem: Session Timeout

**Symptoms:**
Colab disconnects after ~12 hours or during inactivity.

**Solutions:**

1. **Use Colab Pro** for longer sessions (up to 24 hours)

2. **Save checkpoints frequently:**
   - Models are automatically saved to `/content/drive/MyDrive/CloudML/models/`
   - These persist across sessions

3. **Resume training:**
   - The code automatically loads the latest checkpoint if available
   - You can restart the cell and it will continue from the last saved epoch

4. **Keep session alive (browser extension):**
   - Install "Colab Auto Reconnect" or similar extensions
   - Note: This is against Colab's terms if you're using it to bypass usage limits

### Problem: Installation Failures

**Symptoms:**
```
ERROR: Failed building wheel for mamba-ssm
```

**Solutions:**

1. **Restart runtime and try again:**
   - `Runtime > Restart runtime`
   - Re-run all cells

2. **Install packages individually:**
   ```python
   !pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
   !pip install h5py==3.14.0 netCDF4==1.7.2
   !pip install causal-conv1d==1.4.0
   !pip install mamba-ssm==2.2.2
   ```

3. **Skip optional dependencies:**
   - If mamba-ssm fails, you can still use other architectures
   - Use `--architecture_name cnn` or `--architecture_name transformer`

## Performance Tips

### Optimize for Colab's Free Tier

1. **Reduce epochs for testing:**
   ```bash
   !python main.py --config configs/bestComboConfig.yaml --epochs 5
   ```

2. **Disable pretraining for quick experiments:**
   ```bash
   !python main.py --config configs/bestComboConfig.yaml --no_pretrain --epochs 5
   ```

3. **Skip plots to save time:**
   ```bash
   !python main.py --config configs/bestComboConfig.yaml --no_plots
   ```

### Monitor GPU Usage

Check GPU utilization during training:

```python
!nvidia-smi
```

### Download Results

After training, download your results:

```python
from google.colab import files

# Zip results
!zip -r results.zip /content/drive/MyDrive/CloudML/plots/
!zip -r models.zip /content/drive/MyDrive/CloudML/models/

# Download (optional - files are already in Drive)
files.download('results.zip')
files.download('models.zip')
```

## Configuration Tips

### Minimal Config for Testing

For quick tests, create a minimal config:

```yaml
data_directory: "/content/drive/MyDrive/CloudML/data/"
output_directory: "/content/drive/MyDrive/CloudML/plots/"
pretrain_flight: "10Feb25"
validation_flight: "12Feb25"
learning_rate: 0.001
weight_decay: 0.04
epochs: 3
batch_size: 4
temporal_frames: 1
loss_type: "mae"
loo: false
flights:
  - name: "10Feb25"
    iFileName: "10Feb25/GLOVE2025_IRAI_L1B_Rev-_20250210.h5"
    cFileName: "10Feb25/CPL_L2_V1-02_01kmLay_259015_10feb25.hdf5"
    nFileName: "10Feb25/CRS_20250210_nav.hdf"
```

### Using Only One Flight

To test with minimal data:

1. Edit config to include only one flight
2. Set both pretrain and validation to the same flight
3. Code will auto-split into train/val

```yaml
pretrain_flight: "10Feb25"
validation_flight: "10Feb25"
flights:
  - name: "10Feb25"
    # ... file paths
```

## Additional Resources

- **Main README:** See `README.md` for detailed architecture information
- **Data Sources:** 
  - HAR: https://har.gsfc.nasa.gov/index.php?section=77
  - CPL: https://cpl.gsfc.nasa.gov/
- **Issues:** Report bugs at https://github.com/rylanmalarchick/cloudMLPublic/issues
- **Contact:** rylan1012@gmail.com

## Colab Alternatives

If you need longer training sessions or more compute:

1. **Kaggle Notebooks** - Free GPU with 30h/week limit
2. **Paperspace Gradient** - Free tier available
3. **AWS SageMaker** - Pay-per-use
4. **Google Cloud Vertex AI** - Pay-per-use with free credits

## Best Practices

1. **Always save to Drive** - Don't save to `/content/` as it's ephemeral
2. **Test with small epochs first** - Verify everything works before long runs
3. **Monitor memory usage** - Use `!nvidia-smi` to check GPU utilization
4. **Use meaningful save_names** - Makes tracking experiments easier
5. **Keep a log** - Document what experiments you run and their results
6. **Backup your configs** - Save modified configs to Drive

## Example Complete Workflow

```python
# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone repo
%cd /content
!git clone https://github.com/rylanmalarchick/cloudMLPublic.git repo
%cd repo

# 3. Install dependencies
!pip install -q torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
!pip install -q h5py netCDF4 mamba-ssm causal-conv1d

# 4. Update config
import yaml
with open('configs/bestComboConfig.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['data_directory'] = '/content/drive/MyDrive/CloudML/data/'
config['output_directory'] = '/content/drive/MyDrive/CloudML/plots/'
config['epochs'] = 10  # Short run for testing
with open('configs/bestComboConfig.yaml', 'w') as f:
    yaml.dump(config, f)

# 5. Run experiment
!python main.py --config configs/bestComboConfig.yaml --save_name colab_test

# 6. Check results
!ls /content/drive/MyDrive/CloudML/plots/
!ls /content/drive/MyDrive/CloudML/models/
```

Happy training! 🚀