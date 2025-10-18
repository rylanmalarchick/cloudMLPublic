# Changelog

All notable changes to the Cloud-ML project will be documented in this file.

## [Unreleased]

### Fixed (2024-10-18)

#### Google Colab Compatibility
- **Fixed image_shape assertion error**: Corrected assertion in `main.py` to properly validate temporal frame dimensions `(temporal_frames, height, width)` instead of incorrectly expecting single channel `(1, height, width)`
- **Added missing command-line arguments**: 
  - `--angles_mode` to control which solar angles are used (both, sza_only, saa_only, none)
  - `--augment` / `--no-augment` to enable/disable data augmentation
  - `--architecture_name` to specify model architecture (transformer, gnn, ssm, cnn)
- **Fixed ablation study commands**: Updated `colab_training.ipynb` to use correct boolean flag syntax (e.g., `--no-augment` instead of `--augment false`)
- **Fixed architecture argument**: Changed from `--architecture.name` to `--architecture_name` for command-line compatibility

#### Configuration Management
- **Standardized config parameter names**: Renamed `lr` → `learning_rate` and `wd` → `weight_decay` throughout all configuration files for consistency
- **Added backward compatibility**: `main.py` now automatically converts old config keys (`lr`, `wd`) to new format (`learning_rate`, `weight_decay`)
- **Updated all YAML configs**: Applied naming changes to `config.yaml`, `bestComboConfig.yaml`, and all ablation config files

#### Pipeline Improvements
- **Enhanced parameter passing**: Updated `run_pretraining()`, `run_final_training_and_evaluation()`, and `run_loo_evaluation()` to properly pass `augment` and `angles_mode` parameters from config to datasets
- **Fixed sample data preparation**: Added missing `augment` and `angles_mode` parameters to sample data creation in `main.py`

### Added (2024-10-18)

#### Documentation
- **COLAB_SETUP.md**: Comprehensive Google Colab setup and troubleshooting guide including:
  - Step-by-step setup instructions
  - Data upload guidelines
  - Common error solutions
  - Performance optimization tips
  - Complete workflow examples
  
- **Enhanced README.md**: Added sections for:
  - Complete command-line argument reference table
  - Google Colab running instructions
  - Ablation study examples
  - Configuration override examples

#### Colab Notebook
- **Updated colab_training.ipynb**: 
  - Fixed all ablation study commands with correct syntax
  - Simplified config path handling
  - Added proper Google Drive path configuration
  - Improved error handling for package installation

### Changed (2024-10-18)

#### Code Quality
- **Improved argument handling**: Main.py now uses cleaner logic for nested config updates (e.g., architecture name mapping)
- **Better error messages**: Assertions now provide more descriptive error messages with actual vs expected values

## Previous Versions

### Features (Pre-changelog)
- Multimodal regression model with CNN backbone
- Spatial and temporal attention mechanisms
- FiLM layers for metadata integration
- Multiple architecture support (Transformer, GNN, SSM, CNN)
- Leave-one-out cross-validation
- Comprehensive evaluation metrics
- TensorBoard logging
- Conformal prediction for uncertainty quantification
- Data augmentation pipeline
- HDF5 lazy loading for efficient memory usage
- Pre-training and fine-tuning pipeline
- Model ensemble support
- Scene complexity weighting
- Hard example mining

## Migration Guide

### Updating from pre-2024-10-18 version

If you have existing config files or scripts, you need to make the following changes:

1. **Update config files**: Replace `lr:` with `learning_rate:` and `wd:` with `weight_decay:`
   ```bash
   # Automatic update using sed (Linux/Mac)
   sed -i 's/^lr:/learning_rate:/g' configs/*.yaml
   sed -i 's/^wd:/weight_decay:/g' configs/*.yaml
   ```

2. **Update command-line arguments**: 
   - Old: `python main.py --lr 0.001 --wd 0.04`
   - New: `python main.py --learning_rate 0.001 --weight_decay 0.04`

3. **Update boolean flags**:
   - Old: `--use_spatial_attention false`
   - New: `--no-use_spatial_attention`

4. **Update architecture specification**:
   - Old: Not available via CLI
   - New: `--architecture_name gnn`

5. **Update angles mode**:
   - Old: Not available via CLI
   - New: `--angles_mode sza_only`

Note: The code maintains backward compatibility for config files - old `lr` and `wd` keys will be automatically converted.

## Known Issues

### Warnings
- **mamba-ssm FutureWarnings**: The mamba-ssm package generates deprecation warnings about `torch.cuda.amp.custom_fwd/bwd`. These are harmless and will be fixed in future mamba-ssm releases.
- **TensorFlow/XLA registration warnings**: Multiple library registration warnings from TensorFlow/XLA. These do not affect functionality.

### Limitations
- **Google Colab free tier**: Training may be limited by ~12 hour session timeouts and GPU availability
- **Memory constraints**: Large batch sizes or many temporal frames may cause OOM errors on T4 GPUs
- **Data requirements**: All three files (IRAI .h5, CPL .hdf5, navigation .hdf) must be present for each flight

## Upcoming Features

- [ ] Improved memory efficiency for larger batch sizes
- [ ] Support for additional satellite data sources
- [ ] Real-time inference pipeline
- [ ] Model compression for edge deployment
- [ ] Enhanced visualization dashboard
- [ ] Automated hyperparameter tuning
- [ ] Multi-GPU training support

## Contributing

When contributing, please:
1. Update this CHANGELOG with your changes
2. Follow the existing format (Added/Changed/Fixed/Removed)
3. Include the date of changes
4. Reference issue numbers where applicable

## Support

For issues or questions:
- GitHub Issues: https://github.com/rylanmalarchick/cloudMLPublic/issues
- Email: rylan1012@gmail.com
- Documentation: See README.md and COLAB_SETUP.md