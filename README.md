# Cloud-ML Model Training

This project addresses the challenge of accurate cloud base height (CBH) prediction from NASA satellite data, critical for climate modeling and aviation safety. It trains a deep learning model using a multimodal approach, combining a Convolutional Neural Network (CNN) for image analysis with dense layers for processing scalar data like solar zenith and azimuth angles. The model incorporates spatial and temporal attention mechanisms to focus on relevant image regions and time steps, enabling robust performance across diverse atmospheric conditions.

## Technologies and Libraries

- **Deep Learning Framework:** PyTorch, with CUDA acceleration for GPU training.
- **Data Handling:** HDF5 for efficient storage and loading of large satellite datasets; scikit-learn for preprocessing and scaling.
- **Visualization and Logging:** Matplotlib and Plotly for plots; TensorBoard for experiment tracking.
- **Additional Tools:** NumPy and Pandas for data manipulation; YAML for configuration management.
- **Advanced Components:** Mamba-SSM for sequence modeling and PyTorch Geometric for graph-based architectures where applicable.

## Model Architecture

The model, `MultimodalRegressionModel`, is composed of the following key components:

- **CNN Backbone:** A series of convolutional layers to extract features from the input satellite images. The architecture is configurable through `config.yaml`.
- **FiLM Layers:** Feature-wise Linear Modulation (FiLM) layers are used to inject scalar metadata (like solar angles) into the CNN, allowing the model to condition its image processing on this information.
- **Spatial Attention:** A spatial attention mechanism (`SpatialAttention`) is applied to each frame to focus on the most relevant pixels.
- **Temporal Attention:** A temporal attention mechanism (`TemporalAttention`) weighs the importance of each frame in a sequence, allowing the model to focus on the most informative time steps.
- **Dense Layers:** Fully connected layers process the combined output of the CNN and attention mechanisms to produce the final CBH prediction.

## Pipeline

The training and evaluation pipeline is managed by `src/pipeline.py` and consists of the following stages:

1.  **Pre-training:** The model is first pre-trained on a single, large flight to learn a good initial representation of the data.
2.  **Final Training:** The pre-trained model is then fine-tuned on a combination of all flights, with the pre-training flight data being overweighted to retain its learned features. One flight is held out for validation.
3.  **Leave-One-Out (LOO) Cross-Validation:** For robust evaluation, the model is trained and evaluated multiple times, with each flight being held out as the validation set once.

## Features

- **Data Preprocessing:** Scripts for preparing and normalizing satellite imagery data, including flat-field correction and CLAHE enhancement.
- **Model Training:** Train a deep learning model with spatial and temporal attention mechanisms, supporting pre-training and fine-tuning.
- **Evaluation:** Evaluate model performance using comprehensive metrics (MAE, MSE, RMSE, MAPE, R², error quantiles) and leave-one-out cross-validation for robustness.
- **Visualization:** Generate plots, attention maps, and error analyses to interpret model behavior.
- **Calibration and Uncertainty:** Includes conformal prediction for uncertainty quantification and model calibration scripts.
- **Ablation Studies:** Automated command-line overrides for testing angles, attention, loss, augmentation, and architectures.
- **Result Aggregation:** Scripts to combine metrics across runs into summary CSVs for comparison.

## Challenges and Technical Insights

Developing this model involved addressing several key challenges in remote sensing and multimodal learning:

- **Multimodal Integration:** Combining high-resolution satellite imagery with scalar metadata required FiLM layers to condition the CNN without introducing artifacts, improving accuracy in varying solar conditions.
- **Temporal Variability:** Satellite sequences can include irrelevant frames; temporal attention mechanisms were implemented to weigh informative time steps, reducing noise and enhancing focus on cloud dynamics.
- **Data Imbalance and Generalization:** Flights varied in size and conditions, so pre-training on a large flight with overweighting during fine-tuning ensured retention of learned features, boosting cross-flight performance.
- **Scalability:** Handling large HDF5 datasets necessitated streaming loaders and GPU optimization, with early stopping to prevent overfitting on limited labeled data.
- **Evaluation Rigor:** LOO cross-validation provided a realistic assessment of generalization, revealing insights into model reliability across different atmospheric scenarios.

## Results and Performance

[Placeholder: Please provide specific metrics, e.g., average MAE/MSE across flights, comparisons to baselines, or key findings from your experiments.]

## Future Work

- Integrate transformer-based architectures for improved handling of long-range dependencies in image sequences.
- Expand self-supervised pre-training on unlabeled satellite data to reduce reliance on labeled CPL data.
- Add support for real-time inference and integration with additional sensors (e.g., radar).
- Explore ensemble methods and uncertainty-aware predictions for operational deployment.

## Getting Started

### Prerequisites

- Python 3.9+
- CUDA 11.8+

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Data

This project uses satellite imagery and associated metadata from NASA missions. The data is not included in this repository due to size and access restrictions. You will need to obtain the following datasets:

- **Camera (IRAI) and Navigation Data:** Available from the High Altitude Research (HAR) program at NASA. Download from [https://har.gsfc.nasa.gov/index.php?section=77](https://har.gsfc.nasa.gov/index.php?section=77) (see bottom of the page for data links).
- **Cloud Physics Lidar (CPL) Data:** For cloud base height labels. Contact the CPL team at [https://cpl.gsfc.nasa.gov/](https://cpl.gsfc.nasa.gov/) or reach out to Rylan Malarchick (rylan1012@gmail.com) for access.

Once obtained, place the data in the directory specified by `data_directory` in `config.yaml`.

### Configuration

1.  **Update `config.yaml`:**
    -   Set `data_directory` to the path of your dataset.
    -   Configure model hyperparameters, training settings, and flight data.

2.  **(Optional) Update `bestComboConfig.yaml`:**
    -   This file contains the best-performing hyperparameter combination found through ablation studies.

### Running the Model

-   **Run a single experiment:**
    ```bash
    python main.py --config configs/bestComboConfig.yaml
    ```

-   **Override configuration parameters from the command line:**
    ```bash
    python main.py --config configs/bestComboConfig.yaml --learning_rate 0.0001 --epochs 50
    ```

-   **Run with specific ablation settings:**
    ```bash
    # Test without attention mechanisms
    python main.py --config configs/bestComboConfig.yaml --no-use_spatial_attention --no-use_temporal_attention
    
    # Test with only zenith angles
    python main.py --config configs/bestComboConfig.yaml --angles_mode sza_only
    
    # Test without augmentation
    python main.py --config configs/bestComboConfig.yaml --no-augment
    
    # Test different architecture
    python main.py --config configs/bestComboConfig.yaml --architecture_name gnn
    ```

-   **Run a leave-one-out (LOO) evaluation:**
    ```bash
    python main.py --config configs/bestComboConfig.yaml --loo
    ```

#### Available Command-Line Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--config` | str | Path to configuration YAML file |
| `--learning_rate` | float | Learning rate for optimizer |
| `--weight_decay` | float | Weight decay for regularization |
| `--epochs` | int | Number of training epochs |
| `--temporal_frames` | int | Number of temporal frames to use |
| `--loss_type` | str | Loss function type (mae, huber, weighted_huber_mae) |
| `--angles_mode` | str | Which angles to use: both, sza_only, saa_only, none |
| `--architecture_name` | str | Model architecture: transformer, gnn, ssm, cnn |
| `--augment` / `--no-augment` | bool | Enable/disable data augmentation |
| `--use_spatial_attention` / `--no-use_spatial_attention` | bool | Enable/disable spatial attention |
| `--use_temporal_attention` / `--no-use_temporal_attention` | bool | Enable/disable temporal attention |
| `--flat_field_correction` / `--no-flat_field_correction` | bool | Enable/disable flat-field correction |
| `--zscore_normalize` / `--no-zscore_normalize` | bool | Enable/disable z-score normalization |
| `--loo` / `--no-loo` | bool | Enable/disable leave-one-out cross-validation |
| `--save_name` | str | Custom name for saved models and results |
| `--no_pretrain` | flag | Skip pretraining phase |
| `--no_final` | flag | Skip final training/evaluation |
| `--no_plots` | flag | Skip plot generation |

### Running on Google Colab

A Jupyter notebook (`colab_training.ipynb`) is provided for running experiments on Google Colab with free GPU access:

1. **Open the notebook in Google Colab:**
   - Upload `colab_training.ipynb` to Google Colab or open it directly from GitHub

2. **Mount Google Drive:**
   - Run the first cell to mount your Google Drive for persistent storage

3. **Upload your data to Google Drive:**
   - Create a folder: `/content/drive/MyDrive/CloudML/data/`
   - Upload your flight data folders (e.g., `10Feb25/`, `30Oct24/`, etc.) to this location
   - Each flight folder should contain the `.h5`, `.hdf5`, and `.hdf` files

4. **Run the setup cells:**
   - Install dependencies
   - Clone/update the repository
   - Update configuration paths

5. **Run experiments:**
   - Single experiments or ablation studies
   - Results are saved to Google Drive automatically

**Note:** Google Colab free tier has session limits (~12 hours). For longer training runs, consider Colab Pro or download checkpoints periodically.

## Project Structure

```
 README.md
 main.py
 requirements.txt
 .gitignore
 .pre-commit-config.yaml
 configs/
    config.yaml
    bestComboConfig.yaml
    complexity_weights.yaml
    ablation_*.yaml
 scripts/
    aggregate_results.py
    calibrate_model.py
    pretrain_ssl.py
 src/
    __init__.py
    caching.py
    cplCompareSub.py
    data_preprocessing.py
    ensemble.py
    evaluate_model.py
    hdf5_dataset.py
    mae_model.py
    main_utils.py
    pipeline.py
    plot_saved_results.py
    pytorchmodel.py
    scene_complexity.py
    train_model.py
    unlabeled_dataset.py
    utils/
       ...
    visualization.py
```

-   **`main.py`**: Entry point for running the model.
-   **`configs/`**: Configuration files for experiments, including ablation setups.
-   **`scripts/`**: Additional scripts for calibration, pre-training, and result aggregation.
-   **`src/`**: Source code for data processing, model architecture, training, and evaluation pipelines.
-   **`.gitignore`**: Git ignore file to exclude clutter.
-   **`.pre-commit-config.yaml`**: Pre-commit hooks configuration for code quality (linting, formatting).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for bug fixes, feature requests, or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
