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
- **Evaluation:** Evaluate model performance using standard metrics (MAE, MSE, R²) and leave-one-out cross-validation for robustness.
- **Visualization:** Generate plots, attention maps, and error analyses to interpret model behavior.
- **Calibration and Uncertainty:** Includes conformal prediction for uncertainty quantification and model calibration scripts.

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
    python main.py --config bestComboConfig.yaml
    ```

-   **Override configuration parameters from the command line:**
    ```bash
    python main.py --learning_rate 0.0001 --epochs 50
    ```

-   **Run a leave-one-out (LOO) evaluation:**
    ```bash
    python main.py --loo
    ```

## Project Structure

```
├── README.md
├── config.yaml
├── bestComboConfig.yaml
├── complexity_weights.yaml
├── calibrate_model.py
├── pretrain_ssl.py
├── main.py
├── requirements.txt
├── .gitignore
├── .pre-commit-config.yaml
├── src/
│   ├── __init__.py
│   ├── caching.py
│   ├── cplCompareSub.py
│   ├── data_preprocessing.py
│   ├── ensemble.py
│   ├── evaluate_model.py
│   ├── hdf5_dataset.py
│   ├── mae_model.py
│   ├── main_utils.py
│   ├── pipeline.py
│   ├── plot_saved_results.py
│   ├── pytorchmodel.py
│   ├── scene_complexity.py
│   ├── train_model.py
│   ├── unlabeled_dataset.py
│   ├── utils/
│   │   ├── ...
│   ├── visualization.py
```

-   **`config.yaml`**: Main configuration file for experiments.
-   **`bestComboConfig.yaml`**: Configuration file with optimized hyperparameters.
-   **`complexity_weights.yaml`**: Weights for scene complexity calculations.
-   **`calibrate_model.py`**: Script for model calibration.
-   **`pretrain_ssl.py`**: Script for self-supervised pre-training.
-   **`main.py`**: Entry point for running the model.
-   **`src/`**: Source code for data processing, model architecture, training, and evaluation pipelines.
-   **`.gitignore`**: Git ignore file to exclude clutter.
-   **`.pre-commit-config.yaml`**: Pre-commit hooks configuration for code quality (linting, formatting).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for bug fixes, feature requests, or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
