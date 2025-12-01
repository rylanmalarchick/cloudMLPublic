# Cloud Base Height Retrieval from Airborne Imagery

Production-ready machine learning system for cloud base height (CBH) retrieval from NASA ER-2 aircraft data, validated against Cloud Physics Lidar (CPL) measurements.

##  Production Model (Sprint 6)

**Status:**  Production-Ready | Test Coverage: 93.5% | Deployment: Approved

| Model | R² | MAE | Status |
|-------|----|----|--------|
| **GBDT (Tabular)** | **0.744** | **117.4 m** |  Production |
| Ensemble (GBDT+CNN) | 0.7391 | 122.5 m |  Production |

**Key Achievement:** First model to exceed R² = 0.74 target on operational data.

##  Project Structure

```
cloudMLPublic/
 src/
    cbh_retrieval/          # Sprint 6 production module
 tests/
    cbh/                     # Test suite (93.5% coverage)
 scripts/
    cbh/                     # Production utilities & auditing
 docs/
    cbh/                     # Complete documentation
 preprint/                   # Academic publication materials
 configs/                    # Legacy configuration files
 outputs/
     preprocessed_data/       # Integrated features (HDF5)
```

##  Quick Start

### Production Model (Sprint 6 - Recommended)

```bash
# Install dependencies
pip install -r docs/cbh/requirements_production.txt

# Train production model
python -c "
from src.cbh_retrieval import train_production_model
model, scaler = train_production_model()
"

# Run validation (5-fold CV)
python -c "
from src.cbh_retrieval import validate_tabular
validate_tabular()
"

# Run tests
pytest tests/cbh/ --cov=src/cbh_retrieval
```

### Documentation

- **Model Card:** `docs/cbh/MODEL_CARD.md` - Complete model specifications
- **Deployment Guide:** `docs/cbh/DEPLOYMENT_GUIDE.md` - Production deployment
- **Reproducibility:** `docs/cbh/REPRODUCIBILITY_GUIDE.md` - Full reproduction
- **Future Work:** `docs/cbh/FUTURE_WORK.md` - Roadmap & improvements

##  Dataset

- **Samples:** 933 CPL-aligned observations (5 flights, Oct 2024 - Feb 2025)
- **Input Features:** 18 features (12 atmospheric from ERA5 + 6 geometric from shadow analysis)
- **Target:** Cloud base height from CPL lidar (0.12–1.95 km, mean: 0.83 km)
- **Validation:** 5-fold stratified cross-validation
- **Data Location:** `outputs/preprocessed_data/Integrated_Features.hdf5`

## Technologies and Libraries

- **Machine Learning:** scikit-learn for GBDT production model
- **Deep Learning (Research):** PyTorch with CUDA acceleration for experimental models
- **Data Handling:** HDF5 for efficient storage; pandas and NumPy for data manipulation
- **Visualization:** Matplotlib and Plotly for analysis plots
- **Testing:** pytest with 93.5% coverage
- **Configuration:** YAML for experiment management

## Getting Started

### Prerequisites

- Python 3.9+
- For production model: CPU sufficient
- For research models: CUDA 11.8+ (optional)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/rylanmalarchick/CloudMLPublic.git
    cd CloudMLPublic
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    # For production model only
    pip install -r docs/cbh/requirements_production.txt
    
    # For full development (includes research models)
    pip install -r requirements.txt
    ```

### Data

This project uses airborne imagery and atmospheric data from NASA ER-2 field campaigns. The data is not included due to size and access restrictions. You will need:

- **Integrated Features Dataset:** `Integrated_Features.hdf5` containing atmospheric variables (ERA5) and geometric features (shadow analysis)
- **Raw Data (for research):** IRAI camera imagery, CPL lidar measurements, navigation data

Contact: rylan1012@gmail.com for data access.

### Running the Model

#### Production Model (Recommended)

```bash
# Train production GBDT model
python -c "
from src.cbh_retrieval import train_production_model
model, scaler = train_production_model()
"

# Run 5-fold cross-validation
python -c "
from src.cbh_retrieval import validate_tabular
validate_tabular()
"

# Run tests
pytest tests/cbh/ --cov=src/cbh_retrieval
```

#### Research Models (Legacy)

The repository contains legacy deep learning models from prior research phases. These are maintained for reproducibility but not recommended for production use.

```bash
# Run legacy model training
python main.py --config configs/config.yaml
```

See `docs/cbh/REPRODUCIBILITY_GUIDE.md` for detailed instructions on reproducing research results.

## Project Structure

```
cloudMLPublic/
 README.md
 main.py                     # Legacy deep learning training entry point
 requirements.txt
 configs/                    # Legacy configuration files
 scripts/
    cbh/                     # Production utilities & auditing
 src/
    cbh_retrieval/          # Sprint 6 production module
    [legacy modules]        # Research code for reproducibility
 tests/
    cbh/                     # Test suite (93.5% coverage)
 docs/
    cbh/                     # Complete documentation
 preprint/                   # Academic publication materials
 outputs/
    preprocessed_data/      # Integrated features (HDF5)
```

### Key Files

-   **`src/cbh_retrieval/train_production_model.py`**: Production GBDT training
-   **`src/cbh_retrieval/offline_validation_tabular.py`**: Cross-validation framework
-   **`tests/cbh/`**: Comprehensive test suite
-   **`docs/cbh/MODEL_CARD.md`**: Complete model specifications
-   **`docs/cbh/DEPLOYMENT_GUIDE.md`**: Production deployment instructions

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for bug fixes, feature requests, or improvements.

## Citation

If you use this work in your research, please cite our preprint:

```
[Citation details to be added upon publication]
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
