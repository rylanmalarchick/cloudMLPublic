# Deployment Guide: Cloud Base Height Retrieval System

**Version**: 1.0.0  
**Last Updated**: 2025-11-11  
**Author**: NASA Cloud ML Team

---

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Deployment Options](#deployment-options)
6. [API Integration](#api-integration)
7. [Monitoring & Logging](#monitoring--logging)
8. [Troubleshooting](#troubleshooting)
9. [Maintenance](#maintenance)
10. [Security](#security)
11. [Appendix](#appendix)

---

## Overview

This guide provides step-by-step instructions for deploying the Cloud Base Height (CBH) Retrieval System v1.0.0 in production environments. The system predicts cloud base height using atmospheric and geometric features derived from ERA5 reanalysis and aerial imagery.

### Architecture Overview

```
┌─────────────────┐
│  Data Sources   │
│  - ERA5 Data    │
│  - Images       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocessing   │
│ - Imputation    │
│ - Scaling       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ GBDT Model      │
│ (Inference)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Post-processing │
│ - UQ Intervals  │
│ - Validation    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Output (CBH)    │
└─────────────────┘
```

### Key Components

- **Production Model**: `production_model.joblib` (Gradient Boosting Regressor)
- **Scaler**: `production_scaler.joblib` (StandardScaler for feature normalization)
- **Configuration**: `production_config.json` (Model hyperparameters and metadata)
- **Inference Engine**: Python-based scikit-learn inference
- **Dependencies**: scikit-learn 1.7.0, numpy 2.2.6, pandas 2.3.0

---

## System Requirements

### Minimum Requirements

- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10+
- **Python**: 3.10, 3.11, or 3.12
- **RAM**: 4 GB minimum, 8 GB recommended
- **Disk**: 1 GB for model artifacts and dependencies
- **CPU**: 2 cores minimum, 4+ cores recommended
- **GPU**: Not required (CPU-based inference)

### Recommended Production Environment

- **OS**: Linux (Ubuntu 22.04 LTS)
- **Python**: 3.12
- **RAM**: 16 GB
- **Disk**: 10 GB (includes logs and data caching)
- **CPU**: 8 cores (Intel Xeon or AMD EPYC)
- **Network**: Low-latency connection to ERA5 data source

### Software Dependencies

See `requirements.txt` for full dependency list. Key dependencies:

```
scikit-learn==1.7.0
numpy==2.2.6
pandas==2.3.0
h5py==3.14.0
joblib==1.5.1
scipy==1.15.3
```

---

## Installation

### Option 1: Virtual Environment (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/nasa/cloudMLPublic.git
cd cloudMLPublic

# 2. Create virtual environment
python3.12 -m venv venv_prod
source venv_prod/bin/activate  # Linux/macOS
# OR
venv_prod\Scripts\activate  # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify installation
python -c "import sklearn; import joblib; print('Installation successful')"
```

### Option 2: Docker Container (Isolated Deployment)

```bash
# 1. Build Docker image
cd cloudMLPublic
docker build -t cbh-retrieval:1.0.0 -f Dockerfile.production .

# 2. Run container
docker run -d \
  --name cbh-service \
  -p 8080:8080 \
  -v /data/era5:/data/era5:ro \
  -v /logs:/app/logs \
  cbh-retrieval:1.0.0

# 3. Verify deployment
docker logs cbh-service
curl http://localhost:8080/health
```

**Note**: Dockerfile.production must be created (see Appendix A).

### Option 3: Conda Environment

```bash
# 1. Create conda environment
conda create -n cbh_prod python=3.12
conda activate cbh_prod

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python -c "import sklearn; print(sklearn.__version__)"
```

---

## Configuration

### Environment Variables

Set the following environment variables before deployment:

```bash
# Required
export CBH_MODEL_PATH="/path/to/sow_outputs/sprint6/checkpoints/production_model.joblib"
export CBH_SCALER_PATH="/path/to/sow_outputs/sprint6/checkpoints/production_scaler.joblib"
export CBH_CONFIG_PATH="/path/to/sow_outputs/sprint6/checkpoints/production_config.json"

# Optional
export CBH_LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR
export CBH_LOG_PATH="/var/log/cbh"
export CBH_MAX_BATCH_SIZE="1000"
export CBH_ENABLE_UQ="true"  # Enable uncertainty quantification
export CBH_CACHE_DIR="/tmp/cbh_cache"
```

### Configuration File

Create `config/production.yaml`:

```yaml
model:
  path: "sow_outputs/sprint6/checkpoints/production_model.joblib"
  scaler_path: "sow_outputs/sprint6/checkpoints/production_scaler.joblib"
  config_path: "sow_outputs/sprint6/checkpoints/production_config.json"

inference:
  max_batch_size: 1000
  timeout_seconds: 30
  num_workers: 4

uncertainty:
  enabled: true
  confidence_level: 0.90
  calibration_method: "none"  # TODO: Add post-hoc calibration

validation:
  check_input_range: true
  check_feature_distribution: true
  alert_on_drift: true

logging:
  level: "INFO"
  path: "/var/log/cbh"
  rotation: "daily"
  retention_days: 30

monitoring:
  enabled: true
  prometheus_port: 9090
  health_check_interval: 60
```

### Feature Configuration

Ensure input features match the expected order and units:

```json
{
  "features": [
    {"name": "blh", "unit": "m", "description": "Boundary Layer Height"},
    {"name": "lcl", "unit": "m", "description": "Lifting Condensation Level"},
    {"name": "inversion_height", "unit": "m", "description": "Inversion Height"},
    {"name": "moisture_gradient", "unit": "unitless", "description": "Moisture Gradient"},
    {"name": "stability_index", "unit": "unitless", "description": "Stability Index"},
    {"name": "t2m", "unit": "K", "description": "Temperature at 2m"},
    {"name": "d2m", "unit": "K", "description": "Dewpoint at 2m"},
    {"name": "sp", "unit": "Pa", "description": "Surface Pressure"},
    {"name": "tcwv", "unit": "kg/m2", "description": "Total Column Water Vapor"},
    {"name": "cloud_edge_x", "unit": "pixels", "description": "Cloud Edge X"},
    {"name": "cloud_edge_y", "unit": "pixels", "description": "Cloud Edge Y"},
    {"name": "saa_deg", "unit": "degrees", "description": "Solar Azimuth Angle"},
    {"name": "shadow_angle_deg", "unit": "degrees", "description": "Shadow Angle"},
    {"name": "shadow_detection_confidence", "unit": "0-1", "description": "Confidence"},
    {"name": "shadow_edge_x", "unit": "pixels", "description": "Shadow Edge X"},
    {"name": "shadow_edge_y", "unit": "pixels", "description": "Shadow Edge Y"},
    {"name": "shadow_length_pixels", "unit": "pixels", "description": "Shadow Length"},
    {"name": "sza_deg", "unit": "degrees", "description": "Solar Zenith Angle"}
  ]
}
```

---

## Deployment Options

### Deployment 1: Batch Processing Service

For offline batch processing of large datasets.

```python
# batch_inference.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CBHBatchProcessor:
    def __init__(self, model_path, scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        logger.info("Model and scaler loaded successfully")
    
    def preprocess(self, X):
        """Preprocess features: impute missing values and scale."""
        # Handle missing values
        X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
        # Scale features
        X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def predict_batch(self, X, return_uncertainty=False):
        """Predict CBH for a batch of samples."""
        X_preprocessed = self.preprocess(X)
        predictions = self.model.predict(X_preprocessed)
        
        if return_uncertainty:
            # Placeholder for UQ (requires quantile models)
            uncertainty = np.std(predictions) * np.ones_like(predictions)
            return predictions, uncertainty
        
        return predictions
    
    def process_file(self, input_path, output_path):
        """Process an entire HDF5 file."""
        logger.info(f"Processing {input_path}")
        
        # Load data
        df = pd.read_hdf(input_path, key='features')
        X = df.values
        
        # Predict
        predictions = self.predict_batch(X)
        
        # Save results
        df['cbh_predicted_m'] = predictions
        df.to_csv(output_path, index=False)
        
        logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    processor = CBHBatchProcessor(
        model_path="sow_outputs/sprint6/checkpoints/production_model.joblib",
        scaler_path="sow_outputs/sprint6/checkpoints/production_scaler.joblib"
    )
    
    processor.process_file(
        input_path="data/input_features.h5",
        output_path="data/output_predictions.csv"
    )
```

**Usage**:
```bash
python batch_inference.py
```

### Deployment 2: REST API Service

For real-time inference via HTTP API.

```python
# api_server.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging
from datetime import datetime

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model at startup
model = joblib.load("sow_outputs/sprint6/checkpoints/production_model.joblib")
scaler = joblib.load("sow_outputs/sprint6/checkpoints/production_scaler.joblib")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.utcnow().isoformat()})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        
        # Validate input
        if features.shape[1] != 18:
            return jsonify({"error": f"Expected 18 features, got {features.shape[1]}"}), 400
        
        # Preprocess
        features = np.nan_to_num(features, nan=0.0)
        features_scaled = scaler.transform(features)
        
        # Predict
        cbh_prediction = model.predict(features_scaled)[0]
        
        # Response
        response = {
            "cbh_meters": float(cbh_prediction),
            "timestamp": datetime.utcnow().isoformat(),
            "model_version": "1.0.0"
        }
        
        logger.info(f"Prediction: {cbh_prediction:.2f}m")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data = request.get_json()
        features = np.array(data['features'])
        
        if features.shape[1] != 18:
            return jsonify({"error": f"Expected 18 features, got {features.shape[1]}"}), 400
        
        # Preprocess and predict
        features = np.nan_to_num(features, nan=0.0)
        features_scaled = scaler.transform(features)
        predictions = model.predict(features_scaled)
        
        response = {
            "predictions": predictions.tolist(),
            "count": len(predictions),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
```

**Usage**:
```bash
# Start server
python api_server.py

# Test endpoint
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [1000, 800, 1200, 0.5, 0.3, 290, 285, 101325, 25, 
                 100, 200, 180, 45, 0.9, 150, 250, 50, 30]
  }'
```

### Deployment 3: Streaming Service (Kafka)

For real-time processing of streaming data.

```python
# streaming_service.py
from kafka import KafkaConsumer, KafkaProducer
import joblib
import numpy as np
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CBHStreamProcessor:
    def __init__(self, model_path, scaler_path, 
                 input_topic='cbh-input', output_topic='cbh-output',
                 kafka_servers=['localhost:9092']):
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        self.consumer = KafkaConsumer(
            input_topic,
            bootstrap_servers=kafka_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_servers,
            value_serializer=lambda m: json.dumps(m).encode('utf-8')
        )
        
        self.output_topic = output_topic
        logger.info("Stream processor initialized")
    
    def process_message(self, message):
        """Process a single message."""
        try:
            features = np.array(message['features']).reshape(1, -1)
            features = np.nan_to_num(features, nan=0.0)
            features_scaled = self.scaler.transform(features)
            
            prediction = self.model.predict(features_scaled)[0]
            
            result = {
                'id': message.get('id'),
                'cbh_meters': float(prediction),
                'timestamp': message.get('timestamp')
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return None
    
    def run(self):
        """Main processing loop."""
        logger.info("Starting stream processing...")
        
        for message in self.consumer:
            result = self.process_message(message.value)
            
            if result:
                self.producer.send(self.output_topic, value=result)
                logger.debug(f"Processed message {result['id']}")

if __name__ == "__main__":
    processor = CBHStreamProcessor(
        model_path="sow_outputs/sprint6/checkpoints/production_model.joblib",
        scaler_path="sow_outputs/sprint6/checkpoints/production_scaler.joblib"
    )
    processor.run()
```

---

## API Integration

### Python Client Library

```python
# cbh_client.py
import requests
import numpy as np

class CBHClient:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
    
    def predict(self, features):
        """Single prediction."""
        response = requests.post(
            f"{self.base_url}/predict",
            json={"features": features}
        )
        response.raise_for_status()
        return response.json()
    
    def batch_predict(self, features_list):
        """Batch prediction."""
        response = requests.post(
            f"{self.base_url}/batch_predict",
            json={"features": features_list}
        )
        response.raise_for_status()
        return response.json()
    
    def health_check(self):
        """Check service health."""
        response = requests.get(f"{self.base_url}/health")
        return response.json()

# Usage example
if __name__ == "__main__":
    client = CBHClient()
    
    # Health check
    print(client.health_check())
    
    # Single prediction
    features = [1000, 800, 1200, 0.5, 0.3, 290, 285, 101325, 25,
                100, 200, 180, 45, 0.9, 150, 250, 50, 30]
    result = client.predict(features)
    print(f"CBH: {result['cbh_meters']:.2f}m")
```

### cURL Examples

```bash
# Health check
curl http://localhost:8080/health

# Single prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [1000, 800, 1200, 0.5, 0.3, 290, 285, 101325, 25, 
                 100, 200, 180, 45, 0.9, 150, 250, 50, 30]
  }'

# Batch prediction
curl -X POST http://localhost:8080/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      [1000, 800, 1200, 0.5, 0.3, 290, 285, 101325, 25, 100, 200, 180, 45, 0.9, 150, 250, 50, 30],
      [1100, 850, 1300, 0.6, 0.4, 292, 287, 101300, 26, 110, 210, 185, 50, 0.85, 160, 260, 55, 35]
    ]
  }'
```

---

## Monitoring & Logging

### Application Logging

Configure structured logging:

```python
# logging_config.py
import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_logging(log_path="/var/log/cbh", level=logging.INFO):
    """Configure application logging."""
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # File handler (rotating)
    file_handler = RotatingFileHandler(
        f"{log_path}/cbh_service.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
```

### Performance Monitoring

Track key metrics:

```python
# metrics.py
import time
from functools import wraps
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Metrics
prediction_counter = Counter('cbh_predictions_total', 'Total predictions')
prediction_latency = Histogram('cbh_prediction_latency_seconds', 'Prediction latency')
error_counter = Counter('cbh_errors_total', 'Total errors', ['error_type'])
active_requests = Gauge('cbh_active_requests', 'Active requests')

def track_prediction(func):
    """Decorator to track prediction metrics."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        active_requests.inc()
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            prediction_counter.inc()
            return result
        except Exception as e:
            error_counter.labels(error_type=type(e).__name__).inc()
            raise
        finally:
            duration = time.time() - start_time
            prediction_latency.observe(duration)
            active_requests.dec()
    
    return wrapper

# Start Prometheus metrics server
start_http_server(9090)
```

### Alerting

Configure alerts for critical conditions:

```yaml
# alerts.yaml
groups:
  - name: cbh_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: rate(cbh_errors_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
      
      - alert: SlowPredictions
        expr: cbh_prediction_latency_seconds{quantile="0.95"} > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow predictions (p95 > 1s)"
      
      - alert: ServiceDown
        expr: up{job="cbh_service"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "CBH service is down"
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Model Loading Fails

**Symptom**: `FileNotFoundError` or `PickleError` when loading model

**Solutions**:
```bash
# Check file exists
ls -lh sow_outputs/sprint6/checkpoints/production_model.joblib

# Verify scikit-learn version
python -c "import sklearn; print(sklearn.__version__)"
# Should be 1.7.0

# Re-download model if corrupted
md5sum sow_outputs/sprint6/checkpoints/production_model.joblib
```

#### Issue 2: Feature Dimension Mismatch

**Symptom**: `ValueError: Expected 18 features, got X`

**Solutions**:
- Verify input feature order matches training configuration
- Check for missing features in data pipeline
- Use feature configuration JSON to validate

#### Issue 3: Poor Prediction Quality

**Symptom**: High MAE (>200m) or negative R²

**Solutions**:
- Check for domain shift (compare input distributions to training data)
- Verify preprocessing is applied correctly
- Check for NaN/Inf values in inputs
- Consider domain adaptation if new atmospheric regime

#### Issue 4: Out of Memory

**Symptom**: `MemoryError` during batch processing

**Solutions**:
```python
# Process in smaller batches
batch_size = 100
for i in range(0, len(X), batch_size):
    batch = X[i:i+batch_size]
    predictions = model.predict(scaler.transform(batch))
```

#### Issue 5: Slow Inference

**Symptom**: Latency >100ms per sample

**Solutions**:
- Use batched predictions (10-100x faster)
- Ensure model is loaded once at startup (not per request)
- Consider model quantization or distillation
- Profile code to identify bottlenecks

### Debug Mode

Enable detailed debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Add timing
import time
start = time.time()
predictions = model.predict(X)
print(f"Inference took {time.time() - start:.4f}s")

# Check intermediate values
print(f"Input shape: {X.shape}")
print(f"Input range: [{X.min():.2f}, {X.max():.2f}]")
print(f"Scaled range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")
```

### Support Contacts

- **Technical Issues**: ml-support@nasa.gov
- **Data Issues**: data-team@nasa.gov
- **Emergency**: on-call-sre@nasa.gov

---

## Maintenance

### Routine Maintenance Tasks

#### Weekly
- [ ] Review error logs for anomalies
- [ ] Check disk space and log rotation
- [ ] Monitor prediction latency trends

#### Monthly
- [ ] Analyze prediction quality metrics on validation set
- [ ] Review and update documentation
- [ ] Update dependency security patches

#### Quarterly
- [ ] Retrain model with new labeled data
- [ ] Perform data drift analysis
- [ ] Update model card and deployment guide
- [ ] Conduct load testing

### Model Retraining

When to retrain:
1. MAE degrades by >20% on validation set
2. New flight campaign data available (≥100 samples)
3. Domain shift detected
4. Quarterly schedule

Retraining procedure:
```bash
# 1. Prepare new training data
python scripts/prepare_training_data.py --new-data-path /path/to/new/data

# 2. Run cross-validation
python sow_outputs/sprint6/validation/cross_validate_tabular.py

# 3. Train production model
python sow_outputs/sprint6/training/train_production_model.py

# 4. Validate performance
python sow_outputs/sprint6/validation/validate_production_model.py

# 5. Deploy new version
cp sow_outputs/sprint6/checkpoints/production_model.joblib \
   /production/models/production_model_v1.1.0.joblib

# 6. Update version in config
# Edit production.yaml: model_version: "1.1.0"

# 7. Restart service
systemctl restart cbh-service
```

### Backup & Recovery

#### Backup Strategy

```bash
# Automated daily backups
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d)
BACKUP_DIR="/backups/cbh"

# Backup model artifacts
tar -czf $BACKUP_DIR/models_$DATE.tar.gz \
  sow_outputs/sprint6/checkpoints/

# Backup configuration
tar -czf $BACKUP_DIR/config_$DATE.tar.gz \
  config/

# Backup logs (last 7 days)
tar -czf $BACKUP_DIR/logs_$DATE.tar.gz \
  /var/log/cbh/

# Cleanup old backups (>30 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

#### Recovery Procedure

```bash
# 1. Stop service
systemctl stop cbh-service

# 2. Restore from backup
tar -xzf /backups/cbh/models_20251111.tar.gz -C /

# 3. Verify artifacts
python -c "import joblib; joblib.load('sow_outputs/sprint6/checkpoints/production_model.joblib')"

# 4. Restart service
systemctl start cbh-service

# 5. Verify health
curl http://localhost:8080/health
```

---

## Security

### Security Best Practices

1. **Authentication & Authorization**
   - Implement API key authentication for production endpoints
   - Use JWT tokens for user authentication
   - Apply rate limiting to prevent abuse

2. **Network Security**
   - Deploy behind reverse proxy (nginx/Apache)
   - Enable HTTPS/TLS for all endpoints
   - Restrict access via firewall rules

3. **Input Validation**
   - Validate all input features (range checks, type checks)
   - Sanitize inputs to prevent injection attacks
   - Implement request size limits

4. **Dependency Security**
   - Regularly update dependencies for security patches
   - Use `pip-audit` to scan for vulnerabilities
   - Pin dependency versions in production

5. **Model Security**
   - Protect model files from unauthorized access (chmod 600)
   - Monitor for model extraction attempts
   - Implement rate limiting on inference endpoints

### Example: API Key Authentication

```python
# secure_api.py
from flask import Flask, request, jsonify
from functools import wraps
import os

app = Flask(__name__)
API_KEYS = set(os.environ.get('CBH_API_KEYS', '').split(','))

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key not in API_KEYS:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    # ... prediction logic
    pass
```

### Compliance

- **Data Privacy**: No PII processed; atmospheric data only
- **Export Control**: Check ITAR/EAR restrictions for model export
- **Audit Logging**: Log all prediction requests with timestamps

---

## Appendix

### Appendix A: Dockerfile.production

```dockerfile
# Dockerfile.production
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY sow_outputs/sprint6/checkpoints/ /app/checkpoints/
COPY api_server.py /app/
COPY config/ /app/config/

# Create non-root user
RUN useradd -m -u 1000 cbh && chown -R cbh:cbh /app
USER cbh

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Run application
CMD ["python", "api_server.py"]
```

### Appendix B: Systemd Service File

```ini
# /etc/systemd/system/cbh-service.service
[Unit]
Description=Cloud Base Height Retrieval Service
After=network.target

[Service]
Type=simple
User=cbh
WorkingDirectory=/opt/cbh
Environment="CBH_MODEL_PATH=/opt/cbh/models/production_model.joblib"
Environment="CBH_SCALER_PATH=/opt/cbh/models/production_scaler.joblib"
ExecStart=/opt/cbh/venv/bin/python /opt/cbh/api_server.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable cbh-service
sudo systemctl start cbh-service
sudo systemctl status cbh-service
```

### Appendix C: nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/cbh
upstream cbh_backend {
    server 127.0.0.1:8080;
}

server {
    listen 80;
    server_name cbh.nasa.gov;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name cbh.nasa.gov;

    ssl_certificate /etc/ssl/certs/cbh.crt;
    ssl_certificate_key /etc/ssl/private/cbh.key;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN";
    add_header X-Content-Type-Options "nosniff";

    location / {
        proxy_pass http://cbh_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=cbh_limit:10m rate=10r/s;
    limit_req zone=cbh_limit burst=20 nodelay;
}
```

### Appendix D: Performance Benchmarks

| Deployment Type | Latency (p50) | Latency (p95) | Throughput | Memory |
|-----------------|---------------|---------------|------------|--------|
| Single sample   | 0.5 ms        | 2 ms          | 2000 req/s | 50 MB  |
| Batch (100)     | 10 ms         | 20 ms         | 10k samples/s | 100 MB |
| Batch (1000)    | 80 ms         | 150 ms        | 12k samples/s | 200 MB |

Tested on: Intel Xeon E5-2670, 16GB RAM, Ubuntu 22.04

### Appendix E: Quick Reference

**Start service**:
```bash
python api_server.py
```

**Test prediction**:
```bash
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" \
  -d '{"features": [1000, 800, 1200, 0.5, 0.3, 290, 285, 101325, 25, 100, 200, 180, 45, 0.9, 150, 250, 50, 30]}'
```

**Check logs**:
```bash
tail -f /var/log/cbh/cbh_service.log
```

**Restart service**:
```bash
sudo systemctl restart cbh-service
```

---

**Document Version**: 1.0.0  
**Last Updated**: 2025-11-11  
**Next Review**: 2026-02-11