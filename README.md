# Boroondara Traffic Flow Prediction System (TFPS)

A comprehensive traffic flow prediction system that utilizes machine learning models to predict traffic conditions based on historical data. This system was developed as part of the COS30018 Intelligent Systems project assignment.

## Overview

The Traffic Flow Prediction System (TFPS) predicts traffic flow using various deep learning models trained on historical traffic data from VicRoads for the city of Boroondara. The system implements multiple machine learning models:

- Long Short-Term Memory (LSTM) networks
- Gated Recurrent Units (GRU)
- Stacked Autoencoders (SAE)
- Transformer models

These models are trained on traffic flow data (the number of cars passing through intersections every 15 minutes) to predict future traffic conditions.

## System Requirements

- Python 3.8 or later
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Joblib

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/traffic-flow-prediction.git
   cd traffic-flow-prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure data files are in the correct directories:
   - Place SCATS data CSV file in `data/raw/Scats Data October 2006.csv`
   - Place SCATS site metadata in `data/raw/SCATSSiteListingSpreadsheet_VicRoads.csv`

## Project Structure

```
├── boroondara_main.py         # Main entry point
├── config/                    # Configuration files
│   └── boroondara_config.py   # Model configurations
├── data/                      # Data files
│   ├── raw/                   # Raw data files
│   ├── processed/             # Processed data files
│   └── boroondara_preprocessing.py  # Data preprocessing utilities
├── src/                       # Source code
│   ├── boroondara.py          # Main functionality module
│   ├── models/                # Model implementations
│   │   ├── base.py            # Base model interface
│   │   ├── lstm.py            # LSTM model implementation
│   │   ├── gru.py             # GRU model implementation
│   │   ├── saes.py            # Stacked Autoencoder model implementation
│   │   └── transformer.py     # Transformer model implementation
│   └── utils/                 # Utility modules
│       ├── cli_parser.py      # Command-line argument parsing
│       ├── console.py         # Console output utilities
│       ├── data_processor.py  # Data processing utilities
│       ├── evaluation.py      # Evaluation metrics
│       ├── evaluation_utils.py # Evaluation utilities
│       ├── model_utils.py     # Model utilities
│       └── visualization.py   # Visualization utilities
├── saved_models/              # Saved model files
├── logs/                      # TensorBoard logs
└── results/                   # Evaluation results and visualizations
```

## Usage

The system provides four main commands:

1. `preprocess` - Preprocess the SCATS data for training
2. `train` - Train the prediction models
3. `evaluate` - Evaluate model performance
4. `visualize` - Visualize traffic patterns

### 1. Data Preprocessing

Preprocess the data before training:

```bash
python boroondara_main.py preprocess
```

Options:
- `--combined` - Create a combined dataset from all sites
- `--scats_data` - Path to SCATS data CSV file (default: "data/raw/Scats Data October 2006.csv")
- `--metadata` - Path to SCATS metadata CSV file (default: "data/raw/SCATSSiteListingSpreadsheet_VicRoads.csv")
- `--processed_dir` - Directory for processed data (default: "data/processed/boroondara")
- `--seq_length` - Sequence length for prediction (default: 12)
- `--force` - Force preprocessing even if files already exist

Example with all options:
```bash
python boroondara_main.py preprocess --combined --scats_data "path/to/scats_data.csv" --metadata "path/to/metadata.csv" --processed_dir "data/processed/custom" --seq_length 24 --force
```

### 2. Model Training

The system offers four different training strategies, each with its own benefits and use cases:

#### a. Train models for a specific site:
```bash
python boroondara_main.py train --site_id 3120
```
This strategy trains individual models for a single specific traffic site. Use this when:
- You want to focus on prediction accuracy for one particular intersection
- You're experimenting with model parameters for a specific location
- You have limited computational resources

**Output**: Model files saved in `saved_models/boroondara/per_site/3120/[model_type].h5`

#### b. Train separate models for each site:
```bash
python boroondara_main.py train --train_all_sites
```
This strategy trains separate, specialized models for each available site. Use this when:
- You need the highest possible prediction accuracy for each individual site
- Different intersections have very different traffic patterns
- You want to compare site-specific performance variations

**Output**: Multiple model files, one for each site, saved in `saved_models/boroondara/per_site/[site_id]/[model_type].h5`

#### c. Train one model using data from all sites without pre-combining:
```bash
python boroondara_main.py train --one_model_all_sites
```
This strategy loads data from each site separately at runtime and combines them to train a single model. Use this when:
- You want one model that generalizes across all sites
- You want to preserve site-specific information during training
- You're experimenting with cross-site pattern recognition

**Output**: One model file per model type, saved as `saved_models/boroondara/[model_type]_all_sites.h5`

#### d. Train a combined model using pre-processed merged data:
```bash
python boroondara_main.py train --combined_model
```
This strategy uses a pre-combined dataset (created with `preprocess --combined`) to train a single model. Use this when:
- You want a generalized model with the most standardized data
- You need the fastest training time for multi-site data
- You've already created optimized combined datasets

**Output**: One model file per model type, saved as `saved_models/boroondara/[model_type]_boroondara.h5`

#### Training options:
- `--models` - Models to train (comma-separated list: lstm,gru,saes,transformer,all) (default: "all")
- `--batch_size` - Batch size for training (default: 256)
- `--epochs` - Number of training epochs (default: 100)
- `--pretraining_epochs` - Number of pretraining epochs for SAE (default: 50)
- `--patience` - Patience for early stopping (default: 10)
- `--max_sites` - Maximum number of sites to process (optional)
- `--model_dir` - Directory for saved models (default: "saved_models/boroondara")
- `--log_dir` - Directory for TensorBoard logs (default: "logs/boroondara")
- `--results_dir` - Directory for results (default: "results/boroondara")

Example with specific models and parameters:
```bash
python boroondara_main.py train --one_model_all_sites --models lstm,transformer --batch_size 128 --epochs 200 --patience 20
```

### 3. Model Evaluation

Evaluate trained models based on where and how they were trained:

#### Evaluate models for a specific site:
```bash
python boroondara_main.py evaluate --site_id 3120
```
This evaluates models on data from a specific site. It first looks for site-specific models, and if not found, falls back to using combined models.

#### Evaluate on all available sites:
```bash
python boroondara_main.py evaluate --evaluate_all_sites
```
This evaluates models across all sites and produces comparison metrics showing which models perform best on different sites.

#### Evaluate combined model:
```bash
python boroondara_main.py evaluate
```
This evaluates models trained on combined data, using the combined test dataset.

#### Evaluation options:
- `--models` - Models to evaluate (comma-separated list: lstm,gru,saes,transformer,all) (default: "all")
- `--max_sites` - Maximum number of sites to evaluate (optional)
- `--model_dir` - Directory containing trained models (default: "saved_models/boroondara")
- `--results_dir` - Directory for evaluation results (default: "results/boroondara")

Example with specific models:
```bash
python boroondara_main.py evaluate --evaluate_all_sites --models lstm,transformer --max_sites 10
```

### 4. Data Visualization

Visualize traffic patterns:

```bash
python boroondara_main.py visualize
```

Options:
- `--site_id` - Specific SCATS site ID to visualize (optional)
- `--max_sites` - Maximum number of sites to visualize (default: 5)
- `--results_dir` - Directory for visualization outputs (default: "results/boroondara")

Example for a specific site:
```bash
python boroondara_main.py visualize --site_id 3120
```

## Model Descriptions

### 1. LSTM (Long Short-Term Memory)
- Recurrent neural network architecture designed for time series analysis
- Well-suited for capturing long-term dependencies in traffic data
- Default configuration: 2 LSTM layers with 64 units each, followed by dropout and a dense output layer

### 2. GRU (Gated Recurrent Unit)
- Simpler variant of LSTM with fewer parameters
- Often performs similarly to LSTM but with faster training
- Default configuration: 2 GRU layers with 64 units each, followed by dropout and a dense output layer

### 3. SAE (Stacked Autoencoder)
- Deep learning model that learns compressed representations of the input data
- Uses pretrained layers that are then fine-tuned for prediction
- Default configuration: 3 hidden layers (400 units each) with sigmoid activation

### 4. Transformer
- Attention-based model that can capture dependencies without recurrence
- Excellent for time series data with complex patterns
- Default configuration: 64-dimensional model with 4 attention heads, 2 transformer layers

## Choosing The Right Training Strategy

The choice between training strategies depends on your specific needs:

1. **Site-specific models** (--site_id or --train_all_sites)
   - Best for: Maximum accuracy at specific intersections
   - Pros: Highest site-specific prediction accuracy, captures local patterns
   - Cons: Many models to maintain, doesn't generalize across sites
   - When to use: Traffic management for specific problematic intersections

2. **Combined models** (--combined_model or --one_model_all_sites)
   - Best for: City-wide traffic analysis, route planning
   - Pros: Single model to maintain, generalizes across sites, learns global patterns
   - Cons: May lose some site-specific accuracy
   - When to use: Route guidance systems, city-level traffic management

For developing a complete system, you might want to:
1. Start with a combined model for a broad understanding of traffic patterns
2. Train site-specific models for critical intersections
3. Compare performance to determine the best approach for your specific needs

## Running the Complete Pipeline

The following example demonstrates a complete workflow:

```bash
# 1. Preprocess the data
python boroondara_main.py preprocess --combined

# 2. Train models for a specific site
python boroondara_main.py train --site_id 3120 --models lstm,transformer

# 3. Train a model with data from all sites
python boroondara_main.py train --one_model_all_sites --models gru

# 4. Evaluate the models
python boroondara_main.py evaluate --site_id 3120
python boroondara_main.py evaluate --evaluate_all_sites --max_sites 20

# 5. Visualize traffic patterns
python boroondara_main.py visualize --site_id 3120
```

## TensorBoard Visualization

To view training progress in TensorBoard:

```bash
tensorboard --logdir=logs/boroondara
```

Then open your browser and navigate to http://localhost:6006.

## Troubleshooting

### Common Issues:

1. **Missing data files**:
   - Ensure all required data files are in the `data/raw/` directory
   - Check that file names match the expected paths or provide custom paths using the appropriate command-line arguments

2. **Memory errors during training**:
   - Reduce batch size with the `--batch_size` option
   - Limit the number of sites processed with `--max_sites`
   - Consider using a subset of the data for initial experiments

3. **Model training issues**:
   - Increase the patience value with `--patience` to allow more epochs without improvement
   - Check the learning curves in TensorBoard to diagnose overfitting or underfitting
   - Try different model architectures by modifying configurations in `config/boroondara_config.py`

4. **Slow performance**:
   - Use GRU or Transformer models which may train faster than LSTM or SAE
   - Reduce sequence length with `--seq_length` for faster training
   - Use a subset of sites for initial experiments

### Tips for Better Results:

1. Start with a single site to verify the system works correctly
2. Use `--one_model_all_sites` for better generalization across different locations
3. Compare different models to find the best architecture for your specific needs
4. Visualize the data before training to understand traffic patterns
5. Experiment with different sequence lengths to capture relevant patterns

## Route Guidance Functionality

Once your models are trained, you can use the system to provide route guidance based on predicted traffic flows:

```bash
python boroondara_main.py route --origin 2000 --destination 3002 --time "2024-10-15 08:00"
```

This will provide up to five routes from the origin to the destination, with estimated travel times based on the predicted traffic conditions.

### Route Options:
- `--origin` - Origin SCATS site number
- `--destination` - Destination SCATS site number
- `--time` - Departure time (format: "YYYY-MM-DD HH:MM")
- `--max_routes` - Maximum number of routes to return (default: 5)
- `--model` - Model to use for predictions (default: "transformer")

## Team

- Le Quang Hai (Team Leader)
- Thai Duong Bao Tan (Main Developer)
- Trinh Quy Khang (Documentation Lead)
- Tran Thai Duy Khang (Integration Specialist)
