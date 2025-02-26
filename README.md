# Traffic Flow Prediction

Traffic Flow Prediction with Neural Networks (LSTM, GRU, SAE).

## Project Structure

```
traffiflow_prediction/
├── data/
│   ├── processed/      # Processed data files
│   ├── raw/            # Original data files
│   │   ├── test.csv
│   │   └── train.csv
│   └── preprocessing.py # Data processing utilities
├── notebooks/          # Jupyter notebooks for exploration
├── saved_models/       # Trained model files
├── outputs/            # Output files and visualizations
├── scripts/            # Command-line scripts
│   └── train.py        # Script for training models
├── src/                # Source code
│   ├── models/         # Model implementations
│   │   ├── base.py     # Base model class
│   │   ├── lstm.py     # LSTM model
│   │   ├── gru.py      # GRU model
│   │   └── saes.py     # SAE model
│   └── utils/          # Utility functions
│       ├── evaluation.py    # Evaluation metrics
│       └── visualization.py # Plotting functions
├── .gitignore          # Git ignore file
├── LICENSE             # License file
├── main.py             # Main evaluation script
├── pyproject.toml      # Project configuration
├── README.md           # Project documentation
└── requirements.txt    # Project dependencies
```

## Requirements

- Python 3.10+
- TensorFlow 2.10+
- scikit-learn 1.1+
- pandas 1.5+
- numpy 1.23+
- matplotlib 3.6+

## Setup

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/TrafficFlowPrediction.git
   cd TrafficFlowPrediction
   ```

2. Create a virtual environment:

   ```
   python -m venv venv
   ```

3. Activate the virtual environment:

   - Windows:
     ```
     venv\Scripts\activate
     ```
   - Linux/macOS:
     ```
     source venv/bin/activate
     ```

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Train Models

To train a model:

```
python scripts/train.py --model lstm
```

Available options:

- `--model`: Model type to train (`lstm`, `gru`, or `saes`)
- `--train_file`: Path to training data CSV
- `--test_file`: Path to test data CSV
- `--lag`: Number of lag observations (default: 12)
- `--batch_size`: Training batch size (default: 256)
- `--epochs`: Number of training epochs (default: 600)
- `--output_dir`: Directory to save trained models

## Evaluate Models

To evaluate and compare trained models:

```
python main.py
```

Available options:

- `--train_file`: Path to training data CSV
- `--test_file`: Path to test data CSV
- `--lag`: Number of lag observations
- `--models_dir`: Directory containing trained models

## Data

Data are obtained from the Caltrans Performance Measurement System (PeMS). The dataset contains 5-minute interval traffic flow data.

## Results

| Metrics | MAE  |  MSE  | RMSE |  MAPE  |   R²   | Explained variance score |
| ------- | :--: | :---: | :--: | :----: | :----: | :----------------------: |
| LSTM    | 7.21 | 98.05 | 9.90 | 16.56% | 0.9396 |          0.9419          |
| GRU     | 7.20 | 99.32 | 9.97 | 16.78% | 0.9389 |          0.9389          |
| SAEs    | 7.06 | 92.08 | 9.60 | 17.80% | 0.9433 |          0.9442          |

## References

```
@article{SAEs,
  title={Traffic Flow Prediction With Big Data: A Deep Learning Approach},
  author={Y Lv, Y Duan, W Kang, Z Li, FY Wang},
  journal={IEEE Transactions on Intelligent Transportation Systems, 2015, 16(2):865-873},
  year={2015}
}

@article{RNN,
  title={Using LSTM and GRU neural network methods for traffic flow prediction},
  author={R Fu, Z Zhang, L Li},
  journal={Chinese Association of Automation, 2017:324-328},
  year={2017}
}
```

## License

See [LICENSE](LICENSE) for details.
