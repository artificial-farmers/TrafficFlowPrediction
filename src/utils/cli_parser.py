"""
Command-line argument parsing utilities for Boroondara Traffic Flow Prediction System.
"""
import argparse
from typing import Any


def parse_args() -> Any:
    """Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Boroondara Traffic Flow Prediction System"
    )

    # Main command
    subparsers = parser.add_subparsers(
        dest="command",
        help="Command to execute"
    )

    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--scats_data",
        type=str,
        default="data/raw/Scats Data October 2006.csv",
        help="Path to SCATS data CSV file"
    )
    common_parser.add_argument(
        "--metadata",
        type=str,
        default="data/raw/SCATSSiteListingSpreadsheet_VicRoads.csv",
        help="Path to SCATS metadata CSV file"
    )
    common_parser.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed/boroondara",
        help="Directory for processed data"
    )
    common_parser.add_argument(
        "--model_dir",
        type=str,
        default="saved_models/boroondara",
        help="Directory for saved models"
    )
    common_parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/boroondara",
        help="Directory for TensorBoard logs"
    )
    common_parser.add_argument(
        "--results_dir",
        type=str,
        default="results/boroondara",
        help="Directory for results"
    )
    common_parser.add_argument(
        "--seq_length",
        type=int,
        default=12,
        help="Sequence length for prediction"
    )
    common_parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Models to use (comma-separated list: lstm,gru,saes,transformer,all)"
    )
    common_parser.add_argument(
        "--site_id",
        type=str,
        default=None,
        help="Specific SCATS site ID"
    )
    common_parser.add_argument(
        "--max_sites",
        type=int,
        default=None,
        help="Maximum number of sites to process"
    )
    common_parser.add_argument(
        "--force",
        action="store_true",
        help="Force operation even if files already exist"
    )

    # Preprocess command
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Preprocess data",
        parents=[common_parser]
    )
    preprocess_parser.add_argument(
        "--combined",
        action="store_true",
        help="Create combined dataset from all sites"
    )

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train models",
        parents=[common_parser]
    )
    train_parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    train_parser.add_argument(
        "--pretraining_epochs",
        type=int,
        default=50,
        help="Number of pretraining epochs for SAE"
    )
    train_parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Patience for early stopping"
    )
    train_parser.add_argument(
        "--combined_model",
        action="store_true",
        help="Train a combined model using data from all sites"
    )
    train_parser.add_argument(
        "--train_all_sites",
        action="store_true",
        help="Train separate models for each site"
    )
    train_parser.add_argument(
        "--one_model_all_sites",
        action="store_true",
        help="Train one model using data from all sites without combining them first"
    )

    # Evaluate command
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate models",
        parents=[common_parser]
    )
    evaluate_parser.add_argument(
        "--evaluate_all_sites",
        action="store_true",
        help="Evaluate on all available sites"
    )

    # Visualize command
    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Create data visualizations",
        parents=[common_parser]
    )

    return parser.parse_args()


def resolve_model_types(models_arg: str) -> list:
    """Resolve model types from command-line argument.

    Args:
        models_arg: Models argument string (e.g., "all", "lstm,gru")

    Returns:
        List of model types to use
    """
    if models_arg.lower() == 'all':
        return ['lstm', 'gru', 'saes', 'transformer']

    return [model_type.strip().lower() for model_type in models_arg.split(',')]
