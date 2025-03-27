"""
Main functionality module for Boroondara Traffic Flow Prediction System.
"""
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from typing import List, Dict, Optional, Any, Tuple

from src.utils.console import (
    print_header,
    print_section,
    print_success,
    print_warning,
    print_error,
    print_info
)
from src.utils.data_processor import (
    preprocess_data,
    prepare_combined_dataset,
    load_site_data,
    load_combined_data,
    get_available_sites,
    get_site_location
)
from src.utils.model_utils import (
    setup_model_directories,
    setup_tensorflow,
    train_model,
    load_models
)
from src.utils.evaluation_utils import (
    evaluate_site_model,
    create_model_summary,
    visualize_training_history
)
from config.boroondara_config import get_all_configs
from src.utils.visualization import plot_traffic_patterns


def preprocess_command(args: Any) -> None:
    """Execute the preprocess command.

    Args:
        args: Command-line arguments
    """
    print_header("Preprocessing Boroondara Traffic Data")

    # Process site-specific data
    preprocess_data(
        args.scats_data,
        args.metadata,
        args.processed_dir,
        args.seq_length,
        args.force
    )

    # Process combined data if requested
    if args.combined:
        print_section("Creating Combined Dataset")
        prepare_combined_dataset(
            args.scats_data,
            args.metadata,
            args.processed_dir,
            args.seq_length,
            args.force
        )

    print_success("Preprocessing completed successfully!")


def train_command(args: Any) -> None:
    """Execute the train command.

    Args:
        args: Command-line arguments
    """
    print_header("Training Traffic Flow Prediction Models")

    # Setup environment
    setup_tensorflow()
    setup_model_directories(args.model_dir, args.log_dir, args.results_dir)

    # Get model configurations
    configs = get_all_configs()

    # Determine which models to train
    from src.utils.cli_parser import resolve_model_types
    models_to_train = resolve_model_types(args.models)

    # Training strategy
    if args.site_id:
        # Train models for a specific site
        train_site_specific_models(args, args.site_id, models_to_train, configs)
    elif args.one_model_all_sites:
        # Train one model with data from all sites
        train_one_model_all_sites(args, models_to_train, configs)
    elif args.combined_model:
        # Train a combined model with merged data
        train_combined_model(args, models_to_train, configs)
    elif args.train_all_sites:
        # Train separate models for each site
        train_all_sites(args, models_to_train, configs)
    else:
        print_warning("No training mode specified. Use --site_id, --combined_model, --one_model_all_sites, or --train_all_sites")
        return

    print_success("Training completed successfully!")


def train_site_specific_models(
    args: Any,
    site_id: str,
    models_to_train: List[str],
    configs: Dict[str, Dict[str, Any]]
) -> None:
    """Train models for a specific SCATS site.

    Args:
        args: Command-line arguments
        site_id: SCATS site ID
        models_to_train: List of model types to train
        configs: Model configurations
    """
    # Ensure site_id is a string
    site_id = str(site_id)
    print_section(f"Training Models for SCATS Site {site_id}")

    try:
        # Load site data
        X_train, y_train, X_test, y_test, scaler = load_site_data(site_id, args.processed_dir)

        # Get site location if available
        location = get_site_location(site_id, args.processed_dir)
        if location:
            print_info(f"Training models for {location}")

        print_info(f"Data shapes:")
        print_info(f"X_train: {X_train.shape}")
        print_info(f"y_train: {y_train.shape}")
        print_info(f"X_test: {X_test.shape}")
        print_info(f"y_test: {y_test.shape}")

        results = []

        # Train each requested model
        for model_type in models_to_train:
            if model_type not in configs:
                print_warning(f"Unknown model type '{model_type}'. Skipping.")
                continue

            _, result = train_model(
                model_type,
                X_train,
                y_train,
                X_test,
                y_test,
                configs[model_type],
                args,
                site_id
            )

            results.append(result)

            # Visualize training history
            visualize_training_history(
                result['history'],
                model_type,
                args.results_dir,
                site_id
            )

        # Print summary
        print_section(f"Training Summary for Site {site_id}")

        for result in results:
            print_info(f"\n{result['model_type'].upper()}:")
            print_info(f"  Test Loss: {result['loss']:.4f}")
            print_info(f"  Test MAPE: {result['mape']:.4f}")
            print_info(f"  Model saved: {result['model_path']}")

    except FileNotFoundError as e:
        print_error(f"No processed data found for site {site_id}")
        print_error(f"Run preprocessing first: python boroondara_main.py preprocess")
    except Exception as e:
        print_error(f"Error training models for site {site_id}: {str(e)}")
        import traceback
        traceback.print_exc()


def train_combined_model(
    args: Any,
    models_to_train: List[str],
    configs: Dict[str, Dict[str, Any]]
) -> None:
    """Train a combined model using merged data from all sites.

    Args:
        args: Command-line arguments
        models_to_train: List of model types to train
        configs: Model configurations
    """
    print_section("Training Combined Model")

    combined_data_path = os.path.join(args.processed_dir, "X_train.npy")

    if not os.path.exists(combined_data_path):
        print_warning("Combined data not found. Creating combined dataset...")
        prepare_combined_dataset(
            args.scats_data,
            args.metadata,
            args.processed_dir,
            args.seq_length,
            args.force
        )

    try:
        # Load combined data
        X_train, y_train, X_test, y_test = load_combined_data(args.processed_dir)

        print_info(f"Combined data shapes:")
        print_info(f"X_train: {X_train.shape}")
        print_info(f"y_train: {y_train.shape}")
        print_info(f"X_test: {X_test.shape}")
        print_info(f"y_test: {y_test.shape}")

        results = []

        # Train each requested model
        for model_type in models_to_train:
            if model_type not in configs:
                print_warning(f"Unknown model type '{model_type}'. Skipping.")
                continue

            _, result = train_model(
                model_type,
                X_train,
                y_train,
                X_test,
                y_test,
                configs[model_type],
                args
            )

            results.append(result)

            # Visualize training history
            visualize_training_history(
                result['history'],
                model_type,
                args.results_dir
            )

        # Print summary
        print_section("Training Summary for Combined Model")

        for result in results:
            print_info(f"\n{result['model_type'].upper()}:")
            print_info(f"  Test Loss: {result['loss']:.4f}")
            print_info(f"  Test MAPE: {result['mape']:.4f}")
            print_info(f"  Model saved: {result['model_path']}")

    except Exception as e:
        print_error(f"Error training combined model: {str(e)}")
        import traceback
        traceback.print_exc()


def train_one_model_all_sites(
    args: Any,
    models_to_train: List[str],
    configs: Dict[str, Dict[str, Any]]
) -> None:
    """Train one model using data from all sites without combining them first.

    Args:
        args: Command-line arguments
        models_to_train: List of model types to train
        configs: Model configurations
    """
    print_section("Training One Model with All Sites Data")

    try:
        # Get available sites
        sites = get_available_sites(args.processed_dir)

        if not sites:
            print_error("No processed site data found")
            print_error("Run preprocessing first: python boroondara_main.py preprocess")
            return

        print_info(f"Found {len(sites)} sites with processed data")

        # Limit number of sites if specified
        if args.max_sites and args.max_sites < len(sites):
            print_info(f"Limiting to {args.max_sites} sites as specified")
            sites = sites[:args.max_sites]

        # Collect data from all sites
        all_X_train, all_y_train = [], []
        all_X_test, all_y_test = [], []

        for site_id in sites:
            try:
                # Ensure site_id is a string
                site_id = str(site_id)
                X_train, y_train, X_test, y_test, _ = load_site_data(site_id, args.processed_dir)

                all_X_train.append(X_train)
                all_y_train.append(y_train)
                all_X_test.append(X_test)
                all_y_test.append(y_test)

                print_info(f"Loaded data from site {site_id}")
            except Exception as e:
                print_warning(f"Failed to load data for site {site_id}: {str(e)}")

        if not all_X_train:
            print_error("Failed to load data from any site")
            return

        # Combine data from all sites
        X_train_combined = np.vstack(all_X_train)
        y_train_combined = np.concatenate(all_y_train)
        X_test_combined = np.vstack(all_X_test)
        y_test_combined = np.concatenate(all_y_test)

        # Shuffle the training data
        shuffle_idx = np.random.permutation(len(X_train_combined))
        X_train_combined = X_train_combined[shuffle_idx]
        y_train_combined = y_train_combined[shuffle_idx]

        print_info(f"Combined data shapes:")
        print_info(f"X_train: {X_train_combined.shape}")
        print_info(f"y_train: {y_train_combined.shape}")
        print_info(f"X_test: {X_test_combined.shape}")
        print_info(f"y_test: {y_test_combined.shape}")

        results = []

        # Train each requested model
        for model_type in models_to_train:
            if model_type not in configs:
                print_warning(f"Unknown model type '{model_type}'. Skipping.")
                continue

            _, result = train_model(
                model_type,
                X_train_combined,
                y_train_combined,
                X_test_combined,
                y_test_combined,
                configs[model_type],
                args
            )

            results.append(result)

            # Save model path with a special name to indicate it's trained on all sites
            model_path = os.path.join(args.model_dir, f"{model_type}_all_sites.h5")
            os.rename(result['model_path'], model_path)
            result['model_path'] = model_path

            # Visualize training history
            visualize_training_history(
                result['history'],
                f"{model_type}_all_sites",
                args.results_dir
            )

        # Print summary
        print_section("Training Summary for All Sites Model")

        for result in results:
            print_info(f"\n{result['model_type'].upper()}:")
            print_info(f"  Test Loss: {result['loss']:.4f}")
            print_info(f"  Test MAPE: {result['mape']:.4f}")
            print_info(f"  Model saved: {result['model_path']}")

    except Exception as e:
        print_error(f"Error training model with all sites data: {str(e)}")
        import traceback
        traceback.print_exc()


def train_all_sites(
    args: Any,
    models_to_train: List[str],
    configs: Dict[str, Dict[str, Any]]
) -> None:
    """Train separate models for each available site.

    Args:
        args: Command-line arguments
        models_to_train: List of model types to train
        configs: Model configurations
    """
    print_section("Training Models for All Sites")

    try:
        # Get available sites
        sites = get_available_sites(args.processed_dir)

        if not sites:
            print_error("No processed site data found")
            print_error("Run preprocessing first: python boroondara_main.py preprocess")
            return

        print_info(f"Found {len(sites)} sites with processed data")

        # Limit number of sites if specified
        if args.max_sites and args.max_sites < len(sites):
            print_info(f"Limiting to {args.max_sites} sites as specified")
            sites = sites[:args.max_sites]

        # Train models for each site
        for site_id in sites:
            # Ensure site_id is a string
            site_id = str(site_id)
            train_site_specific_models(args, site_id, models_to_train, configs)

    except Exception as e:
        print_error(f"Error training models for all sites: {str(e)}")
        import traceback
        traceback.print_exc()


def evaluate_command(args: Any) -> None:
    """Execute the evaluate command.

    Args:
        args: Command-line arguments
    """
    print_header("Evaluating Traffic Flow Prediction Models")

    # Setup environment
    setup_tensorflow()

    # Determine which models to evaluate
    from src.utils.cli_parser import resolve_model_types
    model_types = resolve_model_types(args.models)

    # Evaluation strategy
    if args.site_id:
        # Evaluate models for a specific site
        evaluate_site_specific(args, args.site_id, model_types)
    elif args.evaluate_all_sites:
        # Evaluate on all available sites
        evaluate_all_sites(args, model_types)
    else:
        # Evaluate combined model
        evaluate_combined_model(args, model_types)

    print_success("Evaluation completed successfully!")


def evaluate_site_specific(args: Any, site_id: str, model_types: List[str]) -> None:
    """Evaluate models on a specific site.

    Args:
        args: Command-line arguments
        site_id: SCATS site ID
        model_types: List of model types to evaluate
    """
    # Ensure site_id is a string
    site_id = str(site_id)

    # Load site-specific models if they exist
    site_model_dir = os.path.join(args.model_dir, "per_site", site_id)
    if os.path.exists(site_model_dir):
        models = load_models(args.model_dir, model_types, site_id)

        if not models:
            print_warning(f"No site-specific models found for site {site_id}")
            print_warning("Falling back to combined models")
            models = load_models(args.model_dir, model_types)
    else:
        # Fallback to combined models
        print_warning(f"No site-specific models directory found for site {site_id}")
        print_warning("Using combined models")
        models = load_models(args.model_dir, model_types)

    if not models:
        print_error("No models could be loaded. Please train models first.")
        return

    try:
        # Load site data
        X_train, y_train, X_test, y_test, scaler = load_site_data(site_id, args.processed_dir)

        # Get site location
        location = get_site_location(site_id, args.processed_dir)

        # Evaluate models on this site
        evaluate_site_model(
            models,
            X_test,
            y_test,
            site_id,
            location,
            args.results_dir,
            scaler
        )

    except FileNotFoundError:
        print_error(f"No processed data found for site {site_id}")
    except Exception as e:
        print_error(f"Error evaluating site {site_id}: {str(e)}")
        import traceback
        traceback.print_exc()


def evaluate_all_sites(args: Any, model_types: List[str]) -> None:
    """Evaluate models on all available sites.

    Args:
        args: Command-line arguments
        model_types: List of model types to evaluate
    """
    try:
        # Get all available sites
        sites = get_available_sites(args.processed_dir)

        if not sites:
            print_error("No processed site data found")
            return

        print_info(f"Found {len(sites)} sites with processed data")

        # Limit number of sites if specified
        if args.max_sites and args.max_sites < len(sites):
            print_info(f"Limiting to {args.max_sites} sites as specified")
            sites = sites[:args.max_sites]

        # Evaluate each site with its own model if available
        all_results = []

        for site_id in sites:
            try:
                # Ensure site_id is a string
                site_id = str(site_id)

                # First try to load site-specific models
                site_model_dir = os.path.join(args.model_dir, "per_site", site_id)
                if os.path.exists(site_model_dir):
                    print_info(f"Using site-specific models for site {site_id}")
                    models = load_models(args.model_dir, model_types, site_id)

                    # If no site-specific models found, fall back to combined models
                    if not models:
                        print_warning(f"No site-specific models found for site {site_id}")
                        print_warning("Falling back to combined models")
                        models = load_models(args.model_dir, model_types)
                else:
                    # Fall back to combined models
                    print_warning(f"No site-specific models directory for site {site_id}")
                    print_warning("Using combined models")
                    models = load_models(args.model_dir, model_types)

                if not models:
                    print_warning(f"No models could be loaded for site {site_id}. Skipping.")
                    continue

                # Load site data
                X_train, y_train, X_test, y_test, scaler = load_site_data(site_id, args.processed_dir)

                # Get site location
                location = get_site_location(site_id, args.processed_dir)

                # Evaluate models on this site
                results = evaluate_site_model(
                    models,
                    X_test,
                    y_test,
                    site_id,
                    location,
                    args.results_dir,
                    scaler
                )

                # Add to results list
                all_results.append({
                    "site_id": site_id,
                    "location": location,
                    "results": results
                })

            except Exception as e:
                print_warning(f"Error evaluating site {site_id}: {str(e)}")

        # Create model summary
        if all_results:
            create_model_summary(all_results, args.results_dir)
        else:
            print_warning("No evaluation results to summarize")

    except Exception as e:
        print_error(f"Error evaluating all sites: {str(e)}")
        import traceback
        traceback.print_exc()

def evaluate_combined_model(args: Any, model_types: List[str]) -> None:
    """Evaluate combined model.

    Args:
        args: Command-line arguments
        model_types: List of model types to evaluate
    """
    # Load combined models
    models = load_models(args.model_dir, model_types)

    if not models:
        print_error("No models could be loaded. Please train models first.")
        return

    try:
        # Load combined test data
        combined_data_path = os.path.join(args.processed_dir, "X_test.npy")

        if not os.path.exists(combined_data_path):
            print_error("No combined data found. Run preprocessing with --combined or train a combined model first.")
            return

        X_train, y_train, X_test, y_test = load_combined_data(args.processed_dir)

        print_info(f"Test data shape: {X_test.shape}")

        # Visualize predictions without a scaler (data is already scaled)
        evaluate_site_model(
            models,
            X_test,
            y_test,
            "combined",
            "Combined Dataset",
            args.results_dir
        )

    except Exception as e:
        print_error(f"Error evaluating combined model: {str(e)}")
        import traceback
        traceback.print_exc()


def visualize_command(args: Any) -> None:
    """Execute the visualize command.

    Args:
        args: Command-line arguments
    """
    print_header("Visualizing Traffic Flow Data")

    # Create visualization directory
    vis_dir = os.path.join(args.results_dir, "data_visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    try:
        # Load SCATS data
        from data.boroondara_preprocessing import load_scats_data
        scats_df = load_scats_data(args.scats_data)
        print_info(f"Loaded {len(scats_df)} records from SCATS data")

        # Get available sites
        from data.boroondara_preprocessing import get_available_scats_sites
        available_sites = get_available_scats_sites(scats_df)
        print_info(f"Found {len(available_sites)} unique SCATS sites in the dataset")

        # Determine which sites to visualize
        sites_to_visualize = []

        if args.site_id:
            # Ensure site_id is a string
            args.site_id = str(args.site_id)

            if args.site_id in available_sites:
                sites_to_visualize = [args.site_id]
            else:
                print_warning(f"Site {args.site_id} not found in the dataset")
                return
        else:
            # Get top sites by data volume
            site_counts = scats_df['SCATS_Site'].value_counts().head(args.max_sites or 5)
            sites_to_visualize = site_counts.index.tolist()

        print_info(f"Visualizing traffic patterns for {len(sites_to_visualize)} sites")

        # Create hourly pattern visualization
        hourly_fig = plot_traffic_patterns(
            scats_df,
            sites_to_visualize,
            group_by='hour',
            title='Hourly Traffic Flow Patterns',
            save_path=os.path.join(vis_dir, "hourly_patterns.png")
        )

        # Create daily pattern visualization
        daily_fig = plot_traffic_patterns(
            scats_df,
            sites_to_visualize,
            group_by='weekday',
            title='Daily Traffic Flow Patterns',
            save_path=os.path.join(vis_dir, "daily_patterns.png")
        )

        print_success(f"Traffic pattern visualizations saved to {vis_dir}")

    except Exception as e:
        print_error(f"Error creating visualizations: {str(e)}")
        import traceback
        traceback.print_exc()
