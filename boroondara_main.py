"""
Main entry point for Boroondara Traffic Flow Prediction System.
"""
import sys
from src.utils.cli_parser import parse_args
from src.boroondara import (
    preprocess_command,
    train_command,
    evaluate_command,
    visualize_command
)
from src.utils.console import print_warning


def main():
    """Main entry point for the application."""
    # Parse command-line arguments
    args = parse_args()

    # Execute the requested command
    if args.command == "preprocess":
        preprocess_command(args)
    elif args.command == "train":
        train_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "visualize":
        visualize_command(args)
    else:
        print_warning("Please specify a command: preprocess, train, evaluate, or visualize")
        print_warning("Example: python boroondara_main.py train --site_id 0970")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
