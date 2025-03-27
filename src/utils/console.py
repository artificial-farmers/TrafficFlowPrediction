"""
Console formatting utilities for Boroondara Traffic Flow Prediction System.
"""
import sys
from typing import Optional


class ConsoleColors:
    """ANSI color codes for console output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(message: str, char: str = "=", width: int = 70) -> None:
    """Print a formatted header message.

    Args:
        message: Header message to print
        char: Character to use for the header line
        width: Width of the header line
    """
    print(f"\n{ConsoleColors.HEADER}{char * width}{ConsoleColors.ENDC}")
    print(f"{ConsoleColors.BOLD}{message}{ConsoleColors.ENDC}")
    print(f"{ConsoleColors.HEADER}{char * width}{ConsoleColors.ENDC}")


def print_section(message: str, char: str = "-", width: int = 50) -> None:
    """Print a formatted section header message.

    Args:
        message: Section message to print
        char: Character to use for the section line
        width: Width of the section line
    """
    print(f"\n{ConsoleColors.BLUE}{char * width}{ConsoleColors.ENDC}")
    print(f"{ConsoleColors.BLUE}{message}{ConsoleColors.ENDC}")
    print(f"{ConsoleColors.BLUE}{char * width}{ConsoleColors.ENDC}")


def print_success(message: str) -> None:
    """Print a success message.

    Args:
        message: Success message to print
    """
    print(f"{ConsoleColors.GREEN}{message}{ConsoleColors.ENDC}")


def print_warning(message: str) -> None:
    """Print a warning message.

    Args:
        message: Warning message to print
    """
    print(f"{ConsoleColors.WARNING}Warning: {message}{ConsoleColors.ENDC}")


def print_error(message: str) -> None:
    """Print an error message.

    Args:
        message: Error message to print
    """
    print(f"{ConsoleColors.FAIL}Error: {message}{ConsoleColors.ENDC}")


def print_info(message: str) -> None:
    """Print an informational message.

    Args:
        message: Informational message to print
    """
    print(f"{ConsoleColors.CYAN}{message}{ConsoleColors.ENDC}")


def print_progress(
    iteration: int,
    total: int,
    prefix: str = '',
    suffix: str = '',
    decimals: int = 1,
    length: int = 50,
    fill: str = '█',
    print_end: str = "\r"
) -> None:
    """Print a progress bar.

    Args:
        iteration: Current iteration
        total: Total iterations
        prefix: Prefix string
        suffix: Suffix string
        decimals: Decimal places in percent complete
        length: Character length of bar
        fill: Bar fill character
        print_end: End character (e.g. "\r", "\n")
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()

    # Print new line on complete
    if iteration == total:
        print()
