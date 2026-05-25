#!/usr/bin/env python3
"""
Quick Start Script for SignalForge

Sets up everything in one command.

Usage:
    python scripts/quickstart.py --dataset telco
"""

import argparse
import subprocess
import sys
from pathlib import Path

from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")


def check_prerequisites():
    """Check if all prerequisites are installed."""
    logger.info("Checking prerequisites...")

    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 11):
        logger.error(f"Python 3.11+ required, found {python_version.major}.{python_version.minor}")
        return False

    logger.success(f"✓ Python {python_version.major}.{python_version.minor}")

    # Check required packages
    required = ['pandas', 'numpy', 'sklearn', 'loguru']
    missing = []

    for package in required:
        try:
            __import__(package)
            logger.success(f"✓ {package}")
        except ImportError:
            missing.append(package)
            logger.warning(f"✗ {package} (missing)")

    if missing:
        logger.info("\nInstall missing packages:")
        logger.info(f"pip install {' '.join(missing)}")
        return False

    # Check Kaggle (optional)
    try:
        subprocess.run(['kaggle', '--version'], capture_output=True, text=True, check=False)
        logger.success("✓ Kaggle CLI")
    except FileNotFoundError:
        logger.warning("✗ Kaggle CLI (optional, for downloading datasets)")
        logger.info("Install with: pip install kaggle")
        logger.info("Get API key from: https://www.kaggle.com/settings")

    return True


def setup_directories():
    """Create necessary directories."""
    logger.info("\nSetting up directories...")

    dirs = [
        'data/raw',
        'data/processed',
        'models/artifacts',
        'logs',
        'notebooks'
    ]

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        logger.success(f"✓ Created {d}/")


def download_dataset(dataset: str):
    """Download dataset from Kaggle."""
    logger.info(f"\n📥 Downloading {dataset} dataset...")

    cmd = [sys.executable, 'scripts/download_real_data.py', '--source', dataset]

    try:
        subprocess.run(cmd, check=True)
        logger.success(f"✓ Downloaded {dataset} dataset")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to download dataset: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="SignalForge Quick Start")
    parser.add_argument('--dataset', default='telco',
                       choices=['telco', 'bank', 'saas', 'cell2cell'],
                       help='Dataset to download')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip prerequisite checks')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("SignalForge Quick Start")
    logger.info("=" * 80)

    # Check prerequisites
    if not args.skip_checks:
        if not check_prerequisites():
            logger.error("\nPrerequisites not met. Please install missing dependencies.")
            return

    # Setup directories
    setup_directories()

    # Download dataset
    if not download_dataset(args.dataset):
        logger.error("\nSetup failed. Check Kaggle configuration.")
        return

    # Success message
    logger.info("\n" + "=" * 80)
    logger.success("🎉 Setup Complete!")
    logger.info("=" * 80)

    logger.info("\n📊 Next Steps:")
    logger.info("\n1. Explore the data:")
    logger.info("   jupyter notebook notebooks/01_eda_telco.ipynb")

    logger.info("\n2. Engineer features:")
    logger.info("   python scripts/engineer_real_features.py")

    logger.info("\n3. Train models (Optuna-tuned, with statistical analysis):")
    logger.info("   python scripts/train_with_optuna.py")

    logger.info("\n4. Launch the dashboard:")
    logger.info("   streamlit run src/app/dashboard.py")

    logger.info("\n📚 Documentation:")
    logger.info("   - docs/SETUP.md - Full setup guide")
    logger.info("   - docs/DATASETS.md - Dataset comparison")

    logger.success("\n🚀 Ready to build!")


if __name__ == "__main__":
    main()
