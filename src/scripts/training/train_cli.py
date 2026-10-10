"""
CLI entry point for training first-stage autoencoders.
"""

import sys
from pathlib import Path
import argparse
import yaml

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from training import Trainer


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Train first-stage autoencoder (KL or VQ)'
    )
    parser.add_argument(
        'config',
        type=str,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--diagnostics',
        action='store_true',
        help='Enable memory diagnostics'
    )
    parser.add_argument(
        '--diagnostic-log',
        type=str,
        default=None,
        help='Path for diagnostic log file'
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get model type
    model_type = config['model']['type']
    
    # Setup diagnostic log path
    diagnostic_log = None
    if args.diagnostics:
        if args.diagnostic_log:
            diagnostic_log = Path(args.diagnostic_log)
        else:
            diagnostic_log = Path(config['paths']['log_dir']) / 'diagnostics.log'
    
    # Create trainer
    trainer = Trainer(
        config=config,
        model_type=model_type,
        enable_diagnostics=args.diagnostics,
        diagnostic_log_path=diagnostic_log
    )
    
    # Train
    results = trainer.train()
    
    print("\nTraining Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
