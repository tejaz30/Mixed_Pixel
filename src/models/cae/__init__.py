"""
Convolutional Autoencoder (CAE) module.

Baseline CAE implementation for thermal image mixed-pixel detection
as described in Section 4.1 of the paper.

Main Components:
    - ConvolutionalAutoencoder: Main CAE model
    - CAEEncoder: Encoder network
    - CAEDecoder: Decoder network
    - CAELoss: Loss functions
    - CAETrainer: Training pipeline
    - CAEInference: Inference handler

Usage:
    from models.cae import ConvolutionalAutoencoder, CAETrainer, CAEInference
    
    # Create model
    model = ConvolutionalAutoencoder()
    
    # Train
    trainer = CAETrainer(model, device, checkpoint_dir)
    trainer.train(train_loader, val_loader)
    
    # Inference
    inference = CAEInference(model, device, checkpoint_path)
    results = inference.predict(images)
"""

from .model import ConvolutionalAutoencoder, create_cae
from .encoder import CAEEncoder
from .decoder import CAEDecoder
from .loss import CAELoss, ReconstructionLoss, compute_reconstruction_metrics
from .trainer import CAETrainer, EarlyStopping
from .inference import CAEInference, load_model_for_inference, visualize_reconstruction
from .utils import (
    get_device,
    count_parameters,
    load_config,
    save_config,
    setup_logging,
    get_model_size,
    create_directories,
    find_latest_checkpoint,
    validate_input_shape,
    get_checkpoint_info,
    format_time,
    seed_everything
)

__all__ = [
    # Model
    'ConvolutionalAutoencoder',
    'create_cae',
    'CAEEncoder',
    'CAEDecoder',
    
    # Loss
    'CAELoss',
    'ReconstructionLoss',
    'compute_reconstruction_metrics',
    
    # Training
    'CAETrainer',
    'EarlyStopping',
    
    # Inference
    'CAEInference',
    'load_model_for_inference',
    'visualize_reconstruction',
    
    # Utils
    'get_device',
    'count_parameters',
    'load_config',
    'save_config',
    'setup_logging',
    'get_model_size',
    'create_directories',
    'find_latest_checkpoint',
    'validate_input_shape',
    'get_checkpoint_info',
    'format_time',
    'seed_everything'
]

__version__ = '1.0.0'
