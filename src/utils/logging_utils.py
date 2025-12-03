"""
Logging utilities for the Hybrid Quantum GAN Drug Discovery project.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get an existing logger or create a basic one."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Set up basic console logging
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class TrainingLogger:
    """Logger specifically for training progress."""
    
    def __init__(
        self,
        name: str,
        log_dir: str,
        log_every: int = 100
    ):
        self.logger = setup_logger(name, log_dir)
        self.log_every = log_every
        self.step = 0
        self.epoch = 0
        self.metrics_history = {}
    
    def log_step(self, metrics: dict, step: Optional[int] = None):
        """Log metrics for a training step."""
        if step is not None:
            self.step = step
        else:
            self.step += 1
        
        # Store metrics
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append((self.step, value))
        
        # Log if needed
        if self.step % self.log_every == 0:
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.logger.info(f"Step {self.step} | {metrics_str}")
    
    def log_epoch(self, epoch: int, metrics: dict):
        """Log metrics for an epoch."""
        self.epoch = epoch
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch} | {metrics_str}")
    
    def log_message(self, message: str, level: str = "info"):
        """Log a general message."""
        log_func = getattr(self.logger, level.lower(), self.logger.info)
        log_func(message)
    
    def get_history(self, metric: str) -> list:
        """Get history for a specific metric."""
        return self.metrics_history.get(metric, [])
    
    def save_history(self, filepath: str):
        """Save metrics history to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)


class ExperimentLogger:
    """Logger for experiment tracking."""
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str,
        config: Optional[dict] = None
    ):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger(
            experiment_name,
            str(self.log_dir),
            log_to_file=True
        )
        
        # Save config
        if config:
            self.save_config(config)
        
        self.start_time = datetime.now()
        self.logger.info(f"Experiment '{experiment_name}' started at {self.start_time}")
    
    def save_config(self, config: dict):
        """Save experiment configuration."""
        import json
        config_path = self.log_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def log_artifact(self, name: str, artifact: any):
        """Log an artifact (model, data, etc.)."""
        import pickle
        artifact_path = self.log_dir / f"{name}.pkl"
        with open(artifact_path, 'wb') as f:
            pickle.dump(artifact, f)
        self.logger.info(f"Saved artifact: {name}")
    
    def log_figure(self, name: str, fig):
        """Log a matplotlib figure."""
        fig_path = self.log_dir / f"{name}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"Saved figure: {name}")
    
    def finish(self):
        """Mark experiment as finished."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        self.logger.info(f"Experiment finished. Duration: {duration}")
