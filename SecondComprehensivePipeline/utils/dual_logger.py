"""
Dual Logger - Logs to both WandB (offline mode) and TensorBoard simultaneously
All metrics are saved locally and work without internet connection.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class DualLogger:
    """
    Unified logger that writes to both WandB (offline) and TensorBoard simultaneously.
    Ensures all metrics are saved locally (offline mode) - works without internet connection.
    WandB logs can be synced later using: wandb sync <run_dir>
    """

    def __init__(self, tensorboard_dir: str, wandb_dir: Optional[str] = None, use_wandb: bool = True):
        """
        Initialize dual logger.

        Args:
            tensorboard_dir: Directory to save TensorBoard logs
            wandb_dir: Directory for WandB offline logs (default: tensorboard_dir/../wandb)
            use_wandb: Whether to also log to WandB in offline mode (default: True)
        """
        self.use_wandb = use_wandb
        self.tensorboard_dir = Path(tensorboard_dir)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

        # Set WandB directory
        if wandb_dir is None:
            self.wandb_dir = self.tensorboard_dir.parent / "wandb"
        else:
            self.wandb_dir = Path(wandb_dir)
        self.wandb_dir.mkdir(parents=True, exist_ok=True)

        # Initialize TensorBoard writer
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
            logger.info(f"✓ TensorBoard logging initialized at: {self.tensorboard_dir}")
        except ImportError:
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")
            self.tb_writer = None

        # Check WandB availability and set offline mode
        self.wandb = None
        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb
                # Force offline mode - works without internet connection
                os.environ["WANDB_MODE"] = "offline"
                os.environ["WANDB_DIR"] = str(self.wandb_dir)
                logger.info(f"✓ WandB logging enabled (OFFLINE mode) at: {self.wandb_dir}")
                logger.info("  Logs can be synced later with: wandb sync <run_dir>")
            except ImportError:
                logger.warning("WandB not available. Only TensorBoard will be used.")
                self.use_wandb = False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
        """
        Log metrics to both WandB and TensorBoard.

        Args:
            metrics: Dictionary of metric_name -> value
            step: Global step/iteration number (optional)
            commit: Whether to commit the log (WandB only)
        """
        # Log to TensorBoard
        if self.tb_writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if step is not None:
                        self.tb_writer.add_scalar(key, value, step)
                    else:
                        self.tb_writer.add_scalar(key, value)

        # Log to WandB
        if self.use_wandb and self.wandb is not None and self.wandb.run is not None:
            try:
                if step is not None:
                    self.wandb.log(metrics, step=step, commit=commit)
                else:
                    self.wandb.log(metrics, commit=commit)
            except Exception as e:
                logger.warning(f"Failed to log to WandB: {e}")

    def log_text(self, tag: str, text: str, step: Optional[int] = None):
        """
        Log text to both systems.

        Args:
            tag: Tag/name for the text
            text: Text content to log
            step: Global step (optional)
        """
        # Log to TensorBoard
        if self.tb_writer is not None:
            if step is not None:
                self.tb_writer.add_text(tag, text, step)
            else:
                self.tb_writer.add_text(tag, text)

        # Log to WandB (as summary)
        if self.use_wandb and self.wandb is not None and self.wandb.run is not None:
            try:
                self.wandb.run.summary[tag] = text
            except Exception as e:
                logger.warning(f"Failed to log text to WandB: {e}")

    def flush(self):
        """Flush both loggers."""
        if self.tb_writer is not None:
            self.tb_writer.flush()

    def close(self):
        """Close both loggers."""
        if self.tb_writer is not None:
            self.tb_writer.close()
            logger.info("TensorBoard writer closed")


# Global dual logger instance
_dual_logger: Optional[DualLogger] = None


def init_dual_logger(tensorboard_dir: str, use_wandb: bool = True) -> DualLogger:
    """
    Initialize the global dual logger.

    Args:
        tensorboard_dir: Directory for TensorBoard logs
        use_wandb: Whether to use WandB

    Returns:
        DualLogger instance
    """
    global _dual_logger
    # Fix: use keyword argument to avoid passing bool as wandb_dir
    _dual_logger = DualLogger(tensorboard_dir, use_wandb=use_wandb)
    return _dual_logger


def get_dual_logger() -> Optional[DualLogger]:
    """Get the global dual logger instance."""
    return _dual_logger


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
    """
    Convenience function to log metrics using the global dual logger.
    Falls back to WandB-only logging if dual logger not initialized.

    Args:
        metrics: Dictionary of metric_name -> value
        step: Global step (optional)
        commit: Whether to commit (WandB only)
    """
    if _dual_logger is not None:
        _dual_logger.log(metrics, step=step, commit=commit)
    else:
        # Fallback to WandB if available
        try:
            import wandb
            if wandb.run is not None:
                if step is not None:
                    wandb.log(metrics, step=step, commit=commit)
                else:
                    wandb.log(metrics, commit=commit)
        except ImportError:
            logger.warning("Neither dual logger nor WandB available for logging")
