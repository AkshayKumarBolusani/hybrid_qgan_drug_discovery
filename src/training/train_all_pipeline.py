"""Master training pipeline."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.train_gan import train_gan
from src.training.train_qsar import train_qsar
from src.training.train_toxicity import train_toxicity
from src.utils import get_logger

logger = get_logger(__name__)

def train_all_pipeline():
    """Run complete training pipeline."""
    logger.info("=" * 80)
    logger.info("STARTING COMPLETE TRAINING PIPELINE")
    logger.info("=" * 80)
    
    # Step 1: Train GAN
    logger.info("\n[1/3] Training GAN...")
    try:
        generator, discriminator = train_gan(epochs=20, batch_size=32)
        logger.info("✓ GAN training complete")
    except Exception as e:
        logger.error(f"GAN training failed: {e}")
    
    # Step 2: Train QSAR
    logger.info("\n[2/3] Training QSAR model...")
    try:
        qsar_model, metrics = train_qsar(model_type='rf')
        logger.info(f"✓ QSAR training complete - R2: {metrics.get('r2', 0):.3f}")
    except Exception as e:
        logger.error(f"QSAR training failed: {e}")
    
    # Step 3: Train Toxicity
    logger.info("\n[3/3] Training toxicity model...")
    try:
        tox_model = train_toxicity()
        logger.info("✓ Toxicity training complete")
    except Exception as e:
        logger.error(f"Toxicity training failed: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 80)

if __name__ == '__main__':
    train_all_pipeline()
