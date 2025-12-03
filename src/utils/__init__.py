"""
Utility modules for the Hybrid Quantum GAN Drug Discovery project.
"""

from .logging_utils import (
    setup_logger,
    get_logger,
    TrainingLogger,
    ExperimentLogger,
)
from .config_utils import (
    load_yaml,
    save_yaml,
    load_config,
    load_all_configs,
    merge_configs,
    get_project_root,
    get_config,
    Config,
)
from .metrics_utils import (
    compute_regression_metrics,
    compute_classification_metrics,
    compute_validity,
    compute_uniqueness,
    compute_novelty,
    compute_diversity,
    compute_all_generation_metrics,
    compute_property_statistics,
    compute_docking_statistics,
)
from .visualization_utils import (
    setup_plotting_style,
    plot_training_curves,
    plot_loss_curves,
    plot_property_distribution,
    plot_metrics_radar,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_molecule_grid,
    plot_docking_scores,
    plot_shap_summary,
    fig_to_base64,
)

__all__ = [
    # Logging
    'setup_logger',
    'get_logger',
    'TrainingLogger',
    'ExperimentLogger',
    # Config
    'load_yaml',
    'save_yaml',
    'load_config',
    'load_all_configs',
    'merge_configs',
    'get_project_root',
    'get_config',
    'Config',
    # Metrics
    'compute_regression_metrics',
    'compute_classification_metrics',
    'compute_validity',
    'compute_uniqueness',
    'compute_novelty',
    'compute_diversity',
    'compute_all_generation_metrics',
    'compute_property_statistics',
    'compute_docking_statistics',
    # Visualization
    'setup_plotting_style',
    'plot_training_curves',
    'plot_loss_curves',
    'plot_property_distribution',
    'plot_metrics_radar',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_molecule_grid',
    'plot_docking_scores',
    'plot_shap_summary',
    'fig_to_base64',
]
