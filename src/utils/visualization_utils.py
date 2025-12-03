"""
Visualization utilities for plotting and displaying results.
"""

import io
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def setup_plotting_style():
    """Set up consistent plotting style."""
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
    })


def plot_training_curves(
    metrics_history: Dict[str, List[Tuple[int, float]]],
    title: str = "Training Progress",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot training curves for multiple metrics.
    
    Args:
        metrics_history: Dict of metric name -> list of (step, value)
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for metric_name, values in metrics_history.items():
        if values:
            steps, metric_values = zip(*values)
            ax.plot(steps, metric_values, label=metric_name, linewidth=2)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_loss_curves(
    g_losses: List[float],
    d_losses: List[float],
    title: str = "GAN Training Losses",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot generator and discriminator loss curves.
    
    Args:
        g_losses: Generator losses
        d_losses: Discriminator losses
        title: Plot title
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(g_losses, label='Generator Loss', color='blue', alpha=0.7)
    ax.plot(d_losses, label='Discriminator Loss', color='red', alpha=0.7)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_property_distribution(
    values: List[float],
    property_name: str,
    reference_values: Optional[List[float]] = None,
    bins: int = 30,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot property distribution histogram.
    
    Args:
        values: Property values to plot
        property_name: Name of the property
        reference_values: Optional reference distribution
        bins: Number of bins
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter NaN values
    values = [v for v in values if v is not None and not np.isnan(v)]
    
    ax.hist(values, bins=bins, alpha=0.7, label='Generated', color='blue', density=True)
    
    if reference_values:
        reference_values = [v for v in reference_values if v is not None and not np.isnan(v)]
        ax.hist(reference_values, bins=bins, alpha=0.5, label='Reference', color='gray', density=True)
    
    ax.set_xlabel(property_name)
    ax.set_ylabel('Density')
    ax.set_title(f'{property_name} Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_metrics_radar(
    metrics: Dict[str, float],
    title: str = "Model Performance",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a radar/spider plot for metrics.
    
    Args:
        metrics: Dictionary of metric name -> value (0-1 scale)
        title: Plot title
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    # Number of variables
    N = len(categories)
    
    # Create angles for radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the loop
    values += values[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    ax.plot(angles, values, 'o-', linewidth=2, color='blue')
    ax.fill(angles, values, alpha=0.25, color='blue')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title(title, y=1.08)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional class names
        title: Plot title
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_roc_curve(
    y_true: List[int],
    y_prob: List[float],
    title: str = "ROC Curve",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        title: Plot title
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_molecule_grid(
    smiles_list: List[str],
    labels: Optional[List[str]] = None,
    mols_per_row: int = 4,
    img_size: Tuple[int, int] = (300, 300),
    save_path: Optional[str] = None
) -> Optional[Any]:
    """
    Create a grid of molecule images.
    
    Args:
        smiles_list: List of SMILES strings
        labels: Optional labels for each molecule
        mols_per_row: Number of molecules per row
        img_size: Image size for each molecule
        save_path: Optional save path
        
    Returns:
        PIL Image or None if rdkit not available
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        
        mols = []
        valid_labels = []
        
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            if mol is not None:
                mols.append(mol)
                if labels:
                    valid_labels.append(labels[i] if i < len(labels) else "")
        
        if not mols:
            return None
        
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=mols_per_row,
            subImgSize=img_size,
            legends=valid_labels if valid_labels else None
        )
        
        if save_path:
            img.save(save_path)
        
        return img
        
    except ImportError:
        print("RDKit not available for molecule visualization")
        return None


def fig_to_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str


def plot_docking_scores(
    scores: List[float],
    labels: Optional[List[str]] = None,
    title: str = "Docking Scores",
    threshold: float = -6.0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot docking scores as bar chart.
    
    Args:
        scores: List of docking scores
        labels: Optional molecule labels
        title: Plot title
        threshold: Score threshold for "hits"
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(scores))
    colors = ['green' if s < threshold else 'gray' for s in scores]
    
    ax.bar(x, scores, color=colors, alpha=0.7)
    ax.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
    
    if labels:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
    
    ax.set_xlabel('Molecule')
    ax.set_ylabel('Docking Score (kcal/mol)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_shap_summary(
    shap_values: np.ndarray,
    feature_names: List[str],
    max_features: int = 20,
    title: str = "SHAP Feature Importance",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a custom SHAP summary plot.
    
    Args:
        shap_values: SHAP values array
        feature_names: Feature names
        max_features: Maximum features to show
        title: Plot title
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    # Calculate mean absolute SHAP values
    mean_shap = np.abs(shap_values).mean(axis=0)
    
    # Sort by importance
    indices = np.argsort(mean_shap)[-max_features:]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.barh(range(len(indices)), mean_shap[indices], color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
