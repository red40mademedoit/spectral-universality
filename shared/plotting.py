"""
Consistent plotting style for Spectral Universality Project.

Usage:
    from shared.plotting import setup_style, COLORS
    setup_style()
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ── Color Palette ────────────────────────────────────────────

COLORS = {
    'poisson': '#2196F3',      # Blue
    'goe': '#FF9800',          # Orange
    'gue': '#E91E63',          # Pink/Red
    'intermediate': '#9E9E9E', # Grey
    'signal': '#4CAF50',       # Green
    'noise': '#BDBDBD',        # Light grey
    'accent': '#673AB7',       # Purple
}

DOMAIN_COLORS = {
    'primes': '#E91E63',
    'embeddings': '#2196F3',
    'proteins': '#4CAF50',
    'photosynthesis': '#FF9800',
}


def setup_style():
    """Apply consistent style to all plots."""
    plt.rcParams.update({
        'figure.figsize': (8, 5),
        'figure.dpi': 150,
        'font.family': 'serif',
        'font.size': 11,
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
        'savefig.bbox': 'tight',
        'savefig.dpi': 300,
    })


def plot_spacing_histogram(spacings, ax=None, label=None, bins=50):
    """Plot spacing ratio histogram with theoretical curves."""
    if ax is None:
        fig, ax = plt.subplots()
    
    from .rmt_utils import R_POISSON, R_GOE, R_GUE
    
    ax.hist(spacings, bins=bins, density=True, alpha=0.6, 
            color=COLORS['signal'], label=label or 'Data')
    
    # Theoretical Poisson: P(r) = 2/(1+r)^2
    r = np.linspace(0, 1, 200)
    ax.plot(r, 2 / (1 + r)**2, '--', color=COLORS['poisson'], 
            label=f'Poisson (⟨r⟩={R_POISSON:.3f})')
    
    ax.axvline(np.mean(spacings), color='k', linestyle='-', alpha=0.5,
               label=f'⟨r⟩ = {np.mean(spacings):.4f}')
    ax.axvline(R_GOE, color=COLORS['goe'], linestyle=':', alpha=0.7,
               label=f'GOE ({R_GOE})')
    ax.axvline(R_GUE, color=COLORS['gue'], linestyle=':', alpha=0.7,
               label=f'GUE ({R_GUE})')
    
    ax.set_xlabel('Spacing ratio r')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    
    return ax
