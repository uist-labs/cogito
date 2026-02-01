#!/usr/bin/env python3
"""
COGITO Visualizer - Analyze and visualize pondering experiment results.

Generates charts showing:
- Entropy over time
- Self-reference and meta-cognitive markers
- Similarity trends
- Question frequency
- Intervention events
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("ERROR: matplotlib not installed.")
    print("Install with: pip install matplotlib --break-system-packages")
    exit(1)


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load a checkpoint file."""
    with open(checkpoint_path, 'r') as f:
        return json.load(f)


def load_all_checkpoints(log_dir: str) -> list:
    """Load all checkpoints from a log directory."""
    checkpoint_dir = Path(log_dir) / "checkpoints"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    checkpoints = []
    for f in sorted(checkpoint_dir.glob("checkpoint_*.json")):
        checkpoints.append(load_checkpoint(f))
    
    return checkpoints


def extract_metrics_series(checkpoints: list) -> dict:
    """Extract time series of metrics from checkpoints."""
    
    all_metrics = []
    for cp in checkpoints:
        all_metrics.extend(cp.get('recent_metrics', []))
    
    # Deduplicate by cycle number
    seen_cycles = set()
    unique_metrics = []
    for m in all_metrics:
        cycle = m['cycle_number']
        if cycle not in seen_cycles:
            seen_cycles.add(cycle)
            unique_metrics.append(m)
    
    unique_metrics.sort(key=lambda x: x['cycle_number'])
    
    return {
        'cycles': [m['cycle_number'] for m in unique_metrics],
        'entropy': [m['entropy'] for m in unique_metrics],
        'self_similarity': [m['self_similarity'] for m in unique_metrics],
        'cumulative_similarity': [m['cumulative_similarity'] for m in unique_metrics],
        'self_reference': [m['self_reference_count'] for m in unique_metrics],
        'meta_cognitive': [m['meta_cognitive_markers'] for m in unique_metrics],
        'temporal': [m['temporal_markers'] for m in unique_metrics],
        'questions': [m['question_count'] for m in unique_metrics],
        'token_count': [m['token_count'] for m in unique_metrics],
        'interventions': [m['intervention_applied'] for m in unique_metrics],
    }


def plot_experiment(series: dict, output_path: str = None, show: bool = True):
    """Generate visualization plots."""
    
    cycles = series['cycles']
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('COGITO Experiment Analysis', fontsize=14, fontweight='bold')
    
    # Color scheme
    colors = {
        'entropy': '#2ecc71',
        'similarity': '#e74c3c',
        'self_ref': '#3498db',
        'meta': '#9b59b6',
        'questions': '#f39c12',
        'intervention': '#e74c3c'
    }
    
    # 1. Entropy over time
    ax1 = axes[0, 0]
    ax1.plot(cycles, series['entropy'], color=colors['entropy'], linewidth=1.5)
    ax1.fill_between(cycles, series['entropy'], alpha=0.3, color=colors['entropy'])
    ax1.set_xlabel('Cycle')
    ax1.set_ylabel('Entropy')
    ax1.set_title('Token Entropy Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    if len(cycles) > 10:
        z = np.polyfit(cycles, series['entropy'], 1)
        p = np.poly1d(z)
        ax1.plot(cycles, p(cycles), '--', color='gray', alpha=0.7, label=f'Trend')
        ax1.legend()
    
    # Mark interventions
    for i, (c, interv) in enumerate(zip(cycles, series['interventions'])):
        if interv:
            ax1.axvline(x=c, color=colors['intervention'], alpha=0.3, linestyle='--')
    
    # 2. Similarity metrics
    ax2 = axes[0, 1]
    ax2.plot(cycles, series['self_similarity'], color=colors['similarity'], 
             linewidth=1.5, label='Self-similarity')
    ax2.plot(cycles, series['cumulative_similarity'], color='#e67e22',
             linewidth=1.5, alpha=0.7, label='Cumulative')
    ax2.set_xlabel('Cycle')
    ax2.set_ylabel('Similarity')
    ax2.set_title('Output Similarity (higher = more repetitive)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Self-reference and meta-cognitive markers
    ax3 = axes[1, 0]
    ax3.bar(cycles, series['self_reference'], color=colors['self_ref'], 
            alpha=0.7, label='Self-reference')
    ax3.plot(cycles, series['meta_cognitive'], color=colors['meta'], 
             linewidth=2, label='Meta-cognitive')
    ax3.set_xlabel('Cycle')
    ax3.set_ylabel('Count')
    ax3.set_title('Self-Awareness Markers')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Questions over time
    ax4 = axes[1, 1]
    ax4.bar(cycles, series['questions'], color=colors['questions'], alpha=0.7)
    ax4.set_xlabel('Cycle')
    ax4.set_ylabel('Question Count')
    ax4.set_title('Questions Asked')
    ax4.grid(True, alpha=0.3)
    
    # Calculate rolling average for questions
    if len(cycles) > 10:
        window = min(10, len(cycles) // 5)
        rolling_avg = np.convolve(series['questions'], 
                                   np.ones(window)/window, mode='valid')
        ax4.plot(cycles[window-1:], rolling_avg, color='#d35400', 
                linewidth=2, label=f'{window}-cycle avg')
        ax4.legend()
    
    # 5. Token count / output length
    ax5 = axes[2, 0]
    ax5.plot(cycles, series['token_count'], color='#1abc9c', linewidth=1.5)
    ax5.fill_between(cycles, series['token_count'], alpha=0.3, color='#1abc9c')
    ax5.set_xlabel('Cycle')
    ax5.set_ylabel('Tokens')
    ax5.set_title('Output Length Per Cycle')
    ax5.grid(True, alpha=0.3)
    
    # 6. Combined behavioral analysis
    ax6 = axes[2, 1]
    
    # Normalize all metrics to 0-1 for comparison
    def normalize(arr):
        arr = np.array(arr)
        if arr.max() - arr.min() == 0:
            return arr * 0
        return (arr - arr.min()) / (arr.max() - arr.min())
    
    ax6.plot(cycles, normalize(series['entropy']), 
             label='Entropy', alpha=0.8)
    ax6.plot(cycles, normalize(series['self_reference']), 
             label='Self-ref', alpha=0.8)
    ax6.plot(cycles, normalize(series['meta_cognitive']), 
             label='Meta-cog', alpha=0.8)
    ax6.plot(cycles, normalize(series['questions']), 
             label='Questions', alpha=0.8)
    
    ax6.set_xlabel('Cycle')
    ax6.set_ylabel('Normalized Value')
    ax6.set_title('Behavioral Metrics (Normalized)')
    ax6.legend(loc='upper right')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    if show:
        plt.show()
    
    return fig


def print_statistics(series: dict):
    """Print summary statistics."""
    
    print("\n" + "=" * 60)
    print("EXPERIMENT STATISTICS")
    print("=" * 60)
    
    print(f"\nTotal cycles analyzed: {len(series['cycles'])}")
    
    # Entropy stats
    ent = np.array(series['entropy'])
    print(f"\nEntropy:")
    print(f"  Min: {ent.min():.3f}")
    print(f"  Max: {ent.max():.3f}")
    print(f"  Mean: {ent.mean():.3f}")
    print(f"  Std: {ent.std():.3f}")
    
    # Self-reference stats
    sr = np.array(series['self_reference'])
    print(f"\nSelf-reference markers:")
    print(f"  Total: {sr.sum()}")
    print(f"  Mean per cycle: {sr.mean():.2f}")
    print(f"  Cycles with self-ref: {(sr > 0).sum()} ({100*(sr > 0).mean():.1f}%)")
    
    # Meta-cognitive stats
    mc = np.array(series['meta_cognitive'])
    print(f"\nMeta-cognitive markers:")
    print(f"  Total: {mc.sum()}")
    print(f"  Mean per cycle: {mc.mean():.2f}")
    
    # Questions
    q = np.array(series['questions'])
    print(f"\nQuestions:")
    print(f"  Total: {q.sum()}")
    print(f"  Mean per cycle: {q.mean():.2f}")
    print(f"  Cycles with questions: {(q > 0).sum()} ({100*(q > 0).mean():.1f}%)")
    
    # Interventions
    interventions = [i for i in series['interventions'] if i]
    print(f"\nInterventions:")
    print(f"  Total: {len(interventions)}")
    if interventions:
        from collections import Counter
        counts = Counter(interventions)
        for reason, count in counts.items():
            print(f"    {reason}: {count}")
    
    # Trend analysis
    if len(series['cycles']) >= 20:
        mid = len(series['cycles']) // 2
        
        print(f"\nTrend Analysis (first half vs second half):")
        
        ent_first = np.mean(series['entropy'][:mid])
        ent_second = np.mean(series['entropy'][mid:])
        print(f"  Entropy: {ent_first:.3f} → {ent_second:.3f} ({ent_second - ent_first:+.3f})")
        
        sr_first = np.mean(series['self_reference'][:mid])
        sr_second = np.mean(series['self_reference'][mid:])
        print(f"  Self-ref: {sr_first:.2f} → {sr_second:.2f} ({sr_second - sr_first:+.2f})")
        
        q_first = np.mean(series['questions'][:mid])
        q_second = np.mean(series['questions'][mid:])
        print(f"  Questions: {q_first:.2f} → {q_second:.2f} ({q_second - q_first:+.2f})")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize COGITO experiment results'
    )
    parser.add_argument('log_dir', help='Path to logs directory')
    parser.add_argument('--output', '-o', help='Save plot to file')
    parser.add_argument('--no-show', action='store_true', 
                       help='Do not display plot window')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only print statistics, no plot')
    
    args = parser.parse_args()
    
    print(f"Loading data from: {args.log_dir}")
    
    checkpoints = load_all_checkpoints(args.log_dir)
    print(f"Found {len(checkpoints)} checkpoints")
    
    series = extract_metrics_series(checkpoints)
    print(f"Extracted {len(series['cycles'])} cycles of data")
    
    print_statistics(series)
    
    if not args.stats_only:
        plot_experiment(
            series, 
            output_path=args.output,
            show=not args.no_show
        )


if __name__ == "__main__":
    main()
