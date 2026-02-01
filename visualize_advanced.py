#!/usr/bin/env python3
"""
Advanced COGITO Visualization Suite
Deeper analysis and comparison tools for pondering experiments.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


def load_all_checkpoints(log_dir: str) -> List[dict]:
    """Load all checkpoint files and extract cycle data."""
    checkpoint_dir = Path(log_dir) / "checkpoints"
    if not checkpoint_dir.exists():
        print(f"No checkpoints directory found in {log_dir}")
        return []
    
    all_cycles = []
    seen_cycles = set()
    
    checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_*.json"))
    
    for cp_file in checkpoint_files:
        with open(cp_file) as f:
            data = json.load(f)
            for metric in data.get('recent_metrics', []):
                cycle_num = metric['cycle_number']
                if cycle_num not in seen_cycles:
                    seen_cycles.add(cycle_num)
                    all_cycles.append(metric)
    
    return sorted(all_cycles, key=lambda x: x['cycle_number'])


def detect_phase_transitions(cycles: List[dict], window: int = 10) -> List[dict]:
    """Detect points where behavior changes significantly."""
    if len(cycles) < window * 2:
        return []
    
    transitions = []
    entropies = [c.get('entropy', 0) for c in cycles]
    questions = [c.get('question_count', 0) for c in cycles]
    
    for i in range(window, len(cycles) - window):
        # Compare windows before and after
        before_entropy = np.mean(entropies[i-window:i])
        after_entropy = np.mean(entropies[i:i+window])
        before_questions = np.mean(questions[i-window:i])
        after_questions = np.mean(questions[i:i+window])
        
        entropy_change = after_entropy - before_entropy
        question_change = after_questions - before_questions
        
        # Significant drop in either metric
        if entropy_change < -1.5 or question_change < -3:
            transitions.append({
                'cycle': cycles[i]['cycle_number'],
                'type': 'collapse',
                'entropy_change': entropy_change,
                'question_change': question_change
            })
        # Significant recovery
        elif entropy_change > 1.5 or question_change > 3:
            transitions.append({
                'cycle': cycles[i]['cycle_number'],
                'type': 'recovery',
                'entropy_change': entropy_change,
                'question_change': question_change
            })
    
    # Deduplicate nearby transitions
    filtered = []
    for t in transitions:
        if not filtered or t['cycle'] - filtered[-1]['cycle'] > window:
            filtered.append(t)
    
    return filtered


def extract_dominant_phrases(log_dir: str, cycle_range: Tuple[int, int]) -> Dict[str, int]:
    """Extract most common phrases from a cycle range."""
    transcript_dir = Path(log_dir) / "transcripts"
    if not transcript_dir.exists():
        return {}
    
    transcript_files = list(transcript_dir.glob("*.txt"))
    if not transcript_files:
        return {}
    
    # Read transcript
    with open(transcript_files[0]) as f:
        content = f.read()
    
    # Simple phrase extraction (3-6 word sequences)
    from collections import Counter
    import re
    
    phrases = Counter()
    words = re.findall(r'\b\w+\b', content.lower())
    
    for n in range(3, 7):
        for i in range(len(words) - n):
            phrase = ' '.join(words[i:i+n])
            phrases[phrase] += 1
    
    # Filter to meaningful phrases (appear more than 5 times)
    return {k: v for k, v in phrases.most_common(50) if v > 5}


def plot_phase_analysis(cycles: List[dict], transitions: List[dict], output_path: str = None):
    """Create a detailed phase analysis plot."""
    if not HAS_MATPLOTLIB:
        print("Cannot create plots without matplotlib")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('COGITO Phase Analysis', fontsize=14, fontweight='bold')
    
    cycle_nums = [c['cycle_number'] for c in cycles]
    entropies = [c.get('entropy', 0) for c in cycles]
    questions = [c.get('question_count', 0) for c in cycles]
    similarities = [c.get('self_similarity', 0) for c in cycles]
    
    # Plot 1: Entropy with phase regions
    ax1 = axes[0]
    ax1.plot(cycle_nums, entropies, 'b-', alpha=0.7, linewidth=1)
    ax1.fill_between(cycle_nums, entropies, alpha=0.3)
    ax1.set_ylabel('Entropy', fontsize=10)
    ax1.set_title('Token Entropy with Phase Transitions', fontsize=11)
    
    # Mark transitions
    for t in transitions:
        color = 'red' if t['type'] == 'collapse' else 'green'
        ax1.axvline(t['cycle'], color=color, linestyle='--', alpha=0.7, linewidth=2)
        label = f"{'↓' if t['type'] == 'collapse' else '↑'} {t['cycle']}"
        ax1.annotate(label, (t['cycle'], max(entropies)*0.9), fontsize=8, 
                    rotation=90, ha='right')
    
    # Plot 2: Questions with rolling average
    ax2 = axes[1]
    ax2.bar(cycle_nums, questions, alpha=0.4, color='orange', width=1)
    # Rolling average
    window = 20
    if len(questions) >= window:
        rolling_avg = np.convolve(questions, np.ones(window)/window, mode='valid')
        ax2.plot(cycle_nums[window-1:], rolling_avg, 'r-', linewidth=2, label=f'{window}-cycle avg')
    ax2.set_ylabel('Questions', fontsize=10)
    ax2.set_title('Questions Asked (Curiosity Indicator)', fontsize=11)
    ax2.legend()
    
    # Plot 3: Self-similarity (loop detection)
    ax3 = axes[2]
    ax3.plot(cycle_nums, similarities, 'purple', alpha=0.7, linewidth=1)
    ax3.fill_between(cycle_nums, similarities, alpha=0.3, color='purple')
    ax3.axhline(0.9, color='red', linestyle=':', alpha=0.5, label='Loop threshold')
    ax3.set_ylabel('Self-Similarity', fontsize=10)
    ax3.set_xlabel('Cycle', fontsize=10)
    ax3.set_title('Output Self-Similarity (1.0 = Perfect Repetition)', fontsize=11)
    ax3.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved phase analysis to {output_path}")
    else:
        plt.show()


def plot_comparison(runs: Dict[str, List[dict]], output_path: str = None):
    """Compare multiple runs side by side."""
    if not HAS_MATPLOTLIB:
        print("Cannot create plots without matplotlib")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('COGITO Run Comparison', fontsize=14, fontweight='bold')
    
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    
    # Plot 1: Entropy comparison
    ax1 = axes[0, 0]
    for i, (name, cycles) in enumerate(runs.items()):
        cycle_nums = [c['cycle_number'] for c in cycles]
        entropies = [c.get('entropy', 0) for c in cycles]
        ax1.plot(cycle_nums, entropies, color=colors[i % len(colors)], 
                alpha=0.7, label=name, linewidth=1.5)
    ax1.set_xlabel('Cycle')
    ax1.set_ylabel('Entropy')
    ax1.set_title('Token Entropy Comparison')
    ax1.legend()
    
    # Plot 2: Questions comparison
    ax2 = axes[0, 1]
    for i, (name, cycles) in enumerate(runs.items()):
        cycle_nums = [c['cycle_number'] for c in cycles]
        questions = [c.get('question_count', 0) for c in cycles]
        # Rolling average for smoothness
        window = 10
        if len(questions) >= window:
            rolling = np.convolve(questions, np.ones(window)/window, mode='valid')
            ax2.plot(cycle_nums[window-1:], rolling, color=colors[i % len(colors)],
                    alpha=0.7, label=name, linewidth=1.5)
    ax2.set_xlabel('Cycle')
    ax2.set_ylabel('Questions (10-cycle avg)')
    ax2.set_title('Curiosity Comparison')
    ax2.legend()
    
    # Plot 3: Time to collapse
    ax3 = axes[1, 0]
    collapse_cycles = []
    names = []
    for name, cycles in runs.items():
        # Find first cycle where entropy drops below 3 for 10+ cycles
        entropies = [c.get('entropy', 0) for c in cycles]
        collapse_cycle = None
        for i in range(len(entropies) - 10):
            if all(e < 3 for e in entropies[i:i+10]):
                collapse_cycle = cycles[i]['cycle_number']
                break
        if collapse_cycle:
            collapse_cycles.append(collapse_cycle)
            names.append(name)
    
    if collapse_cycles:
        bars = ax3.bar(names, collapse_cycles, color=colors[:len(names)], alpha=0.7)
        ax3.set_ylabel('Cycle Number')
        ax3.set_title('Cycles Until Collapse')
        for bar, val in zip(bars, collapse_cycles):
            ax3.annotate(str(val), (bar.get_x() + bar.get_width()/2, val),
                        ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Intervention comparison
    ax4 = axes[1, 1]
    intervention_counts = []
    for name, cycles in runs.items():
        count = sum(1 for c in cycles if c.get('intervention_applied'))
        intervention_counts.append(count)
    
    if intervention_counts:
        bars = ax4.bar(list(runs.keys()), intervention_counts, 
                      color=colors[:len(runs)], alpha=0.7)
        ax4.set_ylabel('Count')
        ax4.set_title('Total Interventions')
        for bar, val in zip(bars, intervention_counts):
            ax4.annotate(str(val), (bar.get_x() + bar.get_width()/2, val),
                        ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {output_path}")
    else:
        plt.show()


def plot_intervention_effectiveness(cycles: List[dict], output_path: str = None):
    """Analyze how effective interventions were."""
    if not HAS_MATPLOTLIB:
        print("Cannot create plots without matplotlib")
        return
    
    # Find intervention points
    interventions = []
    for i, c in enumerate(cycles):
        if c.get('intervention_applied'):
            interventions.append({
                'index': i,
                'cycle': c['cycle_number'],
                'type': c['intervention_applied'],
                'entropy_before': cycles[max(0, i-5):i],
                'entropy_after': cycles[i:min(len(cycles), i+10)]
            })
    
    if not interventions:
        print("No interventions found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Intervention Effectiveness Analysis', fontsize=14, fontweight='bold')
    
    # Plot 1: Entropy before/after each intervention
    ax1 = axes[0]
    for interv in interventions[:15]:  # Limit to first 15 for readability
        before = [c.get('entropy', 0) for c in interv['entropy_before']]
        after = [c.get('entropy', 0) for c in interv['entropy_after']]
        
        x_before = list(range(-len(before), 0))
        x_after = list(range(0, len(after)))
        
        color = 'red' if 'collapse' in interv['type'] else 'orange'
        ax1.plot(x_before + x_after, before + after, alpha=0.3, color=color)
    
    ax1.axvline(0, color='black', linestyle='--', label='Intervention')
    ax1.set_xlabel('Cycles relative to intervention')
    ax1.set_ylabel('Entropy')
    ax1.set_title('Entropy Trajectory Around Interventions')
    ax1.legend()
    
    # Plot 2: Intervention type distribution
    ax2 = axes[1]
    from collections import Counter
    type_counts = Counter(i['type'] for i in interventions)
    ax2.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%',
            colors=['red', 'orange', 'yellow', 'green'][:len(type_counts)])
    ax2.set_title('Intervention Types')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved intervention analysis to {output_path}")
    else:
        plt.show()


def analyze_vocabulary_over_time(cycles: List[dict], window: int = 20) -> List[dict]:
    """Track vocabulary richness over time."""
    results = []
    
    for i in range(0, len(cycles), window):
        window_cycles = cycles[i:i+window]
        if not window_cycles:
            continue
        
        avg_unique = np.mean([c.get('unique_tokens', 0) for c in window_cycles])
        avg_total = np.mean([c.get('token_count', 1) for c in window_cycles])
        
        results.append({
            'start_cycle': window_cycles[0]['cycle_number'],
            'end_cycle': window_cycles[-1]['cycle_number'],
            'avg_unique_tokens': avg_unique,
            'avg_total_tokens': avg_total,
            'vocabulary_ratio': avg_unique / max(avg_total, 1)
        })
    
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_advanced.py <log_dir> [log_dir2] ...")
        print("\nExamples:")
        print("  python visualize_advanced.py ./logs_overnight_mirror")
        print("  python visualize_advanced.py ./logs_mirror ./logs_wonder  # Compare runs")
        sys.exit(1)
    
    log_dirs = sys.argv[1:]
    
    if len(log_dirs) == 1:
        # Single run analysis
        log_dir = log_dirs[0]
        print(f"Loading data from: {log_dir}")
        
        cycles = load_all_checkpoints(log_dir)
        if not cycles:
            print("No data found")
            sys.exit(1)
        
        print(f"Loaded {len(cycles)} cycles")
        
        # Detect phase transitions
        transitions = detect_phase_transitions(cycles)
        print(f"\nDetected {len(transitions)} phase transitions:")
        for t in transitions:
            print(f"  Cycle {t['cycle']}: {t['type']} "
                  f"(entropy Δ={t['entropy_change']:.2f}, questions Δ={t['question_change']:.2f})")
        
        # Vocabulary analysis
        vocab_analysis = analyze_vocabulary_over_time(cycles)
        print(f"\nVocabulary analysis (20-cycle windows):")
        for v in vocab_analysis[:5]:
            print(f"  Cycles {v['start_cycle']}-{v['end_cycle']}: "
                  f"{v['avg_unique_tokens']:.1f} unique / {v['avg_total_tokens']:.1f} total "
                  f"(ratio: {v['vocabulary_ratio']:.2f})")
        if len(vocab_analysis) > 5:
            print("  ...")
            for v in vocab_analysis[-3:]:
                print(f"  Cycles {v['start_cycle']}-{v['end_cycle']}: "
                      f"{v['avg_unique_tokens']:.1f} unique / {v['avg_total_tokens']:.1f} total "
                      f"(ratio: {v['vocabulary_ratio']:.2f})")
        
        # Generate plots
        if HAS_MATPLOTLIB:
            plot_phase_analysis(cycles, transitions, 
                              f"{log_dir}/phase_analysis.png")
            plot_intervention_effectiveness(cycles,
                              f"{log_dir}/intervention_analysis.png")
    
    else:
        # Multi-run comparison
        print(f"Comparing {len(log_dirs)} runs...")
        
        runs = {}
        for log_dir in log_dirs:
            name = Path(log_dir).name
            cycles = load_all_checkpoints(log_dir)
            if cycles:
                runs[name] = cycles
                print(f"  {name}: {len(cycles)} cycles")
        
        if len(runs) >= 2 and HAS_MATPLOTLIB:
            plot_comparison(runs, "comparison.png")


if __name__ == "__main__":
    main()
