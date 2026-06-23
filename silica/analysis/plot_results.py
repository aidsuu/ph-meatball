#!/usr/bin/env python3
"""
Generate publication-quality comparison plots across all pH conditions.

Reads analysis output files and creates:
1. XRD patterns overlay (all pH conditions + reference peaks)
2. Phase percentage bar chart vs pH
3. Ring-size distribution comparison
4. Steinhardt Q4-Q6 scatter comparison

Usage:
    python plot_results.py --results-dir ../results
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import glob
import re


# ── Style ────────────────────────────────────────────────────────────────────

def setup_style():
    """Set publication-quality matplotlib style."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 9,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


# ── Data Loading ─────────────────────────────────────────────────────────────

def find_ph_files(results_dir, pattern):
    """Find files matching pattern for each pH condition."""
    files = {}
    for f in sorted(glob.glob(os.path.join(results_dir, pattern))):
        # Extract pH from filename
        match = re.search(r'pH[_]?(\d+\.?\d*)', os.path.basename(f))
        if match:
            ph = float(match.group(1))
            files[ph] = f
    return files


def load_xrd_data(results_dir):
    """Load XRD data for all pH conditions."""
    files = find_ph_files(results_dir, "xrd_*.dat")
    data = {}
    for ph, f in files.items():
        arr = np.loadtxt(f)
        data[ph] = {'two_theta': arr[:, 0], 'intensity': arr[:, 1]}
    return data


def load_steinhardt_summaries(results_dir):
    """Load Steinhardt phase classification summaries."""
    files = find_ph_files(results_dir, "steinhardt_*_summary.txt")
    data = {}
    for ph, f in files.items():
        phases = {}
        with open(f) as fh:
            for line in fh:
                for phase in ['amorphous', 'cristobalite', 'tridymite']:
                    if line.strip().startswith(phase):
                        pct = float(line.strip().split(':')[1].strip().rstrip('%'))
                        phases[phase] = pct
        data[ph] = phases
    return data


def load_ring_data(results_dir):
    """Load ring-size distributions for all pH conditions."""
    files = find_ph_files(results_dir, "rings_*.dat")
    data = {}
    for ph, f in files.items():
        arr = np.loadtxt(f)
        data[ph] = {'sizes': arr[:, 0].astype(int),
                     'counts': arr[:, 1].astype(int),
                     'fractions': arr[:, 2]}
    return data


def load_steinhardt_peratom(results_dir):
    """Load per-atom Steinhardt data for scatter plots."""
    files = find_ph_files(results_dir, "steinhardt_*.dat")
    # Exclude summary files
    files = {ph: f for ph, f in files.items() if 'summary' not in f}
    data = {}
    for ph, f in files.items():
        arr = np.loadtxt(f)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            data[ph] = {'q4': arr[:, 1], 'q6': arr[:, 2]}
    return data


# ── Plot Functions ───────────────────────────────────────────────────────────

def plot_xrd_overlay(xrd_data, output_dir):
    """Plot XRD patterns for all pH conditions overlaid."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(xrd_data)))
    ph_values = sorted(xrd_data.keys())

    for i, ph in enumerate(ph_values):
        d = xrd_data[ph]
        # Offset for visibility
        offset = i * 0.2
        intensity = d['intensity'] / np.max(d['intensity']) + offset
        ax.plot(d['two_theta'], intensity, color=colors[i],
                linewidth=1.0, label=f'pH {ph:.1f}')

    # Reference peaks
    ax.axvline(21.9, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.axvline(36.1, color='red', linestyle=':', alpha=0.3, linewidth=0.8)
    ax.axvline(20.5, color='green', linestyle='--', alpha=0.3, linewidth=0.8)

    ax.text(22.5, ax.get_ylim()[1] * 0.95, 'C(101)', fontsize=7, color='red')
    ax.text(36.5, ax.get_ylim()[1] * 0.95, 'C(200)', fontsize=7, color='red')
    ax.text(18.5, ax.get_ylim()[1] * 0.95, 'T', fontsize=7, color='green')

    ax.set_xlabel("2θ (degrees)")
    ax.set_ylabel("Intensity (arb. units, offset)")
    ax.set_title("Simulated XRD Patterns — Effect of pH on Silica Structure")
    ax.set_xlim(10, 70)
    ax.legend(loc='upper right')

    fig.tight_layout()
    outpath = os.path.join(output_dir, "xrd_comparison.png")
    fig.savefig(outpath)
    print(f"  Saved: {outpath}")
    plt.close()


def plot_phase_barchart(phase_data, output_dir):
    """Plot phase percentage bar chart vs pH."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ph_values = sorted(phase_data.keys())
    n = len(ph_values)
    x = np.arange(n)
    width = 0.25

    amorphous = [phase_data[ph].get('amorphous', 0) for ph in ph_values]
    cristobalite = [phase_data[ph].get('cristobalite', 0) for ph in ph_values]
    tridymite = [phase_data[ph].get('tridymite', 0) for ph in ph_values]

    ax.bar(x - width, amorphous, width, label='Amorphous',
           color='#95a5a6', edgecolor='#7f8c8d')
    ax.bar(x, cristobalite, width, label='Cristobalite',
           color='#e74c3c', edgecolor='#c0392b')
    ax.bar(x + width, tridymite, width, label='Tridymite',
           color='#2ecc71', edgecolor='#27ae60')

    ax.set_xlabel("pH")
    ax.set_ylabel("Phase Percentage (%)")
    ax.set_title("Effect of pH on Silica Polymorph Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels([f'{ph:.1f}' for ph in ph_values])
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    outpath = os.path.join(output_dir, "phase_vs_pH.png")
    fig.savefig(outpath)
    print(f"  Saved: {outpath}")
    plt.close()


def plot_ring_comparison(ring_data, output_dir):
    """Plot ring-size distributions for all pH conditions."""
    fig, ax = plt.subplots(figsize=(9, 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(ring_data)))
    ph_values = sorted(ring_data.keys())
    n_ph = len(ph_values)

    if n_ph == 0:
        return

    # Grouped bar chart
    all_sizes = set()
    for d in ring_data.values():
        all_sizes.update(d['sizes'].tolist())
    sizes = sorted(all_sizes)
    n_sizes = len(sizes)

    bar_width = 0.8 / n_ph
    x = np.arange(n_sizes)

    for i, ph in enumerate(ph_values):
        d = ring_data[ph]
        fracs = []
        for s in sizes:
            idx = np.where(d['sizes'] == s)[0]
            fracs.append(d['fractions'][idx[0]] if len(idx) > 0 else 0)

        offset = (i - n_ph / 2 + 0.5) * bar_width
        ax.bar(x + offset, fracs, bar_width, label=f'pH {ph:.1f}',
               color=colors[i], edgecolor='white', linewidth=0.5)

    ax.set_xlabel("Ring Size (Si atoms)")
    ax.set_ylabel("Fraction")
    ax.set_title("Ring-Size Distribution — Effect of pH")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend(fontsize=8)

    fig.tight_layout()
    outpath = os.path.join(output_dir, "rings_comparison.png")
    fig.savefig(outpath)
    print(f"  Saved: {outpath}")
    plt.close()


def plot_steinhardt_comparison(steinhardt_data, output_dir):
    """Plot Q4-Q6 scatter plots for all pH conditions."""
    n = len(steinhardt_data)
    if n == 0:
        return

    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows),
                             squeeze=False)

    ph_values = sorted(steinhardt_data.keys())

    # Reference points
    refs = {
        'Amorph.': (0.03, 0.03, '#888888'),
        'Crist.': (0.19, 0.57, '#e74c3c'),
        'Tridy.': (0.13, 0.48, '#2ecc71'),
    }

    for i, ph in enumerate(ph_values):
        row, col = divmod(i, cols)
        ax = axes[row][col]
        d = steinhardt_data[ph]

        ax.scatter(d['q4'], d['q6'], s=5, alpha=0.4, c='steelblue',
                  edgecolors='none')

        for name, (rq4, rq6, color) in refs.items():
            ax.plot(rq4, rq6, '*', color=color, markersize=12, zorder=5)
            ax.annotate(name, (rq4, rq6), fontsize=7,
                       xytext=(3, 3), textcoords='offset points')

        ax.set_xlabel("Q₄")
        ax.set_ylabel("Q₆")
        ax.set_title(f"pH {ph:.1f}")
        ax.set_xlim(-0.02, 0.30)
        ax.set_ylim(-0.02, 0.70)

    # Hide empty subplots
    for i in range(n, rows * cols):
        row, col = divmod(i, cols)
        axes[row][col].set_visible(False)

    fig.suptitle("Steinhardt Q₄–Q₆ Maps — Effect of pH", fontsize=14, y=1.02)
    fig.tight_layout()
    outpath = os.path.join(output_dir, "steinhardt_comparison.png")
    fig.savefig(outpath)
    print(f"  Saved: {outpath}")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality comparison plots")
    parser.add_argument("--results-dir", default="../results",
                        help="Directory containing analysis output files")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for plots (default: same as results)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.results_dir

    os.makedirs(args.output_dir, exist_ok=True)
    setup_style()

    print(f"Loading results from: {args.results_dir}")

    # Load and plot each analysis type
    xrd_data = load_xrd_data(args.results_dir)
    if xrd_data:
        print(f"\nXRD data found for pH: {sorted(xrd_data.keys())}")
        plot_xrd_overlay(xrd_data, args.output_dir)

    phase_data = load_steinhardt_summaries(args.results_dir)
    if phase_data:
        print(f"\nPhase data found for pH: {sorted(phase_data.keys())}")
        plot_phase_barchart(phase_data, args.output_dir)

    ring_data = load_ring_data(args.results_dir)
    if ring_data:
        print(f"\nRing data found for pH: {sorted(ring_data.keys())}")
        plot_ring_comparison(ring_data, args.output_dir)

    stein_data = load_steinhardt_peratom(args.results_dir)
    if stein_data:
        print(f"\nSteinhardt data found for pH: {sorted(stein_data.keys())}")
        plot_steinhardt_comparison(stein_data, args.output_dir)

    if not any([xrd_data, phase_data, ring_data, stein_data]):
        print("\nNo analysis data found. Run analysis scripts first.")
        return

    print(f"\nAll plots saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
