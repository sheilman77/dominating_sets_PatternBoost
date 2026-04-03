"""
find_positive_graphs.py

Scans all experiment run folders in a given experiment directory, identifies
runs that produced positive scores (via distribution.txt), then re-scores every
graph in the corresponding search_output_<number>.txt files.  Every graph with a
positive score is drawn and saved as a PNG.  All console output is also written
to log.txt in the output directory.

A "positive score" means the dominating polynomial of the graph is NOT
log-concave at some position k, i.e. a_k^2 < a_{k-1} * a_{k+1}.

Score definition (from graph_dom_poly_lcc.py):
    score = -log_concave_check(dominating_polynomial coefficients)
    Positive => the polynomial is NOT log-concave.

Usage
-----
    python find_positive_graphs.py [options]

Required arguments
    -e / --experiment-dir   Path to the top-level experiment folder
                            (contains run sub-folders).
                            Default: test_run

    -o / --output-dir       Directory where images and log.txt are written.
                            Created if it does not exist.
                            Default: results

Optional arguments
    --min-score FLOAT       Only report graphs with score >= this threshold.
                            Default: 0 (any positive score)

    --dpi INT               Resolution of saved graph images (dots per inch).
                            Default: 150

    --layout {spring,planar,kamada_kawai}
                            Layout algorithm for drawing graphs.
                            Default: spring

    --no-images             Skip image generation (log only).

    -v / --verbose          Also print skipped runs to the console
                            (they are always written to log.txt).

Output layout
    <output-dir>/
        log.txt                                   full run log
        <run>__<file>__L<line>.png                one image per positive-score graph
"""

import os
import re
import sys
import ast
import math
import glob
import argparse
import textwrap
from itertools import chain, combinations
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')           # non-interactive backend -- no display required
import matplotlib.pyplot as plt
import networkx as nx


# == graph utilities ===========================================================

def find_graph_size(code_length):
    """
    Given a binary code of length n*(n-1)/2, return n (the number of vertices).
    Mirrors draw_graph.py's find_graph_size logic.
    """
    doubled = code_length * 2
    for i in range(doubled + 2):
        if i * (i - 1) == doubled:
            return i
    raise ValueError(f"Cannot determine graph size from code length {code_length}")


def binary_to_graph(binary_code, n):
    """Convert a flat binary code (list of 0/1) to an adjacency dict."""
    adjacency_list = {i: [] for i in range(n)}
    index = 0
    for i in range(n):
        for j in range(i + 1, n):
            if binary_code[index] == 1:
                adjacency_list[i].append(j)
                adjacency_list[j].append(i)
            index += 1
    return adjacency_list


# == scoring functions (from graph_dom_poly_lcc.py) ===========================

def powerset(iterable):
    """Return all non-empty subsets of the input iterable."""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def is_dominating_set(graph, subset):
    """Check if the subset is a dominating set of the graph."""
    dominated = set(subset)
    for node in subset:
        dominated.update(graph.get(node, []))
    return dominated == set(graph.keys())


def dominating_polynomial(graph):
    """
    Compute dominating polynomial coefficients for a graph represented as an
    adjacency dict.  Returns list D where D[k] = number of dominating sets of
    size (k+1) (leading zeros stripped).
    """
    nodes = list(graph.keys())
    dom_list = [0] * len(graph)
    for subset in powerset(nodes):
        if is_dominating_set(graph, subset):
            dom_list[len(subset) - 1] += 1
    # strip leading zeros
    while len(dom_list) > 1 and dom_list[0] == 0:
        dom_list.pop(0)
    return dom_list


def log_concave_check(lis):
    """
    Returns min over k of  2*ln(a_{k+1}) - ln(a_k) - ln(a_{k+2})
    for all consecutive triples of nonzero entries.
    Negative => NOT log-concave at that position.
    Returns 999 if fewer than 3 nonzero values exist.
    """
    check = []
    for i in range(len(lis) - 2):
        if lis[i] != 0 and lis[i + 1] != 0 and lis[i + 2] != 0:
            check.append(
                2 * math.log(lis[i + 1]) - math.log(lis[i]) - math.log(lis[i + 2])
            )
    if len(check) == 0:
        return 999
    return float(np.min(np.array(check)))


def score_binary_code(binary_code):
    """
    Score a binary-encoded graph.
    Returns float; positive means the dominating polynomial is NOT log-concave.
    Returns -inf if the polynomial has fewer than 3 nonzero coefficients.
    """
    n = find_graph_size(len(binary_code))
    graph = binary_to_graph(binary_code, n)
    dom_poly = dominating_polynomial(graph)
    nonzero = [x for x in dom_poly if x != 0]
    if len(nonzero) < 3:
        return float('-inf')
    lcc = log_concave_check(nonzero)
    if lcc == 999:
        return float('-inf')
    return -lcc   # positive  <=>  NOT log-concave


# == distribution.txt helper ===================================================

def has_positive_score_in_distribution(dist_path):
    """
    Return True if distribution.txt contains any line with Score > 0.
    Expected format:   Score: <float>, Count: <int>
    """
    with open(dist_path, 'r') as f:
        for line in f:
            m = re.search(r'Score:\s*([-\d.]+)', line)
            if m:
                try:
                    if float(m.group(1)) > 0:
                        return True
                except ValueError:
                    pass
    return False


# == search_output file helpers ================================================

def iter_search_output_files(run_dir):
    """Return sorted list of search_output_<number>.txt files in run_dir."""
    pattern = os.path.join(run_dir, 'search_output_*.txt')
    files = sorted(glob.glob(pattern))
    return files


def find_positive_graphs_in_file(filepath, min_score=0.0):
    """
    Scan a search_output file for graphs with score >= min_score.
    Each line should contain a JSON/Python list of 0s and 1s.
    Returns list of (lineno, binary_code, score) tuples.
    """
    results = []
    with open(filepath, 'r') as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                binary_code = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue
            if not isinstance(binary_code, list):
                continue
            try:
                s = score_binary_code(binary_code)
            except Exception:
                continue
            if s != float('-inf') and s > min_score:
                results.append((lineno, binary_code, s))
    return results


# == drawing ===================================================================

_NX_LAYOUTS = {
    'spring':       lambda G: nx.spring_layout(G, seed=42),
    'planar':       nx.planar_layout,
    'kamada_kawai': nx.kamada_kawai_layout,
}


def _dominating_polynomial_raw(graph):
    """
    Like dominating_polynomial() but returns the full-length list without
    stripping leading zeros.  dom_list[i] = D(i+1) = number of dominating
    sets of size (i+1).  Used for display so we can recover the minimum
    domination number and label the x-axis correctly.
    """
    nodes    = list(graph.keys())
    dom_list = [0] * len(graph)
    for subset in powerset(nodes):
        if is_dominating_set(graph, subset):
            dom_list[len(subset) - 1] += 1
    return dom_list


def _plot_dom_poly_panel(ax, dom_full, ks_full):
    """
    Plot the dominating polynomial as an annotated bar chart on *ax*.

    Parameters
    ----------
    dom_full : list[int]
        Full polynomial coefficients (leading zeros already stripped).
        dom_full[i] = D(ks_full[i]) = number of dominating sets of size ks_full[i].
    ks_full : list[int]
        Original dominating-set sizes corresponding to each coefficient.
        Internal zeros are kept so that x-axis positions are always the true
        k values; they are only excluded from the log-concavity computation.

    Colour coding
    -------------
    Blue : position satisfies log-concavity  (a_k^2 >= a_{k-1} * a_{k+1})
    Red  : position *breaks* log-concavity   (a_k^2 <  a_{k-1} * a_{k+1})

    The violation label 'k=X  Δ=...' is placed above each red bar.
    ALL violation positions are flagged (the loop never stops early).
    """
    MAX_DISPLAY = 15   # cap bars shown to avoid x-axis overflow

    # --- find ALL log-concavity violations on the FULL polynomial ------------
    # Zeros are skipped in the compressed check (log(0) undefined), but the
    # original index j is preserved so we can colour the right bar.
    # All positions are tested — there is no early exit.
    violation_delta = {}   # j (index into dom_full) -> delta  (negative)
    nonzero_entries = [(j, v) for j, v in enumerate(dom_full) if v > 0]
    for pos in range(1, len(nonzero_entries) - 1):
        _, a = nonzero_entries[pos - 1]
        j, b = nonzero_entries[pos]
        _, c = nonzero_entries[pos + 1]
        delta = 2 * math.log(b) - math.log(a) - math.log(c)
        if delta < 0:
            violation_delta[j] = delta   # store ALL violations

    # --- truncate display to first MAX_DISPLAY coefficients ------------------
    truncated = len(dom_full) > MAX_DISPLAY
    vals = dom_full[:MAX_DISPLAY]
    ks   = ks_full[:MAX_DISPLAY]
    n    = len(vals)

    colors = ['#e74c3c' if j in violation_delta else '#3498db'
              for j in range(n)]

    bars = ax.bar(ks, vals, color=colors, edgecolor='white',
                  linewidth=0.5, width=0.7)

    ymax = max(vals) if any(v > 0 for v in vals) else 1

    # --- bar-top count labels ------------------------------------------------
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.01,
                str(v),
                ha='center', va='bottom', fontsize=6.5,
            )

    # --- violation annotations: label above bar, NO arrow --------------------
    # Placed well above the count label so they never overlap the bar itself.
    # Each label shows the original k index explicitly.
    for j, delta in violation_delta.items():
        if j >= n:          # violation is beyond the displayed range
            continue
        label_y = vals[j] + ymax * 0.18
        ax.text(
            ks[j], label_y,
            f'k={ks[j]}\nΔ={delta:.3f}',
            ha='center', va='bottom',
            fontsize=7, color='#c0392b', fontweight='bold',
        )

    # --- axes formatting -----------------------------------------------------
    title = (
        'Dominating polynomial'
        '   (red bar = log-concavity violation:  '
        r'$a_k^2 < a_{k-1}\,a_{k+1}$)'
    )
    if truncated:
        title += f'\n(showing first {MAX_DISPLAY} of {len(dom_full)} coefficients)'

    ax.set_xlabel('Dominating-set size  k', fontsize=8)
    ax.set_ylabel('D(k)  —  # dominating sets', fontsize=8)
    ax.set_title(title, fontsize=8.5)
    ax.tick_params(labelsize=7)
    ax.set_xticks(ks)
    ax.set_xlim(ks[0] - 0.8, ks[-1] + 0.8)
    ax.set_ylim(0, ymax * 1.50)

    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor='#3498db', label='log-concave'),
            Patch(facecolor='#e74c3c',
                  label=r'violation  ($a_k^2 < a_{k-1}\,a_{k+1}$)'),
        ],
        fontsize=7, loc='upper right',
    )

def draw_and_save_graph(binary_code, img_path, score,
                        run_name, source_file, lineno,
                        layout='spring', dpi=150):
    """
    Draw the graph encoded by binary_code and save as a PNG at img_path.

    The figure has two panels stacked vertically:
      Top (2/3)    : the graph (NetworkX drawing).
      Bottom (1/3) : the dominating polynomial as a bar chart; bars where
                     log-concavity is broken are coloured red and annotated
                     with Δ = 2·ln(a_k) − ln(a_{k-1}) − ln(a_{k+1}).
    """
    n   = find_graph_size(len(binary_code))
    adj = binary_to_graph(binary_code, n)

    # -- dominating polynomial ------------------------------------------------
    # _dominating_polynomial_raw returns dom_raw[i] = D(i+1); we strip leading
    # zeros ourselves so we can recover the minimum domination number for the
    # x-axis labels.
    dom_raw      = _dominating_polynomial_raw(adj)       # dom_raw[i] = D(i+1)
    min_k_idx    = next((i for i, v in enumerate(dom_raw) if v != 0), 0)
    dom_stripped = dom_raw[min_k_idx:]
    dom_ks       = list(range(min_k_idx + 1,
                               min_k_idx + 1 + len(dom_stripped)))

    # -- NetworkX graph -------------------------------------------------------
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for node, neighbors in adj.items():
        for nb in neighbors:
            if node < nb:
                G.add_edge(node, nb)

    layout_fn = _NX_LAYOUTS.get(layout, lambda G: nx.spring_layout(G, seed=42))
    pos = layout_fn(G)

    n_nodes   = G.number_of_nodes()
    node_size = max(20, 600 - 8 * n_nodes)
    font_size = max(4, 9 - n_nodes // 15)

    # -- figure: two-panel layout ---------------------------------------------
    fig, (ax_graph, ax_poly) = plt.subplots(
        2, 1,
        figsize=(11, 12),
        gridspec_kw={'height_ratios': [2, 1]},
    )

    # -- draw graph (top panel) -----------------------------------------------
    nx.draw(
        G, pos, ax=ax_graph,
        with_labels=True,
        node_color='skyblue',
        edge_color='gray',
        node_size=node_size,
        font_size=font_size,
    )
    ax_graph.set_title(
        f"Run: {run_name}   File: {source_file}   Line: {lineno}\n"
        f"Score = {score:+.6f}  |  n = {n}  |  edges = {G.number_of_edges()}",
        fontsize=9,
    )

    # -- draw polynomial (bottom panel) ---------------------------------------
    _plot_dom_poly_panel(ax_poly, dom_stripped, dom_ks)

    fig.tight_layout()
    fig.savefig(img_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# == filename helper ============================================================

def make_image_filename(run_name, source_file, lineno):
    """Build a filesystem-safe image filename encoding run / source / line."""
    safe_run  = re.sub(r'[^\w\-]', '_', run_name)
    safe_file = re.sub(r'[^\w\-]', '_', os.path.splitext(source_file)[0])
    return f"{safe_run}__{safe_file}__L{lineno}.png"


# == logging wrapper ===========================================================

class Tee:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, log_path):
        self._log    = open(log_path, 'w', encoding='utf-8')
        self._stdout = sys.stdout

    def write(self, msg):
        self._stdout.write(msg)
        self._log.write(msg)
        self._log.flush()

    def flush(self):
        self._stdout.flush()
        self._log.flush()

    def close(self):
        self._log.flush()
        self._log.close()


# == argument parsing ==========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Scan experiment runs for graphs whose dominating polynomial is
            not log-concave (positive score), draw them, and save a log.
        """),
    )
    parser.add_argument(
        '-e', '--experiment-dir',
        default='test_run',
        metavar='DIR',
        help='Top-level folder containing run sub-directories  (default: test_run)',
    )
    parser.add_argument(
        '-o', '--output-dir',
        default='results',
        metavar='DIR',
        help='Destination for log.txt and graph images  (default: results)',
    )
    parser.add_argument(
        '--min-score',
        type=float,
        default=0.0,
        metavar='FLOAT',
        help='Minimum score threshold to report a graph  (default: 0.0)',
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        metavar='INT',
        help='Resolution of saved PNG images  (default: 150)',
    )
    parser.add_argument(
        '--layout',
        choices=['spring', 'planar', 'kamada_kawai'],
        default='spring',
        help=(
            'Layout algorithm for drawing graphs  (default: spring).\n'
            '  spring       -- NetworkX spring/force-directed layout\n'
            '  planar       -- NetworkX planar layout (falls back to spring if non-planar)\n'
            '  kamada_kawai -- NetworkX Kamada-Kawai layout\n'
        ),
    )
    parser.add_argument(
        '--no-images',
        action='store_true',
        help='Disable image generation; write log only',
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print skipped runs to the console (always written to log.txt)',
    )
    return parser.parse_args()


# == main ======================================================================

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, 'log.txt')
    tee = Tee(log_path)
    sys.stdout = tee

    def log(msg=''):
        print(msg)

    # header
    log(f"find_positive_graphs.py  --  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Experiment dir : {os.path.abspath(args.experiment_dir)}")
    log(f"  Output dir     : {os.path.abspath(args.output_dir)}")
    log(f"  Min score      : {args.min_score}")
    log(f"  Layout         : {args.layout}")
    log(f"  DPI            : {args.dpi}")
    log(f"  Images         : {'disabled' if args.no_images else 'enabled'}")
    log("=" * 70)

    try:
        if not os.path.isdir(args.experiment_dir):
            log(f"\nERROR: experiment directory '{args.experiment_dir}' not found.")
            sys.exit(1)

        run_dirs = sorted([
            os.path.join(args.experiment_dir, d)
            for d in os.listdir(args.experiment_dir)
            if os.path.isdir(os.path.join(args.experiment_dir, d))
        ])

        if not run_dirs:
            log(f"\nNo sub-folders found inside '{args.experiment_dir}'.")
            return

        log(f"\nFound {len(run_dirs)} run folder(s).\n")

        total_positive = 0
        total_images   = 0

        for run_dir in run_dirs:
            run_name  = os.path.basename(run_dir)
            dist_path = os.path.join(run_dir, 'distribution.txt')

            # ---- step 1: quick filter via distribution.txt -------------------
            if not os.path.isfile(dist_path):
                msg = f"[SKIP] {run_name}  -- no distribution.txt"
                if args.verbose:
                    log(msg)
                else:
                    tee._log.write(msg + '\n')
                continue

            if not has_positive_score_in_distribution(dist_path):
                msg = f"[SKIP] {run_name}  -- no positive scores in distribution.txt"
                if args.verbose:
                    log(msg)
                else:
                    tee._log.write(msg + '\n')
                continue

            log(f"\n{'='*70}")
            log(f"RUN:  {run_name}")
            log(f"  +  Positive score(s) detected in distribution.txt")
            log(f"{'='*70}")

            # ---- step 2: re-score every graph in search_output_*.txt ---------
            search_files = iter_search_output_files(run_dir)
            if not search_files:
                log("  No search_output_<number>.txt files found -- skipping.")
                continue

            run_total = 0
            for sf in search_files:
                sf_base   = os.path.basename(sf)
                log(f"\n  Scanning {sf_base} ...")
                positives = find_positive_graphs_in_file(sf, min_score=args.min_score)
                if not positives:
                    log(f"  (no positive-score graphs found in {sf_base})")
                    continue

                log(f"\n  File: {sf_base}  ({len(positives)} positive-score graph(s))")
                log(f"  {'-'*60}")

                for lineno, code, s in positives:
                    run_total      += 1
                    total_positive += 1
                    n = find_graph_size(len(code))

                    log(f"  Line {lineno:>6}  |  score = {s:+.6f}  |  n = {n}  |  "
                        f"edges = {sum(code)}")
                    log(f"             Binary code: {code}")

                    # ---- step 3: draw and save image -------------------------
                    if not args.no_images:
                        img_name = make_image_filename(run_name, sf_base, lineno)
                        img_path = os.path.join(args.output_dir, img_name)
                        try:
                            draw_and_save_graph(
                                code, img_path, s,
                                run_name, sf_base, lineno,
                                layout=args.layout,
                                dpi=args.dpi,
                            )
                            log(f"             Image saved : {img_name}")
                            total_images += 1
                        except Exception as exc:
                            log(f"             [WARNING] Could not save image: {exc}")

            if run_total == 0:
                log("  (No graphs above the score threshold found after re-scoring.)")
            else:
                log(f"\n  -> {run_total} positive-score graph(s) found in this run.")

        # summary
        log(f"\n{'='*70}")
        log(f"DONE.  Total positive-score graphs : {total_positive}")
        if not args.no_images:
            log(f"       Images saved               : {total_images}")
        log(f"       Log written to             : {os.path.abspath(log_path)}")
        log("=" * 70)

    finally:
        sys.stdout = tee._stdout
        tee.close()


if __name__ == '__main__':
    main()