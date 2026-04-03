"""
find_positive_trees.py

Scans all experiment run folders in the given experiment directory, identifies
runs that produced positive scores (via distribution.txt), then re-scores every
tree in the corresponding search_output_<number>.txt files.  Every tree with a
positive score is drawn and saved as a PNG.  All console output is also written
to log.txt in the output directory.

Usage
-----
    python find_positive_trees.py [options]

Required arguments
    -e / --experiment-dir   Path to the top-level experiment folder
                            (contains randomly-named run sub-folders).
                            Default: test_run

    -o / --output-dir       Directory where images and log.txt are written.
                            Created if it does not exist.
                            Default: results

Optional arguments
    --min-score FLOAT       Only report trees with score >= this threshold.
                            Default: 0 (any positive score)

    --dpi INT               Resolution of saved tree images (dots per inch).
                            Default: 150

    --layout {spring,planar,kamada_kawai,hierarchy}
                            Layout algorithm for drawing trees.
                            spring, planar, kamada_kawai use NetworkX built-ins.
                            hierarchy uses the top-down hierarchy_pos() function.
                            Default: spring

    --no-images             Skip image generation (log only).

    -v / --verbose          Also print skipped runs to the console
                            (they are always written to log.txt).

Scoring (from my_dom_poly.ipynb)
    score = -log_concave_check(dominating_polynomial coefficients)
    Positive => the polynomial is NOT log-concave at some position k,
    i.e. a_k^2 < a_{k-1} * a_{k+1}.

Output layout
    <output-dir>/
        log.txt                              full run log
        <run>__<file>__L<line>.png           one image per positive-score tree
"""

import os
import re
import sys
import ast
import math
import glob
import argparse
import textwrap
from datetime import datetime

import random
import numpy as np
import matplotlib
matplotlib.use('Agg')           # non-interactive backend -- no display required
import matplotlib.pyplot as plt
import networkx as nx


# == scoring functions (from my_dom_poly.ipynb) ================================

def prufer_to_tree(prufer_code):
    """Convert a 1-indexed Prufer code to a 0-indexed adjacency dict."""
    n = len(prufer_code) + 2
    degree = [1] * n
    for node in prufer_code:
        degree[node - 1] += 1

    adjacency_list = {i: [] for i in range(n)}
    prufer_0 = [x - 1 for x in prufer_code]
    degree_copy = degree[:]

    for node in prufer_0:
        for i in range(n):
            if degree_copy[i] == 1:
                adjacency_list[node].append(i)
                adjacency_list[i].append(node)
                degree_copy[node] -= 1
                degree_copy[i] -= 1
                break

    remaining = [i for i in range(n) if degree_copy[i] == 1]
    u, v = remaining
    adjacency_list[u].append(v)
    adjacency_list[v].append(u)
    return adjacency_list


def poly_add(p, q):
    r = [0] * max(len(p), len(q))
    for i, a in enumerate(p): r[i] += a
    for i, b in enumerate(q): r[i] += b
    return r


def poly_sub(p, q):
    r = p[:] + [0] * max(0, len(q) - len(p))
    for i, b in enumerate(q):
        r[i] -= b
    while len(r) > 1 and r[-1] == 0:
        r.pop()
    return r


def poly_mul(p, q):
    r = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            r[i + j] += a * b
    return r


def dominating_polynomial(tree):
    """
    Compute the dominating polynomial of a tree (0-indexed adjacency dict).
    Returns list D where D[k] = number of dominating sets of size k.
    """
    n = len(tree)
    A = [None] * n
    B = [None] * n
    C = [None] * n

    def dfs(v, parent):
        children = [u for u in tree[v] if u != parent]
        if not children:
            A[v] = [0, 1]
            B[v] = [0]
            C[v] = [1]
            return
        prodA     = [1]
        prodBC    = [1]
        prodBonly = [1]
        for u in children:
            dfs(u, v)
            prodA     = poly_mul(prodA,     poly_add(A[u], C[u]))
            prodBC    = poly_mul(prodBC,    poly_add(A[u], B[u]))
            prodBonly = poly_mul(prodBonly, B[u])
        A[v] = poly_mul(prodA, [0, 1])
        C[v] = prodBC
        B[v] = poly_sub(prodBC, prodBonly)

    root = next((i for i, nbrs in tree.items() if len(nbrs) > 1), 0)
    dfs(root, -1)
    return poly_add(A[root], B[root])


def log_concave_check(lis):
    """
    Returns min over k of  2*ln(a_k) - ln(a_{k-1}) - ln(a_{k+1})
    computed over consecutive triples in the zero-free subsequence of lis.
    Zeros are deleted before testing so that log is never called on zero,
    and so that a violation straddling a zero gap is still detected.
    Negative return value => NOT log-concave at that position.
    Returns 999 if fewer than 3 nonzero values exist.
    """
    nonzero = [x for x in lis if x != 0]
    if len(nonzero) < 3:
        return 999
    check = [
        2 * math.log(nonzero[i + 1]) - math.log(nonzero[i]) - math.log(nonzero[i + 2])
        for i in range(len(nonzero) - 2)
    ]
    return float(np.min(np.array(check)))


def score_prufer_code(prufer_1indexed):
    """
    Score a 1-indexed Prufer code using the objective function from the notebook.
    Returns float; positive means the dominating polynomial is not log-concave.
    """
    tree = prufer_to_tree(prufer_1indexed)
    dom_poly = dominating_polynomial(tree)
    lcc = log_concave_check(dom_poly)
    if lcc == 999:
        return float('-inf')
    return -lcc


# == drawing (adapted from draw_tree.py) =======================================

_NX_LAYOUTS = {
    'spring':       nx.spring_layout,
    'planar':       nx.planar_layout,
    'kamada_kawai': nx.kamada_kawai_layout,
}


def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    Licensed under Creative Commons Attribution-Share Alike
   
    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.
   
    G: the graph (must be a tree)
   
    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.
   
    width: horizontal space allocated for this branch - avoids overlap with other branches
   
    vert_gap: gap between levels of hierarchy
   
    vert_loc: vertical location of root
   
    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''
   
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children)
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap,
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

           
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


def _format_prufer_label(code, cols=20):
    """
    Format a Prufer code list as a compact multi-line string, `cols` values per row.
    e.g.  [7, 14, 21, 32, ..., 18,
            4, 11, 40, ...]
    """
    rows = [code[i:i + cols] for i in range(0, len(code), cols)]
    lines = []
    for r_idx, row in enumerate(rows):
        prefix = '[' if r_idx == 0 else ' '
        suffix = ']' if r_idx == len(rows) - 1 else ','
        lines.append(prefix + ', '.join(str(v) for v in row) + suffix)
    return '\n'.join(lines)


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
    MAX_DISPLAY = 4   # cap bars shown to avoid x-axis overflow

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

def save_tree_image(prufer_1indexed, filepath, score, run_name,
                    source_file, lineno, layout='spring', dpi=150):
    """
    Draw the tree encoded by prufer_1indexed (1-indexed) and save as a PNG.

    The figure has two panels stacked vertically:
      Top (2/3)    : the tree, rooted at its center (highlighted in orange).
      Bottom (1/3) : the dominating polynomial as a bar chart; bars where
                     log-concavity is broken are coloured red and annotated
                     with Δ = 2·ln(a_k) − ln(a_{k-1}) − ln(a_{k+1}).

    layout can be one of: 'spring', 'planar', 'kamada_kawai', 'hierarchy'.
    'hierarchy' uses hierarchy_pos() rooted at the tree center.
    """
    prufer_0 = [x - 1 for x in prufer_1indexed]
    nx_tree  = nx.from_prufer_sequence(prufer_0)

    # -- dominating polynomial ------------------------------------------------
    adj_tree     = prufer_to_tree(prufer_1indexed)       # adjacency dict (1-indexed nodes)
    dom_poly     = dominating_polynomial(adj_tree)       # dom_poly[k] = D(k)
    min_k_idx    = next((i for i, v in enumerate(dom_poly) if v != 0), 0)
    dom_stripped = dom_poly[min_k_idx:]
    dom_ks       = list(range(min_k_idx, min_k_idx + len(dom_stripped)))

    # -- tree center ----------------------------------------------------------
    centers     = nx.center(nx_tree)
    center_root = centers[0]

    # -- node positions -------------------------------------------------------
    if layout == 'hierarchy':
        try:
            pos = hierarchy_pos(nx_tree, root=center_root)
        except Exception:
            pos = nx.spring_layout(nx_tree)
    else:
        layout_fn = _NX_LAYOUTS.get(layout, nx.spring_layout)
        try:
            pos = layout_fn(nx_tree)
        except nx.NetworkXException:
            pos = nx.spring_layout(nx_tree)

    n_nodes   = nx_tree.number_of_nodes()
    node_size = max(20, 600 - 8 * n_nodes)
    font_size = max(4, 9 - n_nodes // 15)

    # -- figure: two-panel layout ---------------------------------------------
    n_code_rows = math.ceil(len(prufer_1indexed) / 20)
    fig_height  = 11 + 0.22 * n_code_rows
    fig, (ax_tree, ax_poly) = plt.subplots(
        2, 1,
        figsize=(11, fig_height),
        gridspec_kw={'height_ratios': [2, 1]},
    )

    # -- draw tree (top panel) ------------------------------------------------
    center_set  = set(centers)
    node_colors = [
        '#FF8C42' if node in center_set else 'lightblue'
        for node in nx_tree.nodes()
    ]
    nx.draw(
        nx_tree, pos, ax=ax_tree,
        with_labels=True,
        node_color=node_colors,
        node_size=node_size,
        font_size=font_size,
        edge_color='steelblue',
        width=1.2,
    )
    center_label = (f"center: {centers[0]}" if len(centers) == 1
                    else f"centers: {centers[0]}, {centers[1]}")
    ax_tree.set_title(
        f"Run: {run_name}  |  {source_file}  line {lineno}\n"
        f"Score: {score:+.6f}  |  Nodes: {n_nodes}  |  "
        f"Prufer length: {len(prufer_1indexed)}  |  {center_label}",
        fontsize=8, pad=10,
    )

    # -- draw polynomial (bottom panel) ---------------------------------------
    _plot_dom_poly_panel(ax_poly, dom_stripped, dom_ks)

    # -- Prufer code annotation pinned to the bottom of the figure ------------
    code_label = "Prufer code (1-indexed):\n" + _format_prufer_label(prufer_1indexed)
    fig.text(
        0.5, 0.01,
        code_label,
        ha='center', va='bottom',
        fontsize=7,
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5', edgecolor='#cccccc'),
    )

    bottom_margin = 0.02 + 0.018 * n_code_rows
    fig.tight_layout(rect=[0, bottom_margin, 1, 1])
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# == scanning helpers ==========================================================

def has_positive_score_in_distribution(dist_path):
    """Return True if any Score: line in distribution.txt is positive."""
    with open(dist_path, 'r') as f:
        for line in f:
            m = re.search(r'Score:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
            if m and float(m.group(1)) > 0:
                return True
    return False


def iter_search_output_files(run_dir):
    """Return search_output_<number>.txt paths in run_dir, sorted numerically."""
    files = glob.glob(os.path.join(run_dir, 'search_output_*.txt'))
    numeric = [f for f in files if re.search(r'search_output_(\d+)\.txt$', f)]
    numeric.sort(key=lambda f: int(re.search(r'search_output_(\d+)\.txt$', f).group(1)))
    return numeric


def find_positive_trees_in_file(filepath, min_score=0.0):
    """
    Score every tree in a search_output file.
    Returns list of (line_number, prufer_code, score) where score >= min_score.
    """
    results = []
    with open(filepath, 'r') as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                prufer_code = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue
            if not isinstance(prufer_code, list):
                continue
            try:
                s = score_prufer_code(prufer_code)
            except Exception:
                continue
            if s >= min_score:
                results.append((lineno, prufer_code, s))
    return results


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
            Scan experiment runs for trees whose dominating polynomial is
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
        help='Destination for log.txt and tree images  (default: results)',
    )
    parser.add_argument(
        '--min-score',
        type=float,
        default=0.0,
        metavar='FLOAT',
        help='Minimum score threshold to report a tree  (default: 0.0)',
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
        choices=['spring', 'planar', 'kamada_kawai', 'hierarchy'],
        default='spring',
        help=(
            'Layout algorithm for drawing trees  (default: spring).\n'
            '  spring       -- NetworkX spring/force-directed layout\n'
            '  planar       -- NetworkX planar layout (falls back to spring if non-planar)\n'
            '  kamada_kawai -- NetworkX Kamada-Kawai layout\n'
            '  hierarchy    -- top-down hierarchical layout (hierarchy_pos)'
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
    log(f"find_positive_trees.py  --  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Experiment dir : {os.path.abspath(args.experiment_dir)}")
    log(f"  Output dir     : {os.path.abspath(args.output_dir)}")
    log(f"  Min score      : {args.min_score}")
    log(f"  Layout         : {args.layout}")
    log(f"  DPI            : {args.dpi}")
    log(f"  Images         : {'disabled' if args.no_images else 'enabled'}")
    log("=" * 70)

    try:
        # validate experiment directory
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

            # step 1 -- quick filter via distribution.txt
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

            # step 2 -- score every tree in search_output_*.txt
            search_files = iter_search_output_files(run_dir)
            if not search_files:
                log("  No search_output_<number>.txt files found -- skipping.")
                continue

            run_total = 0
            for sf in search_files:
                sf_base   = os.path.basename(sf)
                positives = find_positive_trees_in_file(sf, min_score=args.min_score)
                if not positives:
                    continue

                log(f"\n  File: {sf_base}  ({len(positives)} positive-score tree(s))")
                log(f"  {'-'*60}")

                for lineno, code, s in positives:
                    run_total      += 1
                    total_positive += 1

                    log(f"  Line {lineno:>6}  |  score = {s:+.6f}")
                    log(f"             Prufer code (1-indexed): {code}")

                    # step 3 -- draw and save image
                    if not args.no_images:
                        img_name = make_image_filename(run_name, sf_base, lineno)
                        img_path = os.path.join(args.output_dir, img_name)
                        try:
                            save_tree_image(
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
                log("  (No trees above the score threshold found after re-scoring.)")
            else:
                log(f"\n  -> {run_total} positive-score tree(s) found in this run.")

        # summary
        log(f"\n{'='*70}")
        log(f"DONE.  Total positive-score trees : {total_positive}")
        if not args.no_images:
            log(f"       Images saved              : {total_images}")
        log(f"       Log written to            : {os.path.abspath(log_path)}")
        log("=" * 70)

    finally:
        sys.stdout = tee._stdout
        tee.close()


if __name__ == '__main__':
    main()