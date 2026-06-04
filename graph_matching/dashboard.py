"""Integrated PyQt5 dashboard for graph_matching.

A single window with an environment dropdown and a grid of visualization
panels. Each panel embeds a matplotlib figure produced by ``visualize_nxgraph_3d``
(reused as-is) — full native interactivity (rotate, zoom, pan, mpl toolbar,
on-motion hover highlight) works inside the dashboard.

Add new visualizations by creating another panel class with the same
``set_graph`` / ``clear`` shape and dropping it into the grid in ``Dashboard``.

Run:
    python dashboard.py
"""

import sys
import copy
import json
import pickle
import queue as _queue
import time
from pathlib import Path

import matplotlib

matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PyQt5.QtCore import Qt, QEvent, QSignalBlocker, QTimer, pyqtSignal
import matplotlib.backend_bases as _mpl_bb
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from situational_graphs_wrapper.GraphWrapper import GraphWrapper
from situational_graphs_datasets.graph_visualizer import visualize_nxgraph_3d
from situational_graphs_datasets.InteractiveGraphVisualizer import InteractiveGraphVisualizer


GRAPH_DICTS_DIR = Path(__file__).parent / "graph_dicts"

# ── Model selection ────────────────────────────────────────────────────────────
# Edit this line to switch the GNN model used by the dashboard.
# Pick one:
#   "ws_room_dropout_noise"               → MatchingModel_GATv2SinkhornTopK   (original, TopK)
#   "ws_room_dropout_noise_inc_BCE"       → MatchingModel_MLPGATv2SinkhornBCE  (MLP + BCE)
#   "ws_room_dropout_noise_inc_BCE_noMLP" → MatchingModel_GATv2Sinkhorn        (no MLP, BCE)
#   "ws_room_dropout_noise_inc_WBCE"      → MatchingModel_MLPGATv2SinkhornWBCE (MLP + weighted BCE)
MODEL = "ws_room_dropout_noise_inc_WBCE"

# Make the dry-run matcher importable: it lives in the sibling graph_matching_gnn repo.
_WORKSPACE_SRC = Path(__file__).resolve().parent.parent.parent
_PGM_PATH = str(_WORKSPACE_SRC / "graph_matching_gnn" / "graph_matching")
if _PGM_PATH not in sys.path:
    sys.path.insert(0, _PGM_PATH)
import dry_run_pgm as _dry_run_pgm  # noqa: E402


def list_environments():
    envs = []
    for p in sorted(GRAPH_DICTS_DIR.iterdir()):
        if p.is_dir() and (p / "Prior.pkl").exists() and (p / "Online.pkl").exists():
            envs.append(p.name)
    return envs


def _load_graph(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, GraphWrapper):
        return obj
    return GraphWrapper(graph_obj=obj)


def _load_ground_truth(env_name):
    """Return a set of (a_id, s_id) string-pair tuples from ground_truth.json.

    The JSON uses prefixed IDs (``a_<id>`` for Prior, ``s_<id>`` for Online)
    but graph node IDs carry no prefix, so the prefixes are stripped here.
    """
    gt_path = GRAPH_DICTS_DIR / env_name / "ground_truth.json"
    if not gt_path.exists():
        return set()
    with open(gt_path, "r") as f:
        raw = json.load(f)

    def strip(node_id, prefix):
        s = str(node_id)
        return s[len(prefix):] if s.startswith(prefix) else s

    pairs = set()
    for o_id, p_id in raw.get("rooms", {}).items():
        if p_id != "??":
            pairs.add((strip(p_id, "a_"), strip(o_id, "s_")))
    for entry in raw.get("ws", []):
        if len(entry) == 2 and entry[1] != "??":
            pairs.add((strip(entry[1], "a_"), strip(entry[0], "s_")))
    return pairs


def load_env_graphs(env_name):
    a = _load_graph(GRAPH_DICTS_DIR / env_name / "Prior.pkl")
    s = _load_graph(GRAPH_DICTS_DIR / env_name / "Online.pkl")
    for g in (a, s):
        for _, attrs in g.graph.nodes(data=True):
            for key in ("center", "normal"):
                if key in attrs:
                    v = attrs[key]
                    if hasattr(v, "tolist"):
                        v = v.tolist()
                    if len(v) > 2:
                        v = v[:2]
                    attrs[key] = v
    gt = _load_ground_truth(env_name)
    return a, s, gt


def _prepare_for_viz(graph):
    g_viz = copy.deepcopy(graph)
    g_viz.from_2D_to_3D()
    g_viz._add_complete_viz_attributes_to_graph()
    return g_viz


def _prepare_for_editor(graph):
    """Like `_prepare_for_viz`, but also surfaces nested `viz.*` keys as the
    flat `viz_type` / `viz_data` / `viz_feat` keys that
    `InteractiveGraphVisualizer` reads directly. SDG-generated graphs ship
    with both flat and nested keys; pickled wrappers loaded by the dashboard
    only carry the nested form, so we copy them across.
    """
    g_viz = _prepare_for_viz(graph)
    for _, attrs in g_viz.graph.nodes(data=True):
        viz = attrs.get("viz") or {}
        vtype = viz.get("type", "Point")
        attrs.setdefault("viz_type", vtype)
        # Editor reads viz_data as the line geometry for "Line", or the
        # render position for "Point".
        if "viz_data" not in attrs:
            if vtype == "Line" and "limits" in viz:
                attrs["viz_data"] = viz["limits"]
            elif "center" in viz:
                attrs["viz_data"] = viz["center"]
            else:
                attrs["viz_data"] = attrs.get("center", [0.0, 0.0, 0.0])
        attrs.setdefault("viz_feat", viz.get("feat", "ko"))
    return g_viz


def _axis_center_and_radius(graph_wrapper, axis):
    vals = []
    for _, attrs in graph_wrapper.graph.nodes(data=True):
        if "limits" in attrs:
            lim = np.asarray(attrs["limits"], dtype=float)
            vals.extend(lim[:, axis].tolist())
        if "center" in attrs:
            vals.append(float(attrs["center"][axis]))
    if not vals:
        return 0.0, 0.0
    v_min, v_max = min(vals), max(vals)
    return (v_min + v_max) / 2.0, (v_max - v_min) / 2.0


def _x_center_and_radius(g):
    return _axis_center_and_radius(g, 0)


def _y_center_and_radius(g):
    return _axis_center_and_radius(g, 1)


def _transform_b_inplace(b_graph, dx, dy, theta_rad):
    """Rotate B around its own center then translate by (dx, dy) in XY."""
    cx, _ = _x_center_and_radius(b_graph)
    cy, _ = _y_center_and_radius(b_graph)
    c = np.array([cx, cy], dtype=float)
    R = np.array(
        [
            [np.cos(theta_rad), -np.sin(theta_rad)],
            [np.sin(theta_rad), np.cos(theta_rad)],
        ]
    )
    offset = np.array([dx, dy], dtype=float)

    def transform_point(p):
        p_arr = np.asarray(p, dtype=float)
        new_xy = R @ (p_arr[:2] - c) + c + offset
        if p_arr.shape[0] >= 3:
            return np.concatenate([new_xy, p_arr[2:]]).tolist()
        return new_xy.tolist()

    for _, attrs in b_graph.graph.nodes(data=True):
        if "center" in attrs:
            attrs["center"] = transform_point(attrs["center"])
        if "limits" in attrs:
            lim = np.asarray(attrs["limits"], dtype=float)
            new_lim = np.array([transform_point(lim[i]) for i in range(lim.shape[0])])
            attrs["limits"] = new_lim.tolist()
        if "normal" in attrs:
            n = np.asarray(attrs["normal"], dtype=float)
            new_n_xy = R @ n[:2]
            if n.shape[0] >= 3:
                attrs["normal"] = np.concatenate([new_n_xy, n[2:]]).tolist()
            else:
                attrs["normal"] = new_n_xy.tolist()


def _initial_dx(a, s):
    """Auto X-shift that places S beside A with 1*a_radius of padding."""
    a_cx, a_r = _x_center_and_radius(a)
    s_cx, s_r = _x_center_and_radius(s)
    padding = a_r
    return (a_cx + a_r + padding + s_r) - s_cx


def _align_transform(a, s):
    """(dx, dy) that snaps S's centroid onto A's centroid (graphs overlap)."""
    a_cx, _ = _x_center_and_radius(a)
    s_cx, _ = _x_center_and_radius(s)
    a_cy, _ = _y_center_and_radius(a)
    s_cy, _ = _y_center_and_radius(s)
    return a_cx - s_cx, a_cy - s_cy


def _combine_graphs(a, s, dx, dy, theta_deg, edge_style_fn):
    """Build the A+S composite graph with B translated/rotated and bipartite
    cross-edges styled by ``edge_style_fn(ai, si)``.

    ``edge_style_fn`` receives the original (untransformed) node ids as strings
    and must return ``(viz_feat, linewidth, alpha)`` or ``None`` to skip the edge.
    """
    s_transformed = copy.deepcopy(s)
    _transform_b_inplace(s_transformed, dx, dy, np.deg2rad(theta_deg))

    ga = nx.relabel_nodes(a.graph, lambda n: f"a_{n}", copy=True)
    gs = nx.relabel_nodes(s_transformed.graph, lambda n: f"s_{n}", copy=True)
    merged = nx.compose(ga, gs)

    for ai in a.graph.nodes():
        for si in s.graph.nodes():
            style = edge_style_fn(str(ai), str(si))
            if style is None:
                continue
            viz_feat, linewidth, alpha = style
            merged.add_edge(
                f"a_{ai}",
                f"s_{si}",
                viz_feat=viz_feat,
                linewidth=linewidth,
                alpha=alpha,
            )

    return GraphWrapper(graph_obj=merged)


def _build_gt_editor_graph(a, s, dx, dy, theta_deg, gt_pairs):
    """Compose A and S (with B's transform applied) into a single graph for
    the GT editor: only the GT cross-edges are drawn (no n_a×n_s mesh), and
    the per-node viz attrs are flattened to the keys IGV reads.

    Node IDs are the same prefixed form as ``_combine_graphs`` (``a_<id>`` /
    ``s_<id>``) so the GT enter handler can split a selection back into the
    A/S originals.
    """
    s_transformed = copy.deepcopy(s)
    _transform_b_inplace(s_transformed, dx, dy, np.deg2rad(theta_deg))

    a_prepared = _prepare_for_editor(a)
    s_prepared = _prepare_for_editor(s_transformed)
    ga = nx.relabel_nodes(a_prepared.graph, lambda n: f"a_{n}", copy=True)
    gs = nx.relabel_nodes(s_prepared.graph, lambda n: f"s_{n}", copy=True)
    merged = nx.compose(ga, gs)

    for ai, si in gt_pairs:
        a_node = f"a_{ai}"
        s_node = f"s_{si}"
        if a_node in merged.nodes and s_node in merged.nodes:
            merged.add_edge(a_node, s_node)

    return GraphWrapper(graph_obj=merged)


def _gt_edge_style_fn(gt_pairs):
    def fn(ai, si):
        in_gt = (ai, si) in gt_pairs
        return ("g", 2.0, 1.0) if in_gt else ("r", 0.5, 0.1)
    return fn


# Neutral color for the GT-agnostic "value" mode. Must be a single-character
# matplotlib color code: the visualizer's `_mpl_color_from_feat` only inspects
# `viz_feat[0]` and looks it up in a fixed dict, so hex strings like "#1f77b4"
# fail with `ValueError: '#' is not a valid value for color`. 'b' (blue) is
# safe and visually distinct from the green/red used in the other modes.
_VALUE_ONLY_COLOR = "b"


def _gt_value_edge_style_fn(matrix, a_nodes, s_nodes, gt_pairs,
                            mode="correct", min_weight=0.05):
    """Color bipartite edges by GT membership; alpha by matrix value.

    Matrix is min-max normalized to [0, 1]; the normalized value drives a
    "highlight weight" that controls both alpha and linewidth. The mode
    selects how the weight relates to GT membership:
      - ``"correct"``: GT-true edges → weight = norm (value→1 visible);
        GT-false edges → weight = 1 − norm (value→0 visible). Correct
        predictions stand out.
      - ``"incorrect"``: mapping swapped, so FN-like (GT-true with low
        value) and FP-like (GT-false with high value) edges stand out.
      - ``"value"``: weight = norm for every edge, no GT-based color
        distinction (single neutral color). Pure "what does the matrix
        say" view.
    Edges with weight below `min_weight` are skipped to reduce clutter.
    """
    a_index = {str(n): i for i, n in enumerate(a_nodes)}
    s_index = {str(n): i for i, n in enumerate(s_nodes)}
    m = np.asarray(matrix, dtype=float)
    m_min = float(m.min())
    m_max = float(m.max())
    span = (m_max - m_min) if m_max > m_min else 1.0

    def fn(ai, si):
        i = a_index.get(ai)
        j = s_index.get(si)
        if i is None or j is None:
            return None
        norm = (m[i, j] - m_min) / span
        in_gt = (ai, si) in gt_pairs
        if mode == "value":
            weight = norm
            color = _VALUE_ONLY_COLOR
        elif mode == "incorrect":
            if in_gt:
                return None
            weight = norm
            color = "r"
        else:  # "correct"
            weight = norm if in_gt else (1.0 - norm)
            color = "g" if in_gt else "r"
        if weight < min_weight:
            return None
        alpha = max(0.1, min(1.0, weight))
        linewidth = 0.5 + 2.5 * weight
        return (color, linewidth, alpha)
    return fn


def _node_aura_weights(matrix, a_nodes, s_nodes, gt_pairs, mode):
    """Per-node mean of the same "highlight weight" used by ``_gt_value_edge_style_fn``.

    For each A node, average over all S columns; same for each S node
    averaging over A rows. The cell weight depends on ``mode``:
      - ``"correct"``: ``norm`` for GT-true cells, ``1 - norm`` otherwise.
      - ``"incorrect"``: swapped.
      - ``"value"``: ``norm`` everywhere (no GT distinction).

    Returns ``({"a_<id>": w}, {"s_<id>": w})`` with weights in ``[0, 1]``.
    """
    m = np.asarray(matrix, dtype=float)
    m_min = float(m.min())
    m_max = float(m.max())
    span = (m_max - m_min) if m_max > m_min else 1.0
    norm = (m - m_min) / span

    in_gt_mat = np.zeros_like(norm, dtype=bool)
    a_idx = {str(n): i for i, n in enumerate(a_nodes)}
    s_idx = {str(n): i for i, n in enumerate(s_nodes)}
    for ai, si in gt_pairs:
        i = a_idx.get(ai)
        j = s_idx.get(si)
        if i is not None and j is not None:
            in_gt_mat[i, j] = True

    if mode == "value":
        weights = norm
    elif mode == "incorrect":
        weights = np.where(in_gt_mat, 0.0, norm)
    else:  # "correct"
        weights = np.where(in_gt_mat, norm, 1.0 - norm)

    if weights.size == 0:
        return {}, {}
    a_means = weights.mean(axis=1) if weights.shape[1] > 0 else np.zeros(weights.shape[0])
    s_means = weights.mean(axis=0) if weights.shape[0] > 0 else np.zeros(weights.shape[1])
    a_weights = {f"a_{n}": float(a_means[i]) for i, n in enumerate(a_nodes)}
    s_weights = {f"s_{n}": float(s_means[j]) for j, n in enumerate(s_nodes)}
    return a_weights, s_weights


def _draw_node_aura(fig, prepared_graph, weights, color, size=900):
    """Scatter a colored halo behind each node; per-point alpha = ``weights[id]``.

    Drawn *after* the visualizer with ``zorder=0`` and ``depthshade=False`` so
    the original node markers stay on top and the aura intensity reads
    consistently regardless of viewing angle.
    """
    if not weights:
        return
    g = prepared_graph.graph
    ax = next((a for a in fig.axes if hasattr(a, "get_zlim")), None)
    if ax is None:
        return
    xs, ys, zs, rgba = [], [], [], []
    for n_id, w in weights.items():
        if n_id not in g.nodes:
            continue
        c = _viz_center(g.nodes[n_id])
        if c is None:
            continue
        c = list(c) + [0.0] * (3 - len(c))
        a = max(0.0, min(1.0, float(w)))
        xs.append(c[0])
        ys.append(c[1])
        zs.append(c[2])
        rgba.append(to_rgba(color, alpha=a))
    if not xs:
        return
    ax.scatter(
        xs, ys, zs,
        s=size, c=rgba, marker="o",
        edgecolors="none", depthshade=False, zorder=0,
    )


def _pairs_from_perm(perm, g1_nodes, g2_nodes):
    rows, cols = np.where(np.asarray(perm) > 0)
    return {(str(g1_nodes[r]), str(g2_nodes[c])) for r, c in zip(rows, cols)}


def _compute_classification_metrics(gt_pairs, pred_pairs, total):
    """Confusion-matrix + precision / recall / F1 / accuracy / specificity.

    Same formulation as ``matching_synthetic_dataset.compute_metrics``: treat
    every (a, s) cell of the universe (size ``total``) as a binary
    classification — class 1 if the pair is "matched". TN is the
    cells in the universe that neither GT nor prediction marked. For the
    combined view, ``total = |A| × |S|``; for per-edge-type views, ``total``
    is restricted to the cells whose endpoint types satisfy the filter
    (e.g. for room-room it's |rooms in A| × |rooms in S|).

    Reimplemented locally to avoid importing the heavy
    ``matching_synthetic_dataset`` module (top-level side effects + sklearn
    pulled in for a handful of divisions).
    """
    total = max(int(total), 0)
    gt = set(gt_pairs)
    pred = set(pred_pairs)
    tp = len(gt & pred)
    fp = len(pred - gt)
    fn = len(gt - pred)
    tn = max(total - tp - fp - fn, 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "f1_score": f1, "accuracy": accuracy, "specificity": specificity,
        "total": total, "gt_total": len(gt), "pred_total": len(pred),
    }


# Order matters: drives the order of rows inside a metrics box.
_METRIC_ROW_SPEC = (
    ("precision",   "Precision"),
    ("recall",      "Recall"),
    ("f1_score",    "F1"),
    ("accuracy",    "Accuracy"),
    ("specificity", "Specificity"),
    ("tp",          "TP"),
    ("fp",          "FP"),
    ("fn",          "FN"),
    ("tn",          "TN"),
    ("gt_total",    "GT pairs"),
    ("pred_total",  "Pred pairs"),
    ("match_time",  "Match time (s)"),
)


def _build_metric_rows(parent_layout):
    """Append metric label/value rows to ``parent_layout``.

    Returns ``{key: QLabel}`` so callers can mutate values in place. Each row
    is its own widget so the labels can sit on the left and the values
    right-aligned via a stretch.
    """
    labels = {}
    for key, text in _METRIC_ROW_SPEC:
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)
        rl.addWidget(QLabel(f"{text}:"))
        rl.addStretch(1)
        v = QLabel("—")
        v.setStyleSheet("font-family: monospace;")
        rl.addWidget(v)
        parent_layout.addWidget(row)
        labels[key] = v
    return labels


class _CollapsibleSection(QWidget):
    """A header with a Down/Right-arrow toggle that hides/shows its body.

    The arrow is a ``QToolButton`` with ``ToolButtonTextBesideIcon`` so the
    title and arrow share a single click target. Callers add metric rows
    into ``body_layout()`` — the body is a vertical layout that the toggle
    `setVisible`s on click.
    """

    def __init__(self, title, parent=None, expanded=True):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self._toggle = QToolButton()
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._toggle.setStyleSheet(
            "QToolButton { border: none; font-weight: bold; padding: 2px 0; }"
        )
        self._toggle.toggled.connect(self._on_toggle)
        outer.addWidget(self._toggle)
        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(12, 2, 8, 6)
        self._body_layout.setSpacing(2)
        outer.addWidget(self._body)
        self._body.setVisible(expanded)

    def _on_toggle(self, checked):
        self._body.setVisible(checked)
        self._toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    def body_layout(self):
        return self._body_layout


# Predicates for the per-edge-type breakdowns. Same A/S sides as everywhere
# else: a pair ``(a_id, s_id)`` is classified by the types of its two
# endpoints. "cross" merges both directions (A=room/S=ws and A=ws/S=room)
# per user request ("room-ws or viceversa").
_EDGE_TYPE_PREDICATES = {
    "room-room": lambda at, st: at == "room" and st == "room",
    "ws-ws":     lambda at, st: at == "ws" and st == "ws",
    "room-ws":   lambda at, st: (at == "room" and st == "ws")
                                 or (at == "ws" and st == "room"),
}


def _node_type_map(graph):
    """``{node_id_as_str: type_string}`` from a GraphWrapper.

    Used to evaluate the edge-type predicates against pair sets and to
    count the per-type universe for the TN denominator.
    """
    return {str(n): a.get("type") for n, a in graph.graph.nodes(data=True)}


class DryRunMatcher:
    """Wraps the random-data dry-run matcher (``dry_run_pgm._get_or_make``)
    in the common ``match(a, s) -> (pairs, ints, a_nodes, s_nodes)`` shape.

    Caching is delegated to ``_get_or_make`` (keyed on ``(name, |V|, |E|)``).
    """

    name = "dry-run"
    display = "dry-run matcher (random embeddings)"

    def clear_cache(self):
        pass  # dry-run cache lives in _dry_run_pgm; not safe to clear globally

    def match(self, a, s):
        g1, g2 = a.graph, s.graph
        ints = _dry_run_pgm._get_or_make(g1, g2)
        g1_nodes = list(g1.nodes())
        g2_nodes = list(g2.nodes())
        pairs = _pairs_from_perm(ints["perm"], g1_nodes, g2_nodes)
        return pairs, ints, g1_nodes, g2_nodes


class GnnMatcher:
    """Real GNN matcher (``PartialGraphMatching`` from ``PGM_class``).

    Loads the checkpoint at construction; ``match(a, s)`` converts the two
    graphs to PyG, runs one model forward with ``return_intermediate=True``,
    and returns a dict shaped like the dry-run output:

    - ``affinity``: raw embedding dot product ``h1 @ h2.T``
    - ``sim_normed``: InstanceNorm of the affinity (what the model uses
      pre-Sinkhorn — collected by the model as ``affinity_list``)
    - ``S``: Sinkhorn doubly-stochastic matrix
    - ``perm``: Hungarian binary one-to-one assignment

    All numpy arrays. Per-pair cache keyed on the same ``(name, |V|, |E|)``
    signature the dry-run uses so editor edits naturally invalidate.
    """

    name = "gnn"
    display = "real GNN (PartialGraphMatching)"

    def __init__(self, model_save_path, data_paths, model_class_name):
        # Heavy imports happen here so dry-run mode doesn't pay for torch /
        # torch_geometric / etc. at startup.
        import torch  # noqa: E402
        from PGM_class import (  # noqa: E402
            PartialGraphMatching,
            MatchingModel_GATv2SinkhornTopK,
            MatchingModel_MLPGATv2SinkhornBCE,
            MatchingModel_GATv2Sinkhorn,
            MatchingModel_MLPGATv2SinkhornWBCE,
            nx_to_pyg_data_preserve_order,
            normalize_graph,
        )

        _class_map = {
            "MatchingModel_GATv2SinkhornTopK":    MatchingModel_GATv2SinkhornTopK,
            "MatchingModel_MLPGATv2SinkhornBCE":  MatchingModel_MLPGATv2SinkhornBCE,
            "MatchingModel_GATv2Sinkhorn":        MatchingModel_GATv2Sinkhorn,
            "MatchingModel_MLPGATv2SinkhornWBCE": MatchingModel_MLPGATv2SinkhornWBCE,
        }
        model_class = _class_map[model_class_name]

        self._torch = torch
        self._nx_to_pyg = nx_to_pyg_data_preserve_order
        self._normalize_graph = normalize_graph

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._pgm = PartialGraphMatching(
            model_class=model_class,
            data_paths=data_paths,
            model_save_path=model_save_path,
            device=self._device,
            in_dim=7,
            inference_only=True,
        )
        self._pgm.load_best_model()
        self._model = self._pgm.model
        self._mean = self._pgm.mean
        self._std = self._pgm.std
        self._cache = {}

    def clear_cache(self):
        self._cache.clear()

    @staticmethod
    def _signature(g1, g2):
        return (
            g1.graph.get("name", ""), g1.number_of_nodes(), g1.number_of_edges(),
            g2.graph.get("name", ""), g2.number_of_nodes(), g2.number_of_edges(),
        )

    def match(self, a, s):
        torch = self._torch
        g1, g2 = a.graph, s.graph
        sig = self._signature(g1, g2)
        cached = self._cache.get(sig)
        if cached is not None:
            return cached

        g1_pyg = self._nx_to_pyg(g1)
        g2_pyg = self._nx_to_pyg(g2)
        g1_pyg, g2_pyg = self._normalize_graph(
            g1_pyg, g2_pyg, self._mean, self._std
        )
        g1_pyg = g1_pyg.to(self._device)
        g2_pyg = g2_pyg.to(self._device)

        batch_idx1 = torch.zeros(g1_pyg.num_nodes, dtype=torch.long,
                                 device=self._device)
        batch_idx2 = torch.zeros(g2_pyg.num_nodes, dtype=torch.long,
                                 device=self._device)

        self._model.eval()
        with torch.no_grad():
            # `return_intermediate=True` makes the model's forward also yield
            # `(soft_list, affinity_list, sinkhorn_list)`. `affinity_list[0]`
            # holds the InstanceNormed `sim_normed`; the raw affinity
            # `h1 @ h2.T` isn't collected, so we recompute it from the
            # embeddings the model returns alongside.
            perm_list, embeddings, _soft, affinity_list, sinkhorn_list = self._model(
                g1_pyg, g2_pyg, batch_idx1, batch_idx2,
                inference=True, return_soft=True, return_intermediate=True,
            )

        perm = perm_list[0].detach().cpu().numpy()
        sim_normed = affinity_list[0].detach().cpu().numpy()
        S = sinkhorn_list[0].detach().cpu().numpy()
        h1_b, h2_b = embeddings[0]
        affinity = (h1_b @ h2_b.T).detach().cpu().numpy()

        g1_nodes = list(g1.nodes())
        g2_nodes = list(g2.nodes())
        pairs = _pairs_from_perm(perm, g1_nodes, g2_nodes)
        ints = {
            "affinity": affinity,
            "sim_normed": sim_normed,
            "S": S,
            "perm": perm,
        }
        result = (pairs, ints, g1_nodes, g2_nodes)
        self._cache[sig] = result
        return result


def _scale_3d_axes(fig, factor):
    """Scale every 3D axis's limits about its center. factor < 1 zooms in."""
    for ax in fig.axes:
        if not hasattr(ax, "get_zlim"):
            continue
        for getter, setter in (
            (ax.get_xlim, ax.set_xlim),
            (ax.get_ylim, ax.set_ylim),
            (ax.get_zlim, ax.set_zlim),
        ):
            lo, hi = getter()
            c = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo) * factor
            setter(c - half, c + half)


def _install_scroll_zoom(fig):
    """Scroll-wheel zoom on whichever 3D axes the cursor is over."""
    def on_scroll(event):
        ax = event.inaxes
        if ax is None or not hasattr(ax, "get_zlim"):
            return
        factor = 0.85 if event.button == "up" else 1.18
        for getter, setter in (
            (ax.get_xlim, ax.set_xlim),
            (ax.get_ylim, ax.set_ylim),
            (ax.get_zlim, ax.set_zlim),
        ):
            lo, hi = getter()
            c = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo) * factor
            setter(c - half, c + half)
        fig.canvas.draw_idle()
    fig.canvas.mpl_connect("scroll_event", on_scroll)


def _make_match_legend_proxy(color, label, linewidth=2.5, alpha=1.0):
    return Line2D([0], [0], color=color, linewidth=linewidth, label=label, alpha=alpha)


# 0 = TN (skipped pair), 1 = FP, 2 = TP, 3 = FN. Order chosen so the cmap below
# lines up with the dry-run-panel edge colors (red/green/orange).
_CLASSIFICATION_CMAP = ListedColormap(["white", "red", "green", "orange"])


def _gt_matrix(gt_pairs, a_nodes, s_nodes):
    a_idx = {str(n): i for i, n in enumerate(a_nodes)}
    s_idx = {str(n): i for i, n in enumerate(s_nodes)}
    m = np.zeros((len(a_nodes), len(s_nodes)), dtype=float)
    for ai, si in gt_pairs:
        i = a_idx.get(ai)
        j = s_idx.get(si)
        if i is not None and j is not None:
            m[i, j] = 1.0
    return m


def _classification_matrix(gt_pairs, pred_pairs, a_nodes, s_nodes):
    """Categorical 0=TN, 1=FP, 2=TP, 3=FN. Used by the dry-run panel's matrix inset."""
    a_idx = {str(n): i for i, n in enumerate(a_nodes)}
    s_idx = {str(n): i for i, n in enumerate(s_nodes)}
    m = np.zeros((len(a_nodes), len(s_nodes)), dtype=int)
    for ai, si in pred_pairs:
        i = a_idx.get(ai)
        j = s_idx.get(si)
        if i is None or j is None:
            continue
        m[i, j] = 2 if (ai, si) in gt_pairs else 1
    for ai, si in gt_pairs:
        i = a_idx.get(ai)
        j = s_idx.get(si)
        if i is None or j is None:
            continue
        if m[i, j] == 0:
            m[i, j] = 3
    return m


def _draw_matrix_inset(fig, matrix, a_labels, s_labels,
                       cmap, title,
                       vmin=None, vmax=None,
                       discrete_legend=None):
    """Shrink the 3D axes and add a 2D heatmap inset on the right of `fig`.

    `discrete_legend`: optional list of (color, label) for a categorical legend
    (used by the dry-run classification heatmap).
    """
    # Make room on the right: 3D axes occupy the left ~60% of the panel.
    for ax in list(fig.axes):
        if hasattr(ax, "get_zlim"):
            ax.set_position([-0.12, -0.12, 0.66, 1.24])

    hm = fig.add_axes([0.66, 0.10, 0.27, 0.78])
    m = np.asarray(matrix)
    im = hm.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", origin="upper", interpolation="nearest")
    hm.set_title(title, fontsize=9)
    hm.set_xlabel("S", fontsize=8)
    hm.set_ylabel("A", fontsize=8)
    hm.set_xticks(range(len(s_labels)))
    hm.set_yticks(range(len(a_labels)))
    hm.set_xticklabels([str(x) for x in s_labels], fontsize=6, rotation=90)
    hm.set_yticklabels([str(x) for x in a_labels], fontsize=6)
    hm.tick_params(length=2)

    if discrete_legend is not None:
        handles = [Patch(facecolor=c, edgecolor="grey", label=l)
                   for c, l in discrete_legend]
        hm.legend(handles=handles, loc="upper left",
                  bbox_to_anchor=(1.02, 1.0),
                  fontsize=6, framealpha=0.8, borderaxespad=0.2)
    else:
        fig.colorbar(im, ax=hm, shrink=0.7, pad=0.04, fraction=0.05)


def _viz_center(node_attrs):
    """Return the *visualization* center used by visualize_nxgraph_3d.

    `_add_complete_viz_attributes_to_graph` adds a per-type Z offset (rooms get
    +2.0) into `node["viz"]["center"]`; the raw `node["center"]` is at floor
    height. Overlay lines must use the same offset coords or they land below
    the rendered markers.
    """
    viz = node_attrs.get("viz")
    if isinstance(viz, dict) and "center" in viz:
        return viz["center"]
    return node_attrs.get("center")


def _draw_gt_overlay(fig, prepared_graph, gt_pairs,
                     color="k", alpha=0.4, linewidth=1.5):
    """Overlay semitransparent lines between each GT (A, S) pair on the 3D axes.

    Run *after* the visualizer so the overlay sits on top of the matcher-styled
    edges without going through the merged-graph edge-attribute pipeline (which
    is a simple Graph and would replace existing edges).
    """
    if not gt_pairs:
        return
    g = prepared_graph.graph
    ax = next((a for a in fig.axes if hasattr(a, "get_zlim")), None)
    if ax is None:
        return
    for ai, si in gt_pairs:
        a_node = f"a_{ai}"
        s_node = f"s_{si}"
        if a_node not in g.nodes or s_node not in g.nodes:
            continue
        ac = _viz_center(g.nodes[a_node])
        sc = _viz_center(g.nodes[s_node])
        if ac is None or sc is None:
            continue
        ac = list(ac) + [0.0] * (3 - len(ac))
        sc = list(sc) + [0.0] * (3 - len(sc))
        ax.plot(
            [ac[0], sc[0]], [ac[1], sc[1]], [ac[2], sc[2]],
            color=color, alpha=alpha, linewidth=linewidth,
        )


_AXES_BOX = (-0.12, -0.12, 1.24, 1.24)  # x0, y0, w, h — let 3D axes overflow
_VIEW_ZOOM = 0.5  # < 1 = closer; smaller axis limits ⇒ data fills more of panel


def _post_process_figure(fig, extra_legend=None):
    """Make the 3D axes fill the panel, install scroll zoom, and rebuild the
    legend at the figure's upper-left.

    The generic 'edge' label produced by the visualizer is dropped — match-edges
    are described via panel-specific entries passed in ``extra_legend`` (a list
    of ``Line2D`` proxies)."""
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    for ax in fig.axes:
        if hasattr(ax, "get_zlim"):
            # mpl's default 3D projection leaves a wide margin inside the
            # subplot. Overflowing the bounds enlarges the visible cube.
            ax.set_position(_AXES_BOX)
        legend = ax.get_legend()
        handles, labels = ax.get_legend_handles_labels()
        if legend is not None:
            legend.remove()
        pairs = [(h, l) for h, l in zip(handles, labels) if l != "edge"]
        if extra_legend:
            for proxy in extra_legend:
                pairs.append((proxy, proxy.get_label()))
        if pairs:
            hs, ls = zip(*pairs)
            ax.legend(
                hs, ls,
                loc="upper left",
                bbox_to_anchor=(0.0, 1.0),
                bbox_transform=fig.transFigure,
                fontsize=8,
                framealpha=0.7,
                borderaxespad=0.2,
            )
    _scale_3d_axes(fig, _VIEW_ZOOM)
    _install_scroll_zoom(fig)


class Zoom3DToolbar(NavigationToolbar2QT):
    """matplotlib 3.6's rectangle-zoom doesn't work on 3D axes. Override the
    zoom button to do a step zoom on every 3D axes in the figure:
    plain click = zoom in, Shift+click = zoom out. Scroll wheel still works
    for finer control."""

    def zoom(self, *_args, **_kwargs):
        modifiers = QApplication.keyboardModifiers()
        factor = 1.25 if modifiers & Qt.ShiftModifier else 0.8
        _scale_3d_axes(self.canvas.figure, factor)
        self.canvas.draw_idle()
        # Our zoom is a one-shot action, not a modal toggle — immediately
        # uncheck the toolbar button so it doesn't appear stuck "active".
        action = self._actions.get("zoom")
        if action is not None:
            action.setChecked(False)


def _all_cids(fig):
    cb_dict = getattr(fig.canvas.callbacks, "callbacks", {})
    cids = set()
    for cids_dict in cb_dict.values():
        cids.update(cids_dict.keys())
    return cids


def _disconnect_except(fig, baseline_cids):
    """Drop every mpl callback not in ``baseline_cids``.

    Used between renders so the visualizer's on-motion hover handler and our
    scroll-zoom handler don't stack (each closure holds refs to dead artists
    after fig.clear, which both leaks memory and slows mouse interaction)."""
    cb_dict = getattr(fig.canvas.callbacks, "callbacks", {})
    for cids_dict in cb_dict.values():
        for cid in list(cids_dict.keys()):
            if cid not in baseline_cids:
                fig.canvas.mpl_disconnect(cid)


def _render_into_figure(graph, name, target_fig, extra_legend=None,
                        include_node_ids=False, enable_hover=False):
    """Clear ``target_fig`` and run visualize_nxgraph_3d into it.

    The visualizer's ``plt.figure(name)`` call is rerouted to return
    ``target_fig`` so the same Figure (and its already-connected Qt canvas)
    is reused across renders — avoiding the per-refresh cost of constructing
    new Figure/Canvas/toolbar widgets.

    ``enable_hover``: if False, the visualizer's ``motion_notify_event``
    hover-highlight handler is dropped entirely (it iterates O(n+e) artists
    + a full repaint per pixel of cursor movement — the dominant lag source
    for combined panels with bipartite cross-edges). If True, the handler
    is wrapped to bail when a mouse button is held, so drag-rotation stays
    smooth even with hover-highlight on.
    """
    target_fig.clear()

    real_figure = plt.figure
    real_close = plt.close
    real_show = plt.show
    real_connect = target_fig.canvas.mpl_connect

    def patched_figure(*args, **kwargs):
        return target_fig

    def patched_connect(event, callback):
        if event == "motion_notify_event":
            if not enable_hover:
                # Skip the visualizer's hover handler — saves O(n+e)
                # projection math + a draw_idle per cursor pixel.
                return -1
            # Hover on but suppress during drag.
            original = callback
            def guarded(evt):
                if evt.button is not None:
                    return
                return original(evt)
            return real_connect(event, guarded)
        return real_connect(event, callback)

    plt.figure = patched_figure
    plt.close = lambda *a, **kw: None
    plt.show = lambda *a, **kw: None
    target_fig.canvas.mpl_connect = patched_connect
    try:
        visualize_nxgraph_3d(
            graph, name, visualize_alone=False, include_node_ids=include_node_ids
        )
    finally:
        plt.figure = real_figure
        plt.close = real_close
        plt.show = real_show
        try:
            del target_fig.canvas.mpl_connect  # restore the class method
        except AttributeError:
            pass
    _post_process_figure(target_fig, extra_legend=extra_legend)


class GraphPanel(QWidget):
    """One visualization slot: header + mpl toolbar + embedded mpl canvas."""

    expand_toggled = pyqtSignal(object)  # emits self
    view_changed = pyqtSignal(float, float)  # elev, azim — user-driven rotation

    def __init__(self, title, parent=None, show_header=True):
        super().__init__(parent)
        self._fig = None
        self._canvas = None
        self._toolbar = None
        self._baseline_cids = frozenset()
        self._last_view = None
        self._title = title
        self._header = None
        self._expand_btn = None

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(2, 2, 2, 2)
        self._layout.setSpacing(2)

        if show_header:
            header_row = QHBoxLayout()
            header_row.setContentsMargins(0, 0, 0, 0)
            self._header = QLabel(title)
            self._header.setStyleSheet("font-weight: 600; font-size: 12px;")
            header_row.addWidget(self._header)
            header_row.addStretch(1)
            self._expand_btn = QToolButton()
            self._expand_btn.setText("⛶")
            self._expand_btn.setToolTip(
                "Toggle full-window — or double-click the plot. Esc to collapse."
            )
            self._expand_btn.setAutoRaise(True)
            self._expand_btn.clicked.connect(lambda: self.expand_toggled.emit(self))
            header_row.addWidget(self._expand_btn)
            self._layout.addLayout(header_row)

    def _ensure_fig(self):
        """First call: build an empty Figure + Canvas + toolbar, snapshot the
        toolbar's mpl callbacks so they survive subsequent figure clears."""
        if self._fig is not None:
            return
        self._fig = Figure()
        FigureCanvasQTAgg(self._fig)  # attaches as self._fig.canvas
        self._canvas = self._fig.canvas
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._canvas.setMinimumHeight(200)
        self._toolbar = Zoom3DToolbar(self._canvas, self)
        self._layout.addWidget(self._toolbar)
        self._layout.addWidget(self._canvas, stretch=1)
        # Double-click on the canvas toggles full-window for this panel.
        # View-sync fires on button-release (not motion) so dragging stays
        # smooth — the cross-panel rotation broadcast happens once per drag
        # end rather than once per pixel of cursor motion.
        self._canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self._canvas.mpl_connect("button_release_event", self._on_canvas_release)
        self._baseline_cids = frozenset(_all_cids(self._fig))

    def _on_canvas_press(self, event):
        if event.dblclick:
            self.expand_toggled.emit(self)

    def _on_canvas_release(self, event):
        if self._fig is None:
            return
        for ax in self._fig.axes:
            if not hasattr(ax, "get_zlim"):
                continue
            view = (ax.elev, ax.azim)
            if view != self._last_view:
                self._last_view = view
                self.view_changed.emit(view[0], view[1])
            break

    def apply_view(self, elev, azim):
        """Programmatically match another panel's rotation. Does not re-emit."""
        if self._fig is None:
            return
        changed = False
        for ax in self._fig.axes:
            if not hasattr(ax, "get_zlim"):
                continue
            if ax.elev != elev or ax.azim != azim:
                ax.view_init(elev=elev, azim=azim)
                changed = True
        if changed:
            self._last_view = (elev, azim)
            self._canvas.draw_idle()

    def set_graph(self, graph, name, extra_legend=None,
                  gt_overlay_pairs=None, include_node_ids=False,
                  matrix_inset=None, node_aura=None, enable_hover=False):
        """`matrix_inset`: optional dict with keys understood by
        `_draw_matrix_inset` (matrix, a_labels, s_labels, cmap, title, plus
        optional vmin/vmax/discrete_legend). When provided, the 3D plot is
        shrunk to ~60% width and a 2D heatmap is drawn on the right.

        `node_aura`: optional dict ``{"weights": {node_id: w}, "color": str,
        "size": int}`` — draws a colored halo behind each node with alpha =
        its weight. Used by the three bottom panels to surface per-node
        average correctness/incorrectness from the matrix.

        `enable_hover`: install the visualizer's hover-highlight handler.
        Off by default — it's expensive enough to dominate camera/toggle
        lag for combined panels with bipartite cross-edges.
        """
        self._ensure_fig()
        _disconnect_except(self._fig, self._baseline_cids)
        g_viz = _prepare_for_viz(graph)
        _render_into_figure(g_viz, name, self._fig,
                            extra_legend=extra_legend,
                            include_node_ids=include_node_ids,
                            enable_hover=enable_hover)
        if gt_overlay_pairs:
            _draw_gt_overlay(self._fig, g_viz, gt_overlay_pairs)
        if node_aura is not None:
            ax = next((a for a in self._fig.axes if hasattr(a, "get_zlim")), None)
            saved_lims = None
            if ax is not None:
                saved_lims = (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())
            _draw_node_aura(self._fig, g_viz, **node_aura)
            # ax.scatter on a 3D axes can retrigger autoscaling; restore the
            # limits set by `_scale_3d_axes` so the aura doesn't zoom the view.
            if saved_lims is not None:
                ax.set_xlim(saved_lims[0])
                ax.set_ylim(saved_lims[1])
                ax.set_zlim(saved_lims[2])
        if matrix_inset is not None:
            _draw_matrix_inset(self._fig, **matrix_inset)
        self._canvas.draw_idle()


class EditorPanel(QWidget):
    """Interactive 3D editor panel — embeds an ``InteractiveGraphVisualizer``.

    Mirrors ``GraphPanel``'s public shape (``set_graph``, ``expand_toggled``,
    ``view_changed`` / ``apply_view``) so the Dashboard can treat it identically
    for grid placement, fullscreen toggling, and view-sync. The IGV owns its own
    drawing of the 3D axes (clicks, hover, key shortcuts) — we just provide it a
    pre-existing Figure + Canvas via the ``fig=`` ctor arg.
    """

    expand_toggled = pyqtSignal(object)
    view_changed = pyqtSignal(float, float)

    def __init__(self, title, parent=None, show_header=True, enter_handler=None):
        super().__init__(parent)
        self._fig = None
        self._canvas = None
        self._toolbar = None
        self._editor = None
        self._last_view = None
        self._group_queue = _queue.Queue()
        self._graph_update_queue = _queue.Queue()
        # Optional pre-empt for Enter key. Signature: (igv) -> bool. Returning
        # True consumes the event before IGV's on_key runs (which would otherwise
        # finalize the active group and call create_room_from_planes).
        self._enter_handler = enter_handler
        self._title = title
        self._header = None
        self._expand_btn = None

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(2, 2, 2, 2)
        self._layout.setSpacing(2)

        if show_header:
            header_row = QHBoxLayout()
            header_row.setContentsMargins(0, 0, 0, 0)
            self._header = QLabel(title)
            self._header.setStyleSheet("font-weight: 600; font-size: 12px;")
            header_row.addWidget(self._header)
            header_row.addStretch(1)
            self._expand_btn = QToolButton()
            self._expand_btn.setText("⛶")
            self._expand_btn.setToolTip(
                "Toggle full-window — or double-click the plot. Esc to collapse."
            )
            self._expand_btn.setAutoRaise(True)
            self._expand_btn.clicked.connect(lambda: self.expand_toggled.emit(self))
            header_row.addWidget(self._expand_btn)
            self._layout.addLayout(header_row)

    def _ensure_fig(self):
        if self._fig is not None:
            return
        self._fig = Figure()
        # Match the GraphPanel layout: axes overflow the figure bounds so the
        # 3D plot (and its legend) fill the panel instead of sitting tiny in
        # the middle. subplots_adjust must be set before the IGV calls
        # add_subplot() so the initial position uses the wider bounds.
        self._fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        FigureCanvasQTAgg(self._fig)  # attaches as self._fig.canvas
        self._canvas = self._fig.canvas
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._canvas.setMinimumHeight(200)
        # `key_press_event` only fires when the canvas has keyboard focus;
        # standalone mpl windows grab focus automatically but an embedded Qt
        # canvas does not. StrongFocus lets clicks and tab give it focus.
        self._canvas.setFocusPolicy(Qt.StrongFocus)
        self._toolbar = Zoom3DToolbar(self._canvas, self)
        self._layout.addWidget(self._toolbar)
        self._layout.addWidget(self._canvas, stretch=1)
        # Double-click → expand. View-sync fires on button-release so drag
        # rotation stays smooth (the cross-panel broadcast happens once per
        # drag end, not once per cursor pixel).
        self._canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self._canvas.mpl_connect("button_release_event", self._on_canvas_release)
        # Belt-and-suspenders for key delivery: install a Qt event filter on
        # the canvas that translates Qt KeyPress → mpl key_press_event and
        # fires the IGV's on_key handler directly. mpl's own keyPressEvent
        # does this too when the canvas has focus, but in some Qt setups
        # focus gets stolen between the click and the next keypress; the
        # filter runs even if mpl's path isn't taken. Also delegate the
        # widget's focus to the canvas via setFocusProxy so tab-into-panel
        # lands on the canvas, not on the panel itself.
        self._canvas.installEventFilter(self)
        self.setFocusProxy(self._canvas)

    def eventFilter(self, obj, event):
        if (
            event.type() == QEvent.KeyPress
            and obj is self._canvas
            and self._editor is not None
        ):
            mpl_key = (
                self._canvas._get_key(event)
                if hasattr(self._canvas, "_get_key") else None
            )
            # GT-edit mode (and any future custom Enter behavior) pre-empts
            # the IGV's create_room_from_planes side effect on Enter. Handler
            # decides whether to consume; if so we skip the mpl callback path.
            if (
                mpl_key == "enter"
                and self._enter_handler is not None
                and self._enter_handler(self._editor)
            ):
                self._editor.draw_graph()
                return True
            if mpl_key is not None:
                mpl_event = _mpl_bb.KeyEvent(
                    "key_press_event", self._canvas, mpl_key, x=0, y=0,
                )
                self._canvas.callbacks.process("key_press_event", mpl_event)
                # Consume the Qt event so mpl's own keyPressEvent doesn't
                # also fire `callbacks.process` — that would double-trigger
                # IGV.on_key (e.g. create two rooms per Enter). mpl's
                # default-key handlers ('s' for save, 'r' for home view…)
                # are also connected through `callbacks.process`, so they
                # still run via our path above.
                return True
        return super().eventFilter(obj, event)

    def _on_canvas_press(self, event):
        # Grab keyboard focus so subsequent key presses are delivered to this
        # canvas (and translated by mpl to key_press_event for the IGV).
        self._canvas.setFocus()
        if event.dblclick:
            self.expand_toggled.emit(self)

    def _on_canvas_release(self, event):
        if self._fig is None:
            return
        for ax in self._fig.axes:
            if not hasattr(ax, "get_zlim"):
                continue
            view = (ax.elev, ax.azim)
            if view != self._last_view:
                self._last_view = view
                self.view_changed.emit(view[0], view[1])
            break

    def apply_view(self, elev, azim):
        if self._fig is None:
            return
        changed = False
        for ax in self._fig.axes:
            if not hasattr(ax, "get_zlim"):
                continue
            if ax.elev != elev or ax.azim != azim:
                ax.view_init(elev=elev, azim=azim)
                changed = True
        if changed:
            self._last_view = (elev, azim)
            self._canvas.draw_idle()

    def set_graph(self, graph, name, **_unused_kwargs):
        """(Re)build the editor for ``graph``.

        Accepts and ignores the extra kwargs used by ``GraphPanel.set_graph``
        (``extra_legend``, ``gt_overlay_pairs``, ``include_node_ids``,
        ``matrix_inset``, ``enable_hover``) so the Dashboard's panel-driving
        code can stay uniform across panel types.
        """
        self._ensure_fig()
        self._fig.clear()
        # subplots_adjust was set on the figure once in _ensure_fig but
        # `fig.clear()` resets the subplot params, so re-apply.
        self._fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        self._editor = InteractiveGraphVisualizer(
            graph=graph,
            image_name=name,
            group_queue=self._group_queue,
            graph_update_queue=self._graph_update_queue,
            logger=None,
            fig=self._fig,
        )
        # Overflow the axes so the 3D cube fills the canvas (same trick the
        # GraphPanels use). IGV.draw_graph doesn't touch ax.set_position, so
        # this stays across redraws (cla doesn't reset the explicit position).
        self._editor.ax.set_position(_AXES_BOX)
        self._canvas.draw_idle()


class SwitchablePanel(QWidget):
    """A panel that hosts a viz (GraphPanel) and an editor (EditorPanel) and
    flips between them via a header button.

    Exposes the same outer signal shape as the inner panels (``expand_toggled``,
    ``view_changed``) so the Dashboard grid + view-sync + fullscreen plumbing
    treats it identically. Inner panels' headers are suppressed; the
    SwitchablePanel owns the title + mode toggle + expand controls.
    """

    expand_toggled = pyqtSignal(object)
    view_changed = pyqtSignal(float, float)
    mode_changed = pyqtSignal(object, bool)  # self, is_editor

    def __init__(self, title, parent=None, enter_handler=None,
                 start_in_editor=False):
        super().__init__(parent)
        self._title = title

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(2, 2, 2, 2)
        self._layout.setSpacing(2)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        self._header = QLabel(title)
        self._header.setStyleSheet("font-weight: 600; font-size: 12px;")
        header_row.addWidget(self._header)
        header_row.addStretch(1)
        self._mode_btn = QToolButton()
        self._mode_btn.setAutoRaise(True)
        self._mode_btn.clicked.connect(self._toggle_mode)
        header_row.addWidget(self._mode_btn)
        self._expand_btn = QToolButton()
        self._expand_btn.setText("⛶")
        self._expand_btn.setToolTip(
            "Toggle full-window — or double-click the plot. Esc to collapse."
        )
        self._expand_btn.setAutoRaise(True)
        self._expand_btn.clicked.connect(lambda: self.expand_toggled.emit(self))
        header_row.addWidget(self._expand_btn)
        self._layout.addLayout(header_row)

        self._viz_panel = GraphPanel(title, show_header=False)
        self._editor_panel = EditorPanel(
            title, show_header=False, enter_handler=enter_handler,
        )
        for inner in (self._viz_panel, self._editor_panel):
            # Dblclick on either inner canvas bubbles up as our expand_toggled
            # so the Dashboard can move the whole SwitchablePanel between
            # grid/expand pages without caring which mode is active.
            inner.expand_toggled.connect(lambda _src: self.expand_toggled.emit(self))
            inner.view_changed.connect(self.view_changed.emit)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._viz_panel)     # index 0
        self._stack.addWidget(self._editor_panel)  # index 1
        self._layout.addWidget(self._stack, stretch=1)

        self.set_mode(start_in_editor, emit=False)

    @property
    def viz_panel(self):
        return self._viz_panel

    @property
    def editor_panel(self):
        return self._editor_panel

    @property
    def editor(self):
        """The IGV instance inside the editor panel, or None if it has never
        been set_graph'd."""
        return self._editor_panel._editor

    @property
    def is_editor(self):
        return self._stack.currentIndex() == 1

    def set_mode(self, editor, emit=True):
        self._stack.setCurrentIndex(1 if editor else 0)
        # Button label = the action the click performs (the *other* mode).
        self._mode_btn.setText("View" if editor else "Edit")
        self._mode_btn.setToolTip(
            "Switch to viewer" if editor else "Switch to editor"
        )
        if emit:
            self.mode_changed.emit(self, editor)

    def _toggle_mode(self):
        self.set_mode(not self.is_editor)

    def apply_view(self, elev, azim):
        # Keep both inner panels' 3D axes in sync so flipping modes doesn't
        # snap the camera back to the mpl default.
        self._viz_panel.apply_view(elev, azim)
        self._editor_panel.apply_view(elev, azim)


class Dashboard(QMainWindow):
    def __init__(self, matcher):
        super().__init__()
        self._matcher = matcher
        self.setWindowTitle(f"Graph Matching Dashboard — {matcher.display}")
        self.resize(1500, 850)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        top = QHBoxLayout()
        top.addWidget(QLabel("Environment:"))
        self._combo = QComboBox()
        envs = list_environments()
        self._combo.addItems(envs)
        top.addWidget(self._combo)
        top.addStretch(1)
        root.addLayout(top)

        # Two-page stack: page 0 = the grid of all panels, page 1 = a single
        # expanded panel. Clicking a panel's ⛶ button moves it between pages.
        self._stack = QStackedWidget()
        self._grid_page = QWidget()
        self._grid = QGridLayout(self._grid_page)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(2)
        self._expand_page = QWidget()
        self._expand_layout = QVBoxLayout(self._expand_page)
        self._expand_layout.setContentsMargins(0, 0, 0, 0)
        self._stack.addWidget(self._grid_page)
        self._stack.addWidget(self._expand_page)
        root.addWidget(self._stack, stretch=1)

        # Enter in any editor commits edits to downstream. For A/S that means
        # pulling the editor's `full_graph` into `_current_a/_current_s` and
        # re-running the matcher (same code path as the Recompute button).
        # For GT it means toggling the selected (a, s) pair in `_current_gt`.
        self._a_panel = SwitchablePanel("A", enter_handler=self._on_ase_editor_enter)
        self._s_panel = SwitchablePanel("S", enter_handler=self._on_ase_editor_enter)
        self._gt_panel = SwitchablePanel(
            "A + S — ground truth",
            enter_handler=self._on_gt_editor_enter,
        )
        self._dryrun_panel = GraphPanel("A + S — dry-run matcher")
        self._affinity_panel = GraphPanel("A + S — affinity")
        self._simnormed_panel = GraphPanel("A + S — sim_normed")

        # Metrics column for the right of the grid. Top is the combined
        # (all-types) box, below it three collapsible sections — one per
        # edge type (room-room, ws-ws, room-ws). Built before the grid
        # placement so the container exists when it's `addWidget`-ed.
        self._metrics_container = QWidget()
        _mc_layout = QVBoxLayout(self._metrics_container)
        _mc_layout.setContentsMargins(0, 0, 0, 0)
        _mc_layout.setSpacing(4)

        self._metrics_box = QGroupBox("Metrics")
        _combined_layout = QVBoxLayout(self._metrics_box)
        _combined_layout.setContentsMargins(8, 6, 8, 6)
        _combined_layout.setSpacing(2)
        self._metric_labels = _build_metric_rows(_combined_layout)
        _mc_layout.addWidget(self._metrics_box)

        # Per-edge-type sections. Each one is a collapsible header + metric
        # rows; refs to value labels live in `_metric_labels_by_type[key]`
        # so `_update_metrics` can fill them. Collapsed by default so the
        # combined box stays the focal point; arrow click expands.
        self._metric_labels_by_type = {}
        for _et_key in _EDGE_TYPE_PREDICATES:
            _sec = _CollapsibleSection(_et_key, expanded=False)
            self._metric_labels_by_type[_et_key] = _build_metric_rows(
                _sec.body_layout()
            )
            _mc_layout.addWidget(_sec)
        _mc_layout.addStretch(1)

        self._panel_positions = {}
        self._panels = []
        self._switchable_panels = (self._a_panel, self._s_panel, self._gt_panel)
        for panel, row, col in (
            (self._a_panel, 0, 0),
            (self._s_panel, 0, 1),
            (self._gt_panel, 0, 2),
            (self._affinity_panel, 1, 0),
            (self._simnormed_panel, 1, 1),
            (self._dryrun_panel, 1, 2),
        ):
            self._grid.addWidget(panel, row, col)
            self._panel_positions[panel] = (row, col)
            self._panels.append(panel)
            panel.expand_toggled.connect(self._toggle_panel_expand)
            panel.view_changed.connect(self._on_panel_view_changed)
        # Metrics column sits at row 1 col 3, aligned with the matching 3D
        # panels (row 0 col 3 stays empty). Column 3 has no stretch factor
        # so the column stays at its content width; the six 3D panels keep
        # their existing column shares.
        self._grid.addWidget(self._metrics_container, 1, 3)
        self._grid.setColumnStretch(0, 1)
        self._grid.setColumnStretch(1, 1)
        self._grid.setColumnStretch(2, 1)
        self._grid.setColumnStretch(3, 0)
        for sp in self._switchable_panels:
            sp.mode_changed.connect(self._on_panel_mode_changed)

        self._expanded_panel = None

        esc = QShortcut(QKeySequence(Qt.Key_Escape), self)
        esc.activated.connect(self._collapse_expanded)
        QApplication.instance().installEventFilter(self)

        # Controls grouped by scope: "General" affects every panel (A/S + all
        # combined views), "Matching" affects only the three bottom panels
        # (Hungarian perm / affinity / sim_normed).
        controls = QHBoxLayout()

        general_box = QGroupBox("General")
        general_row = QHBoxLayout(general_box)
        general_row.setContentsMargins(8, 4, 8, 4)

        general_row.addWidget(QLabel("B transform —"))
        general_row.addWidget(QLabel("X:"))
        self._dx_spin = QDoubleSpinBox()
        self._dx_spin.setRange(-1000.0, 1000.0)
        self._dx_spin.setDecimals(2)
        self._dx_spin.setSingleStep(0.5)
        general_row.addWidget(self._dx_spin)

        general_row.addWidget(QLabel("Y:"))
        self._dy_spin = QDoubleSpinBox()
        self._dy_spin.setRange(-1000.0, 1000.0)
        self._dy_spin.setDecimals(2)
        self._dy_spin.setSingleStep(0.5)
        general_row.addWidget(self._dy_spin)

        general_row.addWidget(QLabel("rot:"))
        self._theta_spin = QDoubleSpinBox()
        self._theta_spin.setRange(-360.0, 360.0)
        self._theta_spin.setDecimals(1)
        self._theta_spin.setSingleStep(1.0)
        self._theta_spin.setSuffix(" °")
        general_row.addWidget(self._theta_spin)

        general_row.addSpacing(6)
        self._align_btn = QPushButton("Align centers")
        self._align_btn.setToolTip(
            "Stores the centroid-to-centroid offset for S internally. "
            "Nothing changes visually — the display layout is untouched. "
            "Press Recompute afterwards to re-run the matcher with S's "
            "features shifted onto A's origin."
        )
        self._align_btn.clicked.connect(self._on_align_graphs)
        general_row.addWidget(self._align_btn)

        general_row.addSpacing(12)
        self._node_ids_cb = QCheckBox("Show node IDs")
        self._node_ids_cb.setChecked(False)
        self._node_ids_cb.toggled.connect(self._kick_refresh)
        general_row.addWidget(self._node_ids_cb)

        self._matrix_cb = QCheckBox("Show matrix")
        self._matrix_cb.setChecked(False)
        self._matrix_cb.toggled.connect(self._kick_refresh)
        general_row.addWidget(self._matrix_cb)

        self._hover_cb = QCheckBox("Hover highlight")
        self._hover_cb.setChecked(False)
        self._hover_cb.setToolTip(
            "Highlight nodes/edges under the cursor. Off by default — the "
            "underlying handler iterates every node/edge with a 3D projection "
            "and triggers a full repaint per cursor pixel, which dominates "
            "lag on panels with bipartite cross-edges. When on, it's still "
            "suppressed during drag rotation."
        )
        self._hover_cb.toggled.connect(self._kick_refresh)
        general_row.addWidget(self._hover_cb)

        controls.addWidget(general_box)

        matching_box = QGroupBox("Matching")
        matching_row = QHBoxLayout(matching_box)
        matching_row.setContentsMargins(8, 4, 8, 4)

        self._gt_overlay_cb = QCheckBox("GT overlay on dry-run panels")
        self._gt_overlay_cb.setChecked(False)
        self._gt_overlay_cb.toggled.connect(self._kick_refresh)
        matching_row.addWidget(self._gt_overlay_cb)

        matching_row.addWidget(QLabel("Highlight:"))
        self._highlight_mode_combo = QComboBox()
        # Order matches the _HIGHLIGHT_MODES list below; keep in sync.
        self._highlight_mode_combo.addItems(["Correct", "Incorrect", "Value only"])
        self._highlight_mode_combo.setCurrentIndex(0)
        self._highlight_mode_combo.setToolTip(
            "Correct: GT-true edges with value→1 and GT-false edges with "
            "value→0 are most visible (correct predictions stand out).\n"
            "Incorrect: only FP (GT-false pairs with high value) are shown in "
            "red — GT-true pairs are hidden entirely.\n"
            "Value only: GT membership is ignored — every edge uses a single "
            "neutral color and transparency tracks the raw normalized matrix "
            "value."
        )
        self._highlight_mode_combo.currentIndexChanged.connect(self._kick_refresh)
        matching_row.addWidget(self._highlight_mode_combo)

        self._node_aura_cb = QCheckBox("Highlight per node")
        self._node_aura_cb.setChecked(False)
        self._node_aura_cb.setToolTip(
            "Draws a colored halo behind every node in the bottom three "
            "panels. Halo alpha = mean of the same per-edge highlight "
            "weight (so it tracks 'Highlight correct'); halo color is green "
            "when highlighting correct matches, red when highlighting "
            "incorrect ones."
        )
        self._node_aura_cb.toggled.connect(self._kick_refresh)
        matching_row.addWidget(self._node_aura_cb)

        matching_row.addSpacing(8)
        self._recompute_btn = QPushButton("Recompute")
        self._recompute_btn.setToolTip(
            "Re-run dry-run matching on the currently edited A and S "
            "graphs and refresh GT / dry-run / affinity / Sinkhorn panels."
        )
        self._recompute_btn.clicked.connect(self._on_recompute)
        matching_row.addWidget(self._recompute_btn)

        controls.addWidget(matching_box)

        controls.addStretch(1)
        root.addLayout(controls)

        self._current_a = None
        self._current_s = None
        self._current_gt = set()
        self._current_pred = set()
        self._current_ints = None
        self._current_a_nodes = []
        self._current_s_nodes = []
        self._current_env = None
        self._match_time_s = None
        self._shared_view = None  # (elev, azim) propagated across all panels
        self._gt_editor_key = None  # guards GT editor rebuilds in _refresh_combined
        self._undo_stack = []

        # Debounce timer: a spinbox drag or a burst of toggles coalesces into
        # one refresh ~120 ms after the user pauses. Connections go through
        # `_kick_refresh` rather than directly to `_refresh_timer.start` —
        # Qt would otherwise pass each signal's payload (the bool from
        # `toggled`, the double from `valueChanged`) into the `start(int msec)`
        # overload and overwrite the configured interval (e.g. toggling a
        # checkbox would set the timer to 0 or 1 ms, defeating the debounce).
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(120)
        self._refresh_timer.timeout.connect(self._refresh_combined)

        self._combo.currentTextChanged.connect(self._on_env_change)
        self._dx_spin.valueChanged.connect(self._kick_refresh)
        self._dy_spin.valueChanged.connect(self._kick_refresh)
        self._theta_spin.valueChanged.connect(self._kick_refresh)

        if envs:
            self._on_env_change(envs[0])

    def _on_env_change(self, env_name):
        if not env_name:
            return
        self._undo_stack.clear()
        try:
            a, s, gt = load_env_graphs(env_name)
            _t0 = time.perf_counter()
            pred, ints, a_nodes, s_nodes = self._matcher.match(a, s)
            self._match_time_s = time.perf_counter() - _t0
        except Exception as exc:
            print(
                f"[ERROR] Failed to load environment '{env_name}': {exc}",
                file=sys.stderr,
            )
            return
        self._current_a, self._current_s = a, s
        self._current_gt = gt
        self._current_pred = pred
        self._current_ints = ints
        self._current_a_nodes = a_nodes
        self._current_s_nodes = s_nodes
        self._current_env = env_name
        # A/S panels host both a viz view and an editor view (toggled per
        # panel). Render both inner panels at env load so flipping modes is
        # instant. Editors get prepared copies (3D + viz attrs flattened to
        # the keys IGV reads) so the matcher inputs stay pristine.
        self._render_side_panel(self._a_panel, a, "A")
        self._render_side_panel(self._s_panel, s, "S")

        init_dx = _initial_dx(a, s)
        # Reset controls to the auto-computed starting transform without firing
        # three intermediate refreshes.
        blockers = [
            QSignalBlocker(self._dx_spin),
            QSignalBlocker(self._dy_spin),
            QSignalBlocker(self._theta_spin),
        ]
        self._dx_spin.setValue(init_dx)
        self._dy_spin.setValue(0.0)
        self._theta_spin.setValue(0.0)
        del blockers

        self._refresh_combined()
        self._apply_shared_view()

    def _render_side_panel(self, panel, graph, role,
                           render_viz=True, render_editor=True):
        """Render the requested inner panels (viz / editor) of an A/S
        SwitchablePanel against ``graph``.

        ``_prepare_for_viz`` / ``_prepare_for_editor`` deepcopy the input,
        so calling after the editor has mutated ``graph`` rebuilds the IGV
        against the latest state without aliasing the editor's working copy.
        Editor re-render rebuilds the IGV and loses transient UI state
        (active_groups, toggles), so callers that don't need it (recompute,
        mode switch) should pass ``render_editor=False``.
        """
        show_ids = self._node_ids_cb.isChecked()
        enable_hover = self._hover_cb.isChecked()
        if render_viz:
            panel.viz_panel.set_graph(
                graph,
                f"{self._current_env} - {role}",
                include_node_ids=show_ids,
                enable_hover=enable_hover,
            )
        if render_editor:
            panel.editor_panel.set_graph(
                _prepare_for_editor(graph),
                f"{self._current_env} - {role} editor",
            )

    def _on_ase_editor_enter(self, igv):
        """Enter in an A or S editor commits the edited graph downstream:
        pull the editor's ``full_graph`` into ``_current_a/_current_s`` and
        re-run the matcher + refresh combined panels. Same flow as the
        Recompute button — Enter is the keyboard shortcut for it.
        Returns True so the eventFilter consumes the key.
        """
        self._on_recompute()
        return True

    def _on_gt_editor_enter(self, igv):
        """Commit GT edits on Enter. Two complementary paths:

        1. **Shift+E + Enter (add a pair).** Scan the IGV's full_graph for
           cross-edges with ``type == "manual_edge"`` (the marker that
           ``create_edge_between_selected`` writes). Strip the ``a_``/``s_``
           prefixes and union the pair into ``_current_gt``. Multiple
           manual edges fold in one go.
        2. **Select-2 + Enter (toggle, used to remove).** If exactly one
           ``a_*`` and one ``s_*`` are selected, toggle that pair in
           ``_current_gt``. Existing pairs get removed this way (you can't
           easily delete edges in IGV).

        Both paths can fire on the same Enter without conflict: Shift+E
        clears the selection, so users rarely have both at once; and the
        manual_edge filter prevents the scan from re-adding pairs the
        selection-toggle just removed (existing GT pair edges from
        ``_build_gt_editor_graph`` carry no ``type`` attribute).

        Always consumes Enter (returns True) so IGV's key handlers — and
        the obsolete Enter→create_room path — never run.
        """
        # 1. Collect selection, then clear (so a stale highlight doesn't
        #    linger after a no-op).
        all_selected = set()
        for grp in igv.active_groups.values():
            all_selected.update(grp)
        for grp_type in list(igv.active_groups.keys()):
            igv.active_groups[grp_type] = set()

        pre = self._snapshot()
        changed = False

        # 2. Selection-toggle path (remove / explicit toggle).
        a_sel = [n for n in all_selected if isinstance(n, str) and n.startswith("a_")]
        s_sel = [n for n in all_selected if isinstance(n, str) and n.startswith("s_")]
        if len(a_sel) == 1 and len(s_sel) == 1:
            pair = (a_sel[0][2:], s_sel[0][2:])
            if pair in self._current_gt:
                self._current_gt.discard(pair)
            else:
                self._current_gt.add(pair)
            changed = True

        # 3. Manual-edge absorption path (add via Shift+E).
        nx_g = igv.full_graph.graph if hasattr(igv.full_graph, "graph") else igv.full_graph
        for u, v, data in nx_g.edges(data=True):
            if data.get("type") != "manual_edge":
                continue
            if not (isinstance(u, str) and isinstance(v, str)):
                continue
            if u.startswith("a_") and v.startswith("s_"):
                pair = (u[2:], v[2:])
            elif u.startswith("s_") and v.startswith("a_"):
                pair = (v[2:], u[2:])
            else:
                continue
            if pair not in self._current_gt:
                self._current_gt.add(pair)
                changed = True

        if changed:
            self._push_undo(pre)
            # Rebuilds GT editor + viz from _current_gt — manual_edge entries
            # disappear from the IGV graph and reappear as proper GT edges.
            self._refresh_combined()
            self._apply_shared_view()
        return True  # always consume Enter in GT-edit mode

    def _on_panel_mode_changed(self, panel, is_editor):
        """When an A/S panel flips edit→view, pull the editor's current
        ``full_graph`` into the dashboard's tracked state so subsequent
        refreshes (and the viz inner panel) reflect any in-flight edits.

        For the GT panel, the dashboard never needs to pull from the editor's
        graph (GT data lives in ``self._current_gt`` and is mutated directly
        via the enter handler), so the GT case is a no-op here.
        """
        if is_editor:
            return
        if panel is self._a_panel and panel.editor is not None:
            self._current_a = panel.editor.full_graph
            self._render_side_panel(
                self._a_panel, self._current_a, "A", render_editor=False,
            )
        elif panel is self._s_panel and panel.editor is not None:
            self._current_s = panel.editor.full_graph
            self._render_side_panel(
                self._s_panel, self._current_s, "S", render_editor=False,
            )

    def _on_align_graphs(self):
        """Shift S's features so its centroid overlaps A's centroid.

        ``_current_s`` is mutated in-place, then the visualization spinboxes
        are compensated by the same delta so the combined panels (GT, dry-run,
        etc.) continue to show A and S side-by-side — only the S standalone
        panel (and its editor) will reflect the translated graph. Press
        Recompute afterwards to re-run the matcher on the aligned S.
        """
        if self._current_a is None or self._current_s is None:
            return
        align_dx, align_dy = _align_transform(self._current_a, self._current_s)
        if abs(align_dx) < 1e-9 and abs(align_dy) < 1e-9:
            return  # already aligned, nothing to do

        self._push_undo()
        # Shift _current_s features permanently.
        _transform_b_inplace(self._current_s, align_dx, align_dy, 0.0)

        # Subtract the same delta from the visualization spinboxes so that
        # combined-panel displays (which add spinbox offset on top of features)
        # continue to render S at the same screen position.
        new_dx = self._dx_spin.value() - align_dx
        new_dy = self._dy_spin.value() - align_dy
        blockers = [QSignalBlocker(self._dx_spin), QSignalBlocker(self._dy_spin)]
        self._dx_spin.setValue(new_dx)
        self._dy_spin.setValue(new_dy)
        del blockers

        # id(_current_s) is unchanged but its content changed — force GT editor
        # key to invalidate so the editor rebuilds with the new coordinates.
        self._gt_editor_key = None

        # Re-render the S standalone panel (viz + editor) to show the shifted
        # graph, then refresh the combined panels so the GT panel (and others)
        # also redraw. The spinbox compensation means S appears at the same
        # visual position in the combined panels — but the GT editor rebuild
        # (triggered by _gt_editor_key = None above) still needs to run, and
        # _apply_shared_view is called at the tail of _refresh_combined.
        self._render_side_panel(self._s_panel, self._current_s, "S")
        self._refresh_combined()

    def _kick_refresh(self, *_args):
        """Restart the debounce timer ignoring whatever payload the signal
        carries. See the comment next to `_refresh_timer` for why we don't
        connect signals directly to `QTimer.start`."""
        self._refresh_timer.start()

    _UNDO_LIMIT = 50

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            print(f"[UNDO DBG] KeyPress key={event.key():#x} modifiers={int(event.modifiers()):#x}")
            if event.key() == Qt.Key_Z and event.modifiers() & Qt.ShiftModifier:
                print(f"[UNDO DBG] Shift+Z intercepted, stack size={len(self._undo_stack)}")
                self._on_undo()
                return True
            if event.key() == Qt.Key_C and event.modifiers() & Qt.ShiftModifier:
                for sp in (self._a_panel, self._s_panel):
                    ep = sp.editor_panel
                    if sp.is_editor and ep._canvas is not None and ep._canvas is obj and sp.editor is not None:
                        self._split_selected_plane(sp.editor)
                        return True
        return False

    # ── Plane split (Shift+C) ─────────────────────────────────────────────────

    _SPLIT_GAP = 0.01  # gap between the two sub-planes so they don't share an endpoint

    def _split_selected_plane(self, igv):
        """Split the single selected Line node at its midpoint into two sub-planes.

        Sub-plane 1 : original start → midpoint
        Sub-plane 2 : (midpoint + gap * tangent) → original end

        Both sub-planes inherit all attributes of the original (normal, type,
        viz style, edges to neighbours). Their center is computed as the
        midpoint of their own start/end points along the tangent direction.
        """
        # ── 1. Identify the selected node ────────────────────────────────────
        all_selected = set()
        for grp in igv.active_groups.values():
            all_selected.update(grp)
        if len(all_selected) != 1:
            print(f"[SPLIT] need exactly 1 node selected, got {len(all_selected)}")
            return

        node_id = next(iter(all_selected))
        nx_g = igv.full_graph.graph if hasattr(igv.full_graph, "graph") else igv.full_graph

        if node_id not in nx_g.nodes:
            print(f"[SPLIT] node {node_id!r} not found in graph")
            return

        attrs = dict(nx_g.nodes[node_id])

        # ── 2. Confirm Line type ──────────────────────────────────────────────
        viz_type = attrs.get("viz_type") or (attrs.get("viz") or {}).get("type", "Point")
        if viz_type != "Line":
            print(f"[SPLIT] node {node_id!r} is '{viz_type}', not 'Line' — nothing to split")
            return

        # ── 3. Extract start / end points from limits ─────────────────────────
        # limits lives in attrs["limits"], attrs["viz"]["limits"], or attrs["viz_data"]
        limits = (
            attrs.get("limits")
            or (attrs.get("viz") or {}).get("limits")
            or attrs.get("viz_data")
        )
        if limits is None:
            print(f"[SPLIT] node {node_id!r} has no limits / viz_data")
            return

        lim = np.asarray(limits, dtype=float)
        if lim.ndim < 2 or lim.shape[0] < 2:
            print(f"[SPLIT] limits shape {lim.shape} is degenerate")
            return

        p_start = lim[0].copy()
        p_end   = lim[-1].copy()

        direction = p_end - p_start
        dir_len   = float(np.linalg.norm(direction))
        if dir_len < 1e-9:
            print("[SPLIT] plane has zero length — nothing to split")
            return

        tangent  = direction / dir_len          # unit vector along the plane
        midpoint = (p_start + p_end) / 2.0

        # ── 4. Compute geometry for each half ─────────────────────────────────
        # Sub-plane 1
        lim1    = np.array([p_start, midpoint])
        center1 = (p_start + midpoint) / 2.0   # midpoint of the first half

        # Sub-plane 2 — start slightly past the midpoint to avoid overlap
        p2_start = midpoint + self._SPLIT_GAP * tangent
        lim2     = np.array([p2_start, p_end])
        center2  = (p2_start + p_end) / 2.0    # midpoint of the second half

        # ── 5. Build attribute dicts for the two new nodes ────────────────────
        def _sub_attrs(lim_arr, c_arr):
            sub = copy.deepcopy(attrs)
            sub["limits"]   = lim_arr.tolist()
            sub["center"]   = c_arr.tolist()
            sub["viz_data"] = lim_arr.tolist()
            nested = sub.get("viz")
            if isinstance(nested, dict):
                nested = dict(nested)
                nested["limits"] = lim_arr.tolist()
                nested["center"] = c_arr.tolist()
                sub["viz"] = nested
            return sub

        attrs1 = _sub_attrs(lim1, center1)
        attrs2 = _sub_attrs(lim2, center2)

        # ── 6. Generate two unique integer node IDs ───────────────────────────
        existing = set(nx_g.nodes())
        int_ids  = {int(n) for n in existing if isinstance(n, int)}
        next_id  = max(int_ids, default=0) + 1
        while next_id in existing:
            next_id += 1
        new_id1 = next_id
        next_id += 1
        while next_id in existing:
            next_id += 1
        new_id2 = next_id

        # ── 7. Swap original node for the two halves ──────────────────────────
        neighbors = [(nbr, dict(nx_g[node_id][nbr])) for nbr in nx_g.neighbors(node_id)]
        nx_g.remove_node(node_id)
        nx_g.add_node(new_id1, **attrs1)
        nx_g.add_node(new_id2, **attrs2)
        for nbr, edata in neighbors:
            if nbr in nx_g.nodes:
                nx_g.add_edge(new_id1, nbr, **edata)
                nx_g.add_edge(new_id2, nbr, **edata)

        # ── 8. Clear selection and redraw ─────────────────────────────────────
        for key in list(igv.active_groups.keys()):
            igv.active_groups[key] = set()
        igv.draw_graph()
        print(f"[SPLIT] {node_id!r} → {new_id1} (start→mid) + {new_id2} (mid+gap→end)")

    def _snapshot(self):
        return {
            "gt": set(self._current_gt),
            "a": copy.deepcopy(self._current_a),
            "s": copy.deepcopy(self._current_s),
            "pred": set(self._current_pred),
            "ints": (
                {k: np.copy(v) for k, v in self._current_ints.items()}
                if self._current_ints is not None else None
            ),
            "a_nodes": list(self._current_a_nodes),
            "s_nodes": list(self._current_s_nodes),
            "dx": self._dx_spin.value(),
            "dy": self._dy_spin.value(),
            "theta": self._theta_spin.value(),
        }

    def _push_undo(self, snapshot=None):
        try:
            self._undo_stack.append(snapshot if snapshot is not None else self._snapshot())
            if len(self._undo_stack) > self._UNDO_LIMIT:
                self._undo_stack.pop(0)
            print(f"[UNDO DBG] pushed, stack size={len(self._undo_stack)}")
        except Exception as exc:
            print(f"[UNDO DBG] push FAILED: {exc}")

    def _on_undo(self):
        if not self._undo_stack:
            print("[UNDO DBG] _on_undo called but stack is empty")
            return
        print(f"[UNDO DBG] restoring from stack (size was {len(self._undo_stack)})")
        snap = self._undo_stack.pop()
        try:
            self._current_gt = snap["gt"]
            self._current_a = snap["a"]
            self._current_s = snap["s"]
            self._current_pred = snap["pred"]
            self._current_ints = snap["ints"]
            self._current_a_nodes = snap["a_nodes"]
            self._current_s_nodes = snap["s_nodes"]
            blockers = [
                QSignalBlocker(self._dx_spin),
                QSignalBlocker(self._dy_spin),
                QSignalBlocker(self._theta_spin),
            ]
            self._dx_spin.setValue(snap["dx"])
            self._dy_spin.setValue(snap["dy"])
            self._theta_spin.setValue(snap["theta"])
            del blockers
            self._gt_editor_key = None  # force GT editor rebuild
            self._render_side_panel(self._a_panel, self._current_a, "A")
            self._render_side_panel(self._s_panel, self._current_s, "S")
            self._refresh_combined()
            self._apply_shared_view()
            print("[UNDO DBG] restore complete")
        except Exception as exc:
            import traceback
            print(f"[UNDO DBG] restore FAILED: {exc}")
            traceback.print_exc()

    def _on_recompute(self):
        """Pull the edited graphs out of the editor panels and re-run the
        dry-run matcher on them, then refresh GT / dry-run / affinity /
        Sinkhorn panels with the new prediction + intermediates.

        The dry-run cache (`_dry_run_pgm._get_or_make`) is keyed on
        (name, |V|, |E|), so any edit that adds/removes a node or edge
        naturally invalidates the entry and forces a fresh computation.

        If the user only ever stayed in view mode and never opened the editor,
        the IGVs are still None — in that case we just recompute against the
        unedited graphs already in ``_current_a`` / ``_current_s``.
        """
        ed_a = self._a_panel.editor
        ed_s = self._s_panel.editor
        new_a = ed_a.full_graph if ed_a is not None else self._current_a
        new_s = ed_s.full_graph if ed_s is not None else self._current_s
        if new_a is None or new_s is None:
            return
        self._push_undo()
        # Always force a fresh GNN forward pass. The cache key only covers
        # (name, |V|, |E|), so feature-only changes (e.g. "Align centers",
        # node position edits) would otherwise return stale cached results.
        self._matcher.clear_cache()
        print("[RECOMPUTING matching WITH GNN matcher]")

        # Editor graphs have 3D numpy coords after from_2D_to_3D(). Deep-copy
        # before sanitizing to 2D so the IGV's live graph object is not mutated
        # (it holds a reference and relies on its coords remaining 3D).
        def _to_2d(gw):
            g = copy.deepcopy(gw)
            for _, attrs in g.graph.nodes(data=True):
                for key in ("center", "normal"):
                    if key in attrs:
                        v = attrs[key]
                        attrs[key] = (v.tolist() if hasattr(v, "tolist") else list(v))[:2]
            return g

        self._current_a = _to_2d(new_a)
        self._current_s = _to_2d(new_s)

        # _current_s already carries aligned features if "Align centers" was
        # pressed before Recompute — no additional transform needed here.
        _t0 = time.perf_counter()
        pred, ints, a_nodes, s_nodes = self._matcher.match(self._current_a, self._current_s)
        self._match_time_s = time.perf_counter() - _t0
        self._current_pred = pred
        self._current_ints = ints
        self._current_a_nodes = a_nodes
        self._current_s_nodes = s_nodes
        # Viz inner panels re-render against the sanitized 2D base graphs;
        # editor IGVs already reflect the edits, so no rebuild needed.
        self._render_side_panel(self._a_panel, self._current_a, "A", render_editor=False)
        self._render_side_panel(self._s_panel, self._current_s, "S", render_editor=False)
        self._refresh_combined()
        self._apply_shared_view()

    def _refresh_combined(self):
        if self._current_a is None or self._current_s is None:
            return
        show_ids = self._node_ids_cb.isChecked()
        show_matrix = self._matrix_cb.isChecked()
        enable_hover = self._hover_cb.isChecked()
        a_nodes = self._current_a_nodes
        s_nodes = self._current_s_nodes
        self._update_metrics(a_nodes, s_nodes)
        dx, dy, theta = (
            self._dx_spin.value(),
            self._dy_spin.value(),
            self._theta_spin.value(),
        )
        gt_combined = _combine_graphs(
            self._current_a,
            self._current_s,
            dx, dy, theta,
            _gt_edge_style_fn(self._current_gt),
        )
        gt_inset = None
        if show_matrix:
            gt_inset = {
                "matrix": _gt_matrix(self._current_gt, a_nodes, s_nodes),
                "a_labels": a_nodes,
                "s_labels": s_nodes,
                "cmap": "Greens",
                "title": "GT permutation",
                "vmin": 0.0,
                "vmax": 1.0,
            }
        self._gt_panel.viz_panel.set_graph(
            gt_combined,
            f"{self._current_env} - A + S (GT)",
            extra_legend=[_make_match_legend_proxy("green", "GT pair")],
            include_node_ids=show_ids,
            matrix_inset=gt_inset,
            enable_hover=enable_hover,
        )
        # GT editor: only GT cross-edges are drawn (vs. the full n_a×n_s
        # mesh on the viz side). Selecting 1 a_* + 1 s_* + Enter toggles a
        # pair — see `_on_gt_editor_enter`. Rebuild only when the underlying
        # data changes (env / graphs / GT pairs) — not on every spinbox tick —
        # so editor state (active groups, node selections) survives purely
        # visual updates like transform adjustments.
        gt_editor_key = (
            self._current_env,
            frozenset(self._current_gt),
            id(self._current_a),
            id(self._current_s),
        )
        if gt_editor_key != self._gt_editor_key:
            self._gt_editor_key = gt_editor_key
            gt_editor_graph = _build_gt_editor_graph(
                self._current_a, self._current_s,
                dx, dy, theta,
                self._current_gt,
            )
            self._gt_panel.editor_panel.set_graph(
                gt_editor_graph,
                f"{self._current_env} - GT editor",
            )
        gt_overlay = self._current_gt if self._gt_overlay_cb.isChecked() else None
        gt_legend = (
            [_make_match_legend_proxy("black", "GT (overlay)", linewidth=1.5, alpha=0.4)]
            if gt_overlay else []
        )

        # Combo index → mode token (kept in sync with the QComboBox addItems
        # order in __init__). Three modes: GT-correct highlighting, GT-incorrect
        # highlighting, and a GT-agnostic value-only view.
        highlight_mode = ("correct", "incorrect", "value")[
            self._highlight_mode_combo.currentIndex()
        ]
        title_suffix = "value only" if highlight_mode == "value" else f"highlight {highlight_mode}"
        # In "value" mode every edge uses a single neutral color (no GT
        # distinction). In the other two modes, green = GT-true, red =
        # GT-false; alpha+linewidth track how strongly that edge matches
        # the chosen "correct" vs "incorrect" emphasis.
        if highlight_mode == "value":
            gt_value_legend = [
                _make_match_legend_proxy(_VALUE_ONLY_COLOR, "matrix value (α∝value)"),
            ]
            aura_color = _VALUE_ONLY_COLOR
            aura_label = "node aura (α∝mean value)"
        elif highlight_mode == "correct":
            gt_value_legend = [
                _make_match_legend_proxy("green", "GT-true (α∝value)"),
                _make_match_legend_proxy("red", "GT-false (α∝1−value)"),
            ]
            aura_color = "green"
            aura_label = "node aura (α∝mean correct)"
        else:  # incorrect
            gt_value_legend = [
                _make_match_legend_proxy("red", "FP — GT-false with high value (α∝value)"),
            ]
            aura_color = "red"
            aura_label = "node aura (α∝mean FP weight)"
        show_aura = self._node_aura_cb.isChecked()
        aura_legend = (
            [_make_match_legend_proxy(aura_color, aura_label)] if show_aura else []
        )

        def _aura_for(matrix):
            a_w, s_w = _node_aura_weights(
                matrix,
                self._current_a_nodes, self._current_s_nodes,
                self._current_gt, highlight_mode,
            )
            return {"weights": {**a_w, **s_w}, "color": aura_color}

        if self._current_ints is not None:
            perm_combined = _combine_graphs(
                self._current_a,
                self._current_s,
                dx, dy, theta,
                _gt_value_edge_style_fn(
                    self._current_ints["perm"],
                    self._current_a_nodes,
                    self._current_s_nodes,
                    self._current_gt,
                    mode=highlight_mode,
                ),
            )
            dryrun_inset = None
            if show_matrix:
                dryrun_inset = {
                    "matrix": _classification_matrix(
                        self._current_gt, self._current_pred, a_nodes, s_nodes),
                    "a_labels": a_nodes,
                    "s_labels": s_nodes,
                    "cmap": _CLASSIFICATION_CMAP,
                    "title": "classification",
                    "vmin": -0.5,
                    "vmax": 3.5,
                    "discrete_legend": [
                        ("white", "TN"),
                        ("red", "FP"),
                        ("green", "TP"),
                        ("orange", "FN"),
                    ],
                }
            self._dryrun_panel.set_graph(
                perm_combined,
                f"{self._current_env} - A + S (Hungarian perm, {title_suffix})",
                extra_legend=[*gt_value_legend, *gt_legend, *aura_legend],
                gt_overlay_pairs=gt_overlay,
                include_node_ids=show_ids,
                matrix_inset=dryrun_inset,
                node_aura=_aura_for(self._current_ints["perm"]) if show_aura else None,
                enable_hover=enable_hover,
            )

            affinity_combined = _combine_graphs(
                self._current_a,
                self._current_s,
                dx, dy, theta,
                _gt_value_edge_style_fn(
                    self._current_ints["affinity"],
                    self._current_a_nodes,
                    self._current_s_nodes,
                    self._current_gt,
                    mode=highlight_mode,
                ),
            )
            affinity_inset = None
            if show_matrix:
                affinity_inset = {
                    "matrix": np.asarray(self._current_ints["affinity"]),
                    "a_labels": a_nodes,
                    "s_labels": s_nodes,
                    "cmap": "magma",
                    "title": "affinity",
                }
            self._affinity_panel.set_graph(
                affinity_combined,
                f"{self._current_env} - A + S (affinity, {title_suffix})",
                extra_legend=[*gt_value_legend, *gt_legend, *aura_legend],
                gt_overlay_pairs=gt_overlay,
                include_node_ids=show_ids,
                matrix_inset=affinity_inset,
                node_aura=_aura_for(self._current_ints["affinity"]) if show_aura else None,
                enable_hover=enable_hover,
            )

            simnormed_combined = _combine_graphs(
                self._current_a,
                self._current_s,
                dx, dy, theta,
                _gt_value_edge_style_fn(
                    self._current_ints["sim_normed"],
                    self._current_a_nodes,
                    self._current_s_nodes,
                    self._current_gt,
                    mode=highlight_mode,
                ),
            )
            simnormed_inset = None
            if show_matrix:
                simnormed_inset = {
                    "matrix": np.asarray(self._current_ints["sim_normed"]),
                    "a_labels": a_nodes,
                    "s_labels": s_nodes,
                    "cmap": "viridis",
                    "title": "sim_normed",
                }
            self._simnormed_panel.set_graph(
                simnormed_combined,
                f"{self._current_env} - A + S (sim_normed, {title_suffix})",
                extra_legend=[*gt_value_legend, *gt_legend, *aura_legend],
                gt_overlay_pairs=gt_overlay,
                include_node_ids=show_ids,
                matrix_inset=simnormed_inset,
                node_aura=_aura_for(self._current_ints["sim_normed"]) if show_aura else None,
                enable_hover=enable_hover,
            )

        self._apply_shared_view()

    def _update_metrics(self, a_nodes, s_nodes):
        """Refresh the combined + per-edge-type Metrics boxes.

        Called from `_refresh_combined` so the panels track GT edits (toggled
        in the GT editor) and matcher re-runs. Denominator for the combined
        box is the full |A|×|S| Cartesian product (same convention as
        `matching_synthetic_dataset.compute_metrics`); for the per-type
        boxes it's the count of cells whose endpoints satisfy the type
        predicate (e.g. |rooms in A| × |rooms in S| for room-room).
        """
        total_all = len(a_nodes) * len(s_nodes)
        combined = _compute_classification_metrics(
            self._current_gt, self._current_pred, total_all,
        )
        combined["match_time"] = self._match_time_s
        self._fill_metric_labels(self._metric_labels, combined)

        a_types = _node_type_map(self._current_a)
        s_types = _node_type_map(self._current_s)
        # Cache node types in the same str-keyed form pairs use, so the
        # predicate lookups below match the pair-set element ids.
        a_str = [str(n) for n in a_nodes]
        s_str = [str(n) for n in s_nodes]
        for key, predicate in _EDGE_TYPE_PREDICATES.items():
            gt_sub = {
                p for p in self._current_gt
                if predicate(a_types.get(p[0]), s_types.get(p[1]))
            }
            pred_sub = {
                p for p in self._current_pred
                if predicate(a_types.get(p[0]), s_types.get(p[1]))
            }
            total_sub = sum(
                1 for an in a_str for sn in s_str
                if predicate(a_types.get(an), s_types.get(sn))
            )
            m = _compute_classification_metrics(gt_sub, pred_sub, total_sub)
            self._fill_metric_labels(self._metric_labels_by_type[key], m)

    @staticmethod
    def _fill_metric_labels(labels, metrics):
        for key, label in labels.items():
            v = metrics.get(key)
            if v is None:
                label.setText("—")
            elif isinstance(v, float):
                label.setText(f"{v:.3f}")
            else:
                label.setText(str(v))

    def _on_panel_view_changed(self, elev, azim):
        self._shared_view = (elev, azim)
        source = self.sender()
        for panel in self._panels:
            if panel is source:
                continue
            panel.apply_view(elev, azim)

    def _apply_shared_view(self):
        if self._shared_view is None:
            return
        elev, azim = self._shared_view
        for panel in self._panels:
            panel.apply_view(elev, azim)

    def _toggle_panel_expand(self, panel):
        if self._expanded_panel is panel:
            self._collapse_expanded()
            return
        # Move panel from grid → expand page. Only one panel can be expanded
        # at a time, so we don't need to handle "swap" cases (the others' buttons
        # are hidden behind the stack switch).
        self._grid.removeWidget(panel)
        self._expand_layout.addWidget(panel)
        self._stack.setCurrentWidget(self._expand_page)
        self._expanded_panel = panel

    def _collapse_expanded(self):
        panel = self._expanded_panel
        if panel is None:
            return
        self._expand_layout.removeWidget(panel)
        row, col = self._panel_positions[panel]
        self._grid.addWidget(panel, row, col)
        self._stack.setCurrentWidget(self._grid_page)
        self._expanded_panel = None


_DEFAULT_GNN_PATH = _WORKSPACE_SRC / "graph_matching_gnn" / "GNN"

# Maps model name → (model_class_name, partial_data_subfolder).
# model_class_name is resolved at runtime inside GnnMatcher to avoid importing
# torch at module load time.
_MODEL_CONFIGS = {
    "ws_room_dropout_noise":               ("MatchingModel_GATv2SinkhornTopK",    "ws_room_dropout_noise"),
    "ws_room_dropout_noise_inc_BCE":       ("MatchingModel_MLPGATv2SinkhornBCE",  "ws_room_dropout_noise_inc"),
    "ws_room_dropout_noise_inc_BCE_noMLP": ("MatchingModel_GATv2Sinkhorn",        "ws_room_dropout_noise_inc"),
    "ws_room_dropout_noise_inc_WBCE":      ("MatchingModel_MLPGATv2SinkhornWBCE", "ws_room_dropout_noise_inc"),
}
_DEFAULT_DATA_EQUAL = _DEFAULT_GNN_PATH / "preprocessed" / "graph_matching" / "equal"


def _build_matcher(args):
    if args.dry_run:
        return DryRunMatcher()

    if MODEL not in _MODEL_CONFIGS:
        raise SystemExit(
            f"Unknown MODEL '{MODEL}'. Choose from: {list(_MODEL_CONFIGS.keys())}"
        )
    model_class_name, data_subfolder = _MODEL_CONFIGS[MODEL]
    model_save_path = _DEFAULT_GNN_PATH / "models" / "partial_graph_matching" / MODEL
    data_equal = _DEFAULT_DATA_EQUAL
    data_partial = _DEFAULT_GNN_PATH / "preprocessed" / "partial_graph_matching" / data_subfolder

    missing = [p for p in (model_save_path, data_equal, data_partial) if not p.exists()]
    if missing:
        raise SystemExit(
            "Real-GNN mode selected but the following path(s) don't exist:\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\n\nEither set MODEL to a valid entry in _MODEL_CONFIGS, or pass "
            + "--dry-run to use the random dry-run matcher."
        )
    try:
        return GnnMatcher(
            model_save_path=str(model_save_path),
            data_paths={"equal": str(data_equal), "partial": str(data_partial)},
            model_class_name=model_class_name,
        )
    except ImportError as exc:
        raise SystemExit(
            f"GNN dependencies missing ({exc}).\n"
            "Install torch / torch_geometric, or pass --dry-run to use the "
            "random dry-run matcher."
        ) from exc


def _parse_args(argv):
    import argparse
    parser = argparse.ArgumentParser(
        description="Interactive graph matching dashboard."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use the random-data dry-run matcher (skips loading the real GNN "
             "model). Default is to load the real GNN.",
    )
    return parser.parse_args(argv)


def main():
    args = _parse_args(sys.argv[1:])
    matcher = _build_matcher(args)
    app = QApplication(sys.argv[:1])  # Qt parses its own; pass only argv[0].
    win = Dashboard(matcher=matcher)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
