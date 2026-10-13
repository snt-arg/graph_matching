"""Score graph_dicts environments with one edge-feature model, the way the dashboard does.

    python3 clk_f1.py [--model adj_glob_65_edgefeat_rooms] [--out FILE.csv] <env> [<env> ...]

Same path as the dashboard's Metrics box: EdgeMatcher.match(A, S) -> Hungarian pairs, scored
against ground_truth.json (prefixes stripped, '??' skipped) over the |A| x |S| universe.
Also reports room-room and ws-ws F1 separately.
"""
import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRAPH_DICTS = HERE / "graph_dicts"
GNN_SRC = HERE.parent.parent / "graph_matching_gnn" / "graph_matching"
sys.path.insert(0, str(GNN_SRC))

from situational_graphs_wrapper.GraphWrapper import GraphWrapper  # noqa: E402
import edge_model  # noqa: E402


def load_graph(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    w = obj if isinstance(obj, GraphWrapper) else GraphWrapper(graph_obj=obj)
    for _, attrs in w.graph.nodes(data=True):       # as dashboard.load_env_graphs
        for key in ("center", "normal"):
            if key in attrs:
                v = attrs[key]
                v = v.tolist() if hasattr(v, "tolist") else v
                attrs[key] = v[:2] if len(v) > 2 else v
    return w


def load_gt(path):
    """dashboard._load_ground_truth: set of (a_id, s_id)."""
    raw = json.load(open(path))
    strip = lambda s, p: str(s)[len(p):] if str(s).startswith(p) else str(s)
    pairs = {(strip(a, "a_"), strip(s, "s_")) for s, a in raw.get("rooms", {}).items() if a != "??"}
    pairs |= {(strip(e[1], "a_"), strip(e[0], "s_")) for e in raw.get("ws", [])
              if len(e) == 2 and e[1] != "??"}
    return pairs


def prf(gt, pred):
    tp = len(gt & pred); fp = len(pred - gt); fn = len(gt - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return tp, fp, fn, p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("envs", nargs="+")
    ap.add_argument("--model", default="adj_glob_65_edgefeat_rooms")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--gt-name", default="ground_truth.json",
                    help="e.g. ground_truth_proposed.json for provisional scores before confirmation")
    a = ap.parse_args()
    model_dir = GNN_SRC.parent / "GNN" / "models" / "partial_graph_matching" / a.model
    matcher = edge_model.EdgeMatcher(str(model_dir))

    rows = []
    for env in a.envs:
        d = GRAPH_DICTS / env
        if not (d / a.gt_name).exists():
            print(f"{env}: no {a.gt_name}, skipped"); continue
        A, S = load_graph(d / "Prior.pkl"), load_graph(d / "Online.pkl")
        gt = load_gt(d / a.gt_name)
        pred, _, a_nodes, s_nodes = matcher.match(A, S)
        at = {str(n): t.get("type") for n, t in A.graph.nodes(data=True)}
        row = {"env": env, "model": a.model, "gt_file": a.gt_name, "A_nodes": len(a_nodes), "S_nodes": len(s_nodes),
               "A_rooms": sum(t == "room" for t in at.values()),
               "S_rooms": sum(t.get("type") == "room" for _, t in S.graph.nodes(data=True)),
               "gt_pairs": len(gt), "pred_pairs": len(pred)}
        for name, keep in (("all", None), ("room", "room"), ("ws", "ws")):
            g = {p for p in gt if keep is None or at.get(p[0]) == keep}
            q = {p for p in pred if keep is None or at.get(p[0]) == keep}
            tp, fp, fn, p, r, f1 = prf(g, q)
            if name == "all":
                row.update({"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 3),
                            "recall": round(r, 3), "f1": round(f1, 3)})
            else:
                row[f"f1_{name}"] = round(f1, 3)
        rows.append(row)
        print(f"{env:22s} |A| {row['A_nodes']:3d} |S| {row['S_nodes']:3d} rooms {row['A_rooms']}/{row['S_rooms']} "
              f"GT {row['gt_pairs']:3d}  P {row['precision']:.3f} R {row['recall']:.3f} F1 {row['f1']:.3f} "
              f"(room {row['f1_room']:.3f}, ws {row['f1_ws']:.3f})")
    if a.out and rows:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
