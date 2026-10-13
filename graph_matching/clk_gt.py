"""Propose a ground_truth.json for a graph_dicts/<env>/ by aligning Online onto Prior, no anchors.

    python3 clk_gt.py <env> [<env> ...]

Online (map frame) and Prior (prior_map frame) are not aligned, so a rigid 2D transform is searched:
every rotation in 1 deg steps x every (online room, prior room) centre pair as the translation
anchor (rooms are emitted complete-or-absent, so their centres are the reliable points). Each
candidate is scored by how many Online ws land on a same-facing Prior ws (perpendicular < 0.2 m,
centre inside the segment); the best is refined on a 0.1 deg / 0.02 m grid. The best DISTINCT
runner-up (> 10 deg or > 1 m away) is reported too: mirror-symmetric houses can fit twice.

Pairing rules are the ones used for example/ (gt_build.py): same-facing normal (dot > 0.95),
perp + along-segment overshoot < 0.25 m; ambiguous when a second, different Prior wall is within
0.05 of the best. Rooms pair with the nearest Prior room centre within 1.0 m.

Writes graph_dicts/<env>/ground_truth_proposed.json and gt_overlay.png, and prints a report. It
does NOT write ground_truth.json -- that is copied over once the proposal is confirmed.
"""
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

GRAPH_DICTS = Path(__file__).parent / "graph_dicts"


def load(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    g = obj.graph if hasattr(obj, "graph") else obj
    ws, rooms = [], []
    for n, d in g.nodes(data=True):
        c = np.asarray(d["center"], float)[:2]
        if d.get("type") == "ws":
            ws.append({"id": str(n), "c": c, "n": np.asarray(d["normal"], float)[:2],
                       "L": float(d["length"])})
        elif d.get("type") == "room":
            rooms.append({"id": str(n), "c": c})
    return ws, rooms


def rot(deg):
    r = np.radians(deg)
    return np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])


class Fit:
    def __init__(self, P, O):
        self.Pc = np.array([w["c"] for w in P]); self.Pn = np.array([w["n"] for w in P])
        self.PL = np.array([w["L"] for w in P]); self.Pt = np.stack([-self.Pn[:, 1], self.Pn[:, 0]], 1)
        self.Oc = np.array([w["c"] for w in O]); self.On = np.array([w["n"] for w in O])

    def cost(self, R, t):
        c = self.Oc @ R.T + t; n = self.On @ R.T; d = c[:, None] - self.Pc[None]
        perp = np.abs((d * self.Pn[None]).sum(-1))
        along = np.clip(np.abs((d * self.Pt[None]).sum(-1)) - self.PL[None] / 2, 0, None)
        par = (n[:, None] * self.Pn[None]).sum(-1)
        return perp + along + 10 * (par < 0.95)

    def score(self, R, t):
        C = self.cost(R, t).min(1)
        return (int((C < 0.2).sum()), -float(np.sort(C)[:20].sum()))


def search(fit, PR, OR):
    cands = []
    for deg in np.arange(0, 360, 1.0):
        R = rot(deg)
        for o in OR:
            for p in PR:
                t = p["c"] - R @ o["c"]
                cands.append((fit.score(R, t), deg, t))
    cands.sort(key=lambda x: x[0], reverse=True)

    def distinct(a, b):
        return abs((a[1] - b[1] + 180) % 360 - 180) > 10 or np.linalg.norm(a[2] - b[2]) > 1.0

    # refine the 6 best coarse basins, then rank the refined fits; a coarse runner-up often
    # refines into the same solution, so distinctness is judged after refinement
    basins = []
    for c in cands:
        if all(distinct(c, b) for b in basins):
            basins.append(c)
            if len(basins) == 6:
                break
    refined = sorted((refine(fit, b) for b in basins), key=lambda x: x[0], reverse=True)
    best = refined[0]
    runner = next((r for r in refined[1:] if distinct(r, best)), None)
    return best, runner


def refine(fit, cand):
    _, deg0, t0 = cand
    best = (fit.score(rot(deg0), t0), deg0, t0)
    for deg in np.arange(deg0 - 1.0, deg0 + 1.001, 0.1):
        R = rot(deg)
        for dx in np.arange(-0.5, 0.501, 0.05):
            for dy in np.arange(-0.5, 0.501, 0.05):
                t = t0 + [dx, dy]; s = fit.score(R, t)
                if s > best[0]:
                    best = (s, deg, t)
    _, deg1, t1 = best
    R = rot(deg1)
    for dx in np.arange(-0.05, 0.051, 0.01):
        for dy in np.arange(-0.05, 0.051, 0.01):
            t = t1 + [dx, dy]; s = fit.score(R, t)
            if s > best[0]:
                best = (s, deg1, t)
    return best


def pairs_for(fit, P, O, PR, OR, deg, t):
    R = rot(deg); C = fit.cost(R, t)
    ws, n_amb = [], 0
    for i, w in enumerate(O):
        order = np.argsort(C[i]); j = int(order[0]); ok = C[i, j] < 0.25
        k = int(order[1]) if len(order) > 1 else j
        same_geom = np.allclose(fit.Pc[k], fit.Pc[j]) and np.allclose(fit.Pn[k], fit.Pn[j])
        amb = ok and C[i, k] < 0.25 and C[i, k] - C[i, j] < 0.05 and not same_geom
        n_amb += int(amb)
        ws.append([f"s_{w['id']}", f"a_{P[j]['id']}" if ok and not amb else "??"])
    rooms, room_res = {}, {}
    for o in OR:
        c = R @ o["c"] + t
        d = [np.linalg.norm(c - p["c"]) for p in PR]
        j = int(np.argmin(d)) if d else None
        ok = j is not None and d[j] < 1.0
        rooms[f"s_{o['id']}"] = f"a_{PR[j]['id']}" if ok else "??"
        room_res[o["id"]] = round(float(d[j]), 2) if j is not None else None
    return {"rooms": rooms, "ws": sorted(ws)}, room_res, n_amb


def overlay(path, P, O, PR, OR, deg, t, title):
    R = rot(deg)
    fig, ax = plt.subplots(figsize=(9, 9))
    for w in P:
        tv = np.array([-w["n"][1], w["n"][0]]) * w["L"] / 2
        a, b = w["c"] - tv, w["c"] + tv
        ax.plot([a[0], b[0]], [a[1], b[1]], color="0.6", lw=3, zorder=1)
    for p in PR:
        ax.plot(*p["c"], "s", color="0.4", ms=9); ax.annotate(f"a_{p['id']}", p["c"], color="0.3", fontsize=8)
    for w in O:
        c = R @ w["c"] + t; n = R @ w["n"]; tv = np.array([-n[1], n[0]]) * w["L"] / 2
        a, b = c - tv, c + tv
        ax.plot([a[0], b[0]], [a[1], b[1]], color="tab:red", lw=1.5, zorder=2)
    for o in OR:
        c = R @ o["c"] + t
        ax.plot(*c, "o", color="tab:red", ms=7); ax.annotate(f"s_{o['id']}", c, color="tab:red", fontsize=8)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=9); ax.grid(alpha=0.3)
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def main():
    for env in sys.argv[1:]:
        d = GRAPH_DICTS / env
        P, PR = load(d / "Prior.pkl"); O, OR = load(d / "Online.pkl")
        if not O or not OR:
            print(f"== {env}: Online has {len(O)} ws / {len(OR)} rooms -- cannot fit"); continue
        fit = Fit(P, O)
        best, runner = search(fit, PR, OR)
        (inl, _), deg, t = best
        gt, room_res, n_amb = pairs_for(fit, P, O, PR, OR, deg, t)
        n_ws = sum(p[1] != "??" for p in gt["ws"])
        rep = {"rotation_deg": round(float(deg), 2), "shift": np.round(t, 3).tolist(),
               "ws_inliers_0.2m": inl, "online_ws": len(O), "online_rooms": len(OR),
               "prior_ws": len(P), "prior_rooms": len(PR), "ws_paired": n_ws, "ws_ambiguous": n_amb,
               "room_residual_m": room_res,
               "runner_up": None if runner is None else {
                   "rotation_deg": round(float(runner[1]), 2), "shift": np.round(runner[2], 3).tolist(),
                   "ws_inliers_0.2m": runner[0][0]}}
        with open(d / "ground_truth_proposed.json", "w") as f:
            json.dump({**gt, "_fit": rep}, f, indent=1)
        overlay(d / "gt_overlay.png", P, O, PR, OR, deg, t,
                f"{env}: rot {deg:.1f} deg, shift {np.round(t, 2)}; {inl}/{len(O)} online ws within 0.2 m")
        if runner is not None:
            overlay(d / "gt_overlay_runner_up.png", P, O, PR, OR, runner[1], runner[2],
                    f"{env} RUNNER-UP: rot {runner[1]:.1f} deg; {runner[0][0]}/{len(O)} ws within 0.2 m")
        print(f"== {env}: rot {deg:.1f} deg shift {np.round(t, 2)}  inliers {inl}/{len(O)}  "
              f"ws paired {n_ws} (+{n_amb} ambiguous)  rooms {gt['rooms']}  residuals {room_res}")
        if runner is not None:
            print(f"   runner-up: rot {runner[1]:.1f} deg shift {np.round(runner[2], 2)}  "
                  f"inliers {runner[0][0]}/{len(O)}")


if __name__ == "__main__":
    main()
