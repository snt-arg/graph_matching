"""Geometric verification of the GNN matcher's pairs before they are published (used live by graph_matching_node and
offline by sim_buildings/verify_eval.py).

The GNN proposes, the classic matcher's own checks verify, so the GNN stays fast (no candidate enumeration: CLIPPER
only scores the GNN's m pairs, O(m^2)) while only geometrically coherent, unambiguous pairs reach the S-Graph:

  1. candidates   pairs the GNN assigned one-to-one with Sinkhorn score >= t_c (applied by the caller, option A)
  2. time         drop candidates that contradict an already published pair (same Online node -> another A node or
                  the reverse); published pairs still present are added as anchors (classic stored_match_graph)
  3. rooms        CLIPPER on the room pairs (classic invariant "points"/EuclideanDistance) -> max consistent subset
  4. walls        CLIPPER on the wall pairs in the frame of the best inlier room pair (classic room-frame transform,
                  invariant "points&normal"/PointNormalDistance) -> max consistent subset
  5. gate         average consistency of each kept subset >= global_consistency (classic 0.8), enough rooms/pairs,
                  and every anchor kept (otherwise the new pairs contradict the map: publish nothing)
  6. uniqueness   enumerate rigid alignments Online -> Prior (yaw from wall directions + k*90 deg, translation from
                  room pairings); refuse if the verified pairs do not follow the best alignment or if another
                  alignment explains the observed rooms/walls almost as well (classic DBSCAN "no rival" rule)

Graphs are passed as plain dicts so the same code runs on the node's classic GraphWrappers and on logged pickles:
  rooms  {id: xyz (3,)}            Geometric_info of a Finite Room (centre)
  planes {id: point+normal (6,)}   Geometric_info of a Plane (closest point of the infinite plane to the origin, normal)
"""
import time

import numpy as np

from graph_matching.Clipper import Clipper
from graph_matching.utils import transform_plane_definition

DEFAULTS = {
    "global_consistency": 0.8,
    "min_rooms": 1,
    "min_pairs": 3,
    # evidence floor for the wall check (2026-10-23: four live wrong publications rested on 1-2 consistent walls, i.e.
    # 1 pairwise comparison, consistent by chance in repetitive layouts; 189/190 correct ones kept >= 3). 0 = off.
    "min_walls": 0,        # wall pairs kept by CLIPPER (anchors included)
    "min_wall_ratio": 0.0, # kept / candidate wall pairs: most of the GNN's walls must agree
    "use_time": True,
    "use_joint": True,
    "use_sym": True,
    "align_tol": 0.5,      # m: an Online element "lands" on a Prior element within this distance
    "normal_tol": 0.95,    # cos: same orientation
    "sym_margin": 0.1,     # refuse if a rival alignment scores >= (1 - sym_margin) * best
    "rival_sep": 1.0,      # m: alignments closer than this (and same yaw) are the same hypothesis
}


def _clipper_subset(datatype, D1, D2, params):
    """-> (kept indices, average consistency of the kept subset, full M). D1[i] <-> D2[i] are the candidate pairs."""
    m = len(D1)
    if m == 0:
        return [], 0.0, None
    if m == 1:
        return [0], 1.0, np.zeros((1, 1))
    A = np.stack([np.arange(m), np.arange(m)], axis=1).astype(np.int32)
    c = Clipper(datatype, "0", params)
    c.score_pairwise_consistency(np.asarray(D1, float), np.asarray(D2, float), A)
    M = c.get_affinity_matrix()
    sel, _ = c.solve_clipper()
    kept = sorted({int(p[0]) for p in sel}) if sel is not None and len(sel) else []
    if len(kept) < 2:
        return kept, (1.0 if kept else 0.0), M
    sub = M[np.ix_(kept, kept)]
    return kept, float(sub.sum() / (len(kept) * (len(kept) - 1))), M


def _plane_lines(planes):
    """{id: 6-vec} -> ids, unit 2D normals (n,2), offsets d with n.x = d."""
    ids = list(planes)
    if not ids:
        return ids, np.zeros((0, 2)), np.zeros(0)
    P = np.array([planes[i] for i in ids], float)
    n = P[:, 3:5]
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    d = np.einsum("ij,ij->i", n, P[:, :2])
    return ids, n, d


def _dominant_yaw(n):
    """Wall direction modulo 90 deg (circular mean of 4*angle)."""
    if len(n) == 0:
        return 0.0
    a = np.arctan2(n[:, 1], n[:, 0]) * 4
    return float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()) / 4)


def _hypotheses(prior_rooms, online_rooms, yaw0):
    """Rigid alignments Online -> Prior: yaw0 + k*90 deg, translation from every Online-room <-> Prior-room pairing."""
    out = []
    pr = np.array([v[:2] for v in prior_rooms.values()], float)
    orr = np.array([v[:2] for v in online_rooms.values()], float)
    for k in range(4):
        yaw = yaw0 + k * np.pi / 2
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        for o in orr:
            for p in pr:
                out.append((yaw, R, p - R @ o))
    return out


def _score_alignment(R, t, prior, online, cfg):
    """Number of Online rooms + walls landing on a Prior element of the same type / orientation."""
    (pr, pn, pd), (orr, on, od) = prior, online
    s = 0
    if len(orr) and len(pr):
        q = orr @ R.T + t
        s += int((np.min(np.linalg.norm(q[:, None, :] - pr[None, :, :], axis=2), axis=1) < cfg["align_tol"]).sum())
    if len(on) and len(pn):
        qn = on @ R.T                       # rotated normals
        qd = od + qn @ t                    # n'.x' = d + n'.t
        cos = qn @ pn.T
        dd = np.abs(qd[:, None] - pd[None, :])
        s += int(((cos > cfg["normal_tol"]) & (dd < cfg["align_tol"])).any(axis=1).sum())
    return s


def uniqueness(prior_rooms, prior_planes, online_rooms, online_planes, verified, cfg):
    """-> (ok, info). verified: list of (prior_id, online_id, kind) with kind 'room' / 'plane'."""
    if not online_rooms or not prior_rooms:
        return False, {"sym": "no rooms"}
    _, pn, pd = _plane_lines(prior_planes)
    _, on, od = _plane_lines(online_planes)
    pr = np.array([v[:2] for v in prior_rooms.values()], float)
    orr = np.array([v[:2] for v in online_rooms.values()], float)
    yaw0 = _dominant_yaw(pn) - _dominant_yaw(on)
    hyps = _hypotheses(prior_rooms, online_rooms, yaw0)
    scores = [_score_alignment(R, t, (pr, pn, pd), (orr, on, od), cfg) for _, R, t in hyps]
    best = int(np.argmax(scores))
    yaw_b, R_b, t_b = hyps[best]
    # the verified pairs must follow the best alignment
    for p_id, o_id, kind in verified:
        if kind == "room":
            if np.linalg.norm(R_b @ np.asarray(online_rooms[o_id][:2]) + t_b - np.asarray(prior_rooms[p_id][:2])) > cfg["align_tol"]:
                return False, {"sym": "verified pairs disagree with the best alignment", "best": scores[best]}
    # a rival alignment explaining (almost) as much is an ambiguity: refuse
    rival = 0
    for (yaw, R, t), s in zip(hyps, scores):
        same_yaw = abs((yaw - yaw_b + np.pi) % (2 * np.pi) - np.pi) < 1e-6
        if same_yaw and np.linalg.norm(t - t_b) < cfg["rival_sep"]:
            continue
        rival = max(rival, s)
    ok = rival < (1 - cfg["sym_margin"]) * scores[best]
    return ok, {"sym_best": scores[best], "sym_rival": rival}


def verify(candidates, prior_rooms, prior_planes, online_rooms, online_planes, published, params, cfg=None):
    """candidates: list of dicts {prior, online, kind ('room'|'plane'), score}; ids as in the room/plane dicts.
    published: {(kind, online_id): prior_id} from earlier calls (read only here).
    -> (pairs to publish (subset of candidates, empty = publish nothing), info dict)."""
    cfg = {**DEFAULTS, **(cfg or {})}
    t0 = time.perf_counter()
    info = {"n_cand": len(candidates)}
    # 2. consistency over time
    anchors, cand = [], []
    used_prior = {(k, p) for (k, _), p in published.items()}
    for (k, o), p in published.items():
        src = online_rooms if k == "room" else online_planes
        dst = prior_rooms if k == "room" else prior_planes
        if o in src and p in dst:
            anchors.append({"prior": p, "online": o, "kind": k, "score": 1.0, "anchor": True})
    for c in candidates:
        key = (c["kind"], c["online"])
        if cfg["use_time"] and key in published:
            if published[key] != c["prior"]:
                info["time_dropped"] = info.get("time_dropped", 0) + 1
            continue  # already published (same) or contradicting: not a new candidate
        if cfg["use_time"] and (c["kind"], c["prior"]) in used_prior:
            info["time_dropped"] = info.get("time_dropped", 0) + 1
            continue
        cand.append(c)
    if not cand:
        info["reason"] = "no new candidates"
        info["ms"] = (time.perf_counter() - t0) * 1e3
        return [], info
    pool = cand + (anchors if cfg["use_time"] else [])
    rooms = [c for c in pool if c["kind"] == "room" and c["online"] in online_rooms and c["prior"] in prior_rooms]
    walls = [c for c in pool if c["kind"] == "plane" and c["online"] in online_planes and c["prior"] in prior_planes]
    if cfg["use_joint"]:
        # 3. rooms
        kr, avg_r, _ = _clipper_subset("points", [prior_rooms[c["prior"]] for c in rooms],
                                       [online_rooms[c["online"]] for c in rooms], params)
        rooms_k = [rooms[i] for i in kr]
        info.update(rooms_in=len(rooms), rooms_kept=len(rooms_k), room_avg=avg_r)
        if len(rooms_k) < cfg["min_rooms"] or (len(rooms_k) >= 2 and avg_r < cfg["global_consistency"]):
            info["reason"] = "rooms"
            info["ms"] = (time.perf_counter() - t0) * 1e3
            return [], info
        # 4. walls in the frame of the best inlier room pair (classic merge_lower_level_groups)
        frame = max(rooms_k, key=lambda c: (c.get("anchor", False), c["score"]))
        fp = np.asarray(prior_rooms[frame["prior"]], float)[:3]
        fo = np.asarray(online_rooms[frame["online"]], float)[:3]
        I = np.eye(3)
        D1 = transform_plane_definition(np.array([prior_planes[c["prior"]] for c in walls], float), -fp, I) if walls else []
        D2 = transform_plane_definition(np.array([online_planes[c["online"]] for c in walls], float), -fo, I) if walls else []
        kw, avg_w, _ = _clipper_subset("points&normal", D1, D2, params)
        walls_k = [walls[i] for i in kw]
        info.update(walls_in=len(walls), walls_kept=len(walls_k), wall_avg=avg_w)
        if len(walls_k) >= 2 and avg_w < cfg["global_consistency"]:
            info["reason"] = "walls"
            info["ms"] = (time.perf_counter() - t0) * 1e3
            return [], info
        if len(walls_k) < cfg["min_walls"] or (walls and len(walls_k) / len(walls) < cfg["min_wall_ratio"]):
            info["reason"] = "too few consistent walls"
            info["ms"] = (time.perf_counter() - t0) * 1e3
            return [], info
        kept = rooms_k + walls_k
    else:
        kept = rooms + walls
    # 5. every anchor must survive: a new set that pushes out an earlier merge contradicts the map
    if cfg["use_time"] and cfg["use_joint"]:
        kept_anchor = {(c["kind"], c["online"]) for c in kept if c.get("anchor")}
        if any((a["kind"], a["online"]) not in kept_anchor for a in anchors
               if a in rooms or a in walls):
            info["reason"] = "anchor rejected"
            info["ms"] = (time.perf_counter() - t0) * 1e3
            return [], info
    new = [c for c in kept if not c.get("anchor")]
    if len(new) + len([c for c in kept if c.get("anchor")]) < cfg["min_pairs"] or not new:
        info["reason"] = "too few pairs"
        info["ms"] = (time.perf_counter() - t0) * 1e3
        return [], info
    # 6. uniqueness / symmetry
    if cfg["use_sym"]:
        ver = [(c["prior"], c["online"], c["kind"]) for c in kept]
        ok, sinfo = uniqueness(prior_rooms, prior_planes, online_rooms, online_planes, ver, cfg)
        info.update(sinfo)
        if not ok:
            info["reason"] = "ambiguous"
            info["ms"] = (time.perf_counter() - t0) * 1e3
            return [], info
    info["reason"] = "published"
    info["n_pub"] = len(new)
    info["ms"] = (time.perf_counter() - t0) * 1e3
    return new, info
