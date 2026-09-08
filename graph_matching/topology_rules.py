"""Geometric rules that derive a scene graph's edges from its node geometry.

These rules are the ones
``graph_matching_node.convert_wrapper_to_gnn_format`` applies when it converts
a live S-Graphs observation into the format the GNN was trained on. They lived
inline inside that ~230-line ROS method, which made them unreachable from
anywhere else; this module lifts them out unchanged so the dashboard's editor
can re-derive edges after a node is moved, and so there is a single definition
to keep in sync.

Three edge families exist in the ``adj`` topology the models are trained on:

- ``ws_belongs_room`` — a wall surface belongs to a room. **Never derived
  here.** It encodes ownership, not geometry, so moving a wall does not
  invalidate it, and a caller that wants it changed must say so explicitly.
- ``ws_same_room``    — walls of one room, linked as an angularly-sorted ring
  (n edges for n walls), *not* a complete subgraph.
- ``connected_by_door`` — two rooms whose walls include a pair of opposing
  faces of the same physical wall.

Everything is computed in the XY plane: ``center`` / ``normal`` / ``limits``
are sliced to their first two components, so graphs carrying either 2D or 3D
coordinates (the editor's are 3D after ``from_2D_to_3D``) work unchanged.
"""

import math

import numpy as np


# Maximum separation, along the wall normal, between two surfaces still
# considered opposing faces of one physical wall. In metres, so it is tied to
# the scale of the graph it is applied to: the real scans are metric, but the
# synthetic MSD samples are not.
WALL_NORMAL_DIST_THRESHOLD = 0.25

# Minimum fraction of the shorter segment that two wall surfaces must share
# along their tangent before they count as the same physical wall.
TANGENT_OVERLAP_MIN = 0.01

WS_SAME_ROOM = "ws_same_room"
ROOM_ROOM = "connected_by_door"


def _xy(value, n=2):
    """First ``n`` components of a coordinate, as a float array."""
    return np.asarray(value, dtype=float).ravel()[:n]


def _node_type(graph, node_id):
    return graph.nodes[node_id].get("type")


def neighbours(graph, node_id):
    """Return every neighbour of ``node_id``, ignoring edge direction.

    The graphs are directed but reciprocal, so a directed lookup would usually
    work — usually. ``InteractiveGraphVisualizer.create_room_from_planes``
    writes its ``ws -> room`` edges one-directionally, so anything reading room
    membership must not assume both directions are present.
    """
    return set(graph.successors(node_id)) | set(graph.predecessors(node_id))


def room_of_ws(graph, ws_id):
    """Return the room a wall belongs to, or ``None`` if it is unattached."""
    for nbr in neighbours(graph, ws_id):
        if _node_type(graph, nbr) == "room":
            return nbr
    return None


def ws_of_room(graph, room_id):
    """Return the wall surfaces belonging to ``room_id``."""
    return [n for n in neighbours(graph, room_id)
            if _node_type(graph, n) == "ws"]


def tangent_overlap_ratio(lims_a, lims_b, tangent_2d):
    """Return overlap / min-length for two segments projected on a tangent."""
    t = _xy(tangent_2d)
    pa = sorted(float(np.dot(_xy(ep), t)) for ep in lims_a)
    pb = sorted(float(np.dot(_xy(ep), t)) for ep in lims_b)
    overlap = max(0.0, min(pa[1], pb[1]) - max(pa[0], pb[0]))
    min_len = min(pa[1] - pa[0], pb[1] - pb[0])
    if min_len < 1e-6:
        return 0.0
    return overlap / min_len


def is_derived_edge(graph, u, v):
    """Return True for the families this module owns (ws-ws and room-room).

    room-ws edges are excluded: they are the ownership relation, which geometry
    does not determine.
    """
    tu, tv = _node_type(graph, u), _node_type(graph, v)
    return (tu == "ws" and tv == "ws") or (tu == "room" and tv == "room")


def drop_derived_edges(graph):
    """Remove every ws-ws and room-room edge. Returns the number removed."""
    doomed = [(u, v) for u, v in graph.edges() if is_derived_edge(graph, u, v)]
    graph.remove_edges_from(doomed)
    return len(doomed)


def rebuild_ws_ws_edges(graph):
    """Re-derive the intra-room wall ring for every room.

    The walls of a room are sorted by their bearing about the room centre and
    linked as a forward chain plus one shortcut closing the loop — n edges for
    n walls. This is the pattern the ``adj`` training data actually uses; a
    complete intra-room subgraph is the ``fully`` variant and does not match.
    """
    added = 0
    for room_id, attrs in list(graph.nodes(data=True)):
        if attrs.get("type") != "room":
            continue
        walls = ws_of_room(graph, room_id)
        if len(walls) < 2:
            continue
        centre = _xy(attrs.get("center", [0.0, 0.0]))
        walls.sort(key=lambda n: math.atan2(
            _xy(graph.nodes[n]["center"])[1] - centre[1],
            _xy(graph.nodes[n]["center"])[0] - centre[0],
        ))
        pairs = [(walls[i], walls[i + 1]) for i in range(len(walls) - 1)]
        pairs.append((walls[0], walls[-1]))  # shortcut closing the ring
        for u, v in pairs:
            if not graph.has_edge(u, v):
                graph.add_edge(u, v, type=WS_SAME_ROOM)
                added += 1
    return added


def rebuild_room_room_edges(graph, threshold=None):
    """Re-derive room-room edges from pairs of opposing wall surfaces.

    Two rooms are linked when one wall of each are the two faces of a single
    physical wall, which takes three checks:

    1. the normals point into opposite half-spaces;
    2. the centres, projected onto the normal axis, are no more than
       ``threshold`` apart — two faces of one wall are only wall-thickness
       apart along the normal however partial the observation is;
    3. the segments overlap along their shared tangent, so they cover the
       same stretch of wall rather than two distant parts of a long one.
    """
    if threshold is None:
        threshold = WALL_NORMAL_DIST_THRESHOLD

    walls = []
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("type") != "ws":
            continue
        room = room_of_ws(graph, node_id)
        if room is None or attrs.get("limits") is None:
            continue  # an unattached wall implies no room relation
        walls.append((node_id, room, _xy(attrs.get("center", [0.0, 0.0])),
                      _xy(attrs.get("normal", [0.0, 0.0])), attrs["limits"]))

    added = 0
    for i in range(len(walls)):
        id_i, room_i, c_i, n_i, lim_i = walls[i]
        mag_i = float(np.linalg.norm(n_i))
        if mag_i < 1e-6:
            continue
        u_i = n_i / mag_i
        tangent = np.array([-u_i[1], u_i[0]])
        for j in range(i + 1, len(walls)):
            id_j, room_j, c_j, n_j, lim_j = walls[j]
            if room_i == room_j or graph.has_edge(room_i, room_j):
                continue
            mag_j = float(np.linalg.norm(n_j))
            if mag_j < 1e-6:
                continue
            if float(np.dot(u_i, n_j / mag_j)) > 0.0:
                continue  # same half-space, so not opposing faces
            gap = abs(float(np.dot(c_i, u_i)) - float(np.dot(c_j, u_i)))
            if gap > threshold:
                continue
            overlap = tangent_overlap_ratio(lim_i, lim_j, tangent)
            if overlap < TANGENT_OVERLAP_MIN:
                continue
            graph.add_edge(room_i, room_j, type=ROOM_ROOM)
            added += 1
    return added


def symmetrize(graph):
    """Add the reverse of every edge. Returns the number added.

    The training data and every real scan are fully reciprocal, and the GNN
    aggregates along edge direction — a one-directional edge silently delivers
    half the message passing the model expects along it. Worth running after
    hand-editing: ``create_room_from_planes`` emits ``ws -> room`` only.
    """
    missing = [(v, u, dict(d)) for u, v, d in graph.edges(data=True)
               if not graph.has_edge(v, u)]
    graph.add_edges_from(missing)
    return len(missing)


def rebuild_all(graph, threshold=None):
    """Drop and re-derive every geometric edge, then restore reciprocity.

    room-ws edges are left alone throughout. Returns a summary dict.
    """
    dropped = drop_derived_edges(graph)
    ws_ws = rebuild_ws_ws_edges(graph)
    room_room = rebuild_room_room_edges(graph, threshold)
    mirrored = symmetrize(graph)
    return {
        "dropped": dropped,
        "ws_ws": ws_ws,
        "room_room": room_room,
        "mirrored": mirrored,
        "edges": graph.number_of_edges(),
    }
