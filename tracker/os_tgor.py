import math

import numpy as np

from tracker.basetrack import TrackState


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


class OcclusionStateGraphReasoner(object):
    """Temporal graph reasoner for per-track occlusion state estimation.

    The module is deliberately lightweight: it builds a frame-level graph over
    active/lost tracks and estimates a smooth occlusion state S_t in [0, 1].
    The state can then control AIS motion prior, association, and lifecycle.
    """

    def __init__(self, future_steps=5, neighbor_radius=250.0, ema=0.70,
                 sigma_d=120.0, sigma_v=35.0, sigma_f=80.0,
                 min_bbox_scale=20.0):
        self.future_steps = int(max(1, future_steps))
        self.neighbor_radius = float(max(1.0, neighbor_radius))
        self.ema = float(np.clip(ema, 0.0, 0.99))
        self.sigma_d = float(max(1.0, sigma_d))
        self.sigma_v = float(max(1.0, sigma_v))
        self.sigma_f = float(max(1.0, sigma_f))
        self.min_bbox_scale = float(max(1.0, min_bbox_scale))
        self.prev_states = {}
        self.last_node_details = {}
        self.last_edge_details = []

    @classmethod
    def from_args(cls, args):
        return cls(
            future_steps=getattr(args, 'tgor_future_steps', 5),
            neighbor_radius=getattr(args, 'tgor_neighbor_radius', 250.0),
            ema=getattr(args, 'tgor_ema', 0.70),
            sigma_d=getattr(args, 'tgor_sigma_d', 120.0),
            sigma_v=getattr(args, 'tgor_sigma_v', 35.0),
            sigma_f=getattr(args, 'tgor_sigma_f', 80.0))

    def compute_occlusion_state(self, tracks, ais_obs_by_id=None, timestamp=None,
                                detection_conf_by_track_id=None):
        if len(tracks) == 0:
            self.prev_states = {}
            return {}

        raw_states = {}
        node_details = {}
        edge_details = []
        for i, track_i in enumerate(tracks):
            graph_risk = 0.0
            for j, track_j in enumerate(tracks):
                if i == j:
                    continue
                edge_risk, edge_detail = self._edge_risk(
                    track_i, track_j, ais_obs_by_id, return_detail=True)
                if edge_detail is not None:
                    edge_details.append(edge_detail)
                graph_risk = 1.0 - (1.0 - graph_risk) * (1.0 - edge_risk)

            history_risk = self._history_risk(
                track_i, detection_conf_by_track_id)
            ais_risk = self._ais_inconsistency(track_i, ais_obs_by_id)
            raw = self._sigmoid(
                2.35 * graph_risk + 1.20 * history_risk +
                0.70 * ais_risk - 1.15)
            raw_states[track_i.track_id] = raw
            node_details[track_i.track_id] = {
                'graph_risk': graph_risk,
                'history_risk': history_risk,
                'ais_inconsistency': ais_risk,
                'raw_state': raw,
            }

        states = {}
        active_ids = set()
        for track in tracks:
            tid = track.track_id
            active_ids.add(tid)
            prev = self.prev_states.get(tid, raw_states[tid])
            smooth = self.ema * prev + (1.0 - self.ema) * raw_states[tid]
            if track.state in (TrackState.Lost, TrackState.Occluded):
                smooth = max(smooth, 0.85 * prev)
            states[tid] = _clip01(smooth)
            node_details[tid]['prev_state'] = prev
            node_details[tid]['smooth_state'] = states[tid]

        self.prev_states = {
            tid: state for tid, state in states.items()
            if tid in active_ids
        }
        self.last_node_details = node_details
        self.last_edge_details = edge_details
        return states

    def compute_reliability(self, track, ais_obs_by_id=None, timestamp=None,
                            ais_config=None):
        s = _clip01(getattr(track, 'occlusion_state', 0.0))
        track_conf = _clip01(getattr(track, 'score', 0.0))
        ais_rel = 0.0
        obs = self._ais_obs(track, ais_obs_by_id)
        if obs is not None:
            if ais_config is not None:
                ais_rel = _clip01(ais_config.reliability(obs, timestamp))
            else:
                ais_rel = _clip01(getattr(obs, 'reliability', 1.0))
        stability = _clip01(getattr(track, 'tracklet_len', 0) / 30.0)
        reliability = track_conf * (1.0 - 0.40 * s) + 0.30 * ais_rel + 0.20 * stability
        return _clip01(reliability)

    def _edge_risk(self, track_i, track_j, ais_obs_by_id, return_detail=False):
        ci = self._center(track_i)
        cj = self._center(track_j)
        if ci is None or cj is None:
            return (0.0, None) if return_detail else 0.0

        dist = float(np.linalg.norm(ci - cj))
        dynamic_radius = self._dynamic_radius(track_i, track_j)
        if dist > max(self.neighbor_radius, dynamic_radius):
            return (0.0, None) if return_detail else 0.0

        vi = self._fused_velocity(track_i, ais_obs_by_id)
        vj = self._fused_velocity(track_j, ais_obs_by_id)
        dv = float(np.linalg.norm(vi - vj))
        direction = self._direction_similarity(vi, vj)
        convergence = self._future_convergence(ci, cj, vi, vj)
        distance_score = math.exp(-dist / max(dynamic_radius, self.sigma_d))
        velocity_score = math.exp(-dv / self.sigma_v)
        ais_consistency = 1.0 - 0.5 * (
            self._ais_inconsistency(track_i, ais_obs_by_id) +
            self._ais_inconsistency(track_j, ais_obs_by_id))

        logit = (
            1.35 * distance_score +
            0.65 * velocity_score +
            0.50 * direction +
            1.55 * convergence +
            0.35 * ais_consistency -
            2.10)
        risk = _clip01(self._sigmoid(logit))
        if not return_detail:
            return risk
        return risk, {
            'track_i': getattr(track_i, 'track_id', None),
            'track_j': getattr(track_j, 'track_id', None),
            'distance': dist,
            'relative_velocity': dv,
            'direction_similarity': direction,
            'future_convergence': convergence,
            'distance_score': distance_score,
            'velocity_score': velocity_score,
            'ais_consistency': ais_consistency,
            'edge_risk': risk,
        }

    def _future_convergence(self, ci, cj, vi, vj):
        min_dist = float(np.linalg.norm(ci - cj))
        for step in range(1, self.future_steps + 1):
            pi = ci + step * vi
            pj = cj + step * vj
            min_dist = min(min_dist, float(np.linalg.norm(pi - pj)))
        return _clip01(math.exp(-min_dist / self.sigma_f))

    def _history_risk(self, track, detection_conf_by_track_id):
        track_conf = _clip01(getattr(track, 'score', 0.0))
        if detection_conf_by_track_id is None:
            det_conf = track_conf
        else:
            det_conf = _clip01(detection_conf_by_track_id.get(track.track_id, 0.0))

        if track.state == TrackState.Occluded:
            state_risk = 1.0
        elif track.state == TrackState.Lost:
            state_risk = 0.85
        else:
            state_risk = 0.0

        lost_frames = 0
        if track.state in (TrackState.Lost, TrackState.Occluded):
            lost_frames = max(0, getattr(track, 'frame_id', 0) - getattr(track, 'end_frame', 0))

        age_stability = 1.0 - _clip01(getattr(track, 'tracklet_len', 0) / 30.0)
        risk = (
            0.35 * (1.0 - track_conf) +
            0.35 * (1.0 - det_conf) +
            0.20 * state_risk +
            0.10 * age_stability +
            0.05 * _clip01(lost_frames / 30.0))
        return _clip01(risk)

    def _ais_inconsistency(self, track, ais_obs_by_id):
        obs = self._ais_obs(track, ais_obs_by_id)
        if obs is None or obs.vx is None or obs.vy is None:
            return 0.0
        if getattr(track, 'mean', None) is None:
            return 0.0
        vis = np.asarray([track.mean[4], track.mean[5]], dtype=float)
        ais = np.asarray([obs.vx, obs.vy], dtype=float)
        vis_norm = np.linalg.norm(vis)
        ais_norm = np.linalg.norm(ais)
        if vis_norm < 1e-6 or ais_norm < 1e-6:
            return 0.0
        cos = np.dot(vis, ais) / (vis_norm * ais_norm)
        direction_gap = (1.0 - np.clip(cos, -1.0, 1.0)) / 2.0
        speed_gap = np.linalg.norm(vis - ais) / max(self.sigma_v, 1.0)
        return _clip01(0.7 * direction_gap + 0.3 * min(speed_gap, 1.0))

    def _dynamic_radius(self, track_i, track_j):
        wi, hi = self._size(track_i)
        wj, hj = self._size(track_j)
        return max(self.sigma_d, 0.75 * (max(wi, hi) + max(wj, hj)), self.min_bbox_scale)

    def _center(self, track):
        if getattr(track, 'mean', None) is not None:
            return np.asarray(track.mean[:2], dtype=float)
        try:
            return np.asarray(track.to_xywh()[:2], dtype=float)
        except Exception:
            return None

    def _size(self, track):
        if getattr(track, 'mean', None) is not None:
            return float(max(track.mean[2], 1.0)), float(max(track.mean[3], 1.0))
        try:
            xywh = track.to_xywh()
            return float(max(xywh[2], 1.0)), float(max(xywh[3], 1.0))
        except Exception:
            return self.min_bbox_scale, self.min_bbox_scale

    def _fused_velocity(self, track, ais_obs_by_id):
        vis = np.zeros(2, dtype=float)
        if getattr(track, 'mean', None) is not None:
            vis = np.asarray([track.mean[4], track.mean[5]], dtype=float)
        obs = self._ais_obs(track, ais_obs_by_id)
        if obs is None or obs.vx is None or obs.vy is None:
            return vis
        ais = np.asarray([obs.vx, obs.vy], dtype=float)
        s = _clip01(getattr(track, 'occlusion_state', self.prev_states.get(track.track_id, 0.0)))
        weight = 0.20 + 0.55 * s
        return (1.0 - weight) * vis + weight * ais

    @staticmethod
    def _direction_similarity(vi, vj):
        ni = np.linalg.norm(vi)
        nj = np.linalg.norm(vj)
        if ni < 1e-6 or nj < 1e-6:
            return 0.0
        cos = np.dot(vi, vj) / (ni * nj)
        return _clip01((1.0 + np.clip(cos, -1.0, 1.0)) / 2.0)

    @staticmethod
    def _ais_obs(track, ais_obs_by_id):
        if ais_obs_by_id is None or getattr(track, 'ais_id', None) is None:
            return None
        return ais_obs_by_id.get(track.ais_id)

    @staticmethod
    def _sigmoid(value):
        value = float(np.clip(value, -50.0, 50.0))
        return 1.0 / (1.0 + math.exp(-value))
