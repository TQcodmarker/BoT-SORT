import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

from tracker import matching
from tracker.gmc import GMC
from tracker.basetrack import BaseTrack, TrackState
from tracker.kalman_filter import KalmanFilter
from tracker.ais_fusion import AISBuffer, AISFrame, AISFusionConfig


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, feat=None, feat_history=50):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9
        self.freeze_feature = False
        if feat is not None:
            self.update_features(feat)
        self.ais_id = None
        self.last_ais_obs = None
        self.last_ais_time = None
        self.ais_reliability = 0.0
        self.occluded_since = None

    def update_features(self, feat):
        if self.freeze_feature:
            return
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks, ais_obs_by_id=None, ais_config=None, timestamp=None):
        if len(stracks) == 0:
            return

        normal_tracks = []
        for i, st in enumerate(stracks):
            obs = None if ais_obs_by_id is None else ais_obs_by_id.get(st.ais_id)
            reliability = 0.0 if obs is None or ais_config is None else ais_config.reliability(obs, timestamp)
            if obs is not None and reliability > 0 and (obs.vx is not None or obs.vy is not None):
                mean_state = st.mean.copy()
                if st.state != TrackState.Tracked and st.state != TrackState.Occluded:
                    mean_state[6] = 0
                    mean_state[7] = 0
                if obs.vx is not None:
                    mean_state[4] = obs.vx
                if obs.vy is not None:
                    mean_state[5] = obs.vy
                motion_cov_scale = max(0.25, 1.0 - 0.5 * reliability)
                st.mean, st.covariance = STrack.shared_kalman.predict(
                    mean_state, st.covariance, motion_cov_scale=motion_cov_scale)
            else:
                normal_tracks.append(st)

        if len(normal_tracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in normal_tracks])
            multi_covariance = np.asarray([st.covariance for st in normal_tracks])
            for i, st in enumerate(normal_tracks):
                if st.state != TrackState.Tracked and st.state != TrackState.Occluded:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                normal_tracks[i].mean = mean
                normal_tracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.freeze_feature = False
        self.occluded_since = None
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.freeze_feature = False
        self.occluded_since = None
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        if new_track.ais_id is not None:
            self.bind_ais(new_track.ais_id, new_track.last_ais_obs, frame_id)

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.freeze_feature = False
        self.occluded_since = None
        self.is_activated = True

        self.score = new_track.score
        if new_track.ais_id is not None:
            self.bind_ais(new_track.ais_id, new_track.last_ais_obs, frame_id)

    def bind_ais(self, ais_id, obs=None, frame_id=None):
        self.ais_id = ais_id
        if obs is not None:
            self.last_ais_obs = obs.raw_copy() if hasattr(obs, 'raw_copy') else obs
            self.last_ais_time = obs.timestamp if obs.timestamp is not None else frame_id
            self.ais_reliability = getattr(obs, 'reliability', 1.0)

    def has_valid_ais(self, config, timestamp=None):
        return self.last_ais_obs is not None and config.reliability(self.last_ais_obs, timestamp) > 0

    def mark_occluded(self, frame_id):
        self.state = TrackState.Occluded
        self.freeze_feature = True
        self.frame_id = frame_id
        if self.occluded_since is None:
            self.occluded_since = frame_id

    def update_by_ais_virtual(self, obs, config, timestamp=None):
        if obs is None or self.mean is None or self.kalman_filter is None:
            return False
        reliability = config.reliability(obs, timestamp)
        if reliability <= 0:
            return False

        measurement = self.mean[:4].copy()
        measurement[0] = obs.x
        measurement[1] = obs.y
        position_var = config.position_variance(obs, timestamp)
        self.mean, self.covariance = self.kalman_filter.update_virtual(
            self.mean, self.covariance, measurement,
            position_var=position_var, scale_var=config.scale_var)
        self.bind_ais(obs.ais_id, obs, self.frame_id)
        self.ais_reliability = reliability
        return True

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BoTSORT(object):
    def __init__(self, args, frame_rate=30):

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.frame_id = 0
        self.args = args

        self.track_high_thresh = args.track_high_thresh
        self.track_low_thresh = args.track_low_thresh
        self.new_track_thresh = args.new_track_thresh

        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        if args.with_reid:
            from fast_reid.fast_reid_interfece import FastReIDInterface
            self.encoder = FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, args.device)

        self.gmc = GMC(method=args.cmc_method, verbose=[args.name, args.ablation])
        self.ais_config = AISFusionConfig.from_args(args)
        self.ais_buffer = None

    def _ais_observations_for_tracks(self, tracks, current_ais_by_id, timestamp):
        obs_by_id = dict(current_ais_by_id)
        for track in tracks:
            if track.ais_id is None or track.ais_id in obs_by_id:
                continue
            obs = track.last_ais_obs
            if obs is not None and self.ais_config.reliability(obs, timestamp) > 0:
                obs_by_id[track.ais_id] = obs
        return obs_by_id

    def _get_ais_frame(self, ais_frame, img, timestamp):
        if ais_frame is not None:
            return AISFrame(ais_frame)
        ais_path = getattr(self.args, 'ais_path', None)
        camera_para = getattr(self.args, 'camera_para', None)
        if ais_path is None or camera_para is None or timestamp is None:
            return AISFrame()
        if self.ais_buffer is None:
            image_shape = [img.shape[1], img.shape[0]]
            self.ais_buffer = AISBuffer(ais_path, camera_para, image_shape)
        return self.ais_buffer.query(timestamp)

    def _point_in_oar(self, xy):
        oar = getattr(self.args, 'oar_polygon', None)
        if oar is None:
            return True
        polygon = np.asarray(oar, dtype=float)
        if len(polygon) < 3:
            return True
        x, y = xy
        inside = False
        j = len(polygon) - 1
        for i in range(len(polygon)):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            denom = yj - yi
            if abs(denom) < 1e-12:
                denom = 1e-12
            intersect = ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / denom + xi)
            if intersect:
                inside = not inside
            j = i
        return inside

    def _can_mark_occluded(self, track, ais_by_track_id, timestamp):
        if track.ais_id not in ais_by_track_id:
            return False
        obs = ais_by_track_id[track.ais_id]
        if self.ais_config.reliability(obs, timestamp) < self.ais_config.occlusion_min_score:
            return False
        if not self._point_in_oar(track.mean[:2]):
            return False
        if track.occluded_since is not None:
            if self.frame_id - track.occluded_since > self.ais_config.occlusion_max_frames:
                return False
        return True

    def _apply_ais_cmc(self, ais_frame, warp):
        mode = self.ais_config.cmc_mode
        if mode == 'none':
            return
        if mode == 'same':
            ais_frame.apply_gmc(warp)
            return
        if mode == 'inverse':
            H = np.eye(3, dtype=float)
            H[:2, :] = warp
            try:
                H_inv = np.linalg.inv(H)[:2, :]
            except np.linalg.LinAlgError:
                H_inv = np.eye(2, 3, dtype=float)
            ais_frame.apply_gmc(H_inv)

    def _assign_ais_to_new_tracks(self, tracks, ais_frame, used_ais_ids, timestamp):
        candidates = [obs for obs in ais_frame.records if obs.ais_id is not None and obs.ais_id not in used_ais_ids]
        if len(tracks) == 0 or len(candidates) == 0:
            return

        pairs = []
        for ti, track in enumerate(tracks):
            center = track.to_xywh()[:2]
            for oi, obs in enumerate(candidates):
                if self.ais_config.reliability(obs, timestamp) <= 0:
                    continue
                dist = np.linalg.norm(center - obs.xy)
                if dist <= self.ais_config.bind_distance:
                    pairs.append((dist, ti, oi))
        pairs.sort(key=lambda item: item[0])

        used_tracks = set()
        used_obs = set()
        for _, ti, oi in pairs:
            if ti in used_tracks or oi in used_obs:
                continue
            tracks[ti].bind_ais(candidates[oi].ais_id, candidates[oi], self.frame_id)
            tracks[ti].update_by_ais_virtual(candidates[oi], self.ais_config, timestamp)
            used_tracks.add(ti)
            used_obs.add(oi)
            used_ais_ids.add(candidates[oi].ais_id)

    def update(self, output_results, img, ais_frame=None, timestamp=None):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
                classes = output_results[:, -1]
            else:
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2
                classes = output_results[:, -1]

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]

        else:
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            classes_keep = []

        '''Extract embeddings '''
        if self.args.with_reid:
            features_keep = self.encoder.inference(img, dets)

        if len(dets) > 0:
            '''Detections'''
            if self.args.with_reid:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f) for
                              (tlbr, s, f) in zip(dets, scores_keep, features_keep)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                              (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ais_frame = self._get_ais_frame(ais_frame, img, timestamp)
        ais_by_id = ais_frame.by_id()

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        ais_by_track_id = self._ais_observations_for_tracks(strack_pool, ais_by_id, timestamp)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool, ais_by_track_id, self.ais_config, timestamp)

        # Fix camera motion
        warp = self.gmc.apply(img, dets)
        self._apply_ais_cmc(ais_frame, warp)
        ais_by_id = ais_frame.by_id()
        ais_by_track_id = self._ais_observations_for_tracks(strack_pool, ais_by_id, timestamp)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        # Associate with high score detection boxes
        ious_dists = matching.iou_distance(strack_pool, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)

            # Popular ReID method (JDE / FairMOT)
            # raw_emb_dists = matching.embedding_distance(strack_pool, detections)
            # dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, strack_pool, detections)
            # emb_dists = dists

            # IoU making ReID
            # dists = matching.embedding_distance(strack_pool, detections)
            # dists[ious_dists_mask] = 1.0
        else:
            dists = ious_dists

        dists = matching.fuse_ais(
            dists, strack_pool, detections, ais_by_track_id,
            self.ais_config, timestamp)

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for track in activated_starcks + refind_stracks:
            if track.ais_id in ais_by_track_id:
                track.update_by_ais_virtual(ais_by_track_id[track.ais_id], self.ais_config, timestamp)

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            classes_second = classes[inds_second]
        else:
            dets_second = []
            scores_second = []
            classes_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track
                             if strack_pool[i].state == TrackState.Tracked
                             or strack_pool[i].state == TrackState.Occluded]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        dists = matching.fuse_ais(
            dists, r_tracked_stracks, detections_second, ais_by_track_id,
            self.ais_config, timestamp)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            if track.ais_id in ais_by_track_id:
                track.update_by_ais_virtual(ais_by_track_id[track.ais_id], self.ais_config, timestamp)

        for it in u_track:
            track = r_tracked_stracks[it]
            if self._can_mark_occluded(track, ais_by_track_id, timestamp):
                track.mark_occluded(self.frame_id)
                track.update_by_ais_virtual(ais_by_track_id[track.ais_id], self.ais_config, timestamp)
                lost_stracks.append(track)
            elif not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        ious_dists = matching.iou_distance(unconfirmed, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)
        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        new_tracks = []
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            new_tracks.append(track)
            activated_starcks.append(track)
        used_ais_ids = set([track.ais_id for track in self.tracked_stracks + self.lost_stracks
                            if track.ais_id is not None])
        self._assign_ais_to_new_tracks(new_tracks, ais_frame, used_ais_ids, timestamp)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if track.state == TrackState.Occluded:
                if track.occluded_since is not None and \
                        self.frame_id - track.occluded_since > self.ais_config.occlusion_max_frames:
                    track.mark_lost()
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.lost_stracks = [t for t in self.lost_stracks if t.state != TrackState.Removed]
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]


        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
