import csv
import datetime
import math
import os

import numpy as np


def _timestamp_to_seconds(timestamp):
    if timestamp is None:
        return None
    timestamp = float(timestamp)
    if timestamp > 10000000000:
        return timestamp / 1000.0
    return timestamp


def _datetime_name_from_timestamp(timestamp):
    seconds = _timestamp_to_seconds(timestamp)
    if seconds is None:
        return None
    dt = datetime.datetime.fromtimestamp(seconds)
    return dt.strftime('%Y_%m_%d_%H_%M_%S')


class AISObservation(object):
    """Frame-level projected AIS observation in image coordinates."""

    def __init__(self, ais_id, x, y, timestamp=None, speed=None, course=None,
                 heading=None, vx=None, vy=None, position_var=None,
                 reliability=1.0, lon=None, lat=None, speed_variance=None,
                 continuity=1.0, vx_ground=None, vy_ground=None):
        self.ais_id = ais_id
        self.x = float(x)
        self.y = float(y)
        self.raw_x = float(x)
        self.raw_y = float(y)
        self.timestamp = timestamp
        self.speed = None if speed is None else float(speed)
        self.course = None if course is None else float(course)
        self.heading = None if heading is None else float(heading)
        self.vx = vx
        self.vy = vy
        self.position_var = position_var
        self.reliability = float(reliability)
        self.lon = lon
        self.lat = lat
        self.speed_variance = speed_variance
        self.continuity = float(continuity)
        self.vx_ground = vx_ground
        self.vy_ground = vy_ground

    @property
    def xy(self):
        return np.asarray([self.x, self.y], dtype=float)

    def copy(self):
        obs = AISObservation(
            self.ais_id, self.x, self.y, timestamp=self.timestamp,
            speed=self.speed, course=self.course, heading=self.heading,
            vx=self.vx, vy=self.vy, position_var=self.position_var,
            reliability=self.reliability, lon=self.lon, lat=self.lat,
            speed_variance=self.speed_variance, continuity=self.continuity,
            vx_ground=self.vx_ground, vy_ground=self.vy_ground)
        obs.raw_x = self.raw_x
        obs.raw_y = self.raw_y
        return obs

    def raw_copy(self):
        obs = self.copy()
        obs.x = obs.raw_x
        obs.y = obs.raw_y
        return obs


class AISFrame(object):
    """Adapter for heterogeneous projected AIS records passed to BoTSORT."""

    def __init__(self, records=None):
        self.records = []
        if records is None:
            return
        if isinstance(records, AISFrame):
            self.records = [obs.copy() for obs in records.records]
            return
        if isinstance(records, dict):
            records = [records]
        for record in records:
            obs = self.from_record(record)
            if obs is not None:
                self.records.append(obs.copy())

    @staticmethod
    def from_record(record):
        if isinstance(record, AISObservation):
            return record

        if isinstance(record, dict):
            xy = record.get('xy')
            x = record.get('x', record.get('img_x', record.get('u')))
            y = record.get('y', record.get('img_y', record.get('v')))
            if xy is not None:
                x, y = xy[0], xy[1]
            if x is None or y is None:
                return None
            return AISObservation(
                record.get('ais_id', record.get('mmsi', record.get('id'))),
                x, y,
                timestamp=record.get('timestamp', record.get('time')),
                speed=record.get('speed', record.get('sog')),
                course=record.get('course', record.get('cog')),
                heading=record.get('heading'),
                vx=record.get('vx'),
                vy=record.get('vy'),
                position_var=record.get('position_var'),
                reliability=record.get('reliability', 1.0),
                lon=record.get('lon'),
                lat=record.get('lat'),
                speed_variance=record.get('speed_variance'),
                continuity=record.get('continuity', 1.0),
                vx_ground=record.get('vx_ground'),
                vy_ground=record.get('vy_ground'))

        try:
            record_len = len(record)
        except TypeError:
            return None

        if record_len >= 3:
            return AISObservation(record[0], record[1], record[2])
        return None

    def by_id(self):
        return {obs.ais_id: obs for obs in self.records if obs.ais_id is not None}

    def nearest(self, xy, max_distance=None):
        if len(self.records) == 0:
            return None
        xy = np.asarray(xy, dtype=float)
        dists = [np.linalg.norm(obs.xy - xy) for obs in self.records]
        index = int(np.argmin(dists))
        if max_distance is not None and dists[index] > max_distance:
            return None
        return self.records[index]

    def apply_gmc(self, H):
        if H is None or len(self.records) == 0:
            return
        H = np.asarray(H, dtype=float)
        R = H[:2, :2]
        t = H[:2, 2]
        for obs in self.records:
            xy = R.dot(obs.xy) + t
            obs.x, obs.y = float(xy[0]), float(xy[1])


class AISProjector(object):
    """Project FVessel AIS lon/lat records into image coordinates."""

    def __init__(self, camera_para, image_shape):
        self.camera_para = camera_para
        self.image_shape = image_shape

    @staticmethod
    def _distance_m(lat1, lon1, lat2, lon2):
        earth_radius = 6371000.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lam = math.radians(lon2 - lon1)
        a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2.0) ** 2
        return 2.0 * earth_radius * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    @staticmethod
    def _bearing_deg(lat1, lon1, lat2, lon2):
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        d_lam = math.radians(lon2 - lon1)
        y = math.sin(d_lam) * math.cos(phi2)
        x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(d_lam)
        return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

    @staticmethod
    def _advance_lonlat(lon, lat, course, speed_knots, dt_sec):
        distance = speed_knots * dt_sec * 1852.0 / 3600.0
        bearing = math.radians(course)
        lat_rad = math.radians(lat)
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = max(1e-6, 111320.0 * math.cos(lat_rad))
        d_north = distance * math.cos(bearing)
        d_east = distance * math.sin(bearing)
        return lon + d_east / meters_per_deg_lon, lat + d_north / meters_per_deg_lat

    def project(self, lon, lat):
        lon_cam = self.camera_para[0]
        lat_cam = self.camera_para[1]
        shoot_hdir = self.camera_para[2]
        shoot_vdir = self.camera_para[3]
        height_cam = self.camera_para[4]
        f_x = self.camera_para[7]
        f_y = self.camera_para[8]
        u0 = self.camera_para[9]
        v0 = self.camera_para[10]

        distance = self._distance_m(lat_cam, lon_cam, lat, lon)
        relative_angle = self._bearing_deg(lat_cam, lon_cam, lat, lon)
        angle_hor = relative_angle - shoot_hdir
        if angle_hor < -180:
            angle_hor += 360
        elif angle_hor > 180:
            angle_hor -= 360

        hor_rad = math.radians(angle_hor)
        shv_rad = math.radians(-shoot_vdir)
        z_w = distance * math.cos(hor_rad)
        x_w = distance * math.sin(hor_rad)
        y_w = height_cam
        z = z_w / math.cos(shv_rad) + (y_w - z_w * math.tan(shv_rad)) * math.sin(shv_rad)
        if abs(z) < 1e-6:
            return None
        x = f_x * x_w / z + u0
        y = f_y * (y_w - z_w * math.tan(shv_rad)) * math.cos(shv_rad) / z + v0
        return float(x), float(y)

    def velocity_px(self, lon, lat, course, speed):
        if course is None or speed is None or speed <= 0:
            return None, None
        xy0 = self.project(lon, lat)
        lon1, lat1 = self._advance_lonlat(lon, lat, course, speed, 1.0)
        xy1 = self.project(lon1, lat1)
        if xy0 is None or xy1 is None:
            return None, None
        return xy1[0] - xy0[0], xy1[1] - xy0[1]

    def from_csv_row(self, row):
        try:
            lon = float(row['lon'])
            lat = float(row['lat'])
            speed = float(row['speed'])
            course = float(row['course'])
            heading = float(row['heading'])
            timestamp = float(row['timestamp'])
        except (KeyError, TypeError, ValueError):
            return None
        if lon <= 0 or lat <= 0 or speed < 0 or course < 0 or course >= 360:
            return None
        xy = self.project(lon, lat)
        if xy is None:
            return None
        vx, vy = self.velocity_px(lon, lat, course, speed)
        return AISObservation(
            row.get('mmsi'), xy[0], xy[1], timestamp=timestamp,
            speed=speed, course=course, heading=heading, vx=vx, vy=vy,
            lon=lon, lat=lat)


class AISBuffer(object):
    """Per-second AIS CSV reader for FVessel-style clips."""

    def __init__(self, ais_path=None, camera_para=None, image_shape=None):
        self.ais_path = ais_path
        self.projector = None
        if ais_path is not None and camera_para is not None and image_shape is not None:
            self.projector = AISProjector(camera_para, image_shape)
        self._cache = {}

    def query(self, timestamp):
        if self.ais_path is None or self.projector is None or timestamp is None:
            return AISFrame()
        name = _datetime_name_from_timestamp(timestamp)
        if name is None:
            return AISFrame()
        if name not in self._cache:
            self._cache[name] = self._read_second(name)
        return AISFrame(self._cache[name])

    def _read_second(self, name):
        path = os.path.join(self.ais_path, name + '.csv')
        if not os.path.exists(path):
            return []
        records = []
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                obs = self.projector.from_csv_row(row)
                if obs is not None:
                    records.append(obs)
        return records


class AISFusionConfig(object):
    def __init__(self, max_age=2.0, kappa=0.5, position_var=4.0,
                 scale_var=1000000.0, bind_distance=120.0, cost_weight=0.25,
                 heading_weight=0.05, occlusion_min_score=0.4,
                 occlusion_max_frames=60, cmc_mode='inverse',
                 motion_max_weight=1.0, max_speed_variance=4.0,
                 enable_virtual_update=False):
        self.max_age = max_age
        self.kappa = kappa
        self.position_var = position_var
        self.scale_var = scale_var
        self.bind_distance = bind_distance
        self.cost_weight = cost_weight
        self.heading_weight = heading_weight
        self.occlusion_min_score = occlusion_min_score
        self.occlusion_max_frames = occlusion_max_frames
        self.cmc_mode = cmc_mode
        self.motion_max_weight = motion_max_weight
        self.max_speed_variance = max_speed_variance
        self.enable_virtual_update = enable_virtual_update

    @classmethod
    def from_args(cls, args):
        return cls(
            max_age=getattr(args, 'ais_max_age', 2.0),
            kappa=getattr(args, 'ais_kappa', 0.5),
            position_var=getattr(args, 'ais_position_var', 4.0),
            scale_var=getattr(args, 'ais_scale_var', 1000000.0),
            bind_distance=getattr(args, 'ais_bind_distance', 120.0),
            cost_weight=getattr(args, 'ais_cost_weight', 0.25),
            heading_weight=getattr(args, 'ais_heading_weight', 0.05),
            occlusion_min_score=getattr(args, 'ais_occlusion_min_score', 0.4),
            occlusion_max_frames=getattr(args, 'ais_occlusion_max_frames', 60),
            cmc_mode=getattr(args, 'ais_cmc_mode', 'inverse'),
            motion_max_weight=getattr(args, 'ais_motion_max_weight', 1.0),
            max_speed_variance=getattr(args, 'ais_max_speed_variance', 4.0),
            enable_virtual_update=getattr(args, 'ais_enable_virtual_update', False))

    def delta_t(self, obs, timestamp):
        obs_time = _timestamp_to_seconds(getattr(obs, 'timestamp', None))
        frame_time = _timestamp_to_seconds(timestamp)
        if obs_time is None or frame_time is None:
            return 0.0
        return max(0.0, frame_time - obs_time)

    def reliability(self, obs, timestamp):
        reliability = getattr(obs, 'reliability', 1.0)
        delta_t = self.delta_t(obs, timestamp)
        if delta_t > self.max_age:
            return 0.0
        time_conf = math.exp(-self.kappa * delta_t)

        speed_variance = getattr(obs, 'speed_variance', None)
        if speed_variance is None:
            speed_conf = 1.0
        else:
            speed_conf = math.exp(
                -max(0.0, float(speed_variance)) /
                max(self.max_speed_variance, 1e-6))

        continuity = np.clip(getattr(obs, 'continuity', 1.0), 0.0, 1.0)
        return float(np.clip(reliability * time_conf * speed_conf * continuity, 0.0, 1.0))

    def occlusion_aware_weight(self, occlusion_score, track_conf=None,
                               detection_conf=None):
        occlusion_score = float(np.clip(occlusion_score, 0.0, 1.0))
        track_conf = 1.0 if track_conf is None else float(np.clip(track_conf, 0.0, 1.0))
        detection_conf = track_conf if detection_conf is None else float(np.clip(detection_conf, 0.0, 1.0))

        if occlusion_score < 0.2 and track_conf >= 0.6 and detection_conf >= 0.5:
            return 0.0
        if occlusion_score < 0.5:
            weight = 0.3 + 0.4 * occlusion_score
        else:
            weight = 0.7 + 0.3 * occlusion_score
        if track_conf < 0.5 or detection_conf < 0.4:
            weight = max(weight, 0.7)
        return float(np.clip(weight, 0.0, 1.0))

    def motion_weight(self, track, obs, timestamp, detection_conf=None):
        if obs is None:
            return 0.0
        ais_conf = self.reliability(obs, timestamp)
        if ais_conf <= 0:
            return 0.0

        state_name = track.state
        if state_name in (2, 5):  # TrackState.Lost / TrackState.Occluded
            occlusion_score = 1.0
        elif getattr(track, 'occluded_since', None) is not None:
            occlusion_score = 0.8
        else:
            track_conf = getattr(track, 'score', 1.0)
            occlusion_score = 1.0 - float(np.clip(track_conf, 0.0, 1.0))

        gate_weight = self.occlusion_aware_weight(
            occlusion_score,
            track_conf=getattr(track, 'score', 1.0),
            detection_conf=detection_conf)
        return float(np.clip(gate_weight * ais_conf * self.motion_max_weight, 0.0, 1.0))

    def can_output_occluded(self, track, obs, timestamp, frame_id):
        if obs is None or track is None or track.mean is None:
            return False
        if getattr(track, 'state', None) != 5:  # TrackState.Occluded
            return False
        occluded_since = getattr(track, 'occluded_since', None)
        if occluded_since is None:
            return False
        if frame_id - occluded_since > self.occlusion_max_frames:
            return False
        if self.reliability(obs, timestamp) < self.occlusion_min_score:
            return False
        if self.motion_weight(track, obs, timestamp) < self.occlusion_min_score:
            return False
        w, h = float(track.mean[2]), float(track.mean[3])
        return w > 1.0 and h > 1.0

    def position_variance(self, obs, timestamp):
        base = obs.position_var if obs.position_var is not None else self.position_var
        rel = max(self.reliability(obs, timestamp), 1e-3)
        return float(base) / rel
