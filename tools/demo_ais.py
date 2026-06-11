import argparse
import os
import os.path as osp
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
from loguru import logger


BOT_SORT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
DEEPSORVF_ROOT = osp.abspath(osp.join(BOT_SORT_ROOT, '..'))
if BOT_SORT_ROOT not in sys.path:
    sys.path.insert(0, BOT_SORT_ROOT)
if DEEPSORVF_ROOT not in sys.path:
    sys.path.insert(0, DEEPSORVF_ROOT)

from tracker.bot_sort import BoTSORT
from tracker.ais_fusion import AISFrame
from tracker.tracking_utils.timer import Timer
from utils.AIS_utils import AISPRO
from utils.draw import DRAW
from utils.file_read import ais_initial, read_all, update_time
from utils.FUS_utils import FUSPRO
from utils.gen_result import gen_result
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess


def normalize_data_path(path):
    path = path.replace('\\', '/')
    if not path.endswith('/'):
        path += '/'
    return path


def resize_by_height(img, height):
    if height <= 0 or img.shape[0] == height:
        return img
    scale = float(height) / float(img.shape[0])
    width = int(round(img.shape[1] * scale))
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)


def timestamp_to_tracker_ms(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    value = float(value)
    if value > 10000000000:
        return value
    return value * 1000.0


def current_ais_vis(AIS_vis, frame_timestamp_ms):
    if AIS_vis is None or len(AIS_vis) == 0:
        return pd.DataFrame(columns=[
            'mmsi', 'lon', 'lat', 'speed', 'course', 'heading',
            'type', 'x', 'y', 'timestamp'
        ])
    frame_sec = int(frame_timestamp_ms // 1000)
    timestamps = AIS_vis['timestamp'].astype(float).astype(int)
    return AIS_vis[timestamps == frame_sec].reset_index(drop=True)


def ais_vis_to_records(AIS_vis, frame_timestamp_ms):
    records = []
    if AIS_vis is None or len(AIS_vis) == 0:
        return records

    for _, row in AIS_vis.iterrows():
        if pd.isna(row.get('mmsi')) or pd.isna(row.get('x')) or pd.isna(row.get('y')):
            continue
        ais_timestamp = timestamp_to_tracker_ms(row.get('timestamp'))
        if ais_timestamp is None:
            continue
        delta_t = abs(frame_timestamp_ms - ais_timestamp) / 1000.0
        records.append({
            'ais_id': int(row['mmsi']),
            'x': float(row['x']),
            'y': float(row['y']),
            'timestamp': ais_timestamp,
            'speed': row.get('speed', None),
            'course': row.get('course', None),
            'heading': row.get('heading', None),
            'lon': row.get('lon', None),
            'lat': row.get('lat', None),
            'reliability': 1.0,
            'delta_t': delta_t,
        })
    return records


def targets_to_vis_cur(online_targets, frame_timestamp_ms, args):
    vis_rows = []

    for target in online_targets:
        tlwh = target.tlwh
        vertical = tlwh[2] / max(tlwh[3], 1e-6) > args.aspect_ratio_thresh
        if tlwh[2] * tlwh[3] <= args.min_box_area or vertical:
            continue

        x1 = int(max(tlwh[0], 0))
        y1 = int(max(tlwh[1], 0))
        x2 = int(max(tlwh[0] + tlwh[2], 0))
        y2 = int(max(tlwh[1] + tlwh[3], 0))
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        track_id = int(target.track_id)
        vis_rows.append({
            'ID': track_id,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'x': cx,
            'y': cy,
            'timestamp': int(frame_timestamp_ms // 1000),
        })

    return pd.DataFrame(
        vis_rows,
        columns=['ID', 'x1', 'y1', 'x2', 'y2', 'x', 'y', 'timestamp'])


def fusion_bindings(Fus_tra):
    bindings = {}
    if Fus_tra is None or len(Fus_tra) == 0:
        return bindings
    for _, row in Fus_tra.iterrows():
        try:
            bindings[int(row['ID'])] = int(row['mmsi'])
        except (TypeError, ValueError):
            continue
    return bindings


def apply_fusion_binding_to_tracks(online_targets, Fus_tra, ais_records, tracker,
                                   timestamp, state_update=False):
    bindings = fusion_bindings(Fus_tra)
    if len(bindings) == 0:
        return

    ais_by_id = AISFrame(ais_records).by_id()
    for target in online_targets:
        track_id = int(target.track_id)
        if track_id not in bindings:
            continue
        mmsi = bindings[track_id]
        obs = ais_by_id.get(mmsi)
        target.bind_ais(mmsi, obs, tracker.frame_id)
        if state_update and obs is not None:
            target.update_by_ais_virtual(obs, tracker.ais_config, timestamp)


class Predictor(object):
    def __init__(self, model, exp, device, fp16=False):
        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {
            'height': img.shape[0],
            'width': img.shape[1],
            'raw_img': img,
        }
        proc_img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info['ratio'] = ratio
        proc_img = torch.from_numpy(proc_img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            proc_img = proc_img.half()

        with torch.no_grad():
            timer.tic()
            outputs = self.model(proc_img)
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        return outputs, img_info


def build_predictor(args):
    if args.with_reid:
        if not osp.isabs(args.fast_reid_config):
            args.fast_reid_config = osp.join(BOT_SORT_ROOT, args.fast_reid_config)
        if not osp.isabs(args.fast_reid_weights):
            args.fast_reid_weights = osp.join(BOT_SORT_ROOT, args.fast_reid_weights)

    exp = get_exp(args.exp_file, args.name)
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    device = torch.device('cuda' if args.device == 'gpu' and torch.cuda.is_available() else 'cpu')
    model = exp.get_model().to(device)
    logger.info('Model Summary: {}'.format(get_model_info(model, exp.test_size)))
    model.eval()

    if args.ckpt is None:
        raise ValueError('Please pass --ckpt for the YOLOX detector checkpoint.')
    logger.info('loading checkpoint')
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt.get('model', ckpt))
    logger.info('loaded checkpoint done.')

    if args.fuse:
        logger.info('Fusing model...')
        model = fuse_model(model)
    if args.fp16:
        model = model.half()
    return Predictor(model, exp, device, args.fp16)


def update_tracker_without_ais(tracker, detections, img, timestamp):
    config = tracker.ais_config
    old_max_age = config.max_age
    old_cost_weight = config.cost_weight
    old_heading_weight = config.heading_weight
    old_occlusion_min_score = config.occlusion_min_score
    try:
        config.max_age = -1.0
        config.cost_weight = 0.0
        config.heading_weight = 0.0
        config.occlusion_min_score = float('inf')
        return tracker.update(detections, img, ais_frame=[], timestamp=timestamp)
    finally:
        config.max_age = old_max_age
        config.cost_weight = old_cost_weight
        config.heading_weight = old_heading_weight
        config.occlusion_min_score = old_occlusion_min_score


def run(args):
    args.mot20 = not args.fuse_score
    args.data_path = normalize_data_path(args.data_path)
    args.result_path = normalize_data_path(args.result_path)
    video_path, ais_path, result_video, result_metric, initial_time, camera_para = read_all(
        args.data_path, args.result_path)
    if args.path:
        if osp.basename(osp.normpath(args.path)) != osp.basename(osp.normpath(video_path)):
            raise ValueError(
                '--path must point to the same clip video as --data_path so AIS time stays aligned. '
                'Use --data_path for the target clip directory.')
        video_path = args.path

    ais_file, timestamp0, time0 = ais_initial(ais_path, initial_time)
    Time = initial_time.copy()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError('Unable to open video: {}'.format(video_path))

    im_shape = [cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = args.fps
    args.fps = fps
    frame_step_ms = int(1000 / fps)

    AIS = AISPRO(ais_path, ais_file, im_shape, frame_step_ms)
    FUS = FUSPRO(min(im_shape) // 2, im_shape, frame_step_ms)
    DRA = DRAW(im_shape, frame_step_ms)
    predictor = build_predictor(args)
    tracker = BoTSORT(args, frame_rate=fps)
    timer = Timer()

    os.makedirs(osp.dirname(result_video), exist_ok=True)
    writer = None
    Vis_tra = pd.DataFrame(columns=['ID', 'x1', 'y1', 'x2', 'y2', 'x', 'y', 'timestamp'])
    bin_inf = pd.DataFrame(columns=['ID', 'mmsi', 'timestamp', 'match'])
    times = 0
    frame_id = 0
    time_i = 0.0
    sum_t = []

    logger.info('Start Time: {} || Stamp: {} || fps: {}'.format(time0, timestamp0, fps))
    while True:
        ok, im = cap.read()
        if not ok or im is None:
            break
        frame_id += 1
        start = time.time()

        Time, timestamp, Time_name = update_time(Time, frame_step_ms)
        ais_update_frame = timestamp % 1000 < frame_step_ms
        AIS_vis, AIS_cur = AIS.process(camera_para, timestamp, Time_name)
        AIS_vis_current = current_ais_vis(AIS_vis, timestamp) if ais_update_frame else current_ais_vis(None, timestamp)
        ais_frame = ais_vis_to_records(AIS_vis_current, timestamp)

        outputs, img_info = predictor.inference(im, timer)
        scale = min(
            predictor.test_size[0] / float(img_info['height']),
            predictor.test_size[1] / float(img_info['width']))

        if outputs[0] is not None:
            detections = outputs[0].cpu().numpy()[:, :7]
            detections[:, :4] /= scale
        else:
            detections = np.empty((0, 7), dtype=float)

        online_targets = update_tracker_without_ais(
            tracker, detections, img_info['raw_img'], timestamp)
        timer.toc()

        Vis_cur = targets_to_vis_cur(online_targets, timestamp, args)
        if len(Vis_cur) > 0:
            Vis_tra = pd.concat([Vis_tra, Vis_cur], ignore_index=True)
            Vis_tra = Vis_tra.drop(
                Vis_tra[Vis_tra['timestamp'] < (timestamp // 1000 - 2 * 60)].index)
        Fus_tra, bin_inf = FUS.fusion(AIS_vis, AIS_cur, Vis_tra, Vis_cur, timestamp)
        apply_fusion_binding_to_tracks(
            online_targets, Fus_tra, ais_frame, tracker, timestamp, args.ais_state_update)

        end = time.time() - start
        time_i += end
        if timestamp % 1000 < frame_step_ms:
            gen_result(times, Vis_cur, Fus_tra, result_metric, im_shape)
            times += 1
            sum_t.append(time_i)
            logger.info('Time: {} || Stamp: {} || Process: {:.6f} || Average: {:.6f} +- {:.6f}'.format(
                Time_name, timestamp, time_i, np.mean(sum_t), np.std(sum_t)))
            time_i = 0.0

        result = DRA.draw_traj(im, AIS_vis, AIS_cur, Vis_tra, Vis_cur, Fus_tra, timestamp)
        result = resize_by_height(result, args.show_size)
        if args.save_result and writer is None:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            writer = cv2.VideoWriter(result_video, fourcc, fps, (result.shape[1], result.shape[0]))
        if args.save_result:
            writer.write(result)

        if args.show:
            cv2.imshow('BoT-SORT AIS', result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if args.max_frames > 0 and frame_id >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    logger.info('Saved metric prefix: {}'.format(result_metric))
    if args.save_result:
        logger.info('Saved video: {}'.format(result_video))


def make_parser():
    parser = argparse.ArgumentParser('BoT-SORT AIS Demo with DeepSORVF-style output')
    parser.add_argument('demo', default='video', choices=['video'], help='demo type')
    parser.add_argument('--path', default='', help='optional video path override')
    parser.add_argument('--data_path', type=str, default='../clip-01/', help='DeepSORVF clip directory')
    parser.add_argument('--result_path', type=str, default='../result_ais/', help='DeepSORVF-style result directory')
    parser.add_argument('--save_result', action='store_true', help='save rendered video')
    parser.add_argument('--show', action='store_true', help='show rendered video')
    parser.add_argument('--show_size', type=int, default=500)
    parser.add_argument('--max_frames', type=int, default=-1)

    parser.add_argument('-f', '--exp_file', default=None, type=str)
    parser.add_argument('-c', '--ckpt', default=None, type=str)
    parser.add_argument('-n', '--name', default=None, type=str)
    parser.add_argument('--device', default='gpu', choices=['gpu', 'cpu'])
    parser.add_argument('--conf', default=None, type=float)
    parser.add_argument('--nms', default=None, type=float)
    parser.add_argument('--tsize', default=None, type=int)
    parser.add_argument('--fps', default=30, type=int)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fuse', action='store_true')

    parser.add_argument('--track_high_thresh', type=float, default=0.6)
    parser.add_argument('--track_low_thresh', default=0.1, type=float)
    parser.add_argument('--new_track_thresh', default=0.7, type=float)
    parser.add_argument('--track_buffer', type=int, default=30)
    parser.add_argument('--match_thresh', type=float, default=0.8)
    parser.add_argument('--aspect_ratio_thresh', type=float, default=1.6)
    parser.add_argument('--min_box_area', type=float, default=10)
    parser.add_argument('--fuse-score', dest='fuse_score', default=False, action='store_true')

    parser.add_argument('--cmc-method', dest='cmc_method', default='orb', type=str)
    parser.add_argument('--with-reid', dest='with_reid', default=False, action='store_true')
    parser.add_argument('--fast-reid-config', dest='fast_reid_config',
                        default='fast_reid/configs/MOT17/sbs_S50.yml')
    parser.add_argument('--fast-reid-weights', dest='fast_reid_weights',
                        default='pretrained/mot17_sbs_S50.pth')
    parser.add_argument('--proximity_thresh', type=float, default=0.5)
    parser.add_argument('--appearance_thresh', type=float, default=0.25)

    parser.add_argument('--ais-max-age', dest='ais_max_age', type=float, default=2.0)
    parser.add_argument('--ais-kappa', dest='ais_kappa', type=float, default=0.5)
    parser.add_argument('--ais-position-var', dest='ais_position_var', type=float, default=4.0)
    parser.add_argument('--ais-scale-var', dest='ais_scale_var', type=float, default=1000000.0)
    parser.add_argument('--ais-bind-distance', dest='ais_bind_distance', type=float, default=120.0)
    parser.add_argument('--ais-cost-weight', dest='ais_cost_weight', type=float, default=0.25)
    parser.add_argument('--ais-heading-weight', dest='ais_heading_weight', type=float, default=0.05)
    parser.add_argument('--ais-occlusion-min-score', dest='ais_occlusion_min_score', type=float, default=0.4)
    parser.add_argument('--ais-occlusion-max-frames', dest='ais_occlusion_max_frames', type=int, default=60)
    parser.add_argument('--ais-cmc-mode', dest='ais_cmc_mode',
                        choices=['none', 'same', 'inverse'], default='inverse')
    parser.add_argument('--ais-state-update', dest='ais_state_update',
                        default=False, action='store_true',
                        help='after FUSPRO binding, correct bound BoT-SORT states with current AIS')
    parser.set_defaults(ablation=False, mot20=False, oar_polygon=None)
    return parser


if __name__ == '__main__':
    run(make_parser().parse_args())
