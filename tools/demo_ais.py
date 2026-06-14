import argparse
import inspect
import os.path as osp
import sys
import time

import cv2
import imutils
import numpy as np
import pandas as pd


BOT_SORT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
DEEPSORVF_ROOT = osp.abspath(osp.join(BOT_SORT_ROOT, '..'))
if BOT_SORT_ROOT not in sys.path:
    sys.path.insert(0, BOT_SORT_ROOT)
if DEEPSORVF_ROOT not in sys.path:
    sys.path.insert(0, DEEPSORVF_ROOT)

from tracker.bot_sort import BoTSORT
from utils.AIS_utils import AISPRO
from utils.VIS_utils_botsort_simple import VISPRO
from utils.FUS_utils import FUSPRO
from utils.gen_result import gen_result
from utils.draw import DRAW
from utils.file_read import ais_initial, read_all, update_time


def normalize_data_path(path):
    path = path.replace('\\', '/')
    if not path.endswith('/'):
        path += '/'
    return path


def main(arg):
    arg.data_path = normalize_data_path(arg.data_path)
    arg.result_path = normalize_data_path(arg.result_path)
    video_path, ais_path, result_video, result_metric, initial_time, camera_para = read_all(
        arg.data_path, arg.result_path)

    arg.video_path = video_path
    arg.ais_path = ais_path
    arg.result_video = result_video
    arg.result_metric = result_metric
    arg.initial_time = initial_time
    arg.camera_para = camera_para
    arg.mot20 = not arg.fuse_score

    ais_file, timestamp0, time0 = ais_initial(arg.ais_path, arg.initial_time)
    Time = arg.initial_time.copy()

    cap = cv2.VideoCapture(arg.video_path)
    im_shape = [cap.get(3), cap.get(4)]
    max_dis = min(im_shape) // 2
    fps = int(cap.get(5))
    if fps <= 0:
        fps = arg.fps
    arg.fps = fps
    t = int(1000 / fps)

    AIS = AISPRO(arg.ais_path, ais_file, im_shape, t)
    FUS = FUSPRO(max_dis, im_shape, t)
    DRA = DRAW(im_shape, t)

    tracker = BoTSORT(arg, frame_rate=fps)
    vispro_params = inspect.signature(VISPRO).parameters
    if 'tracker' in vispro_params:
        VIS = VISPRO(
            arg.anti, arg.anti_rate, t, camera_para=camera_para,
            im_shape=im_shape, tracker=tracker)
    else:
        VIS = VISPRO(arg.anti, arg.anti_rate, t, camera_para, im_shape)
        VIS.tracker = tracker

    name = 'demo'
    show_size = 500
    videoWriter = None
    bin_inf = pd.DataFrame(columns=['ID', 'mmsi', 'timestamp', 'match'])

    print('Start Time: %s || Stamp: %d || fps: %d' % (time0, timestamp0, fps))
    times = 0
    time_i = 0
    sum_t = []

    while True:
        _, im = cap.read()
        if im is None:
            break
        start = time.time()

        Time, timestamp, Time_name = update_time(Time, t)

        AIS_vis, AIS_cur = AIS.process(camera_para, timestamp, Time_name)

        Vis_tra, Vis_cur = VIS.feedCap(im, timestamp, AIS_vis, bin_inf)

        Fus_tra, bin_inf = FUS.fusion(AIS_vis, AIS_cur, Vis_tra, Vis_cur, timestamp)

        end = time.time() - start
        time_i = time_i + end
        if timestamp % 1000 < t:
            gen_result(times, Vis_cur, Fus_tra, arg.result_metric, im_shape)
            times = times + 1
            sum_t.append(time_i)
            print('Time: %s || Stamp: %d || Process: %.6f || Average: %.6f +- %.6f' %
                  (Time_name, timestamp, time_i, np.mean(sum_t), np.std(sum_t)))
            time_i = 0

        im = DRA.draw_traj(im, AIS_vis, AIS_cur, Vis_tra, Vis_cur, Fus_tra, timestamp)

        result = im
        result = imutils.resize(result, height=show_size)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            videoWriter = cv2.VideoWriter(
                arg.result_video, fourcc, fps, (result.shape[1], result.shape[0]))

        videoWriter.write(result)

        cv2.imshow(name, result)
        cv2.waitKey(1)
        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            break

    cap.release()
    if videoWriter is not None:
        videoWriter.release()
    cv2.destroyAllWindows()


def make_parser():
    parser = argparse.ArgumentParser(description='DeepSORVF with YOLOX + BoT-SORT VISPRO')

    parser.add_argument('--anti', type=int, default=1, help='anti-occlusion True/1|False/0')
    parser.add_argument('--anti_rate', type=int, default=0, help='occlusion rate 0-1')

    parser.add_argument('--data_path', type=str, default='./clip-01/', help='data path')
    parser.add_argument('--result_path', type=str, default='./result/', help='result path')

    parser.add_argument('-n', '--name', default=None, type=str)
    parser.add_argument('--device', default='gpu', choices=['gpu', 'cpu'])
    parser.add_argument('--fps', default=30, type=int)

    parser.add_argument('--track_high_thresh', type=float, default=0.6)
    parser.add_argument('--track_low_thresh', default=0.1, type=float)
    parser.add_argument('--new_track_thresh', default=0.7, type=float)
    parser.add_argument('--track_buffer', type=int, default=30)
    parser.add_argument('--match_thresh', type=float, default=0.8)
    parser.add_argument('--aspect_ratio_thresh', type=float, default=10.0)
    parser.add_argument('--min_box_area', type=float, default=10)
    parser.add_argument('--fuse-score', dest='fuse_score', default=False, action='store_true')

    parser.add_argument('--cmc-method', dest='cmc_method', default='none', type=str)
    parser.add_argument('--with-reid', dest='with_reid', default=False, action='store_true')
    parser.add_argument('--fast-reid-config', dest='fast_reid_config',
                        default='fast_reid/configs/MOT17/sbs_S50.yml')
    parser.add_argument('--fast-reid-weights', dest='fast_reid_weights',
                        default='pretrained/mot17_sbs_S50.pth')
    parser.add_argument('--proximity_thresh', type=float, default=0.5)
    parser.add_argument('--appearance_thresh', type=float, default=0.25)

    parser.add_argument('--ais-max-age', dest='ais_max_age', type=float, default=2.0)
    parser.add_argument('--ais-kappa', dest='ais_kappa', type=float, default=0.7)
    parser.add_argument('--ais-position-var', dest='ais_position_var', type=float, default=25.0)
    parser.add_argument('--ais-scale-var', dest='ais_scale_var', type=float, default=1000000.0)
    parser.add_argument('--ais-bind-distance', dest='ais_bind_distance', type=float, default=80.0)
    parser.add_argument('--ais-cost-weight', dest='ais_cost_weight', type=float, default=0.15)
    parser.add_argument('--ais-heading-weight', dest='ais_heading_weight', type=float, default=0.0)
    parser.add_argument('--ais-occlusion-min-score', dest='ais_occlusion_min_score', type=float, default=0.5)
    parser.add_argument('--ais-occlusion-max-frames', dest='ais_occlusion_max_frames', type=int, default=60)
    parser.add_argument('--ais-cmc-mode', dest='ais_cmc_mode',
                        choices=['none', 'same', 'inverse'], default='none')
    parser.add_argument('--ais-debug-enabled', dest='ais_debug_enabled',
                        action='store_true', default=True)
    parser.add_argument('--ais-debug-path', dest='ais_debug_path',
                        default='result/ais_motion_prior_debug.csv')
    parser.add_argument('--tgor-debug-enabled', dest='tgor_debug_enabled',
                        action='store_true', default=True)
    parser.add_argument('--tgor-node-debug-path', dest='tgor_node_debug_path',
                        default='result/os_tgor_nodes.csv')
    parser.add_argument('--tgor-edge-debug-path', dest='tgor_edge_debug_path',
                        default='result/os_tgor_edges.csv')
    parser.add_argument('--tgor-edge-debug-min-risk', dest='tgor_edge_debug_min_risk',
                        type=float, default=0.05)
    parser.add_argument('--tgor-future-steps', dest='tgor_future_steps', type=int, default=5)
    parser.add_argument('--tgor-neighbor-radius', dest='tgor_neighbor_radius', type=float, default=250.0)
    parser.add_argument('--tgor-ema', dest='tgor_ema', type=float, default=0.70)
    parser.add_argument('--tgor-sigma-d', dest='tgor_sigma_d', type=float, default=120.0)
    parser.add_argument('--tgor-sigma-v', dest='tgor_sigma_v', type=float, default=35.0)
    parser.add_argument('--tgor-sigma-f', dest='tgor_sigma_f', type=float, default=80.0)
    parser.add_argument('--tgor-occlusion-mark-thresh', dest='tgor_occlusion_mark_thresh',
                        type=float, default=0.35)
    parser.add_argument('--tgor-output-occlusion-thresh', dest='tgor_output_occlusion_thresh',
                        type=float, default=0.35)
    parser.add_argument('--tgor-lifecycle-extend', dest='tgor_lifecycle_extend',
                        type=float, default=1.0)
    parser.set_defaults(ablation=False, mot20=False, oar_polygon=None)
    return parser


if __name__ == '__main__':
    argspar = make_parser().parse_args()

    print('\nVesselSORT')
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(argspar)
