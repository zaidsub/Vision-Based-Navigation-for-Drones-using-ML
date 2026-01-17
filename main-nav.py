#!/usr/bin/env python3
# main_navigation.py

import os
import cv2
import argparse
import yaml
import logging
import numpy as np
from tqdm import tqdm

# SLAM imports
from SLAM.utils.tools import plot_keypoints, draw_matches
from SLAM.DataLoader.KITTILoader import KITTILoader
from SLAM.DataLoader.SequenceImageLoader import SequenceImageLoader
from SLAM.DataLoader.TUMRGBLoader import TUMRGBLoader
from SLAM.Detectors.SuperPointDetector import SuperPointDetector
#from SLAM.Detectors.HandcraftDetector import HandcraftDetector
from SLAM.Matchers.SuperGlueMatcher import SuperGlueMatcher
from SLAM.VO.VisualOdometry import VisualOdometry, AbosluteScaleComputer

# Segmentation imports
from CNN.segmenter import load_segmentation_model, segment_frame


def keypoints_plot(img, vo):
    return plot_keypoints(img.copy(), vo.kptdescs['cur']['keypoints'], vo.kptdescs['cur'].get('scores'))

class TrajPlotter:
    def __init__(self, size=600, origin=(300,300)):
        self.errors = []
        self.origin = origin
        self.traj = np.zeros((size, size, 3), dtype=np.uint8)

    def update(self, est_t, gt_t):
        x, z = est_t[0,0], est_t[2,0]
        gx, gz = gt_t[0], gt_t[2]
        est_pt = (int(self.origin[0] + x), int(self.origin[1] + z))
        gt_pt  = (int(self.origin[0] + gx), int(self.origin[1] + gz))

        err = np.linalg.norm([x-gx, z-gz])
        self.errors.append(err)
        avg_err = np.mean(self.errors)

        cv2.circle(self.traj, est_pt, 2, (0,255,0), -1)
        cv2.circle(self.traj, gt_pt,  2, (0,0,255), -1)
        cv2.rectangle(self.traj, (10,10),(200,50),(0,0,0),-1)
        cv2.putText(self.traj, f"[AvgError] {avg_err:.4f}m", (15,35), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
        return self.traj


def create_loader(cfg):
    name = cfg['name'].lower()
    if name == 'kittiloader': return KITTILoader(cfg)
    if name == 'sequenceimageloader': return SequenceImageLoader(cfg)
    if name == 'tumrgbdloader': return TUMRGBLoader(cfg)
    raise ValueError(f"Unknown loader: {name}")


def create_detector(cfg):
    if cfg['name'].lower() == 'superpointdetector': return SuperPointDetector(cfg)
    #return HandcraftDetector(cfg)


def create_matcher(cfg):
    if cfg['name'].lower() == 'supergluematcher': return SuperGlueMatcher(cfg)
    raise ValueError(f"Unknown matcher: {cfg['name']}")


def main():
    parser = argparse.ArgumentParser(description="Integrated SLAM + Segmentation Navigation Loop")
    parser.add_argument('--config', type=str, required=True, help="YAML config for SLAM pipeline")
    parser.add_argument('--seg_model', type=str, required=True, help="Path to .h5 segmentation model")
    parser.add_argument('--logging', type=str, default='INFO')
    args = parser.parse_args()

    logging.basicConfig(level=logging._nameToLevel[args.logging])
    # load SLAM config
    cfg_path = args.config
    if not os.path.isfile(cfg_path):
        alt = os.path.join(os.path.dirname(__file__), 'params', args.config)
        if os.path.isfile(alt): cfg_path = alt
        else: raise FileNotFoundError(f"Config '{args.config}' not found.")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # GPU flags
    import torch
    use_cuda = torch.cuda.is_available()
    logging.info(f"Using CUDA: {use_cuda}")
    cfg['detector']['cuda'] = 1 if use_cuda else 0
    cfg['matcher']['cuda']  = 1 if use_cuda else 0

    # build SLAM
    loader   = create_loader(cfg['dataset'])
    detector = create_detector(cfg['detector'])
    matcher  = create_matcher(cfg['matcher'])
    absscale = AbosluteScaleComputer()
    trajplt  = TrajPlotter()
    vo       = VisualOdometry(detector, matcher, loader.cam)

        # load segmentation model
    import tensorflow as tf
    # enable dynamic memory growth if GPU present
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"TensorFlow: enabled memory growth on {len(gpus)} GPU(s)")
        except Exception as e:
            logging.warning(f"Could not set memory growth: {e}")
    seg_model = load_segmentation_model(args.seg_model)

    seg_model = load_segmentation_model(args.seg_model)

    # create display windows
    cv2.namedWindow('keypoints',    cv2.WINDOW_NORMAL)
    cv2.namedWindow('matches',      cv2.WINDOW_NORMAL)
    cv2.namedWindow('segmentation', cv2.WINDOW_NORMAL)
    cv2.namedWindow('trajectory',   cv2.WINDOW_NORMAL)

    prev_gray = None

        # === Main loop: grab frame from AirSim, segment, SLAM, plan, control ===
    import airsim
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    while True:
        # 1) Acquire frame from AirSim
        response = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        frame = img1d.reshape(response.height, response.width, 3)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 2) Semantic Segmentation
        seg_mask = segment_frame(seg_model, img)
        mask_vis = cv2.applyColorMap((seg_mask*20).astype('uint8'), cv2.COLORMAP_JET)
        overlay  = cv2.addWeighted(img, 0.6, mask_vis, 0.4, 0)
        cv2.imshow('segmentation', overlay)

        # 3) SLAM Update
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        R, t = vo.update(img, absscale.update(None))  # no GT in sim

        # Keypoints + matches
        kp_vis = keypoints_plot(img, vo)
        cv2.imshow('keypoints', kp_vis)
        if prev_gray is not None:
            md = matcher({'ref': vo.kptdescs['ref'], 'cur': vo.kptdescs['cur']})
            mv = draw_matches(prev_gray, gray,
                               vo.kptdescs['ref']['keypoints'],
                               vo.kptdescs['cur']['keypoints'],
                               md['match_score'])
            cv2.imshow('matches', mv)
        prev_gray = gray

        # Trajectory live
        traj_vis = trajplt.update(t, t)  # use estimated as pseudo-GT
        cv2.imshow('trajectory', traj_vis)

                # 4) Cost-map + path planning
        # compute cost map from segmentation and current pose
        from planner.costmap import update_cost_map
        from planner.astar import astar_plan
        # project semantic mask into 2D costmap grid
        cost_map = update_cost_map(seg_mask, t)
        # determine start and goal cells (example: start at robot's grid position, goal from config)
        # here, we map t[0],t[2] to grid indices
        start_cell = (int(off[0] + t[0,0]*scale), int(off[1] + t[2,0]*scale))
        # goal hardcoded or loaded from config
        goal_cell = (off[0] + 100, off[1] + 0)  # example goal 20m ahead
        path = astar_plan(cost_map, start_cell, goal_cell)
        if path and len(path) > 1:
            # send velocity command toward next waypoint
            next_cell = path[1]
            # compute direction in meters
            dx = (next_cell[0] - start_cell[0]) / scale
            dy = (next_cell[1] - start_cell[1]) / scale
            # move in body frame: vx forward=dy, vy right=dx
            client.moveByVelocityAsync(dy, dx, 0, 1)

        # Exit
        if cv2.waitKey(1) == 27:
            break

    client.armDisarm(False)
    client.enableApiControl(False)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
