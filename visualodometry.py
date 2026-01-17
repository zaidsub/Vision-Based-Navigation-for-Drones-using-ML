# VO/VisualOdometry.py

import numpy as np
import cv2


class VisualOdometry(object):
    """
    A simple frame-by-frame visual odometry.
    """
    def __init__(self, detector, matcher, cam):
        """
        :param detector: feature detector returning dict with 'keypoints','descriptors','scores'
        :param matcher: matching module expecting a dict {'ref':..., 'cur': ...}
        :param cam: PinholeCamera with fx, cy, cx attributes
        """
        self.detector = detector
        self.matcher = matcher

        # camera intrinsics
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)

        self.index = 0
        self.kptdescs = {}

        # current pose
        self.cur_R = None
        self.cur_t = None

    def update(self, image, absolute_scale=1.0):
        """
        Process a new image and update pose.

        Returns:
            cur_R: 3x3 rotation matrix
            cur_t: 3x1 translation vector
        """
        # detect features
        kptdesc = self.detector(image)
        # augment with image size for matcher
        h, w = image.shape[:2]
        kptdesc['image_size'] = np.array([h, w], dtype=np.float32)

        if self.index == 0:
            # first frame initialization
            self.kptdescs['cur'] = kptdesc
            self.cur_R = np.eye(3, dtype=np.float32)
            self.cur_t = np.zeros((3,1), dtype=np.float32)
        else:
            # previous features present
            self.kptdescs['cur'] = kptdesc
            # perform matching
            match_dict = self.matcher({'ref': self.kptdescs['ref'], 'cur': self.kptdescs['cur']})
            pts_cur = match_dict['cur_keypoints']
            pts_ref = match_dict['ref_keypoints']

            # essential matrix
            E, mask = cv2.findEssentialMat(
                pts_cur, pts_ref,
                focal=self.focal, pp=self.pp,
                method=cv2.RANSAC, prob=0.999, threshold=1.0
            )
            _, R, t, mask_pose = cv2.recoverPose(
                E, pts_cur, pts_ref,
                focal=self.focal, pp=self.pp
            )

            # update pose with scale
            if absolute_scale > 0.1:
                self.cur_t += absolute_scale * (self.cur_R @ t)
                self.cur_R = R @ self.cur_R

        # shift buffers
        self.kptdescs['ref'] = self.kptdescs['cur']
        self.index += 1
        return self.cur_R, self.cur_t


class AbosluteScaleComputer(object):
    def __init__(self):
        self.prev_pose = None
        self.cur_pose = None
        self.count = 0

    def update(self, pose):
        self.cur_pose = pose
        scale = 1.0
        if self.count != 0:
            # Euclidean distance between translation components
            tx = self.cur_pose[0,3] - self.prev_pose[0,3]
            ty = self.cur_pose[1,3] - self.prev_pose[1,3]
            tz = self.cur_pose[2,3] - self.prev_pose[2,3]
            scale = np.linalg.norm([tx, ty, tz])
        self.count += 1
        self.prev_pose = self.cur_pose
        return scale
