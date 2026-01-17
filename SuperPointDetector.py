# Detectors/SuperPointDetector.py

import cv2
import numpy as np
import torch
import logging
from pathlib import Path
from utils.tools import image2tensor, plot_keypoints
from Detectors.superpoint.superpoint import SuperPoint

class SuperPointDetector:
    """
    Wrapper for the PyTorch SuperPoint model.
    Accepts both color and grayscale input images.
    Provides a unified detect() API returning cv2.KeyPoint list and descriptors.
    """
    default_config = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": -1,
        "remove_borders": 4,
        "path": Path(__file__).parent / "superpoint" / "superpoint_v1.pth",
        "cuda": True
    }

    def __init__(self, config=None):
        cfg = self.default_config.copy()
        if config:
            cfg.update(config)
        self.config = cfg
        logging.info("SuperPoint detector config: %s", self.config)

        self.device = torch.device('cuda' if torch.cuda.is_available() and self.config['cuda'] else 'cpu')
        logging.info(f"Using device: {self.device}")

        # Load model
        self.model = SuperPoint(self.config).to(self.device)
        self.model.eval()
        logging.info("Loaded SuperPoint model")

    def __call__(self, image: np.ndarray) -> dict:
        """
        Run SuperPoint inference on a single image.

        Args:
            image: np.ndarray of shape (H,W,3) or (H,W) grayscale

        Returns:
            dict with keys:
              'keypoints': (N,2) float array [x,y]
              'scores':     (N,) float array
              'descriptors': (N,D) float array
        """
        # Normalize input to grayscale
        if image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 2:
            gray = image
        else:
            # Unexpected channel count: take first three channels
            gray = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY)

        # Convert to tensor ([1,1,H,W]) and normalize [0,1]
        tensor = image2tensor(gray, self.device)
        with torch.no_grad():
            pred = self.model({'image': tensor})

        # Extract outputs
        kpts = pred['keypoints'][0].cpu().numpy()      # (N,2)
        scores = pred['scores'][0].cpu().numpy()       # (N,)
        descs = pred['descriptors'][0].cpu().numpy()   # (D, N)
        descriptors = descs.T                          # (N, D)

        return {
            'keypoints': kpts,
            'scores': scores,
            'descriptors': descriptors
        }

    def detect(self, image: np.ndarray):
        """
        Detect keypoints and descriptors and return in OpenCV format.

        Args:
            image: np.ndarray of shape (H,W,3) or (H,W)

        Returns:
            keypoints: List[cv2.KeyPoint]
            descriptors: np.ndarray of shape (N, D)
        """
        out = self(image)
        pts = out['keypoints']
        descs = out['descriptors']
        keypoints = [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in pts]
        return keypoints, descs
