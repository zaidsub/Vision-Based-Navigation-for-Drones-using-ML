import cv2
import numpy as np
import torch
import collections
import matplotlib.cm as cm


def image2tensor(frame, device):
    return torch.from_numpy(frame / 255.).float()[None, None].to(device)


# --- VISUALIZATION ---
# based on: https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/utils.py
def plot_keypoints(image, kpts, scores=None):
    kpts = np.round(kpts).astype(int)

    if scores is not None:
        # get color
        smin, smax = scores.min(), scores.max()
        assert (0 <= smin <= 1 and 0 <= smax <= 1)

        color = cm.gist_rainbow(scores * 0.4)
        color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
        # text = f"min score: {smin}, max score: {smax}"

        for (x, y), c in zip(kpts, color):
            c = (int(c[0]), int(c[1]), int(c[2]))
            cv2.drawMarker(image, (x, y), tuple(c), cv2.MARKER_CROSS, 6)

    else:
        for x, y in kpts:
            cv2.drawMarker(image, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 6)

    return image


# based on: https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/utils.py
def plot_matches(image0, image1, kpts0, kpts1, scores=None, layout="lr"):
    """
    plot matches between two images. If score is nor None, then red: bad match, green: good match
    :param image0: reference image
    :param image1: current image
    :param kpts0: keypoints in reference image
    :param kpts1: keypoints in current image
    :param scores: matching score for each keypoint pair, range [0~1], 0: worst match, 1: best match
    :param layout: 'lr': left right; 'ud': up down
    :return:
    """
    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]

    if layout == "lr":
        H, W = max(H0, H1), W0 + W1
        out = 255 * np.ones((H, W, 3), np.uint8)
        out[:H0, :W0, :] = image0
        out[:H1, W0:, :] = image1
    elif layout == "ud":
        H, W = H0 + H1, max(W0, W1)
        out = 255 * np.ones((H, W, 3), np.uint8)
        out[:H0, :W0, :] = image0
        out[H0:, :W1, :] = image1
    else:
        raise ValueError("The layout must be 'lr' or 'ud'!")

    kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)

    # get color
    if scores is not None:
        smin, smax = scores.min(), scores.max()
        assert (0 <= smin <= 1 and 0 <= smax <= 1)

        color = cm.gist_rainbow(scores * 0.4)
        color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
    else:
        color = np.zeros((kpts0.shape[0], 3), dtype=int)
        color[:, 1] = 255

    for (x0, y0), (x1, y1), c in zip(kpts0, kpts1, color):
        c = c.tolist()
        if layout == "lr":
            cv2.line(out, (x0, y0), (x1 + W0, y1), color=c, thickness=1, lineType=cv2.LINE_AA)
            # display line end-points as circles
            cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x1 + W0, y1), 2, c, -1, lineType=cv2.LINE_AA)
        elif layout == "ud":
            cv2.line(out, (x0, y0), (x1, y1 + H0), color=c, thickness=1, lineType=cv2.LINE_AA)
            # display line end-points as circles
            cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x1, y1 + H0), 2, c, -1, lineType=cv2.LINE_AA)

    return out
import cv2
import numpy as np

def draw_matches(img1, img2, pts1, pts2, scores=None, max_matches=100, color=(0,255,0)):
    """
    Draws line matches between two grayscale or color images.

    Args:
        img1, img2: H×W or H×W×3 arrays
        pts1, pts2: (N×2) float arrays of [x,y] coords
        scores: Optional float array of match confidences [0,1]
        max_matches: How many top matches to draw
        color: BGR tuple

    Returns:
        Combined image with img1 on left, img2 on right, and lines drawn.
    """
    def to_color(im):
        if im.ndim == 2:
            return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        return im.copy()

    im1c = to_color(img1)
    im2c = to_color(img2)
    h1, w1 = im1c.shape[:2]
    h2, w2 = im2c.shape[:2]
    H = max(h1, h2)
    out = np.zeros((H, w1 + w2, 3), dtype=np.uint8)
    out[:h1, :w1] = im1c
    out[:h2, w1:] = im2c

    # choose match indices
    idxs = np.arange(len(pts1))
    if scores is not None:
        idxs = idxs[np.argsort(-scores)]
    for i in idxs[:max_matches]:
        x1, y1 = int(pts1[i][0]), int(pts1[i][1])
        x2, y2 = int(pts2[i][0]) + w1, int(pts2[i][1])
        cv2.line(out, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    return out

class PinholeCamera(object):
    def __init__(self, width, height, fx, fy, cx, cy,
                 k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = [k1, k2, p1, p2, k3]
