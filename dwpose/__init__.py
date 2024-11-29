# https://github.com/IDEA-Research/DWPose
# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import copy
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import torch
from controlnet_aux.util import HWC3, resize_image
from PIL import Image

from . import util
from .wholebody import Wholebody


def draw_pose(pose, H, W, depth_img=None):
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    intersec_flag = None
    if depth_img:
        # canvas = util.draw_bodypose_depth(canvas, candidate, subset, depth_img)
        canvas, intersec_flag = util.draw_bodypose_depth_with_mouth(canvas, candidate, subset, depth_img, faces)
        canvas = util.draw_handpose(canvas, hands)
    else:
        canvas = util.draw_bodypose(canvas, candidate, subset)
        canvas = util.draw_handpose(canvas, hands)
        # canvas = util.draw_facepose(canvas, faces)

    return canvas, intersec_flag


class DWposeDetector:
    def __init__(self):
        pass

    def to(self, device):
        self.pose_estimation = Wholebody(device)
        return self

    def cal_height(self, input_image):
        input_image = cv2.cvtColor(
            np.array(input_image, dtype=np.uint8), cv2.COLOR_RGB2BGR
        )

        input_image = HWC3(input_image)
        H, W, C = input_image.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(input_image)
            nums, keys, locs = candidate.shape
            # candidate[..., 0] /= float(W)
            # candidate[..., 1] /= float(H)
            body = candidate
        return body[0, ..., 1].min(), body[..., 1].max() - body[..., 1].min()

    def __call__(
        self,
        input_image,
        detect_resolution=512,
        image_resolution=512,
        output_type="pil",
        depth_img=None,
        pose_draw_flag=True,
        **kwargs,
    ):
        input_image = cv2.cvtColor(
            np.array(input_image, dtype=np.uint8), cv2.COLOR_RGB2BGR
        )

        input_image = HWC3(input_image)
        # input_image = resize_image(input_image, detect_resolution)
        H, W, C = input_image.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(input_image)
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            score = subset[:, :18]
            max_ind = np.mean(score, axis=-1).argmax(axis=0) # to process multi persons, so only one person reserved
            score = score[[max_ind]]
            body = candidate[:, :18].copy()
            body = body[[max_ind]]
            nums = 1
            body = body.reshape(nums * 18, locs)
            body_score = copy.deepcopy(score)

            # mouth_score = subset[:, [24+48, 24+54]]

            thresh = 0.3
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > thresh:
                        score[i][j] = int(18 * i + j)
                    else:
                        score[i][j] = -1
            
            un_visible = subset < thresh
            candidate[un_visible] = -1

            foot = candidate[:, 18:24]

            faces = candidate[[max_ind], 24:92]

            hands = candidate[[max_ind], 92:113]
            hands = np.vstack([hands, candidate[[max_ind], 113:]])

            # bodies = dict(candidate=body, subset=score, mouth_score=mouth_score)
            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)
            
            if pose_draw_flag:
                detected_map, intersec_flag = draw_pose(pose, H, W, depth_img=depth_img)
                detected_map = HWC3(detected_map)
                if output_type == "pil":
                    detected_map = Image.fromarray(detected_map)
            else:
                detected_map = None
                intersec_flag = None

            return detected_map, body_score, pose, intersec_flag


    def get_multi_person(
        self,
        input_image,
        detect_resolution=512,
        image_resolution=512,
        output_type="pil",
        depth_img=None,
        **kwargs,
    ):
        input_image = cv2.cvtColor(
            np.array(input_image, dtype=np.uint8), cv2.COLOR_RGB2BGR
        )

        input_image = HWC3(input_image)
        H, W, C = input_image.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(input_image)
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            scores = subset[:, :18]
            colors = [128, 255]
            person_identity_mask = np.zeros(shape=(H, W), dtype=np.uint8)
            
            assert len(colors) == scores.shape[0]
            detected_maps = []
            for ind in range(scores.shape[0]):
                score = scores[[ind]]
                body = candidate[:, :18].copy()
                body = body[[ind]]
                nums = 1
                body = body.reshape(nums * 18, locs)
                body_score = copy.deepcopy(score)

                thresh = 0.3
                for i in range(len(score)):
                    for j in range(len(score[i])):
                        if score[i][j] > thresh:
                            score[i][j] = int(18 * i + j)
                        else:
                            score[i][j] = -1
                
                un_visible = subset < thresh
                candidate[un_visible] = -1

                foot = candidate[:, 18:24]

                faces = candidate[[ind], 24:92]
                hands = candidate[[ind], 92:113]
                hands = np.vstack([hands, candidate[[ind], 113:]])

                # bodies = dict(candidate=body, subset=score, mouth_score=mouth_score)
                bodies = dict(candidate=body, subset=score)
                pose = dict(bodies=bodies, hands=hands, faces=faces)
                detected_map, intersec_flag = draw_pose(pose, H, W, depth_img=depth_img) # numpy (H,W,3)
                detected_map = HWC3(detected_map) # nochange, numpy (H,W,3)
                
                detected_maps.append(detected_map)

                # generate_mask
                tmp = ((detected_map/255.0).sum(-1)>=0.1)
                person_identity_mask[tmp] += colors[ind]

                                
            if output_type == "pil":
                detected_map = np.maximum(detected_maps[0], detected_maps[1])
                detected_map = Image.fromarray(detected_map)
                person_identity_mask = np.minimum(person_identity_mask, np.ones_like(person_identity_mask)*255)
                person_identity_mask = Image.fromarray(person_identity_mask)

            return detected_map, None, pose, intersec_flag, person_identity_mask
