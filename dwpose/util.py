# https://github.com/IDEA-Research/DWPose
import math
import numpy as np
import matplotlib
import cv2
import itertools


eps = 0.01


def smart_resize(x, s):
    Ht, Wt = s
    if x.ndim == 2:
        Ho, Wo = x.shape
        Co = 1
    else:
        Ho, Wo, Co = x.shape
    if Co == 3 or Co == 1:
        k = float(Ht + Wt) / float(Ho + Wo)
        return cv2.resize(
            x,
            (int(Wt), int(Ht)),
            interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4,
        )
    else:
        return np.stack([smart_resize(x[:, :, i], s) for i in range(Co)], axis=2)


def smart_resize_k(x, fx, fy):
    if x.ndim == 2:
        Ho, Wo = x.shape
        Co = 1
    else:
        Ho, Wo, Co = x.shape
    Ht, Wt = Ho * fy, Wo * fx
    if Co == 3 or Co == 1:
        k = float(Ht + Wt) / float(Ho + Wo)
        return cv2.resize(
            x,
            (int(Wt), int(Ht)),
            interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4,
        )
    else:
        return np.stack([smart_resize_k(x[:, :, i], fx, fy) for i in range(Co)], axis=2)


def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :] * 0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :] * 0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights[
            ".".join(weights_name.split(".")[1:])
        ]
    return transfered_model_weights


def draw_bodypose(canvas, candidate, subset):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    limbSeq = [
        [2, 3],
        [2, 6],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [2, 9],
        [9, 10],
        [10, 11],
        [2, 12],
        [12, 13],
        [13, 14],
        [2, 1],
        [1, 15],
        [15, 17],
        [1, 16],
        [16, 18],
        [3, 17],
        [6, 18],
    ]

    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
    ]

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    return canvas


def check_intersection(A, B, C, D):
    # cal cross product
    def cross_product(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    # test intersect or not
    if cross_product(A, B, C) * cross_product(A, B, D) < 0 and cross_product(C, D, A) * cross_product(C, D, B) < 0:
        return True
    else:
        return False

def get_line_points(x1, y1, x2, y2, step=0.3):
    points = []
    
    # 计算斜率
    dx = x2 - x1
    dy = y2 - y1
    
    # 确定步长
    distance  = max(abs(dx), abs(dy))
    if distance == 0:
        return [(x1, y1)]
    
    step = distance * step

    x_step = dx / distance * step
    y_step = dy / distance * step
    
    # 生成所有点的坐标
    for i in range(int(distance / step) + 1):
        x = x1 + i * x_step
        y = y1 + i * y_step
        points.append((round(x), round(y)))  # 可以选择四舍五入或其他取整方式
    
    return points

def calculate_threshold(depth_values, factor):
    std_dev = np.std(depth_values)
    threshold = std_dev * factor
    return threshold

def check_depth_continuity(depth_values, threshold):
    for i in range(len(depth_values) - 1):
        if abs(depth_values[i] - depth_values[i+1]) > threshold:
            return False
    return True

def concat_mouth_kp(candidate, subset, faces):
    '''
    candidate: coordinates of points, (18,2)
    subset: (1, 18), if conf > thresh, global rank of keypoint in the image, else -1
    faces: (1, 68, 2)
    '''
    # According to DWposeDetector, only one person is reserved
    mouth_points = faces[0, [48, 54]] # (2, 2)
    candidate = np.concatenate([candidate, mouth_points], 0) # (20, 2)
    mouth_subset = []
    # if 15 in subset:
    #     mouth_subset.append(18)
    # else:
    #     mouth_subset.append(-1)
    
    # if 16 in subset:
    #     mouth_subset.append(19)
    # else:
    #     mouth_subset.append(-1)

    mouth_subset.append(18)
    mouth_subset.append(19)

    subset = np.concatenate([subset, np.array(mouth_subset)[None,...]], 1)

    # print('candidate', candidate.shape)
    # print('subset', subset.shape)
    
    return candidate, subset


def draw_bodypose_depth_with_mouth(canvas, candidate, subset, depth_img=None, faces=None):
    '''
    candidate: coordinates of points, (18,2)
    subset: (1, 18), if conf > thresh, global rank of keypoint in the image, else -1
    '''
    # 0.鼻子 # 1.颈部 # 2.右肩 # 3.右肘 # 4.右手腕 # 5.左肩 # 6.左肘 # 7.左手腕 # 8.右髋部 # 9.右膝盖 # 10.右脚踝 # 11.左髋部 # 12.左膝盖 # 13.左脚踝 # 14.右眼 # 15.左眼 # 16.右耳 # 17.左耳
    # 1.鼻子 # 2.颈部 # 3.右肩 # 4.右肘 # 5.右手腕 # 6.左肩 # 7.左肘 # 8.左手腕 # 9.右髋部 # 10.右膝盖 # 11.右脚踝 # 12.左髋部 # 13.左膝盖 # 14.左脚踝 # 15.右眼 # 16.左眼 # 17.右耳 # 18.左耳
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    if faces is not None:
        # concat faces' candidates and subset with body's
        candidate, subset = concat_mouth_kp(candidate, subset, faces)


    stickwidth = 4

    limbSeq = [
        [2, 3],
        [2, 6],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [2, 9],
        [9, 10],
        [10, 11],
        [2, 12],
        [12, 13],
        [13, 14],
        [2, 1],
        [1, 15],
        [15, 17],
        [1, 16],
        [16, 18],
        [3, 17],
        [6, 18],
        [1, 19], # right mouth corner
        [1, 20], # left mouth corner
    ]

    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
        [255, 128, 0],
        [42, 255, 0],
        [255, 0, 42],
    ]

    # import pdb;pdb.set_trace()

    body_idx = [0,1,6,9,12,13,14,15,16] # 14 16
    limbs_idx = [2,3,4,5,7,8,10,11]
    mouth_idx = [19, 20]
    
    def draw_line():
        Y = candidate[index.astype(int), 0] * float(W)
        X = candidate[index.astype(int), 1] * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly(
            (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
        )
        cv2.fillConvexPoly(canvas, polygon, colors[i])

    for i in body_idx:
        if i in [14, 16]:
            continue
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            draw_line()

    for i in limbs_idx:
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            draw_line()

    # draw line for mouth  
    for i in mouth_idx:
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]

            if (Y < 0).any() or (X < 0).any():
                pass
            else:
                draw_line()   
    
    def cross_limb_seq(limb1, limb2, seq_id, depth_img=None):
        depth_img = depth_img.convert("L")
        depth_img = np.array(depth_img)
        h,w = depth_img.shape

        limb1[0] = np.clip(limb1[0], 0, [w-1, h-1])
        limb1[1] = np.clip(limb1[1], 0, [w-1, h-1])
        limb2[0] = np.clip(limb2[0], 0, [w-1, h-1])
        limb2[1] = np.clip(limb2[1], 0, [w-1, h-1])
        intersection_flag = check_intersection(limb1[0], limb1[1], limb2[0], limb2[1])
        if intersection_flag:
            
            limb1_start = limb1[0]  
            limb1_end = limb1[1]  
            
            limb2_start = limb2[0]  
            limb2_end = limb2[1]  
      
            limb1_points = get_line_points(limb1_start[0], limb1_start[1], limb1_end[0],limb1_end[1],step=0.2)
            
            # 检查limb1的像素值是否连续
            limb1_values = []
            for x,y in limb1_points:
            #     limb1_values.append(depth_img[y, x])
            # limb1_gap =  max([abs(int(limb1_values[i])-int(limb1_values[i+1])) for i in range(len(limb1_values)-1)])
                if y<=h and x<=w:
                    limb1_values.append(depth_img[y, x])
            if len(limb1_values)>1:
                limb1_gap =  max([abs(int(limb1_values[i])-int(limb1_values[i+1])) for i in range(len(limb1_values)-1)])
            else:
                limb1_gap = 0
            
            limb2_points = get_line_points(limb2_start[0], limb2_start[1], limb2_end[0],limb2_end[1],step=0.2)
            # 检查limb1的像素值是否连续
            limb2_values = []
            for x,y in limb2_points:
            #     limb2_values.append(depth_img[y, x])
            # limb2_gap =  max([abs(int(limb2_values[i])-int(limb2_values[i+1])) for i in range(len(limb2_values)-1)])
                if y<=h and x<=w:
                    limb2_values.append(depth_img[y, x])
            if len(limb2_values)>1:
                limb2_gap =  max([abs(int(limb2_values[i])-int(limb2_values[i+1])) for i in range(len(limb2_values)-1)])
            else:
                limb2_gap = 0
                

            # # 自动计算阈值
            # left_threshold = calculate_threshold(limb1_values, 2)
            # right_threshold = calculate_threshold(limb2_values, 2)

            # # 检查左臂和右臂的深度值连续性
            # limb1_continuous = check_depth_continuity(limb1_values, left_threshold)
            # limb2_continuous = check_depth_continuity(limb2_values, right_threshold)


            # 判断交叉的手臂在前面还是后面
            if limb1_gap < limb2_gap:
                # draw limb1
                mX = np.mean(limb1[...,1])
                mY = np.mean(limb1[...,0])

                length = ((limb1_end[0] - limb1_start[0]) ** 2 + (limb1_end[1] - limb1_start[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(limb1_end[1] - limb1_start[1], limb1_end[0] - limb1_start[0]))
                polygon = cv2.ellipse2Poly(
                    (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
                )
                cv2.fillConvexPoly(canvas, polygon, colors[seq_id[0]])
            else:
                # draw limb2
                mX = np.mean(limb2[...,1])
                mY = np.mean(limb2[...,0])

                length = ((limb2_end[0] - limb2_start[0]) ** 2 + (limb2_end[1] - limb2_start[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(limb2_end[1] - limb2_start[1], limb2_end[0] - limb2_start[0]))
                polygon = cv2.ellipse2Poly(
                    (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
                )
                cv2.fillConvexPoly(canvas, polygon, colors[seq_id[1]])

        return intersection_flag


    num_intersec = 0
    for seq_id in list(itertools.combinations(limbs_idx, 2)):
        intersection_flag = cross_limb_seq((candidate[np.array(limbSeq[seq_id[0]]) - 1]*[W,H]).astype(int), (candidate[np.array(limbSeq[seq_id[1]]) - 1]*[W,H]).astype(int),seq_id, depth_img)
        if intersection_flag:
            num_intersec += 1
            

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(20): # Add two points
        if i in [17, 16]: # ignore ears
            continue
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
            
    return canvas, (num_intersec > 0)


def draw_bodypose_depth(canvas, candidate, subset, depth_img=None):
    # 0.鼻子 # 1.颈部 # 2.右肩 # 3.右肘 # 4.右手腕 # 5.左肩 # 6.左肘 # 7.左手腕 # 8.右髋部 # 9.右膝盖 # 10.右脚踝 # 11.左髋部 # 12.左膝盖 # 13.左脚踝 # 14.右眼 # 15.左眼 # 16.右耳 # 17.左耳
    # 1.鼻子 # 2.颈部 # 3.右肩 # 4.右肘 # 5.右手腕 # 6.左肩 # 7.左肘 # 8.左手腕 # 9.右髋部 # 10.右膝盖 # 11.右脚踝 # 12.左髋部 # 13.左膝盖 # 14.左脚踝 # 15.右眼 # 16.左眼 # 17.右耳 # 18.左耳
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    limbSeq = [
        [2, 3],
        [2, 6],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [2, 9],
        [9, 10],
        [10, 11],
        [2, 12],
        [12, 13],
        [13, 14],
        [2, 1],
        [1, 15],
        [15, 17],
        [1, 16],
        [16, 18],
        [3, 17],
        [6, 18],
    ]

    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
    ]

    body_idx = [0,1,6,9,12,13,14,15,16]
    limbs_idx = [2,3,4,5,7,8,10,11]
    
    def draw_line():
        Y = candidate[index.astype(int), 0] * float(W)
        X = candidate[index.astype(int), 1] * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly(
            (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
        )
        cv2.fillConvexPoly(canvas, polygon, colors[i])

    for i in body_idx:
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            draw_line()

    for i in limbs_idx:
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            draw_line()
         
    
    def cross_limb_seq(limb1, limb2, seq_id, depth_img=None):


        depth_img = depth_img.convert("L")
        depth_img = np.array(depth_img)

        if check_intersection(limb1[0], limb1[1], limb2[0], limb2[1]):
            
            
            limb1_start = limb1[0]  
            limb1_end = limb1[1]  
            
            limb2_start = limb2[0]  
            limb2_end = limb2[1]  
      
            limb1_points = get_line_points(limb1_start[0], limb1_start[1], limb1_end[0],limb1_end[1],step=0.2)
            
            # 检查limb1的像素值是否连续
            limb1_values = []
            for x,y in limb1_points:
                limb1_values.append(depth_img[y, x])
            limb1_gap =  max([abs(int(limb1_values[i])-int(limb1_values[i+1])) for i in range(len(limb1_values)-1)])
            
            limb2_points = get_line_points(limb2_start[0], limb2_start[1], limb2_end[0],limb2_end[1],step=0.2)
            # 检查limb1的像素值是否连续
            limb2_values = []
            for x,y in limb2_points:
                limb2_values.append(depth_img[y, x])
            limb2_gap =  max([abs(int(limb2_values[i])-int(limb2_values[i+1])) for i in range(len(limb2_values)-1)])
           

            # # 自动计算阈值
            # left_threshold = calculate_threshold(limb1_values, 2)
            # right_threshold = calculate_threshold(limb2_values, 2)

            # # 检查左臂和右臂的深度值连续性
            # limb1_continuous = check_depth_continuity(limb1_values, left_threshold)
            # limb2_continuous = check_depth_continuity(limb2_values, right_threshold)


            # 判断交叉的手臂在前面还是后面
            if limb1_gap < limb2_gap:
                # draw limb1
                mX = np.mean(limb1[...,1])
                mY = np.mean(limb1[...,0])

                length = ((limb1_end[0] - limb1_start[0]) ** 2 + (limb1_end[1] - limb1_start[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(limb1_end[1] - limb1_start[1], limb1_end[0] - limb1_start[0]))
                polygon = cv2.ellipse2Poly(
                    (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
                )
                cv2.fillConvexPoly(canvas, polygon, colors[seq_id[0]])
            else:
                # draw limb2
                mX = np.mean(limb2[...,1])
                mY = np.mean(limb2[...,0])

                length = ((limb2_end[0] - limb2_start[0]) ** 2 + (limb2_end[1] - limb2_start[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(limb2_end[1] - limb2_start[1], limb2_end[0] - limb2_start[0]))
                polygon = cv2.ellipse2Poly(
                    (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
                )
                cv2.fillConvexPoly(canvas, polygon, colors[seq_id[1]])


            
    for seq_id in list(itertools.combinations(limbs_idx, 2)):
        cross_limb_seq((candidate[np.array(limbSeq[seq_id[0]]) - 1]*[W,H]).astype(int), (candidate[np.array(limbSeq[seq_id[1]]) - 1]*[W,H]).astype(int),seq_id, depth_img)
            

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
            
    return canvas

def draw_handpose(canvas, all_hand_peaks):
    H, W, C = canvas.shape

    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ]

    for peaks in all_hand_peaks:
        peaks = np.array(peaks)

        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(
                    canvas,
                    (x1, y1),
                    (x2, y2),
                    matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0])
                    * 255,
                    thickness=2,
                )

        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    return canvas


def draw_facepose(canvas, all_lmks):
    H, W, C = canvas.shape
    for lmks in all_lmks:
        lmks = np.array(lmks)
        for lmk in lmks:
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
    return canvas


# detect hand according to body pose keypoints
# please refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/hand/handDetector.cpp
def handDetect(candidate, subset, oriImg):
    # right hand: wrist 4, elbow 3, shoulder 2
    # left hand: wrist 7, elbow 6, shoulder 5
    ratioWristElbow = 0.33
    detect_result = []
    image_height, image_width = oriImg.shape[0:2]
    for person in subset.astype(int):
        # if any of three not detected
        has_left = np.sum(person[[5, 6, 7]] == -1) == 0
        has_right = np.sum(person[[2, 3, 4]] == -1) == 0
        if not (has_left or has_right):
            continue
        hands = []
        # left hand
        if has_left:
            left_shoulder_index, left_elbow_index, left_wrist_index = person[[5, 6, 7]]
            x1, y1 = candidate[left_shoulder_index][:2]
            x2, y2 = candidate[left_elbow_index][:2]
            x3, y3 = candidate[left_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, True])
        # right hand
        if has_right:
            right_shoulder_index, right_elbow_index, right_wrist_index = person[
                [2, 3, 4]
            ]
            x1, y1 = candidate[right_shoulder_index][:2]
            x2, y2 = candidate[right_elbow_index][:2]
            x3, y3 = candidate[right_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, False])

        for x1, y1, x2, y2, x3, y3, is_left in hands:
            # pos_hand = pos_wrist + ratio * (pos_wrist - pos_elbox) = (1 + ratio) * pos_wrist - ratio * pos_elbox
            # handRectangle.x = posePtr[wrist*3] + ratioWristElbow * (posePtr[wrist*3] - posePtr[elbow*3]);
            # handRectangle.y = posePtr[wrist*3+1] + ratioWristElbow * (posePtr[wrist*3+1] - posePtr[elbow*3+1]);
            # const auto distanceWristElbow = getDistance(poseKeypoints, person, wrist, elbow);
            # const auto distanceElbowShoulder = getDistance(poseKeypoints, person, elbow, shoulder);
            # handRectangle.width = 1.5f * fastMax(distanceWristElbow, 0.9f * distanceElbowShoulder);
            x = x3 + ratioWristElbow * (x3 - x2)
            y = y3 + ratioWristElbow * (y3 - y2)
            distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
            distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            width = 1.5 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)
            # x-y refers to the center --> offset to topLeft point
            # handRectangle.x -= handRectangle.width / 2.f;
            # handRectangle.y -= handRectangle.height / 2.f;
            x -= width / 2
            y -= width / 2  # width = height
            # overflow the image
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            width1 = width
            width2 = width
            if x + width > image_width:
                width1 = image_width - x
            if y + width > image_height:
                width2 = image_height - y
            width = min(width1, width2)
            # the max hand box value is 20 pixels
            if width >= 20:
                detect_result.append([int(x), int(y), int(width), is_left])

    """
    return value: [[x, y, w, True if left hand else False]].
    width=height since the network require squared input.
    x, y is the coordinate of top left 
    """
    return detect_result


# Written by Lvmin
def faceDetect(candidate, subset, oriImg):
    # left right eye ear 14 15 16 17
    detect_result = []
    image_height, image_width = oriImg.shape[0:2]
    for person in subset.astype(int):
        has_head = person[0] > -1
        if not has_head:
            continue

        has_left_eye = person[14] > -1
        has_right_eye = person[15] > -1
        has_left_ear = person[16] > -1
        has_right_ear = person[17] > -1

        if not (has_left_eye or has_right_eye or has_left_ear or has_right_ear):
            continue

        head, left_eye, right_eye, left_ear, right_ear = person[[0, 14, 15, 16, 17]]

        width = 0.0
        x0, y0 = candidate[head][:2]

        if has_left_eye:
            x1, y1 = candidate[left_eye][:2]
            d = max(abs(x0 - x1), abs(y0 - y1))
            width = max(width, d * 3.0)

        if has_right_eye:
            x1, y1 = candidate[right_eye][:2]
            d = max(abs(x0 - x1), abs(y0 - y1))
            width = max(width, d * 3.0)

        if has_left_ear:
            x1, y1 = candidate[left_ear][:2]
            d = max(abs(x0 - x1), abs(y0 - y1))
            width = max(width, d * 1.5)

        if has_right_ear:
            x1, y1 = candidate[right_ear][:2]
            d = max(abs(x0 - x1), abs(y0 - y1))
            width = max(width, d * 1.5)

        x, y = x0, y0

        x -= width
        y -= width

        if x < 0:
            x = 0

        if y < 0:
            y = 0

        width1 = width * 2
        width2 = width * 2

        if x + width > image_width:
            width1 = image_width - x

        if y + width > image_height:
            width2 = image_height - y

        width = min(width1, width2)

        if width >= 20:
            detect_result.append([int(x), int(y), int(width)])

    return detect_result


# get max index of 2d array
def npmax(array):
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j
