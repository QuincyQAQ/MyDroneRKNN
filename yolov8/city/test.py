import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN
import pickle

ONNX_MODEL = 'best.onnx'
RKNN_MODEL = 'best.rknn'
IMG_PATH = 'image.jpg'
DATASET = 'dataset.txt'
image_save = 'result.jpg'

QUANTIZE_ON = True  # 量化

OBJ_THRESH = 0.25
NMS_THRESH = 0.6
IMG_SIZE = 640

num_classes = 10  # 数据集类别数量
# 'NonViolence','violence','scale1fire', 'scale2fire', 'scale3fire','Casualty-PPE','Helicopter','Ship','Swimmer-No-PPE','night-person'
CLASSES = ('NonViolence', 'violence', 'scale1fire', 'scale2fire', 'scale3fire', 'Casualty-PPE', 'Helicopter', 'Ship',
           'Swimmer-No-PPE', 'night-person')


def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def process(input, mask, anchors):
    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = input[..., 4]
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = input[..., 5:]

    box_xy = input[..., :2] * 2 - 0.5

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= int(IMG_SIZE / grid_h)

    box_wh = pow(input[..., 2:4] * 2, 2)
    box_wh = box_wh * anchors

    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs


def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    boxes = boxes.reshape(-1, 4)
    box_confidences = box_confidences.reshape(-1)
    box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

    _box_pos = np.where(box_confidences >= OBJ_THRESH)
    boxes = boxes[_box_pos]
    box_confidences = box_confidences[_box_pos]
    box_class_probs = box_class_probs[_box_pos]

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score >= OBJ_THRESH)

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    scores = (class_max_score * box_confidences)[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def post_process(input_data):
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
               [59, 119], [116, 90], [156, 198], [373, 326]]

    boxes, classes, scores = [], [], []
    for input, mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


#def draw(image, boxes, scores, classes):
#    print("{:^12} {:^12}  {}".format('class', 'score', 'xmin, ymin, xmax, ymax'))
#    print('-' * 50)
#    for box, score, cl in zip(boxes, scores, classes):
#        top, left, right, bottom = box
#        top = int(top)
#        left = int(left)
#        right = int(right)
#        bottom = int(bottom)

#        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
#        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
#                    (top, left - 6),
#                    cv2.FONT_HERSHEY_SIMPLEX,
#                    0.6, (0, 0, 255), 2)

#        print("{:^12} {:^12.3f} [{:>4}, {:>4}, {:>4}, {:>4}]".format(CLASSES[cl], score, top, left, right, bottom))


#def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
#    shape = im.shape[:2]  # current shape [height, width]
#    if isinstance(new_shape, int):
#        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
#    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
#    ratio = r, r  # width, height ratios
#    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
#    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

#    dw /= 2  # divide padding into 2 sides
#    dh /= 2

#    if shape[::-1] != new_unpad:  # resize
#        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
#    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
#    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
#    return im, ratio, (dw, dh)



if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3566')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread(IMG_PATH) #3x3,(高度, 宽度, 通道数)
    # img, ratio, (dw, dh) = letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Inference
    print('--> Running model')
    img2 = np.expand_dims(img, 0)#增加维度,(批次大小/1个批次图片数量, 高度, 宽度, 通道数)
    #img2.shape(1, 640, 640, 3),官方(1, 640, 640, 3)
    outputs = rknn.inference(inputs=[img2])#(1, 1, 14, 8400)
    #data_format,指定输入数据的排列顺序或格式的参数，N:批次，H:高度，W:宽度，C：通道数
    print(outputs[0][0][0][0])
    print(outputs[0][0][1][0])
    print(outputs[0][0][2][0])
    print(outputs[0][0][3][0])
    print(outputs[0][0][4][0])

    #boxes, classes, scores = post_process(outputs)

    # 提取边界框信息
    boxes, class_probs = np.split(outputs, [4], axis=-1)
    boxes = boxes.reshape((-1, 4))
    class_probs = class_probs.reshape((-1, num_classes))

    # 过滤出置信度较高的边界框
    confidence_threshold = 0.5
    class_threshold = 0.5
    mask = (class_probs.max(axis=1) > confidence_threshold) & (class_probs.argmax(axis=1) > class_threshold)
    filtered_boxes = boxes[mask]
    filtered_class_probs = class_probs[mask]

    # 绘制边界框
    image_with_boxes = img.copy()  # 创建一个副本以免修改原始图像
    for box, class_prob in zip(filtered_boxes, filtered_class_probs):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
        class_id = np.argmax(class_prob)
        class_name = f"Class {class_id}"
        confidence = class_prob[class_id]
        label = f"{class_name}: {confidence:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #draw(img_1, boxes, scores, classes)
        #cv2.imwrite(image_save, img_1)
        # 保存带有边界框的图像
    output_image_path = "output_image.jpg"  # 指定输出图像的文件路径
    cv2.imwrite(output_image_path, image_with_boxes)
    print('Save results.jpg!')
    rknn.release()
