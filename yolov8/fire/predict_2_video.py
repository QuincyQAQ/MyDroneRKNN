import onnxruntime as rt
import numpy as np
import cv2
from rknn.api import RKNN

#from utils.visualize import plot_tracking
#from tracker.byte_tracker import BYTETracker
#from tracking_utils.timer import Timer

CLASSES =['fire']
std_h, std_w = 640, 640  # 标准输入尺寸
ONNX_MODEL = 'fire1.onnx'
RKNN_MODEL = 'fire1.rknn'
DATASET = 'dataset.txt'

QUANTIZE_ON = False

OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = 640

def nms(pred, conf_thres, iou_thres):
    conf = pred[..., 4] > conf_thres
    box = pred[conf == True]
    cls_conf = box[..., 5:]
    cls = []
    for i in range(len(cls_conf)):
        cls.append(int(np.argmax(cls_conf[i])))
    total_cls = list(set(cls))
    output_box = []
    for i in range(len(total_cls)):
        clss = total_cls[i]
        cls_box = []
        for j in range(len(cls)):
            if cls[j] == clss:
                box[j][5] = clss
                cls_box.append(box[j][:6])
        cls_box = np.array(cls_box)
        box_conf = cls_box[..., 4]
        box_conf_sort = np.argsort(box_conf)
        max_conf_box = cls_box[box_conf_sort[len(box_conf) - 1]]
        output_box.append(max_conf_box)
        cls_box = np.delete(cls_box, 0, 0)
        while len(cls_box) > 0:
            max_conf_box = output_box[len(output_box) - 1]
            del_index = []
            for j in range(len(cls_box)):
                current_box = cls_box[j]
                interArea = getInter(max_conf_box, current_box)
                iou = getIou(max_conf_box, current_box, interArea)
                if iou > iou_thres:
                    del_index.append(j)
            cls_box = np.delete(cls_box, del_index, 0)
            if len(cls_box) > 0:
                output_box.append(cls_box[0])
                cls_box = np.delete(cls_box, 0, 0)
    return output_box


def getIou(box1, box2, inter_area):
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - inter_area
    iou = inter_area / union
    return iou


def getInter(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2, \
                                         box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[0] - box2[2] / 2, box2[1] - box1[3] / 2, \
                                         box2[0] + box2[2] / 2, box2[1] + box2[3] / 2
    if box1_x1 > box2_x2 or box1_x2 < box2_x1:
        return 0
    if box1_y1 > box2_y2 or box1_y2 < box2_y1:
        return 0
    x_list = [box1_x1, box1_x2, box2_x1, box2_x2]
    x_list = np.sort(x_list)
    x_inter = x_list[2] - x_list[1]
    y_list = [box1_y1, box1_y2, box2_y1, box2_y2]
    y_list = np.sort(y_list)
    y_inter = y_list[2] - y_list[1]
    inter = x_inter * y_inter
    return inter


def draw(img, xscale, yscale, pred):
    # img_ = img.copy()
    if len(pred):
        for detect in pred:
            box = [int((detect[0] - detect[2] / 2) * xscale), int((detect[1] - detect[3] / 2) * yscale),
                   int((detect[0] + detect[2] / 2) * xscale), int((detect[1] + detect[3] / 2) * yscale)]
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
            font = cv2. FONT_HERSHEY_SIMPLEX

            cv2.putText(img, '{}, {:.2f}'.format( CLASSES[int(detect[5])], detect[4]), (box[0], box[1]),
                        font, 0.7, (0,255,127), 2)
    return img

if __name__ == '__main__':
    height, width = 640, 640


    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588')
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


    cap = cv2.VideoCapture('fire3.mp4')
    while True:
        # t0 = time.time()
        # if frame_id % 20 == 0:
        #     logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:

            # img0 = cv2.imread('test.jpg')
            img0 = frame
            x_scale = img0.shape[1] / width
            y_scale = img0.shape[0] / height
            #img = img0 / 255.
            img = cv2.resize(img0, (width, height))
            #img = np.transpose(img, (2, 0, 1))
            data = np.expand_dims(img, axis=0)
            data = data.astype(np.float32)
            print(data.shape)
            #sess = rt.InferenceSession('yolov8n.onnx')
            #input_name = sess.get_inputs()[0].name
            #label_name = sess.get_outputs()[0].name

            #print(input_name, label_name)
            # print(sess.run([label_name], {input_name: data.astype(np.float32)}))
            #pred = sess.run([label_name], {input_name: data.astype(np.float32)})[0]
            # print(pred.shape)  # (1, 84, 8400)
            pred=rknn.inference([data])
            pred = np.squeeze(pred)
            pred = np.transpose(pred, (1, 0))
            pred_class = pred[..., 4:]
            # print(pred_class)
            pred_conf = np.max(pred_class, axis=-1)
            pred = np.insert(pred, 4, pred_conf, axis=-1)
            result = nms(pred, 0.01, 0.3)

            # print(result[0].shape)

            # bboxes = []
            # scores = []
            #
            # # print(result)
            #
            # for detect in result:
            #     box = [int((detect[0] - detect[2] / 2) * x_scale), int((detect[1] - detect[3] / 2) * y_scale),
            #            int((detect[0] + detect[2] / 2) * x_scale), int((detect[1] + detect[3] / 2) * y_scale)]
            #
            #     bboxes.append(box)
            #     score = detect[4]
            #     scores.append(score)


            # print(result)
            ret_img = draw(img0, x_scale, y_scale, result)
            # ret_img = ret_img[:, :, ::-1]
            # plt.imshow(ret_img)
            # plt.show()
            cv2.imshow("frame", ret_img)
            # vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break


            # online_targets = tracker.update(bboxes, scores)
            # online_tlwhs = []
            # online_ids = []
            # online_scores = []
            # for i, t in enumerate(online_targets):
            #     # tlwh = t.tlwh
            #     tlwh = t.tlwh_yolox
            #     tid = t.track_id
            #     # vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
            #     # if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
            #     if tlwh[2] * tlwh[3] > args.min_box_area:
            #         online_tlwhs.append(tlwh)
            #         online_ids.append(tid)
            #         online_scores.append(t.score)
            #         results.append(
            #             f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
            #         )
            # t1 = time.time()
            # time_ = (t1 - t0) * 1000
            #
            # online_im = plot_tracking(frame, online_tlwhs, online_ids, frame_id=frame_id + 1,
            #                           fps=1000. / time_)
            #
            # cv2.imshow("frame", online_im)
            #
            # # vid_writer.write(online_im)
            # ch = cv2.waitKey(1)
            # if ch == 27 or ch == ord("q") or ch == ord("Q"):
            #     break
        else:
            break
    cv2.destroyAllWindows()
    cap.release()
        # frame_id += 1