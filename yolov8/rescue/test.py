import os
import urllib
import traceback
import sys
import numpy as np

from rknn.api import RKNN

ONNX_MODEL = 'best.onnx'
RKNN_MODEL = 'best.rknn'
IMG_PATH = 'image.jpg'
DATASET = './dataset.txt'

# 是否打开量化
QUANTIZE_ON = True

# 修改成自己的类即可
CLASSES = ('NonViolence','violence','fire','','','Casualty-PPE','Helicopter','Ship','Swimmer-No-PPE','night-person')

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                target_platform="rk3588")
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Build model.')
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