from rknn.api import RKNN

ONNX_MODEL = "yolov5n.onnx"
RKNN_MODEL = "yolov5n.rknn"
DATASET = "dataset.txt"
QUANTIZE_ON = False

if __name__ == "__main__":
    # Create RKNN object
    rknn = RKNN()

    # pre-process config
    print("--> Config model")
    rknn.config(
        mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3588"
    )
    print("done")

    # Load ONNX model
    print("--> Loading model")
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print("Load model failed!")
        exit(ret)
    print("done")

    # Build model
    print("--> Building model")
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print("Build model failed!")
        exit(ret)
    print("done")

    # Export RKNN model
    print("--> Export rknn model")
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print("Export rknn model failed!")
        exit(ret)
    print("done")
