import os
import torch
import tensorrt as trt
import sys

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def GiB(val):
    return val * 1 << 30

def conver_engine(onnx_file_path, engine_file_path="", max_batch_size=1):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = GiB(1)
        builder.max_batch_size = max_batch_size
        # Parse model file
        if not os.path.exists(onnx_file_path):
            print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
            exit(0)
        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None
        # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
        engine = builder.build_cuda_engine(network)
        print("Completed creating Engine")
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        print("Completed writing Engine. Well done!")

if __name__ == "__main__":
    onnx_file_path = 'models/fast_rcnn.onnx'
    engine_file_path = "models/fast_rcnn.trt"
    conver_engine(onnx_file_path, engine_file_path)
