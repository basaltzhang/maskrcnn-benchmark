import os
import torch
import tensorrt as trt
import numpy as np
from tools.convert_model import conver_engine
import time
import cv2
import glob
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))# * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

# This function is generalized for multiple inputs/outputs for full dimension networks.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    # Run inference.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

if __name__ == "__main__":
    onnx_file_path = 'models/fast_rcnn.onnx'
    engine_file_path = "models/fast_rcnn.trt.bak"
    threshold = 0.5
    image_name = "demo/ILSVRC2012_val_00050000.JPEG"
    if not os.path.exists(engine_file_path):
        print("no engine file")
        # conver_engine(onnx_file_path, engine_file_path)
    print(f"Reading engine from file {engine_file_path}")
    preprocess_time = 0
    process_time = 0
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        with runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as context:
            inputs, outputs, bindings, stream = allocate_buffers(engine)
            image = cv2.imread(image_name)
            a = time.time()
            image_height, image_width = image.shape[:2]
            # image = cv2.resize(image, (768, 768)).transpose((2, 0, 1))
            image = np.array(cv2.resize(image, (768, 768)), dtype=np.float)
            image -= np.array([102.9801, 115.9465, 122.7717])
            image = np.transpose(image, (2, 0, 1)).ravel()
            # image_batch = np.stack([image], 0).ravel()
            np.copyto(inputs[0].host, image)
            preprocess_time += time.time() - a
            a = time.time()
            trt_outputs = do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            process_time += time.time() - a
            x = trt_outputs[0].reshape((100, 5))
            # imshow
            image = cv2.imread(image_name)
            indices = x[:, -1] > threshold
            polygons = x[indices, :-1]
            scores = x[indices, -1]
            polygons[:, ::2] *= 1. * image.shape[1] / 768
            polygons[:, 1::2] *= 1. * image.shape[0] / 768
        
            for polygon, score in zip(polygons, scores):
                print(polygon, score)
                cv2.rectangle(image, (int(polygon[0]), int(polygon[1])), (int(polygon[2]), int(polygon[3])), color=(0, 255, 0), thickness=2)
                cv2.putText(image, str("%.3f" % score), (int(polygon[0]), int(polygon[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, False)
            cv2.imwrite("tensorrt_out.jpg", image)

            print("preprocess time: ", preprocess_time, ", inference time: ", process_time)
