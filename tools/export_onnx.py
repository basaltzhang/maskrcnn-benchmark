import time
import numpy as np
import cv2
import torch
from maskrcnn_benchmark.config import cfg
from demo.predictor import COCODemo
from PIL import Image

if __name__ == "__main__":
    config_file = "configs/e2e_faster_rcnn_R_50_C4_1x.yaml"
    image_name = "demo/ILSVRC2012_val_00050000.JPEG"
    out_file = "models/fast_rcnn.onnx"

    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cuda", "MODEL.WEIGHT", "models/e2e_faster_rcnn_R_50_C4_1x.pth", "MODEL.EXPORT_ON", True])

    coco_demo = COCODemo(
        cfg,
        min_image_size=768,
        confidence_threshold=0.7,
    )
    # load image and then run prediction
    image = cv2.imread(image_name)
    height, width = image.shape[:2]
    coco_demo.model.eval()
    image = np.array(cv2.resize(image, (768, 768)), dtype=np.float)
    image -= np.array([102.9801, 115.9465, 122.7717])
    image = np.stack([image] * 1, 0)
    images = torch.from_numpy(image).to(torch.float).to("cuda").permute(0, 3, 1, 2)
    print(images.size())

    with torch.no_grad():
        a = time.time()
        features = coco_demo.model(images)
        b = features.cpu().numpy()
        print("pytorch use time: ", time.time() - a)

    trace_backbone = torch.jit.trace(coco_demo.model, images, check_trace=False)
    torch.onnx.export(trace_backbone, images, out_file, verbose=True, export_params=True, training=False, opset_version=10, example_outputs=features)

    print("export done.")
