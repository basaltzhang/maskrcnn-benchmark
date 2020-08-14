import cv2
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import time

if __name__ == "__main__":
    image_name = "demo/ILSVRC2012_val_00050000.JPEG"
    config_file = "configs/e2e_faster_rcnn_R_50_C4_1x.yaml"
    weight_file = "models/e2e_faster_rcnn_R_50_C4_1x.pth"

    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cuda", "MODEL.WEIGHT", weight_file, "INPUT.MIN_SIZE_TEST", 768, "INPUT.MAX_SIZE_TEST", 768])

    coco_demo = COCODemo(
        cfg,
        min_image_size=768,
        confidence_threshold=0.7,
    )
    # load image and then run prediction
    image = cv2.imread(image_name)
    height, width = image.shape[:2]
    a = time.time()
    predictions = coco_demo.run_on_opencv_image(image)
    print("use time: ", time.time() - a)

    cv2.imwrite("out.jpg", predictions)
    print("done.")
