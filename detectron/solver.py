from mlpm.solver import Solver

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

import cv2

class DetectionSolver(Solver):
    def __init__(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.DEVICE = "cpu"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the detectron2:// shorthand
        cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
        self.predictor = DefaultPredictor(cfg)

    def infer(self, data):
        im = cv2.imread(data['input_file_path'])
        outputs = self.predictor(im)
        results = []
        for index, each in enumerate(outputs["instances"].pred_classes):
            class_id = each.tolist()
            class_box = outputs["instances"].pred_boxes[index].tensor.flatten(
            ).tolist()
            class_score = outputs["instances"].scores[index].tolist()
            results.append(
                {'id': class_id, 'loc': class_box, 'score': class_score})
        return results
