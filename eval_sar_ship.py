import numpy as np
import numpy as np
from mmeval import DOTAMeanAP
from mmdet.datasets.api_wrappers import COCO
import cv2

dota_metric = DOTAMeanAP(num_classes=1)

def _gen_bboxes(num_bboxes, img_w=256, img_h=256):
    # random generate bounding boxes in 'xywha' formart.
    x = np.random.rand(num_bboxes, ) * img_w
    y = np.random.rand(num_bboxes, ) * img_h
    w = np.random.rand(num_bboxes, ) * (img_w - x)
    h = np.random.rand(num_bboxes, ) * (img_h - y)
    a = np.random.rand(num_bboxes, ) * np.pi / 2
    return np.stack([x, y, w, h, a], axis=1)


predict = np.load('result_dict.npy', allow_pickle=True).item()


predictions = []
groundtruths = []

_coco_api = COCO('/home/zytx121/mmrotate/data/HRSID/annotations/test2017.json')
img_ids = _coco_api.get_img_ids()

for i in range(len(img_ids)):
    ann_ids = _coco_api.get_ann_ids(img_ids=img_ids[i])
    ann_info = _coco_api.load_anns(ann_ids) 
    file_name = _coco_api.load_imgs(img_ids[i])[0]['file_name']
    try:
        predict_box = predict[file_name]
    except:
        continue

    pred_bboxes = []
    for pred in predict_box:
        pred = [p / 512 * 800 for p in pred]
        (x, y), (w, h), angle = cv2.minAreaRect(np.array(pred, dtype=np.float32).reshape(-1,2))
        pred_bboxes.append([x, y, w, h, angle / 180 * np.pi])
    pred_len = len(pred_bboxes)

    prediction = {
        'bboxes': np.array(pred_bboxes),
        'scores': np.ones((pred_len, )),
        'labels': np.zeros((pred_len, ))
    }
    predictions.append(prediction)

    gt_bboxes = []
    for ann in ann_info:
        (x, y), (w, h), angle = cv2.minAreaRect(np.array(ann['segmentation'], dtype=np.float32).reshape(-1,2))
        gt_bboxes.append([x, y, w, h, angle / 180 * np.pi])
    box_len = len(gt_bboxes)

    groundtruth = {
        'bboxes': np.array(gt_bboxes),
        'labels': np.zeros((box_len, )),
        'bboxes_ignore': _gen_bboxes(1),
        'labels_ignore': np.zeros((1, ))
    }
    groundtruths.append(groundtruth)

print(dota_metric(predictions=predictions, groundtruths=groundtruths))
