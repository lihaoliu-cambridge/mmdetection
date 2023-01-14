_base_ = [
    '../../_base_/models/mask_rcnn_r50_fpn.py',
    '../../_base_/datasets/coco_instance_sup.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]

work_dir = "./work_dirs/noisyboundaries/coco/mask_rcnn_r50_fpn_1x_coco_sup/"