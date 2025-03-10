_base_ = [
    '../../_base_/models/mask_rcnn_r50_fpn.py',
    '../../_base_/datasets/coco_instance_generatepl_1_4.py',
    '../../_base_/schedules/schedule_2x.py', '../../_base_/default_runtime.py'
]
model=dict(test_cfg=dict(rcnn=dict(mask_thr_binary=0.44)))

work_dir = "./work_dirs/noisyboundaries/coco/mask_rcnn_r50_fpn_2x_coco_generatepl_1_4/"