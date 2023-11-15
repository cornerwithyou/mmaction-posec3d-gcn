# python demo/demo_skeleton.py demo/demo_skeleton.mp4 demo/demo_skeleton_out.mp4 \
#     --config configs/skeleton/posec3d/spfocal_ntu60_xsub.py \
#     --checkpoint work_dirs/spfocal_ntu60_xsub/epoch_10.pth \
#     --det-config demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py \
#     --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
#     --det-score-thr 0.9 \
#     --det-cat-id 0 \
#     --pose-config demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py \
#     --pose-checkpoint https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
#     --label-map tools/data/skeleton/label_map_ntu60.txt

# python demo/demo_skeleton.py demo/demo_skeleton.mp4 demo/demo_skeleton_out.mp4 \
#     --config configs/skeleton/posec3d/spres_ntu60_xsub.py \
#     --checkpoint https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint/slowonly_r50_u48_240e_ntu60_xsub_keypoint-f3adabf1.pth \
#     --det-config demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py \
#     --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
#     --det-score-thr 0.9 \
#     --det-cat-id 0 \
#     --pose-config demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py \
#     --pose-checkpoint https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
#     --label-map tools/data/skeleton/label_map_ntu60.txt

python demo/demo_skeleton.py demo/demo_skeleton.mp4 demo/demo_skeleton_out.mp4 \
    --config configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py \
    --checkpoint https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint/slowonly_r50_u48_240e_ntu60_xsub_keypoint-f3adabf1.pth \
    --det-config demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py \
    --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --det-score-thr 0.9 \
    --det-cat-id 0 \
    --pose-config demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py \
    --pose-checkpoint https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
    --label-map tools/data/skeleton/label_map_ntu60.txt