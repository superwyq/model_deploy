# polygraphy run /home/wyq/hobby/model_deploy/models/yolov5/yolov5s_v7.onnx \
#     --trt  --onnxrt\
#     --onnx-outputs mark all \
#     --trt-outputs mark all \
#     --builder-optimization-level 5 \
#     --verbose \
#     > result-01.log 2>&1

# # /home/wyq/hobby/model_deploy/tensorRT_from_scratch/yolov5_trt/yolov5s.plan
# polygraphy surgeon sanitize /home/wyq/hobby/model_deploy/models/yolov5/yolov5s_v7.onnx \
#     --fold-constant \
#     -o modelA-FoldConstant.onnx \
#     > result.log
    
polygraphy inspect capability /home/wyq/hobby/model_deploy/models/yolov5/yolov5s.onnx > result_capability.log