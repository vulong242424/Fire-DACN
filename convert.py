from pytorch_to_tflite.pytorch_to_tflite import *
import torch
import yaml
import os
import mmcv
from nanodet.model.arch import build_model

PATH_TO_CONFIG = 'trained-models/model_final.pth'
cfg = yaml.safe_load(open(PATH_TO_CONFIG))
cfg = mmcv.Config(cfg)
model = build_model(cfg.model)

img = torch.randn(1,3,416,416)
out = model(img)

!mkdir -p cache/
onnx_out_path = 'model_final.onnx'
torch.onnx.export(model, img, onnx_out_path)

onnx_path = onnx_out_path
tf_path = onnx_path + '.tf'
onnx_to_tf(onnx_path=onnx_path, tf_path=tf_path)
assert os.path.exists(tf_path)

tflite_path = tf_path+'.tflite'
tf_to_tf_lite(tf_path, tflite_path)
assert os.path.exists(tflite_path)
tflite_path