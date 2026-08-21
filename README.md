# Export-TensorFlow-Lite-INT8
from ultralytics import YOLO  
model = YOLO( "/kaggle/working/runs/malaria_bbox_yolov8n_aug_img800/weights/best.pt" )  
model.export(     
format="tflite",     
imgsz=800,     
int8=True )
