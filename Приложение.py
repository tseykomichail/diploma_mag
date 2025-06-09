# coding: cp1251
import gradio as gr
import numpy as np
import cv2
from ultralytics import YOLO
import pydicom

model = YOLO("best.pt")

def process_dicom(file):
    dicom_file = pydicom.dcmread(file.name)
    rescale_slope=dicom_file.RescaleSlope
    rescale_intercept=dicom_file.RescaleIntercept
    image = dicom_file.pixel_array.astype(np.float32)
    image=image*rescale_slope+rescale_intercept
    if image.shape[0]!=512 or image.shape[1]!=512 :
        image = cv2.resize(image, (512, 512))  
    image = np.clip(image, -1000, 500)
    image = (image +1000) / 1500     
    image = np.stack([image]*3, axis=-1)
    image = np.round(image * 255).astype(np.uint8)
    results = model(image, conf=0.005)
    return results[0].plot()

iface = gr.Interface(
    fn=process_dicom,
    inputs=gr.File(label="Загрузите DICOM (.dcm) файл", type="filepath"),
    outputs=gr.Image(type="numpy"),
    title="Детекция злокачественных узлов в легких",
)

iface.launch()
