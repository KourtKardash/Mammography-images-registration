import os
import pydicom
from PIL import Image
import numpy as np
import cv2

input_folder = "C:/Users/skras/OneDrive/Desktop/Mammography-images-registration/InBreastDynamics/AllDICOMs/after"
output_folder = "C:/Users/skras/OneDrive/Desktop/Mammography-images-registration/InBreastDynamicsPng/after"

os.makedirs(output_folder, exist_ok=True)

def save_dicom_as_png(dicom_path, output_path):
    try:
        ds = pydicom.dcmread(dicom_path)
        
        img_array = ds.pixel_array

        img_8bit = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img = Image.fromarray(img_8bit)
        
        img.save(output_path)
        print(f"Сохранено: {output_path}")
    except Exception as e:
        print(f"[Ошибка] Не удалось обработать {dicom_path}: {e}")

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".dcm"):
        dicom_file = os.path.join(input_folder, filename)
        png_file = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")
        save_dicom_as_png(dicom_file, png_file)

print("Конвертация завершена.")