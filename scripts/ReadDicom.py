import pydicom
import matplotlib.pyplot as plt
import numpy as np

file_path1 = "C:/Users/skras/OneDrive/Desktop/Mammography-images-registration/InBreastDynamics/AllDICOMs/before/50993670_b03f1dd34eb3c55f_MG_L_ML_ANON.dcm"
file_path2 = "C:/Users/skras/OneDrive/Desktop/Mammography-images-registration/InBreastDynamics/AllDICOMs/after/50993616_b03f1dd34eb3c55f_MG_L_ML_ANON.dcm"

ds1 = pydicom.dcmread(file_path1)
ds2 = pydicom.dcmread(file_path2)

img1 = ds1.pixel_array
img2 = ds2.pixel_array

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(img1, cmap='gray')
axes[0].set_title('Image 1')
axes[0].axis('off')


axes[1].imshow(img2, cmap='gray')
axes[1].set_title('Image 2')
axes[1].axis('off')


plt.tight_layout()
plt.show()