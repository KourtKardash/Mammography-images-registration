import pydicom
import matplotlib.pyplot as plt
from xml.etree import ElementTree as ET
import numpy as np

dcm_path = "InBreastDynamics/AllDICOMs/after/51049489_6f64793857feb5d0_MG_R_ML_ANON.dcm"
xml_path = "InBreastDynamics/AllXML/51049489.xml"

ds = pydicom.dcmread(dcm_path)
img = ds.pixel_array
tree = ET.parse(xml_path)
root = tree.getroot()

height, width = img.shape
print(height, width)
dpi = 100
figsize = width / dpi, height / dpi

fig = plt.figure(figsize=figsize, dpi=dpi, frameon=False)
ax = plt.Axes(fig, [0, 0, 1, 1])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(img, cmap='gray', aspect='auto')

def get_roi_data(roi_dict, key):
    for i, elem in enumerate(roi_dict):
        if elem.tag == "key" and elem.text == key:
            return roi_dict[i+1]
    return None

for roi in root.findall(".//array/dict"):
    try:
        name_elem = get_roi_data(roi, "Name")
        name = name_elem.text if name_elem is not None else "Unnamed"
        
        points_elem = get_roi_data(roi, "Point_px")
        if points_elem is None:
            continue
            
        points = []
        for point in points_elem.findall("string"):
            try:
                x, y = map(float, point.text.strip("()").split(", "))
                points.append([x, y])
            except (ValueError, AttributeError):
                continue
        
        if not points:
            continue
            
        points = np.array(points)
        if len(points) > 1:
            ax.plot(points[:, 0], points[:, 1], 'r-', linewidth=1, solid_capstyle='round', solid_joinstyle='round')
        elif len(points) == 1:
            ax.scatter(points[0, 0], points[0, 1], c='red', s=1, marker='o', linewidths=0)
        
    except Exception as e:
        print(f"Ошибка при обработке ROI: {e}")
        continue

plt.savefig("out_after.png", bbox_inches='tight', pad_inches=0, dpi=dpi)
plt.close()