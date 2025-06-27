import pydicom
import numpy as np

import os
import glob

import cv2
import ants


def load_dcm_to_array(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array
    return img.astype(np.float32), ds


def parse_filename(filename):
    base = os.path.basename(filename)
    parts = base.split("_")
    scan_id = parts[0]
    patient_id = parts[1]
    modality = parts[2]  # MG
    side = parts[3]   # L or R
    projection = parts[4]  # ML or CC
    return {
        "scan_id": scan_id,
        "patient_id": patient_id,
        "modality": modality,
        "side": side,
        "projection": projection,
        "filename": filename
    }


def find_patient_image_pairs(before_dir, after_dir):
    before_files = glob.glob(os.path.join(before_dir, "*.dcm"))
    after_files = glob.glob(os.path.join(after_dir, "*.dcm"))
    before_info = [parse_filename(f) for f in before_files]
    after_info = [parse_filename(f) for f in after_files]
    from collections import defaultdict
    before_dict = defaultdict(list)
    for info in before_info:
        key = (info["patient_id"], info["side"], info["projection"])
        before_dict[key].append(info)
    pairs = []
    for info in after_info:
        key = (info["patient_id"], info["side"], info["projection"])
        if key in before_dict:
            for before_info_single in before_dict[key]:
                pairs.append({
                    "fixed": info["filename"],
                    "moving": before_info_single["filename"]
                })

    return pairs


after_dir = "InBreastDynamics/AllDICOMs/after"
before_dir = "InBreastDynamics/AllDICOMs/before"

pairs = find_patient_image_pairs(before_dir, after_dir)


def resize_image(img, scale_factor):
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)


for pair in pairs:
    fixed_path = pair['fixed']
    moving_path = pair['moving']

    print(f"\nРегистрация:\nFixed: {fixed_path}\nMoving: {moving_path}")

    fixed_img, fixed_ds = load_dcm_to_array(fixed_path)
    moving_img, moving_ds = load_dcm_to_array(moving_path)

    h1 = fixed_img.shape[0]
    h2 = moving_img.shape[0]
    
    min_height = min(h1, h2)
    
    if h1 != min_height:
        ratio = min_height / h1
        new_width = int(fixed_img.shape[1] * ratio)
        fixed_img = cv2.resize(fixed_img, 
                               (new_width, min_height),
                               interpolation=cv2.INTER_LANCZOS4)
    
    if h2 != min_height:
        ratio = min_height / h2
        new_width = int(moving_img.shape[1] * ratio)
        moving_img = cv2.resize(moving_img,
                                (new_width, min_height),
                                interpolation=cv2.INTER_LANCZOS4)

    fixed_img = resize_image(fixed_img, scale_factor=0.5)
    moving_img = resize_image(moving_img, scale_factor=0.5)

    fixed_ants = ants.from_numpy(fixed_img)
    moving_ants = ants.from_numpy(moving_img)

    reg = ants.registration(
        fixed=fixed_ants,
        moving=moving_ants,
        type_of_transform="SyN"
    )

    warped_img = reg['warpedmovout']
    warped_np = warped_img.numpy()
    
    fixed_info = parse_filename(fixed_path)
    output_path = f"{fixed_info['patient_id']}_{fixed_info['side']}_{fixed_info['projection']}_reg.png"

    from PIL import Image, ImageDraw,ImageFont
    fixed = ((fixed_img - fixed_img.min()) / (fixed_img.max() - fixed_img.min()) * 255).astype(np.uint8)
    fixed_img = Image.fromarray(fixed)
    
    moving_img = ((moving_img - moving_img.min()) / (moving_img.max() - moving_img.min()) * 255).astype(np.uint8)
    moving_img = Image.fromarray(moving_img)
    
    warped = ((warped_np - warped_np.min()) / (warped_np.max() - warped_np.min()) * 255).astype(np.uint8)
    warped_uint8 = Image.fromarray(warped)

    collage = Image.new('RGB', (4 * fixed_img.width, fixed_img.height + 55), color=(255, 255, 255))
    collage.paste(moving_img, (0, 0))
    collage.paste(fixed_img, (fixed_img.width, 0))
    collage.paste(warped_uint8, (fixed_img.width * 2, 0, fixed_img.width * 2 + warped_uint8.width, warped_uint8.height))
    collage.paste(Image.fromarray(cv2.absdiff(fixed, warped)), (fixed_img.width * 3, 0, fixed_img.width * 3 + warped_uint8.width, warped_uint8.height))

    draw = ImageDraw.Draw(collage)

    try:
        font = ImageFont.truetype("arial.ttf", 32)  # Windows
    except:
        font = ImageFont.load_default()  # Linux / macOS

    labels = ["Moving (before)", "Fixed (after)", "Registered", "Difference"]

    for i, label in enumerate(labels):
        x = i * fixed_img.width + 10
        y = fixed_img.height + 5
        draw.text((x, y), label, fill="black", font=font)


    collage.save(output_path)
    print(f"Created collage: {output_path}")
