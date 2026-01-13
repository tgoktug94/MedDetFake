# ddpm_interface.py

import torch
from zipfile import ZipFile
import os
from PIL import Image
import numpy as np
def detect_fake_synthesis(model_path="./models/resnet_8class_best.pth", image_path="original-2.jpg"):
    from medical8_detect import detect_medical8

    result = detect_medical8(
        model_path=model_path,
        image_path=image_path
    )
    return result
def detect_manipulated_density(model_path, image_path, thr=0.28, min_area=20):
    from manip_density_detect import detect_manipulation_density
    return detect_manipulation_density(
        model_path=model_path,
        image_path=image_path,
        thr=thr,
        min_area=min_area,
        return_density_u8=True
    )
