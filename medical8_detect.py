# medical8_detect.py
# 8 sınıflı (brain/chest/kidney/lung × real/fake) medikal deepfake tespit sistemi

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128

# 8 sınıf isimleri
CLASS_NAMES = [
    "brain_real", "brain_fake",
    "chest_real", "chest_fake",
    "kidney_real", "kidney_fake",
    "lung_real", "lung_fake"
]


# ============================================================
# IMAGE PREPROCESS (EĞİTİMDEKİNİN AYNISI)
# ============================================================

def load_image_medical8(img_path):
    img = Image.open(img_path).convert("RGB")

    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]),
    ])

    return transform(img).unsqueeze(0)   # [1,3,H,W]


# ============================================================
# MODEL KURULUMU (ResNet18 + 8 class head)
# ============================================================

def build_resnet8(num_classes=8):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


# ============================================================
# GÜVENLİ WEIGHT LOADER (mismatch-skip)
# ============================================================

def safe_load(model, ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    model_dict = model.state_dict()
    new_state = {}

    for k, v in state.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            new_state[k] = v
        else:
            print(f"[SKIP] {k} (shape mismatch / unused)")

    model_dict.update(new_state)
    model.load_state_dict(model_dict)
    print(f"[OK] Loaded {len(new_state)} parameters.")
    return model


# ============================================================
# ANA FONKSIYON — TEK GÖRÜNTÜDEN TAHMIN
# ============================================================

@torch.no_grad()
def detect_medical8(model_path, image_path):
    """
    return:
        {
            "class_id": int,
            "class_name": str,
            "probs": [8 sınıflı olasılık listesi]
        }
    """

    # model
    model = build_resnet8(num_classes=8).to(DEVICE)
    model = safe_load(model, model_path)
    model.eval()

    # image
    img = load_image_medical8(image_path).to(DEVICE)

    # forward
    logits = model(img)[0]        # [8]
    probs = torch.softmax(logits, dim=0)

    cls_id = torch.argmax(probs).item()
    cls_name = CLASS_NAMES[cls_id]

    return {
        "class_id": cls_id,
        "class_name": cls_name,
        "probs": probs.cpu().numpy().tolist()
    }


# ============================================================
# CLI çalıştırma
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    out = detect_medical8(args.model, args.image)
    print(out)
