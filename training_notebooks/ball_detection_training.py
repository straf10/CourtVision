import os
import shutil
import yaml
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO

load_dotenv()

# 1. Κατέβασμα του Dataset από το Roboflow
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("fyp-3bwmg").project("reloc2-den7l")
version = project.version(1)
dataset = version.download("yolov8")

# 2. Χρήση του dataset.location για σωστό path
current_dir = os.getcwd()
dataset_dir = dataset.location

# Αντιμετώπιση του bug με διπλό φάκελο αν υπάρχει
dataset_name = os.path.basename(dataset_dir)
nested_dir = os.path.join(dataset_dir, dataset_name)

if os.path.isdir(nested_dir):
    for folder in ['train', 'valid', 'test']:
        src_path = os.path.join(nested_dir, folder)
        dst_path = os.path.join(dataset_dir, folder)
        if os.path.exists(src_path):
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.move(src_path, dataset_dir)
            print(f"✅ Μετακινήθηκε ο φάκελος: {folder}")
    shutil.rmtree(nested_dir, ignore_errors=True)

# 3. Ενημέρωση του data.yaml
yaml_path = os.path.join(dataset_dir, 'data.yaml')

with open(yaml_path, 'r') as f:
    old_data = yaml.safe_load(f)

data = {
    'path': dataset_dir,
    'train': 'train/images',
    'val': 'valid/images',
    'names': old_data.get('names', {0: 'ball'}),
    'nc': old_data.get('nc', 1),
}

with open(yaml_path, 'w') as f:
    yaml.dump(data, f)
print(f"✅ Το data.yaml ενημερώθηκε — classes: {data['names']}")

# 4. Εκπαίδευση — YOLOv12l, βελτιστοποιημένη για RTX 3090 (24GB VRAM)
model = YOLO('yolo12l.pt')

results = model.train(
    data=yaml_path,
    epochs=1000,
    patience=100,
    imgsz=1280,
    batch=4,
    plots=True,

    # Optimizer
    optimizer="AdamW",
    lr0=0.0005,
    lrf=0.01,
    weight_decay=0.001,
    warmup_epochs=20,
    cos_lr=True,

    # Augmentation
    augment=True,
    mosaic=1.0,
    close_mosaic=50,
    mixup=0.15,
    copy_paste=0.2,
    scale=0.7,
    fliplr=0.5,
    hsv_h=0.02,
    hsv_s=0.7,
    hsv_v=0.5,
    degrees=10.0,
    translate=0.2,
    erasing=0.1,

    # Performance — utilize 10-core CPU and 128GB RAM
    workers=10,
    amp=True,
    cache="ram",

    project=current_dir,
    name="ball_detection_v1"
)