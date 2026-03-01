import os
import shutil
import yaml
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO

load_dotenv()

# 1. Κατέβασμα του Dataset από το Roboflow
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("roboflow-universe-projects").project("basketball-players-fy4c2")
version = project.version(25)
dataset = version.download("yolov12")

# Ορίζουμε το τρέχον path του φακέλου μας αντί για το /content/
current_dir = os.getcwd()
dataset_dir = os.path.join(current_dir, "Basketball-Players-25")
nested_dir = os.path.join(dataset_dir, "Basketball-Players-25")

# 2. Μετακίνηση Φακέλων (αντιμετώπιση του bug με το διπλό φάκελο)
try:
    shutil.move(os.path.join(dataset_dir, "train"), os.path.join(nested_dir, "train"))
    shutil.move(os.path.join(dataset_dir, "valid"), os.path.join(nested_dir, "valid"))
except Exception as e:
    pass # Αν έχουν ήδη μετακινηθεί, προχωράμε

folders_to_move = ['train', 'valid', 'test']

for folder in folders_to_move:
    src_path = os.path.join(nested_dir, folder)
    dst_path = os.path.join(dataset_dir, folder)

    if os.path.exists(src_path):
        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)
        shutil.move(src_path, dataset_dir)
        print(f"✅ Μετακινήθηκε ο φάκελος: {folder}")

# 3. Ενημέρωση του data.yaml
yaml_path = os.path.join(dataset_dir, 'data.yaml')

data = {
    'path': dataset_dir,
    'train': 'train/images',
    'val': 'valid/images',
    'names': {0: 'player', 1: 'referee', 2: 'ball'}
}

try:
    with open(yaml_path, 'r') as f:
        old_data = yaml.safe_load(f)
        if 'names' in old_data:
            data['names'] = old_data['names']
        if 'nc' in old_data:
            data['nc'] = old_data['nc']
except:
    pass

with open(yaml_path, 'w') as f:
    yaml.dump(data, f)
print("✅ Το data.yaml ενημερώθηκε σωστά.")

# 4. Εκπαίδευση — βελτιστοποιημένη για RTX 3090 (24GB VRAM)
model = YOLO('yolo12l.pt')

results = model.train(
    data=yaml_path,
    epochs=500,
    patience=80,
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

    # Augmentation — aggressive to compensate for few Ball samples
    augment=True,
    mosaic=1.0,
    close_mosaic=30,
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
    name="train3"
)