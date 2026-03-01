from ultralytics import YOLO
import sys 
import supervision as sv

sys.path.append("../")
from utils import read_stub, save_stub

class BallTracker:

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            batch_detections = self.model.predict(batch_frames, conf=0.5)
            detections+=batch_detections
        return detections

    def get_object_tracks(self, frames, read_from_stub = False, stub_path = None):

        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks
        
        detections = self.detect_frames(frames)
        tracks = []

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {}

            for key,value in cls_names.items():
                cls_names_inv[value] = key

            detection_supervision = sv.Detections.from_ultralytics(detection)

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks.append({})
            chosen_bbox = None
            max_confidence = 0

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                confidence = frame_detection[2]

                if cls_id == cls_names_inv["Ball"]:
                    if max_confidence < confidence:
                        chosen_bbox = bbox
                        max_confidence = confidence

            if chosen_bbox is not None:
                tracks[frame_num]["ball"] = {"bbox": chosen_bbox}

        save_stub(stub_path, tracks)

        return tracks