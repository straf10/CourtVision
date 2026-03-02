from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import read_stub, save_stub

class BallTracker:

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in tqdm(range(0, len(frames), batch_size), desc="Detecting ball"):
            batch_frames = frames[i:i+batch_size]
            batch_detections = self.model.predict(batch_frames, conf=0.5)
            detections += batch_detections
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
                tracks[frame_num][1] = {"bbox": chosen_bbox}

        save_stub(stub_path, tracks)

        return tracks

    def remove_wrong_detections(self, ball_positions):
            
            max_allowed_distance = 25 # pixels
            last_good_frame_index = -1

            for i in range(len(ball_positions)):
                current_bbox = ball_positions[i].get(1, {}).get("bbox", [])

                if len(current_bbox) == 0:
                    continue

                if last_good_frame_index == -1:
                    last_good_frame_index = i
                    continue

                last_good_box = ball_positions[last_good_frame_index].get(1, {}).get("bbox", [])
                frame_gap = i - last_good_frame_index
                adjusted_max_distance = max_allowed_distance * frame_gap

                # Calculate the distance between the current and last good box
                distance = np.linalg.norm(np.array(last_good_box[:2]) - np.array(current_bbox[:2]))

                if distance > adjusted_max_distance:
                    ball_positions[i] = {}
                    
                else :
                    last_good_frame_index = i

            return ball_positions
                
    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [ x.get(1, {}).get("bbox", []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill() # backfill missing values

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions