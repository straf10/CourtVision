import sys
sys.path.append("../")
from utils import measure_distance, get_center_of_bbox

class BallAcquisitionDetector:
    def __init__(self):
        self.possession_threshold = 50
        self.min_frames = 11
        self.containment_threshold = 0.8

    def get_key_basketball_player_assignment_points(self, player_bbox, ball_center):
        ball_center_x = ball_center[0]
        ball_center_y = ball_center[1]

        x1, y1, x2, y2 = player_bbox
        width = x2 - x1
        height = y2 -y1
        
        output_points = []

        if ball_center_y > y1 and ball_center_y < y2:
            output_points.append((x1, ball_center_y))
            output_points.append((x2, ball_center_y))

        if ball_center_x > x1 and ball_center_x < x2:
            output_points.append((ball_center_x, y1))
            output_points.append((ball_center_x, y2))

        output_points += [
            (x1 + width // 2, y1),
            (x2, y1),
            (x1, y1),
            (x2, y1 + height // 2),
            (x1, y1 + height // 2),
            (x1 + width // 2, y1 + height // 2),
            (x2, y2),
            (x1, y2),
            (x1 + width // 2, y2),
            (x1 + width // 2, y1 + height // 3),
        ]

        return output_points

    def find_minimum_distance_to_ball(self, ball_center, player_bbox):
        key_points = self.get_key_basketball_player_assignment_points(player_bbox, ball_center)
        min = 9999

        for key_point in key_points:
            distance = measure_distance(ball_center, key_point)
            if min > distance:
                min = distance
        
        return min

        
    def calculate_ball_containment_ratio(self, player_bbox, ball_bbox):

        px1, py1, px2, py2 = player_bbox
        bx1, by1, bx2, by2 = ball_bbox

        player_area = (px2 - px1) * (py2 - py1)
        ball_area = (bx2 - bx1) * (by2 - by1)

        intersection_x1 = max(px1, bx1)
        intersection_y1 = max(py1, by1)
        intersection_x2 = min(px2, bx2)
        intersection_y2 = min(py2, by2)

        intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
        containment_ratio = intersection_area / ball_area

        return containment_ratio
        
    def find_best_candidate_for_ball_possession(self, ball_center, player_tracks_frame, ball_bbox):
        
        high_containment_players = []
        regular_distance_players = []

        for player_id, player_info in player_tracks_frame.items():
            player_bbox = player_info.get("bbox", [])
            if not player_bbox:
                continue
            
            containment = self.calculate_ball_containment_ratio(player_bbox, ball_bbox)
            min_distance = self.find_minimum_distance_to_ball(ball_center, player_bbox)

            if containment > self.containment_threshold:
                high_containment_players.append((player_id, min_distance))
            else:
                regular_distance_players.append((player_id, min_distance))

        # First priority is high containment players
        if high_containment_players:
            best_candidate = max(high_containment_players, key=lambda x: x[1])
            return best_candidate[0]

        if regular_distance_players:
            best_candidate = min(regular_distance_players, key=lambda x: x[1])
            if best_candidate[1] < self.possession_threshold:
                return best_candidate[0]

        return -1

    def detect_ball_possession(self, player_tracks, ball_tracks):
        num_frames = len(player_tracks)
        possession_list = [-1] * num_frames

        consecutive_possession_count= {}
        for frame_num in range(num_frames):
            ball_info = ball_tracks[frame_num].get(1, {})
            if not ball_info:
                continue

            ball_bbox = ball_info.get("bbox", [])
            if not ball_bbox:
                continue

            ball_center = get_center_of_bbox(ball_bbox)

            best_player_id = self.find_best_candidate_for_ball_possession(ball_center, player_tracks[frame_num], ball_bbox)

            if best_player_id != -1:
                number_of_consecutive_frames = consecutive_possession_count.get(best_player_id, 0)+1
                consecutive_possession_count = {best_player_id: number_of_consecutive_frames}

                if number_of_consecutive_frames >= self.min_frames:
                    possession_list[frame_num] = int(best_player_id)

            else:
                consecutive_possession_count = {}
        return possession_list