import os
from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer, TeamBallControlDrawer
from team_assigner import TeamAssigner
from ball_acquisition import BallAcquisitionDetector

INPUT_VIDEO = "input_videos/video_1.mp4"
PLAYER_MODEL = "models/best.pt"
BALL_MODEL = "models/best.pt"
OUTPUT_VIDEO = "output_videos/output_video.avi"

def validate_paths(*paths):
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

def main():

    validate_paths(INPUT_VIDEO, PLAYER_MODEL, BALL_MODEL)

    video_frames, fps = read_video(INPUT_VIDEO)

    player_tracker = PlayerTracker(PLAYER_MODEL)
    ball_tracker = BallTracker(BALL_MODEL)

    player_tracks = player_tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/player_tracks.pkl")
    ball_tracks = ball_tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/ball_tracks.pkl")

    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

    # Team Assignment
    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_player_teams_across_frames(video_frames, player_tracks, read_from_stub=True, stub_path="stubs/player_teams.pkl")


    # Ball Acquisition
    ball_acquisition_detector = BallAcquisitionDetector()
    ball_acquisition = ball_acquisition_detector.detect_ball_possession(player_tracks, ball_tracks)

    print(ball_acquisition)
    
    # Draw Player Tracks
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    team_ball_control_drawer = TeamBallControlDrawer()

    output_video_frames = player_tracks_drawer.draw(video_frames, player_tracks, player_assignment, ball_acquisition)

    # Draw Team Ball Control
    output_video_frames = team_ball_control_drawer.draw(output_video_frames, player_assignment, ball_acquisition)

    # Save Video
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)

    save_video(output_video_frames, OUTPUT_VIDEO, fps)

if __name__ == "__main__":
    main()
