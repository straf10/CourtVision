import os
from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer

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

    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()

    output_video_frames = player_tracks_drawer.draw(video_frames, player_tracks)
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)

    save_video(output_video_frames, OUTPUT_VIDEO, fps)

if __name__ == "__main__":
    main()
