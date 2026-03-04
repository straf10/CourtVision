import os
from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer, TeamBallControlDrawer, PassAndInterceptionsDrawer
from team_assigner import TeamAssigner
from ball_acquisition import BallAcquisitionDetector
from pass_and_interception import PassAndInterceptionDetector

INPUT_VIDEO = "input_videos/video_2.mp4"
PLAYER_MODEL = "models/player_detector_pretained.pt"
BALL_MODEL = "models/ball_detector_pretrained.pt"
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

    # Detect Passes and Interceptions
    pass_and_interception_detector = PassAndInterceptionDetector()
    passes = pass_and_interception_detector.detect_passes(ball_acquisition, player_assignment)
    interceptions = pass_and_interception_detector.detect_interceptions(ball_acquisition, player_assignment)
    
    # Draw Player Tracks
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    team_ball_control_drawer = TeamBallControlDrawer()
    pass_and_interceptions_drawer = PassAndInterceptionsDrawer()

    output_video_frames = player_tracks_drawer.draw(video_frames, player_tracks, player_assignment, ball_acquisition)

    # Draw Team Ball Control
    output_video_frames = team_ball_control_drawer.draw(output_video_frames, player_assignment, ball_acquisition)

    # Draw Passes and Interceptions
    output_video_frames = pass_and_interceptions_drawer.draw(output_video_frames, passes, interceptions)

    # Save Video
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)

    save_video(output_video_frames, OUTPUT_VIDEO, fps)

if __name__ == "__main__":
    main()
