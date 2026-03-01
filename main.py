from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer

def main():
    
    #Read the video
    video_frames = read_video("input_videos/video_1.mp4")

    # Initialize the player tracker
    player_tracker = PlayerTracker("models/player_detector_model.pt")
    ball_tracker = BallTracker("models/ball_detector_model.pt")

    # Run the player tracker
    player_tracks = player_tracker.get_object_tracks(video_frames, read_from_stub = True, stub_path = "stubs/player_tracks.pkl")
    ball_tracks = ball_tracker.get_object_tracks(video_frames, read_from_stub = True, stub_path = "stubs/ball_tracks.pkl")

    # Draw output
    # Initialize the player tracks drawer
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()

    # Draw object tracks
    output_video_frames = player_tracks_drawer.draw(video_frames, player_tracks)
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)

    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()
