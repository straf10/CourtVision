from utils import read_video, save_video
from trackers import PlayerTracker

def main():
    
    #Read the video
    video_frames = read_video("input_videos/video_1.mp4")

    # Initialize the player tracker
    player_tracker = PlayerTracker("models/player_detector_model.pt")

    # Run the player tracker
    player_tracks = player_tracker.get_object_tracks(video_frames, read_from_stub = True, stub_path = "stubs/player_tracks.pkl")

    print(player_tracks)


    save_video(video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()
