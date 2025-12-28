import cv2
import os
import numpy as np
from pytubefix import YouTube
from pytubefix.cli import on_progress
import uuid


def download_yt_video(
    url="https://www.youtube.com/watch?v=U26RWmIf12c", output_dir="video_frames"
):
    os.makedirs(output_dir, exist_ok=True)

    yt = YouTube(url, on_progress_callback=on_progress)
    ys = yt.streams.get_lowest_resolution()
    file_name = f"VIDEO_{uuid.uuid4()}.mp4"
    ys.download(filename=file_name, output_path=output_dir)
    return f"{output_dir}/{file_name}"


def generate_frames_and_save(change_threshold=0.10):
    """
    change_threshold = percentage of pixels that must change
    0.02 = 2% of the frame
    """

    video_path = download_yt_video()
    output_dir = f"video_frames/snaps/{video_path.split(".")[0]}"
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    success, prev_frame = cap.read()
    if not success:
        print("Failed to read video.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_index = 0
    saved_count = 0
    i = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        print("Sec count, ",i)
        i+=1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Absolute difference
        diff = cv2.absdiff(prev_gray, gray)

        # Threshold the difference
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Calculate changed pixel ratio
        changed_pixels = np.count_nonzero(thresh)
        total_pixels = thresh.size
        change_ratio = changed_pixels / total_pixels

        if change_ratio > change_threshold:
            cv2.imwrite(f"{output_dir}/frame_{saved_count}.jpg", frame)
            print(f"Saved frame {saved_count} | Change ratio: {change_ratio:.3f}")
            saved_count += 1

        prev_gray = gray
        frame_index += 1

    cap.release()
    print("Frame extraction complete.")
