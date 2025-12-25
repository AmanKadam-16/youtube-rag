import os
from functools import partial
from multiprocessing.pool import Pool
import cv2
import yt_dlp


# TODO: Need to Understand this code functioning.
def process_video_parallel(url, skip_frames, process_number):
    cap = cv2.VideoCapture(url)
    num_process = os.cpu_count()
    frames_per_second = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // num_process
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames_per_second * process_number)
    x, count = 0
    while x < 10 and count < frames_per_second:
        ret, frame = cap.read()
        if not ret:
            break
        filename = r"Frame_No_" + str(x) + ".png"
        x += 1
        cv2.imwrite(filename.format(count), frame)
        count += skip_frames
        cap.set(1, count)

    cap.release()


def obtain_frames():
    youtube_video_url = "https://www.youtube.com/watch?v=5sLYAQS9sWQ"
    ydl_opts = {}
    ydl = yt_dlp.YoutubeDL(ydl_opts)
    info_dict = ydl.extract_info(youtube_video_url, download=False)
    formats = info_dict.get("formats", None)

    print("Obtaining frames...")
    for f in formats:
        if f.get("format_note", None) == "144p":
            url = f.get("url", None)
            cpu_count = os.cpu_count()
            with Pool(cpu_count) as pool:
                pool.map(partial(process_video_parallel, url, 300), range(cpu_count))
