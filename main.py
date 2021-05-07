import cv2
import numpy as np
import os
import errno
from os import path
import time

import video_summarization

INPUT_FRAMES_FOLDER = "input/frames/"
INPUT_AUDIO_FOLDER = "input/audio/"
OUT_FOLDER = "output/"


def main(frames_dir, audio_dir, output_dir):
    print("Processing frames folder '{}', audio folder '{}' save to '{}' ...".format(frames_dir, audio_dir,
                                                                                     output_dir))
    frames = video_summarization.read_frames_from_folder(frames_dir)
    video_file = video_summarization.create_video_from_frames(frames, output_dir)
    video_summarization.combine_frames_and_audio(video_file, audio_dir, output_dir)


if __name__ == "__main__":
    frame_folders = os.walk(INPUT_FRAMES_FOLDER)
    next(frame_folders)
    for dirpath, dirnames, fnames in frame_folders:
        video_name = dirpath.split("/")[-1]
        frames_folder_dir = os.path.join(INPUT_FRAMES_FOLDER, video_name)
        audio_file_dir = os.path.join(INPUT_AUDIO_FOLDER, video_name+".wav")
        output_dir = os.path.join(OUT_FOLDER, video_name)
        try:
            os.makedirs(output_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        print("Processing frames folder '{}', audio folder '{}' save to '{}' ...".format(frames_folder_dir, audio_file_dir, output_dir))
        video_summarization.create_summarized_video(frames_folder_dir, audio_file_dir, output_dir)

    # main("input/frames/concert", "input/audio/concert.wav", "output/concert")
