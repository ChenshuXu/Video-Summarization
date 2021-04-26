import cv2
import numpy as np
import os
import errno
from os import path
import time

import video_summarization

INPUT_FRAMES_FOLDER = "input/frames"
INPUT_AUDIO_FOLDER = "input/audio"
OUT_FOLDER = "output"


def main(frames_dir, audio_dir):
    print("processing {} + {}".format(frames_dir, audio_dir))


if __name__ == "__main__":
    frame_folders = os.walk(INPUT_FRAMES_FOLDER)
    next(frame_folders)
    for dirpath, dirnames, fnames in frame_folders:
        folder_name = dirpath.split("/")[-1]
        print("Processing folder '" + folder_name + "' ...")

    # main("input/frames/concert", "input/audio/concert.wav")
    main("input/frames/meridian", "input/audio/meridian.wav")
    # main("input/frames/soccer", "input/audio/soccer.wav")
