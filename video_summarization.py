import os
import numpy as np
import scipy as sp
import cv2
from os import path
import time
import ffmpeg
from moviepy.editor import *


def create_video_from_frames(frames_dir, out_dir):
    """
    create a video from frames image in frames_dir, save to out_dir
    :param frames_dir: string
    :param out_dir: string
    :return: string
    """
    frame_names = [img for img in os.listdir(frames_dir)]

    # take frame number out from "frame***.jpg"
    def take_frame_num(name):
        return int(name[5:-4])

    frame_names.sort(key=take_frame_num)
    # print(frame_names)

    # get the size of first frame
    r, c, ch = cv2.imread(os.path.join(frames_dir, frame_names[0])).shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_file = os.path.join(out_dir, "frames.mp4")
    out = cv2.VideoWriter(video_file, fourcc, 30, (c, r))
    for i in range(len(frame_names)):
        out.write(cv2.imread(os.path.join(frames_dir, frame_names[i])))
    out.release()
    return video_file


def combine_frames_and_audio(video_file, audio_file, out_dir):
    """
    combine frame video with audio
    :param video_file: string
    :param audio_file: string
    :param out_dir: string
    :return:
    """
    clip = VideoFileClip(video_file)
    audio_clip = AudioFileClip(audio_file)
    video_clip = clip.set_audio(audio_clip)
    video_clip.write_videofile(os.path.join(out_dir, "combined.mp4"))
