import os
import numpy as np
import scipy as sp
import cv2
from matplotlib import pyplot as plt
from os import path
import time
import ffmpeg
from moviepy.editor import *


def create_video_from_frames(frames, out_dir):
    """
    create a video from frames image in frames_dir, save to out_dir
    :param frames: np.ndarray, dtype=np.uint8, shape=(n, r, c, ch)
    :param out_dir: string
    :return: string, dir of the new video file
    """
    n, r, c, ch = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_file = os.path.join(out_dir, "frames.mp4")
    out = cv2.VideoWriter(video_file, fourcc, 30, (c, r))
    for i in np.arange(n):
        out.write(frames[i])
    out.release()
    return video_file


def combine_frames_and_audio(video_file, audio_file, out_dir):
    """
    combine frame video with audio, create a new video
    :param video_file: string
    :param audio_file: string
    :param out_dir: string
    :return:
    """
    clip = VideoFileClip(video_file)
    audio_clip = AudioFileClip(audio_file)
    video_clip = clip.set_audio(audio_clip)
    video_clip.write_videofile(os.path.join(out_dir, "combined.mp4"))


def read_frames_from_folder(frames_dir, start=0, end=-1):
    """
    read all frames from frames folder
    :param frames_dir: string
    :return: np.ndarray, dtype=np.uint8, shape=(n, r, c, ch)
        image array
    """
    frame_names = [img for img in os.listdir(frames_dir)]

    # take frame number out from "frame***.jpg"
    def take_frame_num(name):
        return int(name[5:-4])

    frame_names.sort(key=take_frame_num)

    frame_names = frame_names[start:end]
    # get the size of first frame
    r, c, ch = cv2.imread(os.path.join(frames_dir, frame_names[0])).shape
    frames = np.zeros((len(frame_names), r, c, ch), dtype=np.uint8)
    for i in range(len(frame_names)):
        frames[i] = cv2.imread(os.path.join(frames_dir, frame_names[i]))
    return frames


def compute_SAD_between_two_frames(frame1, frame2):
    """

    :param frame1: np.ndarray, dtype=np.uint8, shape=(r,c,ch)
    :param frame2: np.ndarray, dtype=np.uint8, shape=(r,c,ch)
    :return: float
    """
    diff = np.abs(frame1.astype(np.float64) - frame2.astype(np.float64))
    diff = np.sum(diff, axis=2)
    average = np.average(diff)
    return average


def compute_difference_SAD(frames):
    """
    use sum of absolute differences (SAD) to calculate the cutting score
    :param frames: np.ndarray, dtype=np.uint8, shape=(n, r, c, ch)
        image array
    :return: np.ndarray, dtype=np.float64, shape=(n-1)
        difference array
    """
    n, r, c, ch = frames.shape
    diffs = np.zeros((n-1), dtype=np.float64)
    for i in np.arange(1, n):
        diffs[i-1] = compute_SAD_between_two_frames(frames[i - 1], frames[i])
    return diffs


def compute_HD_between_two_frames(frame1, frame2):
    """

    :param frame1: np.ndarray, dtype=np.uint8, shape=(r,c,ch)
    :param frame2: np.ndarray, dtype=np.uint8, shape=(r,c,ch)
    :return: float
    """
    r, c, ch = frame1.shape
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame1_h = cv2.calcHist(frame1_gray, [0], None, [256], [0, 256])

    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    frame2_h = cv2.calcHist(frame2_gray, [0], None, [256], [0, 256])
    diff = 1000 * np.sum((frame1_h - frame2_h) ** 2) / (r * c)
    return diff


def compute_difference_HD(frames):
    """
    use histogram differences (HD) to calculate the cutting score
    :param frames: np.ndarray, dtype=np.uint8, shape=(n, r, c, ch)
    :return: np.ndarray, dtype=np.float64, shape=(n-1)
        difference array
    """
    n, r, c, ch = frames.shape
    diffs = np.zeros((n - 1), dtype=np.float64)
    for i in np.arange(1, n):
        diffs[i - 1] = compute_HD_between_two_frames(frames[i - 1], frames[i])
    return diffs


def compute_edge_diff_between_two_frames(frame1, frame2):
    pass


def plot_difference_scores(diffs, plot_filename="diff_score.png", start=0):
    """

    :param diffs: np.ndarray, dtype=np.float64, shape=(n-1)
    :param plot_filename:
    :param start: int, start frame number
    :return:
    """
    start_second = start / 30
    end_second = (start + diffs.shape[0]) / 30
    total_seconds = diffs.shape[0] / 30
    seconds = np.arange(start, start + diffs.shape[0]) / 30
    width = total_seconds * 0.2

    plt.figure(figsize=(width, 4))
    p = plt.plot(seconds, diffs)
    plt.xticks(np.arange(start_second, end_second, 2))
    plt.savefig(plot_filename)
    # plt.show()
    plt.clf()


def get_scene_last_frame_index(frames, threshold=200):
    """
    return index of frame of last scene
    :param threshold: int
    :param frames: np.ndarray, dtype=np.uint8, shape=(n, r, c, ch)
    :return: np.ndarray, dtype=np.int64, shape=(m, 1)
    """
    diffs = compute_difference_HD(frames)
    res = np.argwhere(diffs >= threshold)
    return res
