import math
import os
import numpy as np
import scipy as sp
import cv2
from matplotlib import pyplot as plt
from os import path
import time
from moviepy.editor import *

# Standard PySceneDetect imports:
from scenedetect import VideoManager
from scenedetect import SceneManager

# For content-aware scene detection:
from scenedetect.detectors import ContentDetector


def create_video_from_frames_folder(frames_dir, out_dir):
    """

    :param frames_dir: string
    :param out_dir: string
    :return: string, dir of the new video file
    """
    frames = read_frames_from_folder(frames_dir)
    return create_video_from_frames(frames, out_dir)


def create_video_from_frames(frames, out_dir, file_name="frames.mp4"):
    """
    create a video from frames image in frames_dir, save to out_dir
    :param frames: np.ndarray, dtype=np.uint8, shape=(n, r, c, ch)
    :param out_dir: string
    :param file_name: string
    :return: string, dir of the new video file
    """
    n, r, c, ch = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_file = os.path.join(out_dir, file_name)
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


def compute_difference_SAD_with_edge(frames):
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
        frame1 = cv2.Canny(frames[i - 1], 100, 200)
        frame2 = cv2.Canny(frames[i], 100, 200)
        diff = np.abs(frame1.astype(np.float64) - frame2.astype(np.float64))
        diff = np.sum(diff)
        diffs[i-1] = diff
    return diffs


def compute_HD_between_two_frames(frame1, frame2):
    """

    :param frame1: np.ndarray, dtype=np.uint8, shape=(r,c,ch)
    :param frame2: np.ndarray, dtype=np.uint8, shape=(r,c,ch)
    :return: float
    """
    r = frame1.shape[0]
    c = frame1.shape[1]
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


def compute_difference_HD_with_edge(frames):
    n, r, c, ch = frames.shape
    diffs = np.zeros((n - 1), dtype=np.float64)
    for i in np.arange(1, n):
        frame1 = cv2.Canny(frames[i - 1], 100, 200)
        frame2 = cv2.Canny(frames[i], 100, 200)
        frame1_h = cv2.calcHist(frame1, [0], None, [256], [0, 256])
        frame2_h = cv2.calcHist(frame2, [0], None, [256], [0, 256])
        diff = 1000 * np.sum((frame1_h - frame2_h) ** 2) / (r * c)
        diffs[i - 1] = diff
    return diffs


def compute_edge_diff_between_two_frames(frame1, frame2):
    pass


def compute_homography(prev_frame, curr_frame, feature_params=None, lk_params=None):
    # params for ShiTomasi corner detection
    if feature_params is None:
        feature_params = dict(maxCorners=1000,
                              qualityLevel=0.1,
                              minDistance=10,
                              blockSize=10)

    # Parameters for lucas kanade optical flow
    if lk_params is None:
        lk_params = dict(winSize=(50, 50),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    if p0 is None or p0.shape[0] <= 6:
        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        H = np.append(M, np.array([0.0, 0.0, 1.0]).reshape((1, 3)), axis=0)
        return H

    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)

    # select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    if good_new.shape[0] > 6 and good_old.shape[0] > 6:
        M, _ = cv2.estimateAffine2D(good_new, good_old, method=cv2.RANSAC)
    else:
        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        H = np.append(M, np.array([0.0, 0.0, 1.0]).reshape((1, 3)), axis=0)
        return H

    if M is None:
        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)

    H = np.append(M, np.array([0.0, 0.0, 1.0]).reshape((1, 3)), axis=0)
    return H


def compute_homographies_from_frames(frames, feature_params=None, lk_params=None):
    """
    get array of homographies from frames
    Parameters
    ----------
    frames: np.ndarray(), dtype=np.uint8, shape=(n,r,c,3)
    feature_params: dict
    lk_params: dict
    show_frames: bool
    output_dir: str
    first_frame_idx: int
    Returns
    -------
    homographies: np.ndarray(), dtype=np.float64, shape=(n-1, 3, 3)
    """
    n, r, c, ch = frames.shape
    homographies = np.zeros((n - 1, 3, 3), dtype=np.float64)
    for i in np.arange(1, n):
        h = compute_homography(frames[i - 1], frames[i], feature_params, lk_params)
        homographies[i - 1] = h

    return homographies


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


def compute_scene_idx(frames, threshold=100, minimum_length=30):
    """
    split scenes, return tuple of index (begin, end)
    :param minimum_length:
    :param threshold: int
    :param frames: np.ndarray, dtype=np.uint8, shape=(n, r, c, ch)
    :return: list if tuple(begin frame index, end frame index)
    """
    diffs = compute_difference_SAD(frames)
    result = []
    last_idx = 0
    for i in np.arange(1, frames.shape[0]):
        if diffs[i-1] >= threshold and i-last_idx >= minimum_length:
            result.append((last_idx, i))
            last_idx = i+1
    result.append((last_idx, frames.shape[0]))
    return result


def create_scene_frames_from_breaks(frames, breaks):
    """

    :param frames: np.ndarray, dtype=np.uint8, shape=(n, r, c, ch)
        the whole video frames
    :param breaks: list of tuples (begin, end)
    :return: list of np.ndarray, dtype=np.float64
        array of shot frames
    """
    shots_frames = []

    for i in range(len(breaks)):
        shots_frames.append(frames[breaks[i][0]:breaks[i][1]])

    return shots_frames


def create_scene_video_from_breaks(frames, breaks, out_dir):
    """

    :param frames: np.ndarray, dtype=np.uint8, shape=(n, r, c, ch)
        the whole video frames
    :param breaks: list of tuples (begin, end)
    :param out_dir: string
    :return:
    """
    n, r, c, ch = frames.shape
    shots_frames = create_scene_frames_from_breaks(frames, breaks)
    for i in range(len(breaks)):
        shot_frames = shots_frames[i]
        create_video_from_frames(shot_frames, out_dir, "frames_{}_{}.mp4".format(breaks[i][0], breaks[i][1]))


def compute_movement_score_from_frames(frames):
    """

    :param frames: np.ndarray, dtype=np.uint8, shape=(n, r, c, ch)
    :return:
    """
    n, r, c, ch = frames.shape



def plot_homographies(raw_homography, output_dir, prefix="", smoothed_homography=None, start=None, end=None):
    """
    plot homography array
    Parameters
    ----------
    prefix
    raw_homography: np.ndarray(), dtype=np.float64, shape of (n, 3, 3)
    output_dir: string, output directory string
    smoothed_homography: np.ndarray(), dtype=np.float64, shape of (n, 3, 3)
    start: int, start frame index
    end: int, end frame index

    Returns
    -------

    """
    if start is None:
        start = 0
    if end is None:
        end = start + raw_homography.shape[0]
    else:
        end = min(end, start + raw_homography.shape[0])
    frame_idx = np.arange(start, end)
    n = end - start
    sub_prefix = prefix + " frame {} - {} ".format(start, end)
    raw_x_path = np.zeros(n)
    raw_y_path = np.zeros(n)
    raw_dx = np.zeros(n)
    raw_dy = np.zeros(n)
    smoothed_x_path = np.zeros(n)
    smoothed_y_path = np.zeros(n)
    smoothed_dx = np.zeros(n)
    smoothed_dy = np.zeros(n)
    pt = np.array([1, 1, 1], dtype=np.float64)

    for i in np.arange(n):
        pt = np.matmul(raw_homography[i], pt)
        raw_x_path[i] = pt[0]
        raw_y_path[i] = pt[1]
        raw_dx[i] = raw_homography[i, 0, 2]
        raw_dy[i] = raw_homography[i, 1, 2]
        if smoothed_homography is not None:
            smooth_pt = np.matmul(smoothed_homography[i], pt)
            smoothed_x_path[i] = smooth_pt[0]
            smoothed_y_path[i] = smooth_pt[1]
            smoothed_dx[i] = smoothed_homography[i, 0, 2]
            smoothed_dy[i] = smoothed_homography[i, 1, 2]

    plt.plot(frame_idx, raw_dx, label="raw dx")
    if smoothed_homography is not None:
        plt.plot(frame_idx, smoothed_dx, label="smoothed dx")
    plt.title("dx")
    plt.xlabel("frame")
    plt.ylabel("dx")
    plot_filename = path.join(output_dir, sub_prefix + "motion_dx.png")
    plt.legend()
    plt.savefig(plot_filename)
    # plt.show()
    plt.clf()

    plt.plot(frame_idx, raw_dy, label="raw dy")
    if smoothed_homography is not None:
        plt.plot(frame_idx, smoothed_dy, label="smoothed dy")
    plt.title("dy")
    plt.xlabel("frame")
    plt.ylabel("dy")
    plot_filename = path.join(output_dir, sub_prefix + "motion_dy.png")
    plt.legend()
    plt.savefig(plot_filename)
    # plt.show()
    plt.clf()

    plt.plot(frame_idx, raw_x_path, label="raw x path")
    if smoothed_homography is not None:
        plt.plot(frame_idx, smoothed_x_path, label="smoothed x path")
    plt.title("x transform")
    plt.xlabel("frame")
    plt.ylabel("x transform")
    plot_filename = path.join(output_dir, sub_prefix + "motion_x.png")
    plt.legend()
    plt.savefig(plot_filename)
    # plt.show()
    plt.clf()

    plt.plot(frame_idx, raw_y_path, label="raw y path")
    if smoothed_homography is not None:
        plt.plot(frame_idx, smoothed_y_path, label="smoothed y path")
    plt.title("y transform")
    plt.xlabel("frame")
    plt.ylabel("y transform")
    plot_filename = path.join(output_dir, sub_prefix + "motion_y.png")
    plt.legend()
    plt.savefig(plot_filename)
    # plt.show()
    plt.clf()



def find_scenes(video_path, threshold=30.0):
    """
    using scene detect library
    :param video_path:
    :param threshold:
    :return:
    """
    # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))

    # Improve processing speed by downscaling before processing.
    video_manager.set_downscale_factor()

    # Start the video manager and perform the scene detection.
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    # Each returned scene is a tuple of the (start, end) timecode.
    scene_list = scene_manager.get_scene_list()

    result = []
    for i in scene_list:
        result.append(i[0].get_frames())
    return result

