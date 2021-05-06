import unittest
import os
import numpy as np
import scipy as sp
import cv2
from os import path
import time
import errno
import ffmpeg
from moviepy.editor import *
import video_summarization


class MyTestCase(unittest.TestCase):
    def setUp(self):
        try:
            os.makedirs("test/")
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

    def test_combine_video_audio(self):
        video_summarization.combine_frames_and_audio("output/concert/combined.mp4", "input/audio/concert.wav", ".")

    def test_SAD(self):
        frame76 = cv2.imread("test/frame76.jpg")
        frame77 = cv2.imread("test/frame77.jpg")
        frame78 = cv2.imread("test/frame78.jpg")
        diff_76_77 = video_summarization.compute_SAD_between_two_frames(frame76, frame77)
        diff_77_78 = video_summarization.compute_SAD_between_two_frames(frame77, frame78)
        print(diff_76_77)
        print(diff_77_78)

    def test_HD(self):
        frame76 = cv2.imread("test/frame76.jpg")
        frame77 = cv2.imread("test/frame77.jpg")
        frame78 = cv2.imread("test/frame78.jpg")
        diff_76_77 = video_summarization.compute_HD_between_two_frames(frame76, frame77)
        diff_77_78 = video_summarization.compute_HD_between_two_frames(frame77, frame78)
        print(diff_76_77)
        print(diff_77_78)

    def test_plot_diff_SAD_between_frames(self):
        start = 0
        end = -1
        frames = video_summarization.read_frames_from_folder("input/frames/soccer", start=start, end=end)
        diffs = video_summarization.compute_difference_SAD(frames)
        video_summarization.plot_difference_scores(diffs, plot_filename="test/diff_SAD_soccer.png", start=start)

    def test_plot_diff_SAD_edge_between_frames(self):
        start = 0
        end = -1
        frames = video_summarization.read_frames_from_folder("input/frames/meridian", start=start, end=end)
        diffs = video_summarization.compute_difference_SAD_with_edge(frames)
        video_summarization.plot_difference_scores(diffs, plot_filename="test/diff_SAD_edge_meridian.png", start=start)

    def test_plot_diff_HD_between_frames(self):
        start = 0
        end = -1
        frames = video_summarization.read_frames_from_folder("input/frames/meridian", start=start, end=end)
        diffs = video_summarization.compute_difference_HD(frames)
        video_summarization.plot_difference_scores(diffs, plot_filename="test/diff_HD_meridian.png", start=start)

    def test_plot_diff_HD_edge_between_frames(self):
        start = 0
        end = -1
        frames = video_summarization.read_frames_from_folder("input/frames/meridian", start=start, end=end)
        diffs = video_summarization.compute_difference_HD_with_edge(frames)
        video_summarization.plot_difference_scores(diffs, plot_filename="test/diff_HD_edge_meridian.png", start=start)

    def test_compute_difference_distance(self):
        frames = video_summarization.read_frames_from_folder("input/frames/concert")
        diffs = video_summarization.compute_difference_distance(frames)
        video_summarization.plot_difference_scores(diffs, plot_filename="test/diff_distance_concert.png")

    def test_break_scene_meridian(self):
        frames = video_summarization.read_frames_from_folder("input/frames/meridian")
        cuts = video_summarization.compute_scene_idx(frames)
        video_summarization.create_scene_video_from_breaks(frames, cuts, "test/meridian")

    def test_break_scene_concert(self):
        frames = video_summarization.read_frames_from_folder("input/frames/concert")
        cuts = video_summarization.compute_scene_idx(frames)
        video_summarization.create_scene_video_from_breaks(frames, cuts, "test/concert")

    def test_break_scene_soccer(self):
        frames = video_summarization.read_frames_from_folder("input/frames/soccer")
        cuts = video_summarization.compute_scene_idx(frames)
        video_summarization.create_scene_video_from_breaks(frames, cuts, "test/soccer")

    def test_compute_cut(self):
        frames = video_summarization.read_frames_from_folder("input/frames/soccer")
        cuts = video_summarization.compute_scene_idx(frames)
        print(cuts)

    def test_something(self):
        a = np.array([[1], [2]]).flatten()
        print(a.tolist())

    def test_find_scene(self):
        breaks = video_summarization.find_scenes("output/soccer/combined.mp4")
        frames = video_summarization.read_frames_from_folder("input/frames/soccer")
        video_summarization.create_scene_video_from_breaks(frames, breaks, "test/soccer_PySceneDetect")

    def test_compute_distance_between_frames1(self):
        frames = video_summarization.read_frames_from_folder("input/frames/soccer")
        h = video_summarization.compute_homographies_from_frames(frames)
        video_summarization.plot_homographies(h, "test/", prefix="soccer")

    def test_compute_distance_between_frames(self):
        frames = video_summarization.read_frames_from_folder("input/frames/soccer")
        breaks = video_summarization.find_scenes("output/soccer/combined.mp4")
        scenes = video_summarization.create_scene_frames_from_breaks(frames, breaks)
        h = video_summarization.compute_homographies_from_frames(scenes[1])
        video_summarization.plot_homographies(h, "test/", prefix="soccer_scene1")

if __name__ == '__main__':
    unittest.main()
