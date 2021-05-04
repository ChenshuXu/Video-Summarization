import unittest
import os
import numpy as np
import scipy as sp
import cv2
from os import path
import time
import ffmpeg
from moviepy.editor import *
import video_summarization


class MyTestCase(unittest.TestCase):
    def setUp(self):
        pass

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

    def test_plot_diff_HD_between_frames(self):
        start = 0
        end = -1
        frames = video_summarization.read_frames_from_folder("input/frames/meridian", start=start, end=end)
        diffs = video_summarization.compute_difference_HD(frames)
        video_summarization.plot_difference_scores(diffs, plot_filename="test/diff_HD_meridian.png", start=start)

    def test_get_scene_break_index(self):
        start = 0
        end = -1
        frames = video_summarization.read_frames_from_folder("input/frames/meridian", start=start, end=end)
        breaks = video_summarization.get_scene_last_frame_index(frames)
        print(breaks)


if __name__ == '__main__':
    unittest.main()
