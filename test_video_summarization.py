import unittest
import os
import numpy as np
import cv2
from os import path
import time
import errno
from moviepy.editor import *
from pydub import AudioSegment

import video_summarization


class MyTestCase(unittest.TestCase):
    def setUp(self):
        try:
            os.makedirs("test/")
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

    def test_something(self):
        a = np.array([[1], [2]]).flatten()
        print(a.tolist())

    def test_combine_video_audio(self):
        video_summarization.combine_frames_and_audio("output/concert/combined.mp4", "input/audio/concert.wav", ".")

    def test_SAD(self):
        start = 0
        end = -1
        frames = video_summarization.read_frames_from_folder("input/frames/soccer", start=start, end=end)
        diffs = video_summarization.compute_difference_SAD(frames)
        video_summarization.plot_difference_scores(diffs, plot_filename="test/diff_SAD_soccer.png", start=start)

    """
    test scene breaks
    """
    def test_compute_cut(self):
        frames = video_summarization.read_frames_from_folder("input/frames/soccer")
        cuts = video_summarization.compute_scene_cuts(frames)
        print(cuts)

    def test_break_scene_meridian(self):
        frames = video_summarization.read_frames_from_folder("input/frames/meridian")
        audio_file = AudioSegment.from_file("input/audio/meridian.wav")
        cuts = video_summarization.compute_scene_cuts(frames)
        video_summarization.create_scene_video_from_breaks(frames, audio_file, cuts, "test/meridian")

    def test_break_scene_concert(self):
        frames = video_summarization.read_frames_from_folder("input/frames/concert")
        audio_file = AudioSegment.from_file("input/audio/concert.wav")
        cuts = video_summarization.compute_scene_cuts(frames)
        video_summarization.create_scene_video_from_breaks(frames, audio_file, cuts, "test/concert")

    def test_break_scene_concert_2(self):
        frames = video_summarization.read_frames_from_folder("input/frames/concert_2")
        audio_file = AudioSegment.from_file("input/audio/concert_2.wav")
        cuts = video_summarization.compute_scene_cuts(frames)
        video_summarization.create_scene_video_from_breaks(frames, audio_file, cuts, "test/concert_2")

    def test_break_scene_soccer(self):
        frames = video_summarization.read_frames_from_folder("input/frames/soccer")
        audio_file = AudioSegment.from_file("input/audio/soccer.wav")
        cuts = video_summarization.compute_scene_cuts(frames)
        video_summarization.create_scene_video_from_breaks(frames, audio_file, cuts, "test/soccer")

    def test_break_scene_soccer_2(self):
        frames = video_summarization.read_frames_from_folder("input/frames/soccer_2")
        audio_file = AudioSegment.from_file("input/audio/soccer_2.wav")
        cuts = video_summarization.compute_scene_cuts(frames)
        video_summarization.create_scene_video_from_breaks(frames, audio_file, cuts, "test/soccer_2")

    def test_break_scene_steel(self):
        frames = video_summarization.read_frames_from_folder("input/frames/steel")
        audio_file = AudioSegment.from_file("input/audio/steel.wav")
        cuts = video_summarization.compute_scene_cuts(frames)
        video_summarization.create_scene_video_from_breaks(frames, audio_file, cuts, "test/steel")

    def test_break_scene_superbowl_2(self):
        frames = video_summarization.read_frames_from_folder("input/frames/superbowl_2")
        audio_file = AudioSegment.from_file("input/audio/superbowl_2.wav")
        cuts = video_summarization.compute_scene_cuts(frames)
        video_summarization.create_scene_video_from_breaks(frames, audio_file, cuts, "test/superbowl_2")

    """
    test finding motion characteristics
    """
    def test_compute_distance_between_frames1(self):
        frames = video_summarization.read_frames_from_folder("input/frames/soccer")
        h = video_summarization.compute_homographies_from_frames(frames)
        video_summarization.plot_homographies(h, "test/", prefix="soccer")

    def test_compute_distance_between_frames(self):
        frames = video_summarization.read_frames_from_folder("input/frames/soccer", start=0, end=3000)
        breaks = video_summarization.compute_scene_cuts(frames)
        scenes = video_summarization.create_scene_frames_from_breaks(frames, breaks)
        h = video_summarization.compute_homographies_from_frames(scenes[0])
        video_summarization.plot_homographies(h, "test/", prefix="soccer_scene1")

    def test_movement_score(self):
        frames = video_summarization.read_frames_from_folder("input/frames/soccer", start=0, end=3000)
        breaks = video_summarization.compute_scene_cuts(frames)
        score = video_summarization.compute_movement_score(frames, breaks[0][0], breaks[0][1])
        print(score)

    """
    test finding color characteristics
    """

    """
    test final pipeline
    """
    def test_create_summarized_video(self):
        video_summarization.create_summarized_video_2("input/test_data/frames_test/test_video", "input/test_data/test_video.wav", "test/test_data")

    def test_dataset3(self):
        video_summarization.create_summarized_video_2("input/test_data_3/frames_test/test_video_3", "input/test_data_3/test_video_3.wav", "test/test_data3")


if __name__ == '__main__':
    unittest.main()
