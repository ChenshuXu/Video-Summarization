import sys
import os
import errno
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, QSlider, QStyle,\
    QSizePolicy, QFileDialog
from PyQt5.QtGui import QIcon, QPalette
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QUrl
import video_summarization


class Window(QWidget):

    def __init__(self):

        super().__init__()

        # directory of input frames and audio
        self.frames_folder_abs_dir = None
        self.frames_folder_dir = None
        self.audio_file_abs_dir = None
        self.audio_file_dir = None
        self.output_dir = None

        # directory of generated videos
        self.video_path = None
        self.summarized_video_path = None

        # check the status
        self.generated = False
        self.on_play_video = None

        self.setWindowTitle("Video Summarize Player")
        self.setGeometry(350, 100, 700, 500)
        self.setWindowIcon(QIcon('player.png'))

        p = self.palette()
        p.setColor(QPalette.Window, Qt.gray)
        self.setPalette(p)

        self.init_ui()

        self.show()

    def init_ui(self):

        # create mediaPlayer object
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        # create videoWidget object
        videoWidget = QVideoWidget()

        # create load frames button
        self.loadFramesButton = QPushButton('Load Frames')
        self.loadFramesButton.clicked.connect(self.on_clicked_load_frames_button)

        # create load Wav button
        self.loadWavButton = QPushButton('Load WAV File')
        self.loadWavButton.setEnabled(False)
        self.loadWavButton.clicked.connect(self.on_clicked_load_wav_file_button)

        # create generate button
        self.generateButton = QPushButton('Generate Video')
        self.generateButton.setEnabled(False)
        self.generateButton.clicked.connect(self.on_clicked_generate_video)

        # create process button
        self.processButton = QPushButton('Generate|Summarize')
        self.processButton.setEnabled(False)
        self.processButton.clicked.connect(self.on_clicked_summarize_video)

        # create button for playing and pausing
        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play_video)

        # create button for playing summarized video
        self.playSummarizedButton = QPushButton()
        self.playSummarizedButton.setEnabled(False)
        self.playSummarizedButton.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        self.playSummarizedButton.clicked.connect(self.play_summarized_video)

        # create slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.set_position)

        # create label
        self.label = QLabel()
        self.label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.label.setText('Please choose the frames folder.')

        # create HBox layout1
        hBoxLayout1 = QHBoxLayout()
        hBoxLayout1.setContentsMargins(0, 0, 0, 0)

        # set widgets to the HBox layout1
        hBoxLayout1.addWidget(self.loadFramesButton)
        hBoxLayout1.addWidget(self.loadWavButton)
        hBoxLayout1.addWidget(self.generateButton)
        hBoxLayout1.addWidget(self.processButton)

        # create HBox layout2
        hBoxLayout2 = QHBoxLayout()
        hBoxLayout2.setContentsMargins(0, 0, 0, 0)

        # set widgets to the HBox layout2
        hBoxLayout2.addWidget(self.playButton)
        hBoxLayout2.addWidget(self.playSummarizedButton)
        hBoxLayout2.addWidget(self.slider)

        # create VBox layout
        vBoxLayout = QVBoxLayout()
        vBoxLayout.addWidget(videoWidget)
        vBoxLayout.addLayout(hBoxLayout1)
        vBoxLayout.addLayout(hBoxLayout2)
        vBoxLayout.addWidget(self.label)

        self.setLayout(vBoxLayout)

        self.mediaPlayer.setVideoOutput(videoWidget)

        # media player signals
        self.mediaPlayer.stateChanged.connect(self.media_state_changed)
        self.mediaPlayer.positionChanged.connect(self.position_changed)
        self.mediaPlayer.durationChanged.connect(self.duration_changed)

    def on_clicked_load_frames_button(self):
        self.frames_folder_abs_dir \
            = QFileDialog.getExistingDirectory(self, 'Load Frames', "./", QFileDialog.ShowDirsOnly)
        print(self.frames_folder_abs_dir)
        self.frames_folder_dir = os.path.relpath(self.frames_folder_abs_dir, os.path.curdir)
        print(self.frames_folder_dir)
        self.label.setText('Frames folder path:' + self.frames_folder_dir + ', please load WAV file.')
        self.loadWavButton.setEnabled(True)

    def on_clicked_load_wav_file_button(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load WAV File", "./")
        self.audio_file_abs_dir = file_name
        print(self.audio_file_abs_dir)
        self.audio_file_dir = os.path.relpath(self.audio_file_abs_dir, os.path.curdir)
        print(self.audio_file_dir)
        self.label.setText('WAV file path:' + self.audio_file_dir)
        self.generateButton.setEnabled(True)
        self.processButton.setEnabled(True)

    def video_combination(self):

        video_name = self.frames_folder_dir.split("\\")[-1]
        self.output_dir = os.path.join('output/', video_name)
        print(video_name, self.frames_folder_dir, self.audio_file_dir, self.output_dir)
        try:
            os.makedirs(self.output_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        self.frames = video_summarization.read_frames_from_folder(self.frames_folder_dir)
        video_summarization.create_combined_video(self.frames, self.audio_file_dir, self.output_dir)
        path = os.path.abspath(self.output_dir)
        play_path = os.path.join(path, 'combined.mp4')
        self.video_path = play_path
        self.generated = True
        # self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(self.video_path)))

    def on_clicked_generate_video(self): # for Generate Video btn

        # process
        self.video_combination()
        self.playButton.setEnabled(True)
        self.label.setText('Video generation completed, please click on the play button.')

        # file_name, _ = QFileDialog.getOpenFileName(self, "Open Video", "./")
        #
        # if file_name != '':
        #     print("open video: "+file_name)
        #     self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(file_name)))
        #     self.playButton.setEnabled(True)

    def video_summarization(self):

        if self.generated:
            video_summarization.create_summarized_video_from_combined_video(self.frames, self.video_path,
                                                                            self.audio_file_dir, self.output_dir)
            path = os.path.abspath(self.output_dir)
            play_path = os.path.join(path, 'final.mp4')
            self.summarized_video_path = play_path
        else:
            video_name = self.frames_folder_dir.split("\\")[-1]
            self.output_dir = os.path.join('output/', video_name)
            self.frames = video_summarization.read_frames_from_folder(self.frames_folder_dir)
            video_summarization.create_summarized_video(self.frames, self.audio_file_dir, self.output_dir)
            path = os.path.abspath(self.output_dir)
            play_path = os.path.join(path, 'combined.mp4')
            self.video_path = play_path
            play_path = os.path.join(path, 'final.mp4')
            self.summarized_video_path = play_path

    def on_clicked_summarize_video(self): # for Generate/Summarize btn

        self.video_summarization()
        self.playButton.setEnabled(True)
        self.playSummarizedButton.setEnabled(True)
        self.label.setText('Video generation and summarization completed, please click on the play/computer button.')

    def play_video(self):

        if self.on_play_video != 1:
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(self.video_path)))
            self.on_play_video = 1

        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def play_summarized_video(self):

        if self.on_play_video != 2:
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(self.summarized_video_path)))
            self.on_play_video = 2

        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def media_state_changed(self, state):

        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPause)
            )
        else:
            self.playButton.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay)
            )

    def position_changed(self, position):
        self.slider.setValue(position)

    def duration_changed(self, duration):
        self.slider.setRange(0, duration)

    def set_position(self, position):
        self.mediaPlayer.setPosition(position)

    def handle_errors(self):
        self.playButton.setEnabled(False)
        self.label.setText("Error: " + self.mediaPlayer.errorString())


if __name__ == '__main__':

    app = QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec_())
