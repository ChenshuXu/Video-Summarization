import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, QSlider, QStyle,\
    QSizePolicy, QFileDialog
from PyQt5.QtGui import QIcon, QPalette
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QUrl
import video_summarization


INPUT_FRAMES_FOLDER = "input/frames"
INPUT_AUDIO_FOLDER = "input/audio"
OUT_FOLDER = "output"


class Window(QWidget):

    def __init__(self):

        super().__init__()

        # directory of input frames and audio
        self.frames_folder_dir = None
        self.audio_file_dir = None

        # combined all frames this video
        self.combined_video_file_dir = None

        self.setWindowTitle("Video Summarize Player")
        self.setGeometry(350, 100, 400, 320)
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
        self.loadWavButton = QPushButton('Load Wav File')
        self.loadWavButton.clicked.connect(self.on_clicked_load_audio_button)

        # create open button
        openButton = QPushButton('Open Video')
        openButton.clicked.connect(self.open_file)

        # create button for playing and pausing
        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play_video)

        # create generate button
        self.generateButton = QPushButton()
        self.generateButton.setEnabled(False)
        self.generateButton.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))

        # create slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0,0)
        self.slider.sliderMoved.connect(self.set_position)

        # create label
        self.label = QLabel()
        self.label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        # create HBox layout1
        hBoxLayout1 = QHBoxLayout()
        hBoxLayout1.setContentsMargins(0,0,0,0)

        # set widgets to the HBox layout1
        hBoxLayout1.addWidget(self.loadFramesButton)
        hBoxLayout1.addWidget(self.loadWavButton)
        hBoxLayout1.addWidget(openButton)
        hBoxLayout1.addWidget(self.generateButton)

        # create HBox layout2
        hBoxLayout2 = QHBoxLayout()
        hBoxLayout2.setContentsMargins(0, 0, 0, 0)

        # set widgets to the HBox layout2
        hBoxLayout2.addWidget(self.playButton)
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

    def open_file(self):

        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video", "./")

        if file_name != '':
            print("open video: "+file_name)
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(file_name)))
            self.playButton.setEnabled(True)

    def on_clicked_load_frames_button(self):
        folder_dir = QFileDialog.getExistingDirectory(self, "Open Frame Folder", "./", QFileDialog.ShowDirsOnly)
        print("load frames from :" + folder_dir)
        self.frames_folder_dir = folder_dir

        # start process
        self.load_frames()

    def load_frames(self):
        video_name = self.frames_folder_dir.split("/")[-1]
        output_dir = os.path.join(OUT_FOLDER, video_name)
        return_path = video_summarization.create_video_from_frames_folder(self.frames_folder_dir, output_dir)
        self.combined_video_file_dir = os.path.abspath(return_path)
        print(self.combined_video_file_dir)
        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(self.combined_video_file_dir)))
        self.playButton.setEnabled(True)

    def on_clicked_load_audio_button(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Wave File", "./")
        print("load audio file from :" + file_name)
        self.audio_file_dir = file_name

    def play_video(self):

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
