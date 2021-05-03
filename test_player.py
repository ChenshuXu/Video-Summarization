import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, QSlider, QStyle,\
    QSizePolicy, QFileDialog
from PyQt5.QtGui import QIcon, QPalette
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QUrl


class Window(QWidget):

    def __init__(self):

        super().__init__()

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

        # create load Wav button
        self.loadWavButton = QPushButton('Load Wav File')

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

        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video")

        if file_name != '':
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(file_name)))
            self.playButton.setEnabled(True)

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
