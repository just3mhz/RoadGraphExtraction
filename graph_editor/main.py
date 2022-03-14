import sys

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget

from ui_mainwindow import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent: QWidget = None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

    def load_tile(self):
        self.main_scene.load_tile()

    def load_graph(self):
        self.main_scene.load_graph()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec()
