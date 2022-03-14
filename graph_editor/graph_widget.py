import pickle
from typing import List, Optional, Dict

from PyQt6.QtCore import QTimerEvent, QRectF, QDir, QPointF
from PyQt6.QtGui import QKeyEvent, QPainter, QMouseEvent, QPixmap
from PyQt6.QtWidgets import QWidget, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QFileDialog

from nodeview import NodeView
from edge import Edge


class GraphWidget(QGraphicsView):
    def __init__(self, parent: QWidget = None):
        super(GraphWidget, self).__init__(parent)
        self._scene = QGraphicsScene(self)
        self._scene.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)
        self.setScene(self._scene)
        self.setCacheMode(QGraphicsView.CacheModeFlag.CacheBackground)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setWindowTitle("Elastic Nodes")

        self._edges: List['Edge'] = []
        self._nodes: Dict[(int, int), 'NodeView'] = {}
        self._background_image: Optional[QPixmap] = None

    def keyPressEvent(self, event: QKeyEvent) -> None:
        match event.key():
            case _:
                super(GraphWidget, self).keyPressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        node = NodeView(self)
        node.setPos(self.mapToScene(event.pos()))
        self._scene.addItem(node)

    def item_moved(self):
        pass

    def timerEvent(self, a0: QTimerEvent) -> None:
        super(GraphWidget, self).timerEvent(a0)

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        super(GraphWidget, self).drawBackground(painter, rect)

    def scale_view(self, scale_factor):
        factor = self.transform().scale(scale_factor, scale_factor).mapRect(QRectF(0, 0, 1, 1)).width()
        if not 0.7 < factor < 100:
            return
        self.scale(scale_factor, scale_factor)

    def load_tile(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load tile", QDir.currentPath(), filter='Images (*.png)')
        if not file_name:
            return
        self._set_image(file_name)

    def _set_image(self, path):
        image = QPixmap(path)
        if self._background_image is not None:
            self._scene.removeItem(image)
        self._background_image = QGraphicsPixmapItem(image)
        pos = self._scene.sceneRect().topLeft()
        self._background_image.setPos(pos)
        self._background_image.setZValue(-10)
        self._scene.addItem(self._background_image)
        self.setMaximumSize(image.width(), image.height())

    def load_graph(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load tile", QDir.currentPath(), filter='(*.pickle)')
        with open(file_name, 'rb') as graph_fo:
            graph = pickle.load(graph_fo)
        self.clean_graph_layer()
        for (yu, xu), _ in graph.items():
            node = NodeView.create_node(self, QPointF(xu, yu))
            self._scene.addItem(node)
            self._nodes[(xu, yu)] = node
        for (yu, xu), neighbours in graph.items():
            for (yv, xv) in neighbours:
                self._edges.append(Edge(self._nodes[(xu, yu)], self._nodes[(xv, yv)]))
                self._scene.addItem(self._edges[-1])

    def clean_graph_layer(self):
        for item in self._nodes.values():
            self._scene.removeItem(item)
        self._nodes = {}

        for item in self._edges:
            self._scene.removeItem(item)
        self._edges = []
