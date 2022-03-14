from typing import Optional

from PyQt6.QtCore import Qt, QPointF, QLineF, QRectF, QSizeF
from PyQt6.QtGui import QPainter, QPen
from PyQt6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget

import nodeview


class Edge(QGraphicsItem):
    def __init__(self, source: 'nodeview.NodeView' = None, destination: 'nodeview.NodeView' = None):
        super(QGraphicsItem, self).__init__()
        self._source: 'nodeview.NodeView' = source
        self._destination: 'nodeview.NodeView' = destination
        self._source_point: Optional[QPointF] = None
        self._destination_point: Optional[QPointF] = None

        self._source.add_edge(self)
        self._destination.add_edge(self)

        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

        self.adjust()

    @property
    def source(self):
        return self._source

    @property
    def destination(self):
        return self._destination

    def boundingRect(self) -> QRectF:
        if self._source is None or self._destination is None:
            return QRectF()
        pen_width = 2
        extra = pen_width / 2.

        rect_size = QSizeF(
            self._destination_point.x() - self._source_point.x(),
            self._destination_point.y() - self._source_point.y())

        return QRectF(self._source_point, rect_size).normalized().adjusted(-extra, -extra, extra, extra)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: Optional[QWidget] = None) -> None:
        if self.source is None or self._destination is None:
            return

        line = QLineF(self._source_point, self._destination_point)
        if abs(line.length()) < 0.1:
            return

        painter.setPen(Qt.GlobalColor.yellow)
        painter.drawLine(line)

    def adjust(self):
        if self.source is None or self._destination is None:
            return

        line = QLineF(self.mapFromItem(self._source, 0, 0), self.mapFromItem(self._destination, 0, 0))

        self.prepareGeometryChange()

        self._source_point = line.p1()
        self._destination_point = line.p2()

