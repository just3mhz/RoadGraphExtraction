from typing import List, Optional, Any

from PyQt6.QtGui import QPainterPath, QPainter, QPen
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtWidgets import QGraphicsItem
from PyQt6.QtWidgets import QGraphicsSceneMouseEvent
from PyQt6.QtWidgets import QStyleOptionGraphicsItem
from PyQt6.QtWidgets import QWidget

import edge
import graph_widget


class NodeView(QGraphicsItem):
    def __init__(self, widget: 'graph_widget.GraphWidget'):
        super(QGraphicsItem, self).__init__()

        self._graph_widget: 'graph_widget.GraphWidget' = widget
        self._edges: List['edge.Edge'] = []
        self._position: Optional[QPointF] = None

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
        self.setZValue(-1)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self.update()
        super(NodeView, self).mouseMoveEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self.update()
        super(NodeView, self).mousePressEvent(event)

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value: Any) -> Any:
        match change:
            case QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
                for edge in self._edges:
                    edge.adjust()
                self._graph_widget.item_moved()
            case _:
                pass
        return super(NodeView, self).itemChange(change, value)

    def boundingRect(self) -> QRectF:
        return QRectF(-5, -5, 5, 5)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addEllipse(QRectF(-5, -5, 5, 5))
        return path

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: Optional[QWidget] = None) -> None:
        painter.setBrush(Qt.GlobalColor.yellow)
        painter.setPen(QPen(Qt.GlobalColor.black, 0))
        painter.drawEllipse(-5, -5, 5, 5)

    def type(self) -> int:
        return QGraphicsItem.UserType + 1

    def add_edge(self, edge: 'edge.Edge'):
        self._edges.append(edge)
        edge.adjust()

    @property
    def edges(self):
        return self._edges

    def advance_position(self):
        if self._position == self.pos():
            return False
        self.setPos(self._position)
        return True

    @staticmethod
    def create_node(parent: 'graph_widget.GraphWidget', pos: 'QPointF') -> 'NodeView':
        """
        Create node and add it to scene
        """
        node = NodeView(parent)
        node.setPos(pos)
        return node
