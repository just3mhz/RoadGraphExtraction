import numpy as np

class Transform:
    def __init__(self, lon0: float, lat0: float, lon1: float, lat1: float, w: int, h: int):
        self.x0 = Transform.to_radians(lon0)
        self.y0 = Transform.to_radians(lat0)
        self.x1 = Transform.to_radians(lon1)
        self.y1 = Transform.to_radians(lat1)
        self.lat0 = self.y0

        self.x0, self.y0 = Transform.equirectangular_projection(self.x0, self.y0, self.lat0)
        self.x1, self.y1 = Transform.equirectangular_projection(self.x1, self.y1, self.lat0)

        self.w = w
        self.h = h

    def __call__(self, lon, lat):
        x = Transform.to_radians(lon)
        y = Transform.to_radians(lat)
        x, y = Transform.equirectangular_projection(x, y, self.lat0)
        x = (x - self.x0) * self.w / (self.x1 - self.x0)
        y = (y - self.y0) * self.h / (self.y1 - self.y0)
        return x, y

    @staticmethod
    def to_radians(degrees):
        return np.pi * degrees / 180.
    
    @staticmethod
    def equirectangular_projection(lon, lat, lat0):
        x = lon * np.cos(lat0)
        y = lat
        return x, y
