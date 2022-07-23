from dataclasses import dataclass

import numpy as np


@dataclass
class Point:
    """Point class to represent a pixel coordinate/2D object dimension in pixel"""

    x: int
    y: int

    @property
    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def surround(self, half_width: int, half_height: int):
        """Generate a bounding box surrounding the point of size (width, height)

        :return: Tuple/Couple of Points as start (top-left) end (bottom-right)
        """
        return Point(self.x - half_width, self.y - half_height), Point(
            self.x + half_width, self.y + half_height
        )

    def translate(self, translation: "Point"):
        return Point(self.x + translation.x, self.y + translation.y)


@dataclass
class ImageDimension(Point):
    @property
    def width(self):
        return self.x

    @property
    def height(self):
        return self.y

    @classmethod
    def from_point(cls, point:Point) -> "ImageDimension":
        return ImageDimension(x=point.x, y=point.y)


@dataclass
class BoundingBox:
    top_left: Point
    bottom_right: Point

    def translate(self, translation: Point):
        return BoundingBox(
            top_left=self.top_left.translate(translation),
            bottom_right=self.bottom_right.translate(translation),
        )

    @property
    def dimension(self) -> ImageDimension:
        """return bounding box dimension"""
        return ImageDimension(*(self.bottom_right.as_array - self.top_left.as_array))


@dataclass
class RatioPoint:
    """Ratio Point class to represent a pixel coordinate
    as a ratio of height and width"""

    # todo add validation
    width: float
    height: float

    @property
    def as_array(self) -> np.ndarray:
        return np.array([self.width, self.height])

    def project(self, image_dimension: ImageDimension) -> Point:
        return Point(
            *np.multiply(self.as_array, image_dimension.as_array - 1).astype("int")
        )

    def surround(self, half_width: float, half_height: float):
        """generate a bounding box surrounding the point of size (width, height)
        :return: Tuple/Couple of Points as start (top-left) end (bottom-right)
        """
        if half_width > 1:
            raise ValueError("half_width should be in [0, 1]")
        if half_height > 1 or half_height < 0:
            raise ValueError("half_height should be in [0, 1]")

        return RatioPoint(
            max(0, self.width - half_width), max(0, self.height - half_height)
        ), RatioPoint(
            min(1, self.width + half_width), min(1, self.height + half_height)
        )


@dataclass
class RatioBoundingBox:
    top_left: RatioPoint
    bottom_right: RatioPoint

    def crop(self, dimension: ImageDimension, offset: ImageDimension):
        bounding_box = BoundingBox(
            top_left=self.top_left.project(dimension),
            bottom_right=self.bottom_right.project(dimension),
        ).translate(offset)
        return (
            bounding_box,
            ImageDimension(*bounding_box.top_left.as_array),
        )


@dataclass
class Crop:
    bbox: RatioBoundingBox
    dimension: ImageDimension

    def project(self):
        return self.bbox.top_left.project(
            self.dimension
        ), self.bbox.bottom_right.project(self.dimension)
