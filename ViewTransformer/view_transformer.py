import numpy.typing as npt
import numpy as np
import cv2
from typing import Tuple

class ViewTransformer:
  def __init__(self,
               source: npt.NDArray[np.float32],
               target: npt.NDArray[np.float32]) -> None:
    if source.shape != target.shape:
      raise ValueError("The shape of the source and target must be the same.")
    if source.shape[1] != 2:
      raise ValueError("The dimension of the source[1] must be 2")
    source = source.astype(np.float32)
    target = target.astype(np.float32)
    self.m, _ = cv2.findHomography(source, target)
    if self.m is None:
      raise ValueError("Homography of source and target matrix couldn't be calculated.")

  def transform_points(self,points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    if points.size == 0:
      return points
    if points.shape[1] != 2:
      raise ValueError("Points must be 2D coordinates")

    reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
    transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
    return transformed_points.reshape(-1, 2).astype(np.float32)

  def transform_image(self,
                      image: npt.NDArray[np.uint8],
                      resolution_wh: Tuple[int, int]
                      ) -> npt.NDArray[np.uint8]:
    if len(image.shape) not in {2,3}:
      raise ValueError("Image must be either grayscale or color.")
    return cv2.warpPerspective(image, self.m, resolution_wh)