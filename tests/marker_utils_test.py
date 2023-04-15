"""Tests for marker_utils."""

from absl.testing import absltest
import cv2
import numpy as np

from wellplate_image import marker_utils


class MarkerUtilsTest(absltest.TestCase):

  def test_extract_aruco_rectangle_from_corners(self):
    test_image = cv2.imread('tests/testdata/masked_aruco_markers.jpg')
    assert test_image is not None, "Unable to load test image."
    rectangles = [
      marker_utils.extract_aruco_rectangle_from_corners(
        cv2.rotate(test_image, rot_value) if rot_value is not None
        else test_image
      )
      for rot_value in (
        None,
        cv2.ROTATE_90_CLOCKWISE,
        cv2.ROTATE_90_COUNTERCLOCKWISE,
        cv2.ROTATE_180
      )
    ]
    for r in rectangles[1:]:
      np.testing.assert_array_equal(rectangles[0], r)


  def test_extract_aruco_rectangle_from_edges(self):
    test_image = cv2.imread('tests/testdata/plate_with_mcc_and_aruco.jpg')
    assert test_image is not None, "Unable to load test image."
    rectangles = [
      marker_utils.extract_aruco_rectangle_from_edges(
        cv2.rotate(test_image, rot_value) if rot_value is not None
        else test_image
      )
      for rot_value in (
        None,
        cv2.ROTATE_90_CLOCKWISE,
        cv2.ROTATE_90_COUNTERCLOCKWISE,
        cv2.ROTATE_180
      )
    ]
    for r in rectangles[1:]:
      np.testing.assert_array_equal(rectangles[0], r)

if __name__ == '__main__':
  absltest.main()
