"""Tests for marker_utils."""

from absl.testing import absltest
import numpy as np
from matplotlib import pyplot as plt

from wellplate_image import plate_utils

_WIKIPEDIA_PLATE = (
  'tests/testdata/perspective_transformed_elisa_tmb_wikipedia.jpg')

# Generated images are shown if the test is directly run from command
# line, instead of via pytest.
_SHOW_TEST_IMAGES = (__name__ == '__main__')


class MarkerUtilsTest(absltest.TestCase):

  def test_smoke_find_circles_and_match_grid_to_well_centers(self):
    test_image = plt.imread(_WIKIPEDIA_PLATE)
    plate = plate_utils.PLATE_96_WELL
    pixels_per_mm = plate_utils.estimate_pixels_per_mm(test_image)
    centers, median_radius = plate_utils.find_circles_on_plate(
      test_image, pixels_per_mm=pixels_per_mm,
      hough_circles_acc_thresh=15,
      hough_circles_canny_thresh=20,
    )
    approximate_grid =  plate_utils.get_approximate_plate_grid(
      test_image, pixels_per_mm)
    matched_centers = plate_utils.match_grid_to_well_center_estimates(
      approximate_grid, centers.reshape(-1, 2), pixels_per_mm)
    if _SHOW_TEST_IMAGES:
      import cv2
      for p in matched_centers.reshape(-1, 2):
        test_image=cv2.circle(
          test_image,
          p.astype(int),
          color=(255, 0, 0),
          radius=int(median_radius),
          thickness=5,
        )
      plt.imshow(test_image)
      plt.show()


  def test_extract_pixels_from_circles(self):
    test_image = np.linspace(0, 1, 100 * 100 * 3).reshape((100, 100, 3))
    centers = np.array([[10.1, 15.9], [20.1, 28.9], [80.5, 30.5]])
    radius = 4.5
    pixels = plate_utils.extract_pixels_from_circles(
      test_image, centers, radius
    )
    self.assertLen(pixels, 3)
    # Check that the pixels are within the circle.
    for c, pixels_c in zip(centers, pixels):
      c_x, c_y = c.round().astype(np.int64)
      min_sums_center = np.sum(
        test_image[c_y - 3:c_y + 3, c_x - 3:c_x + 3], axis=(0, 1))
      max_sums_center = np.sum(
        test_image[c_y - 5:c_y + 5, c_x - 5:c_x + 5], axis=(0, 1))
      sum_pixels = np.sum(pixels_c, axis=0)
      np.testing.assert_array_less(min_sums_center, sum_pixels)
      np.testing.assert_array_less(sum_pixels, max_sums_center)


if __name__ == '__main__':
  absltest.main()
