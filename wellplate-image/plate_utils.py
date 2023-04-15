"""Utilities for understanding plates."""
import collections
import dataclasses

import cv2
import numpy as np
import jaxtyping as jt
from sklearn import neighbors

Image = jt.Num[np.ndarray, 'rows cols depth']


@dataclasses.dataclass(frozen=True)
class Plate:
  num_rows: int
  num_cols: int
  # This is in the direction of the columns.
  length_mm: float
  # This is in the direction of the rows.
  width_mm: float
  well_spacing_mm: float
  min_swatch_diameter_mm: float
  max_swatch_diameter_mm: float
  # positions are in (x, y); and should be scaled to pixels via `pixels_per_mm`.
  estimated_upper_left_position_mm: jt.Float[np.ndarray, '2']
  estimated_lower_right_position_mm: jt.Float[np.ndarray, '2']


PLATE_96_WELL = Plate(
  num_rows=8,
  num_cols=12,
  length_mm=127.76,
  width_mm=85.48,
  well_spacing_mm=9.0,
  min_swatch_diameter_mm=3.0,
  max_swatch_diameter_mm=5.0,
  estimated_upper_left_position_mm=np.array([16.0, 12.0]),
  estimated_lower_right_position_mm=np.array([111.0, 73.5]),
)


@dataclasses.dataclass(frozen=True)
class ApproximatePlateGrid:
  """Stores plate grid locations in pixel units."""
  upper_left_point: jt.Float[np.ndarray, '2']
  lower_right_point: jt.Float[np.ndarray, '2']
  well_delta_x: float
  well_delta_y: float
  x_indices: jt.Integer[np.ndarray, 'num_plate_cols']  # noqa: F821
  y_indices: jt.Integer[np.ndarray, 'num_plate_rows']  # noqa: F821
  grid_points: jt.Float[np.ndarray, 'num_plate_cols num_plate_rows 2']


def estimate_pixels_per_mm(
    image: Image,
    plate: Plate = PLATE_96_WELL
) -> float:
  width_pixels = image.shape[0]
  length_pixels = image.shape[1]
  return np.mean([
    length_pixels / plate.length_mm,
    width_pixels / plate.width_mm,
  ])


def get_approximate_plate_grid(
    image: Image,
    pixels_per_mm: float,
    plate: Plate = PLATE_96_WELL,
) -> ApproximatePlateGrid:
  """Returns a grid of (x, y) positions of an estimated plate grid."""
  upper_left = plate.estimated_upper_left_position_mm * pixels_per_mm
  lower_right = plate.estimated_lower_right_position_mm * pixels_per_mm
  well_delta_x = (lower_right[0] - upper_left[0]) / (plate.num_cols - 1)
  well_delta_y = (lower_right[1] - upper_left[1]) / (plate.num_rows - 1)
  x_indices, y_indices = np.meshgrid(
      range(plate.num_cols), range(plate.num_rows))
  x_indices = x_indices.ravel()
  y_indices = y_indices.ravel()
  x_positions = upper_left[0] + well_delta_x * x_indices.ravel()
  y_positions = upper_left[1] + well_delta_y * y_indices.ravel()
  xy_positions = np.stack((x_positions, y_positions), axis=-1)
  return ApproximatePlateGrid(
      upper_left_point=upper_left,
      lower_right_point=lower_right,
      well_delta_x=well_delta_x,
      well_delta_y=well_delta_y,
      x_indices=x_indices,
      y_indices=y_indices,
      grid_points=xy_positions.reshape((plate.num_cols, plate.num_rows, 2)),
  )


def find_circles_on_plate(
    image: Image,
    pixels_per_mm: float,
    hough_circles_canny_thresh: int = 30,
    hough_circles_acc_thresh: int = 20,
    plate: Plate = PLATE_96_WELL,
) -> tuple[
    jt.Float[np.ndarray, 'num_circles 2'],
    float
]:
  """Attempt to find some circles on the plate.

  This function first converts the RGB `image` to XYZ colorspace,
  extracts the `Z` grayscale image, then attempts to find circles
  using the Hough Circle transform, using constraints on expected well
  sizes given by `plate`.

  The parameters `hough_circles_param1` and `hough_circles_param2` control
  the precision/recall trade-off.  Smaller values tend to increase precision
  at the risk of more false negatives, while larger values tend to increase
  recall at the risk of more false positives.

  If not enough circles are found, try changing `param1` first; e.g., increase
  from the default value to `50`.  If too many are found, decrease it, e.g., to
  `20`.

  In general, it is okay to have many false negatives and false positives;
  the method `match_grid_to_well_center_estimates` generally needs only
  about 5 true positives and throws out obvious outliers.

  Args:
    image: The extracted image of the well plate.
    pixels_per_mm: How many pixels of `image` constitute a millimeter.
    hough_circles_canny_thresh: Argument to the Canny detector in the
      Hough Circles method.  Values in `[0, 100]`.
    hough_circles_acc_thresh: Argument to the accumulation threshold in the
      Hough Circles method.  Values in `[0, 100]`.
    plate: The parameters of the plate being imaged.

  Returns:
    The tuple `(circle_center_points, median_radius)`.

  Raises:
    ValueError: If no circles were found.
  """
  # Convert image to XYZ and extract the 'Z' color axis.
  image_z = cv2.cvtColor(image, cv2.COLOR_RGB2XYZ)[:, :, 2]
  min_dist_between = np.floor(
      (plate.well_spacing_mm - 1) * pixels_per_mm).astype(int)

  min_radius = np.floor(
      plate.min_swatch_diameter_mm / 2 * pixels_per_mm
  ).astype(int)
  max_radius = np.ceil(
      (0.5 * (plate.min_swatch_diameter_mm + plate.max_swatch_diameter_mm))
      / 2 * pixels_per_mm
  ).astype(int)
  circles = cv2.HoughCircles(
      image=image_z,
      method=cv2.HOUGH_GRADIENT,
      dp=1,
      minDist=min_dist_between,
      param1=hough_circles_canny_thresh,
      param2=hough_circles_acc_thresh,
      minRadius=min_radius,
      maxRadius=max_radius,
  )

  if circles is None:
    raise ValueError('Unable to find any well circles in image')

  circles = np.squeeze(circles)
  return circles[:, :2], np.median(circles[:, 2])


def match_grid_to_well_center_estimates(
    approximate_grid: ApproximatePlateGrid,
    well_center_estimates: jt.Float[np.ndarray, 'num_centers 2'],
    pixels_per_mm: float,
    plate: Plate = PLATE_96_WELL,
) -> jt.Float[np.ndarray, 'num_rows num_cols 2']:
  """Transforms the approximate grid to better match well centers.

  This uses a homography transform over nearest neighbors between approximate
  grid points and a list of noisy well centers as identified by
  `find_circles_on_plate`.

  Args:
    approximate_grid: The approximate grid.  Output of
      `get_approximate_plate_grid`.
    well_centers: The well centers, e.g., output of
      `find_circles_on_plate`.
    pixels_per_mm: The number of pixels per mm in the image.

  Returns:
    The transformed grid points, for each well in the plate.
  """
  num_rows, num_cols = approximate_grid.grid_points.shape[:2]
  flat_grid_points = approximate_grid.grid_points.reshape((-1, 2))

  neighbor_search = neighbors.NearestNeighbors(n_neighbors=1).fit(
      flat_grid_points)
  distance_to_grid_neighbor, grid_neighbor_index = neighbor_search.kneighbors(
      well_center_estimates)
  distance_to_grid_neighbor = np.squeeze(distance_to_grid_neighbor)
  grid_neighbor_index = np.squeeze(grid_neighbor_index)
  grid_neighbor_cutoff = (
      (plate.max_swatch_diameter_mm / 3) * pixels_per_mm)
  true_well_neighbor_indices = np.where(
      distance_to_grid_neighbor < grid_neighbor_cutoff)[0]
  true_distance_to_grid_neighbor = distance_to_grid_neighbor[
      true_well_neighbor_indices
  ]
  true_grid_neighbor_index = grid_neighbor_index[true_well_neighbor_indices]

  # When there is more than one well associated with a grid point, remove all
  # but the closest.
  grid_to_well_neighbors = (
      collections.defaultdict[int, list[tuple[float, int]]](list))
  for grid_index, well_index, distance in zip(
      true_grid_neighbor_index,
      true_well_neighbor_indices,
      true_distance_to_grid_neighbor
  ):
    grid_to_well_neighbors[grid_index].append((distance, well_index))

  chosen_grid_well_pairs: list[tuple[int, int]] = []
  for grid_index, nearest_wells in grid_to_well_neighbors.items():
    sorted_nearest_wells = sorted(nearest_wells, key=lambda x: x[0])
    chosen_grid_well_pairs.append((grid_index, sorted_nearest_wells[0][1]))

  # Pull out the grid and well points that are associated with each other.
  chosen_grid_points = flat_grid_points[
      [grid_ix for grid_ix, _ in chosen_grid_well_pairs]
  ]
  chosen_well_points = well_center_estimates[
      [well_ix for _, well_ix in chosen_grid_well_pairs]
  ]

  transformation_matrix, inlier_mask = cv2.findHomography(
      srcPoints=chosen_grid_points,
      dstPoints=chosen_well_points,
      method=cv2.RHO)

  # Apply the transformation to the grid points to get the well points.
  transformed_grid_points = cv2.perspectiveTransform(
      src=np.expand_dims(flat_grid_points, axis=0),
      m=transformation_matrix)

  transformed_grid_points = np.squeeze(transformed_grid_points)
  return transformed_grid_points.reshape((num_rows, num_cols, 2))


def extract_pixels_from_circles(
    image: Image,
    centers: jt.Float[np.ndarray, 'num_well_centers 2'],
    radius: float
) -> list[jt.Float[np.ndarray, 'pixels_in_well 3']]:
  """Extracts pixels from each well in the grid.

  Args:
    image: The image to extract pixels from.
    centers: The locations of the wells, e.g.,
      `match_grid_to_well_center_estimates(...).reshape((-1, 2))`.
    radius: The radius of the wells, in pixels.

  Returns:
    A list containing the pixels in each well; length `num_well_centers`.
    Each entry has rows equal to the number of pixels identified in that
    well, and each row contains the associated pixel.
  """
  pixels_in_wells = []
  for well_location in centers:
    # Create a circle mask around well_location
    pixels_including_circle = np.meshgrid(
        np.arange(int(well_location[0] - radius),
                  1 + int(well_location[0] + radius)),
        np.arange(int(well_location[1] - radius),
                  1 + int(well_location[1] + radius))
    )
    pixels_including_circle = (
      np.stack(pixels_including_circle, axis=-1)
      .reshape((-1, 2)))
    circle_mask = np.where(
      np.linalg.norm(
        pixels_including_circle - well_location, axis=-1)
      <= radius)[0]
    circle_x_positions, circle_y_positions = tuple(
      pixels_including_circle[circle_mask].reshape((-1, 2)).T
    )
    pixels_in_wells.append(image[circle_y_positions, circle_x_positions])
  return pixels_in_wells
