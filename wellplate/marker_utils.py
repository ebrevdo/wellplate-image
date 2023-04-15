"""Utilities for finding markers (e.g., ArUCO)."""

import cv2
from cv2 import aruco
from imutils import perspective
import numpy as np
from typing import Optional

import jaxtyping as jt

Image = jt.Integer[np.ndarray, 'rows cols depth']
Positions = tuple[jt.Float[np.ndarray, '4 2'], ...]
Ids = jt.Integer[np.ndarray, 'n_markers']

# Indices into the list returned by `get_sorted_indices_corners`.
_TOP_LEFT_IX = 0
_BOTTOM_LEFT_IX = 1
_TOP_RIGHT_IX = 2
_BOTTOM_RIGHT_IX = 3

# Indices into the list returned by `get_sorted_indices_edges`.
_LEFT_IX = 0
_TOP_IX = 1
_BOTTOM_IX = 2
_RIGHT_IX = 3

# When the ArUCO ids are in the corners of the rectangle.
_DEFAULT_UPPER_LEFT_ARUCO_ID = 0

# When the ArUCO ids are at the edges of the rectangle.
_DEFAULT_TOP_ARUCO_ID = 0


def find_rectangle_aruco_markers(
    image: Image,
    dictionary_code: int = aruco.DICT_4X4_50,
) -> tuple[Positions, Ids]:
  """Find ArUCO markers of a rectangle in an image."""
  parameters = aruco.DetectorParameters()
  parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
  dictionary = aruco.getPredefinedDictionary(dictionary_code)
  detector = aruco.ArucoDetector(dictionary, parameters)
  corners, ids, rejected = detector.detectMarkers(image)
  if len(corners) != 4:
    raise ValueError(
      f'Found {len(corners)} ArUCO markers, expected 4: {ids}. '
      f'Rejected: {rejected}.')
  return tuple(np.squeeze(c) for c in corners), np.squeeze(ids)


def extract_aruco_rectangle_from_corners(
    image: Image,
    dictionary_code: int = aruco.DICT_4X4_50,
    upper_left_aruco_id: int = _DEFAULT_UPPER_LEFT_ARUCO_ID,
):
  """Extract the rectangle defined by ArUCO markers from an image.
  
  The markers are assumed to be at the corners of the rectangle.
  """
  corners, ids = find_rectangle_aruco_markers(
      image, dictionary_code=dictionary_code)
  assert upper_left_aruco_id in ids, "Unable to find upper-left marker in ids."
  rotation_code = get_rotation_code_for_corners(
    corners, ids, upper_left_aruco_id=upper_left_aruco_id)
  if rotation_code is not None:
    rotated_image = cv2.rotate(image, rotation_code)
  else:
    rotated_image = image
  rotated_corners, rotated_ids = find_rectangle_aruco_markers(
      rotated_image, dictionary_code=dictionary_code)

  # identify the inner corner coordinates and extract the rectangle.
  rotated_corner_centers = np.array(
      [np.mean(c, axis=0) for c in rotated_corners])
  sorted_indices = rectangle_sorted_indices_corners(rotated_corner_centers)
  new_corners = [rotated_corners[ix] for ix in sorted_indices]
  new_ids = rotated_ids[sorted_indices]
  assert new_ids[_TOP_LEFT_IX] == upper_left_aruco_id

  inner_corners = np.array([
      new_corners[_TOP_LEFT_IX][
          rectangle_sorted_indices_corners(
            new_corners[_TOP_LEFT_IX]
          )[_BOTTOM_RIGHT_IX]
      ],
      new_corners[_TOP_RIGHT_IX][
          rectangle_sorted_indices_corners(
            new_corners[_TOP_RIGHT_IX]
          )[_BOTTOM_LEFT_IX]
      ],
      new_corners[_BOTTOM_LEFT_IX][
          rectangle_sorted_indices_corners(
            new_corners[_BOTTOM_LEFT_IX]
          )[_TOP_RIGHT_IX]
      ],
      new_corners[_BOTTOM_RIGHT_IX][
          rectangle_sorted_indices_corners(
            new_corners[_BOTTOM_RIGHT_IX]
          )[_TOP_LEFT_IX]
      ],
  ])

  rectangle = four_point_transform(rotated_image, inner_corners)

  return rectangle


def extract_aruco_rectangle_from_edges(
    image: Image,
    dictionary_code: int = aruco.DICT_4X4_50,
    top_aruco_id: int = _DEFAULT_TOP_ARUCO_ID,
):
  """Extract the rectangle defined by ArUCO markers from an image.
  
  The markers are assumed to be at the edges of the rectangle.
  """
  edges, ids = find_rectangle_aruco_markers(
      image, dictionary_code=dictionary_code)
  assert top_aruco_id in ids, "Unable to find top marker in ids."
  rotation_code = get_rotation_code_for_edges(
    edges, ids, top_aruco_id=top_aruco_id)
  if rotation_code is not None:
    rotated_image = cv2.rotate(image, rotation_code)
  else:
    rotated_image = image
  rotated_edges, rotated_ids = find_rectangle_aruco_markers(
      rotated_image, dictionary_code=dictionary_code)  # rotated_image

  # identify the inner corner coordinates and extract the rectangle.
  rotated_edge_centers = np.array(
      [np.mean(c, axis=0) for c in rotated_edges])
  sorted_indices = rectangle_sorted_indices_edges(rotated_edge_centers)
  (ix_left, ix_top, ix_bottom, ix_right) = sorted_indices
  new_edges = [rotated_edges[ix] for ix in sorted_indices]
  new_ids = rotated_ids[sorted_indices]
  assert new_ids[_TOP_IX] == top_aruco_id

  bottom_edge_ids = np.array([_BOTTOM_LEFT_IX, _BOTTOM_RIGHT_IX])
  right_edge_ids = np.array([_BOTTOM_RIGHT_IX, _TOP_RIGHT_IX])
  top_edge_ids = np.array([_TOP_LEFT_IX, _TOP_RIGHT_IX])
  left_edge_ids = np.array([_BOTTOM_LEFT_IX, _TOP_LEFT_IX])

  inner_corners = np.array([
    # Top left point
    intersecting_point(
      new_edges[_TOP_IX][
          rectangle_sorted_indices_corners(new_edges[_TOP_IX])[bottom_edge_ids]
      ],
      new_edges[_LEFT_IX][
          rectangle_sorted_indices_corners(new_edges[_LEFT_IX])[right_edge_ids]
      ]),
    # Top right point
    intersecting_point(
      new_edges[_TOP_IX][
          rectangle_sorted_indices_corners(new_edges[_TOP_IX])[bottom_edge_ids]
      ],
      new_edges[_RIGHT_IX][
          rectangle_sorted_indices_corners(new_edges[_RIGHT_IX])[left_edge_ids]
      ]),
    # Bottom left point
    intersecting_point(
      new_edges[_BOTTOM_IX][
          rectangle_sorted_indices_corners(new_edges[_BOTTOM_IX])[top_edge_ids]
      ],
      new_edges[_LEFT_IX][
          rectangle_sorted_indices_corners(new_edges[_LEFT_IX])[right_edge_ids]
      ]),
    # Bottom right point
    intersecting_point(
      new_edges[_BOTTOM_IX][
          rectangle_sorted_indices_corners(new_edges[_BOTTOM_IX])[top_edge_ids]
      ],
      new_edges[_RIGHT_IX][
          rectangle_sorted_indices_corners(new_edges[_RIGHT_IX])[left_edge_ids]
      ]),
  ])
  rectangle = four_point_transform(rotated_image, inner_corners)

  return rectangle


def intersecting_point(
  line1: jt.Float[np.ndarray, '2 2'],
  line2: jt.Float[np.ndarray, '2 2'],
) -> jt.Float[np.ndarray, '2']:
  """Calculate the point of intersection between lines line1 and line2.
  
  Args:
    line1: An array of shape (2, 2), each row is a point (line1_0, line1_1),
      and the columns are (x, y) coordinates.
    line2: An array of shape (2, 2), each row is a point (line2_0, line2_1),
      and the columns are (x, y) coordinates.
  """
  x1, y1, x2, y2 = line1.ravel()
  x3, y3, x4, y4 = line2.ravel()
  denominator = ((x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4))
  if abs(denominator) < 1e-6:
    raise ValueError('Lines are parallel or coincident')
  px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / denominator
  py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denominator
  return np.array([px, py])


def get_rotation_code_for_corners(
  corners: Positions,
  ids: Ids,
  upper_left_aruco_id: int
) -> Optional[int]:
  corner_centers = np.array([np.mean(c, axis=0) for c in corners])
  upper_left_corner_center = corner_centers[
      np.where(ids == upper_left_aruco_id)[0][0]]
  # Identify center of rectangle.
  center = np.mean(corner_centers, axis=0)

  # Rotate the image so that the upper-left corner is in the upper-left.
  if upper_left_corner_center[0] < center[0]:
    if upper_left_corner_center[1] < center[1]:
      # Top Left
      return None
    else:
      # Top right
      return cv2.ROTATE_90_CLOCKWISE
  else:
    if upper_left_corner_center[1] < center[1]:
      # Bottom left
      return cv2.ROTATE_90_COUNTERCLOCKWISE
    else:
      # Bottom right
      return cv2.ROTATE_180


def get_rotation_code_for_edges(
  edges: Positions,
  ids: Ids,
  top_aruco_id: int
) -> Optional[int]:
  edge_centers = np.array([np.mean(c, axis=0) for c in edges])
  top_edge_center = edge_centers[np.where(ids == top_aruco_id)[0][0]]
  # Identify center of rectangle.
  center = np.mean(edge_centers, axis=0)

  # Rotate the image so that the upper-left edge is in the upper-left.
  top_edge_to_center = np.abs(top_edge_center - center)
  if top_edge_to_center[0] > top_edge_to_center[1]:
    # Top edge is closer to the center vertically than horizontally;
    # thus top edge is actually on the left or right.
    if top_edge_center[0] < center[0]:
      # Left
      return cv2.ROTATE_90_CLOCKWISE
    else:
      # Right
      return cv2.ROTATE_90_COUNTERCLOCKWISE
  else:
    # Top edge is closer to the center horizontally than vertically;
    # thus top edge is actually on the top or bottom.
    if top_edge_center[1] < center[1]:
      # Top
      return None
    else:
      # Bottom
      return cv2.ROTATE_180


def rectangle_sorted_indices_corners(
    coordinates: jt.Float[np.ndarray, '4 2'],
) -> jt.Integer[np.ndarray, '4']:
  """Indices of corner coordinates (top_l, bottom_l, top_r, bottom_r)."""
  # First sort by x, so left is first.
  assert coordinates.shape == (4, 2), 'Coordinates must have shape (4, 2).'
  left_right_indices = np.argsort(coordinates[:, 0])
  left = coordinates[left_right_indices[:2]]
  right = coordinates[left_right_indices[2:]]
  # Then sort by y, so that bottom is first within this.
  left_lower_upper_indices = np.argsort(left[:, 1])
  right_lower_upper_indices = np.argsort(right[:, 1])
  return np.concatenate((
    left_right_indices[:2][left_lower_upper_indices],
    left_right_indices[2:][right_lower_upper_indices],
  ))


def rectangle_sorted_indices_edges(
    coordinates: jt.Float[np.ndarray, '4 2'],
) -> jt.Integer[np.ndarray, '4']:
  """Indices of edge coordinates as (left, top, bottom, right)."""
  # First sort by x, so [left, (top/bottom), right]
  assert coordinates.shape == (4, 2), 'Coordinates must have shape (4, 2).'
  left_right_indices = np.argsort(coordinates[:, 0])
  center = coordinates[left_right_indices[1:3]]

  # Then sort the center by y, so it's (top, bottom).
  top_bottom_indices = np.argsort(center[:, 1])

  return np.concatenate((
    [left_right_indices[0]],  # Left
    left_right_indices[1:3][top_bottom_indices],  # Top, Bottom.
    [left_right_indices[-1]],  # Right
  ))  


def four_point_transform(
    image: Image,
    rectangle_coordinates: jt.Float[np.ndarray, '4 2'],
) -> Image:
  """Extract a rectangle from an image given the corner positions.

  This is a slightly modified version of `perspective.four_point_transform.`.
  """
  rectangle = perspective.order_points(rectangle_coordinates)

  (tl, tr, br, bl) = rectangle

  # compute the width of the new image, which will be the
  # maximum distance between bottom-right and bottom-left
  # x-coordiates or the top-right and top-left x-coordinates
  widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
  widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
  maxWidth = max(int(widthA), int(widthB))

  # compute the height of the new image, which will be the
  # maximum distance between the top-right and bottom-right
  # y-coordinates or the top-left and bottom-left y-coordinates
  heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
  heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
  maxHeight = max(int(heightA), int(heightB))

  # now that we have the dimensions of the new image, construct
  # the set of destination points to obtain a "birds eye view",
  # (i.e. top-down view) of the image, again specifying points
  # in the top-left, top-right, bottom-right, and bottom-left
  # order
  destination = np.array([
      [0, 0],
      [maxWidth - 1, 0],
      [maxWidth - 1, maxHeight - 1],
      [0, maxHeight - 1]], dtype=np.float32)

  # compute the perspective transform matrix and then apply it
  perspective_transform = cv2.getPerspectiveTransform(rectangle, destination)
  warped = cv2.warpPerspective(
      image,
      perspective_transform,
      dsize=(maxWidth, maxHeight),
      flags=cv2.INTER_LANCZOS4)

  return warped
