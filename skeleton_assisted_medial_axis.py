# -*- coding: utf-8 -*-
"""
skeleton_assisted_medial_axis.py
Created on Fri Mar 28 10:35:34 2025
@author: Alexandros Papagiannakis, PhD

Extracts the medial axis of a rod-shaped bacterial cell from its binary mask
using morphological skeletonisation followed by bivariate polynomial fitting.

Pipeline
--------
1. Skeletonise the binary mask (medial_axis from skimage.morphology).
2. Count pixel neighbours to identify endpoints (degree 2 nodes).
3. Trace the skeleton from one endpoint to the other.
4. Fit bivariate polynomials to (i→x) and (i→y) to smooth the skeleton.
5. Apply a rolling mean for final smoothing.
6. Interpolate to 0.1-pixel resolution along the skeleton.
7. Compute cumulative arc-length and centred / scaled versions.
"""

import sys
import matplotlib.pyplot as plt
import skimage.morphology as morph
import numpy as np
import pandas as pd

# get_next_node is recursive; the default limit of 1000 is too shallow for
# long or densely-sampled skeletons.
sys.setrecursionlimit(20000)


def image_framing(image, frame_width):
    """Pads a 2-D binary array with zeros on all four sides.

    Adding a border prevents the skeletonisation from mis-classifying boundary
    pixels as endpoints when the cell touches the image edge.

    Parameters
    ----------
    image : 2-D np.ndarray
        Binary cell mask.
    frame_width : int
        Number of zero-padded pixels to add on each side.

    Returns
    -------
    2-D np.ndarray
        Zero-padded image.
    """
    y_array = np.zeros((image.shape[0], frame_width))
    image = np.concatenate((y_array, image), axis=1)
    image = np.concatenate((image, y_array), axis=1)

    x_array = np.zeros((frame_width, image.shape[1]))
    image = np.concatenate((x_array, image), axis=0)
    image = np.concatenate((image, x_array), axis=0)
    return image


def get_intersection_points(num_neighbors):
    """Returns pixel coordinates where the neighbour count exceeds 4 (crossings)."""
    return np.nonzero(num_neighbors > 4)


def get_merging_points(num_neighbors):
    """Returns pixel coordinates where the neighbour count exceeds 3 (branch points)."""
    return np.nonzero(num_neighbors > 3)


def get_square(skeleton, x, y):
    """Returns the 3×3 neighbourhood around pixel (x, y) in a skeleton image."""
    return skeleton[y - 1:y + 2, x - 1:x + 2]


def skeletonize_mask(cropped_mask):
    """Computes the morphological skeleton and distance transform of a binary mask.

    Parameters
    ----------
    cropped_mask : 2-D bool np.ndarray
        Binary cell mask (True = cell).

    Returns
    -------
    skeleton : 2-D bool np.ndarray
        Skeletonised mask.
    distance : 2-D float np.ndarray
        Distance transform (distance of each skeleton pixel from mask boundary).
    """
    return morph.medial_axis(cropped_mask, return_distance=True)


def get_pixel_neighbors(skeleton):
    """Counts the number of True neighbours (including self) for each skeleton pixel.

    The 3×3 sum around each pixel gives: 1 (isolated) → 2 (endpoint) →
    3 (straight run) → 4 (bend) → 5+ (branch point).

    Parameters
    ----------
    skeleton : 2-D bool np.ndarray

    Returns
    -------
    num_neighbors : 2-D int np.ndarray
        Neighbour count at each pixel (0 for background).
    """
    num_neighbors = np.zeros(skeleton.shape, dtype=int)
    for i in range(skeleton.shape[0]):
        for j in range(skeleton.shape[1]):
            if skeleton[i, j]:
                region = get_square(skeleton, j, i)
                num_neighbors[i, j] = np.sum(region)
    return num_neighbors


def get_position_identity(nodes, x, y):
    """Classifies a skeleton pixel as 'edge', 'merge', or 'cross'.

    A pixel is an 'edge' (endpoint) if its neighbour count is 2 and the maximum
    neighbour count in its 3×3 region is 3 (i.e. it connects to exactly one
    other skeleton pixel of degree ≥ 3).

    Parameters
    ----------
    nodes : 2-D int np.ndarray
        Neighbour count array from get_pixel_neighbors.
    x, y : int
        Column and row of the pixel to classify.

    Returns
    -------
    str or None
        'edge', 'merge', 'cross', or None if the pixel does not qualify.
    """
    if nodes[y][x] == 2:
        square = get_square(nodes, x, y)
        mx = np.max(square.ravel())
        if mx == 3:
            return 'edge'
        elif mx == 4:
            return 'merge'
        elif mx > 4:
            return 'cross'


def get_edge_coordinates(nodes):
    """Finds the two endpoint pixels of the skeleton.

    Parameters
    ----------
    nodes : 2-D int np.ndarray
        Neighbour count array from get_pixel_neighbors.

    Returns
    -------
    start_x : list of int
    start_y : list of int
        Coordinates of skeleton endpoints.
    """
    edge_coords = np.nonzero(nodes == 2)
    start_x = []
    start_y = []
    for i in range(edge_coords[0].shape[0]):
        if get_position_identity(nodes, edge_coords[1][i], edge_coords[0][i]) == 'edge':
            start_x.append(edge_coords[1][i])
            start_y.append(edge_coords[0][i])
    return start_x, start_y


def get_next_node(coord_list, nodes):
    """Recursively traces the skeleton from the last coordinate in coord_list.

    At each step, looks for unvisited neighbours with count ≥ 3 in the 3×3
    neighbourhood and appends the first such pixel to the list.

    Parameters
    ----------
    coord_list : list of (x, y) tuples
        Growing ordered list of skeleton coordinates, starting from one endpoint.
    nodes : 2-D int np.ndarray
        Neighbour count array.

    Returns
    -------
    list of (x, y) tuples
        Complete skeleton path from start endpoint to the pixel before the
        other endpoint (the final endpoint is appended by scan_skeleton).
    """
    recurs = 0
    last_x, last_y = coord_list[-1]

    square = get_square(nodes, last_x, last_y)
    nonzero_coords = np.nonzero(square >= 3)

    for cord_i in range(nonzero_coords[0].shape[0]):
        new_x = nonzero_coords[1][cord_i] + last_x - 1
        new_y = nonzero_coords[0][cord_i] + last_y - 1
        if (new_x, new_y) not in coord_list:
            coord_list.append((new_x, new_y))
            recurs = 1

    if recurs == 0:
        return coord_list
    else:
        return get_next_node(coord_list, nodes)


def prune_skeleton_to_main_axis(skeleton):
    """Removes spurious branches by retaining only the longest path through the skeleton.

    Converts the skeleton to an adjacency graph and runs two BFS passes to find
    the graph diameter (farthest-endpoint pair).  All pixels not on that path
    are discarded, leaving a clean single-strand skeleton with exactly 2 endpoints.

    Parameters
    ----------
    skeleton : 2-D bool np.ndarray

    Returns
    -------
    2-D bool np.ndarray
        Skeleton containing only the main-axis pixels.
    """
    skel_coords = [tuple(c) for c in np.argwhere(skeleton)]
    if not skel_coords:
        return skeleton

    coord_set = set(skel_coords)

    def neighbors(y, x):
        return [(y + dy, x + dx)
                for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if (dy, dx) != (0, 0) and (y + dy, x + dx) in coord_set]

    def bfs(start):
        parent = {start: None}
        queue = [start]
        last = start
        for node in queue:
            last = node
            for nb in neighbors(*node):
                if nb not in parent:
                    parent[nb] = node
                    queue.append(nb)
        path = []
        n = last
        while n is not None:
            path.append(n)
            n = parent[n]
        return last, path

    # Two-pass BFS: first pass finds one true endpoint, second finds the full diameter path
    far1, _ = bfs(skel_coords[0])
    _, longest_path = bfs(far1)

    pruned = np.zeros_like(skeleton)
    for y, x in longest_path:
        pruned[y, x] = True
    return pruned


def scan_skeleton(skeleton):
    """Traces the skeleton from one endpoint to the other.

    Parameters
    ----------
    skeleton : 2-D bool np.ndarray

    Returns
    -------
    coord_list : list of (x, y) tuples
        Ordered medial-axis pixel coordinates from pole to pole.

    Raises
    ------
    ValueError
        If the skeleton does not have exactly two endpoints (indicates a
        branched or disconnected skeleton, e.g. due to a touching cell pair).
    """
    neigh = get_pixel_neighbors(skeleton)
    start_x, start_y = get_edge_coordinates(neigh)
    print(start_x, start_y)

    if len(start_x) == 2 and len(start_y) == 2:
        coord_list = [(start_x[0], start_y[0])]
        coord_list = get_next_node(coord_list, neigh)
    else:
        raise ValueError('Abnormal edges detected — skeleton has ≠ 2 endpoints.')

    coord_list.append((start_x[1], start_y[1]))
    return coord_list


def unwrap_coordinates(coord_list, frame):
    """Converts a list of (x, y) tuples to separate arrays and corrects for padding.

    Parameters
    ----------
    coord_list : list of (x, y) tuples
        Ordered skeleton coordinates in the *padded* image.
    frame : int
        Padding width subtracted to restore original mask coordinates.

    Returns
    -------
    x_coords : list of int
    y_coords : list of int
    i_coords : list of int  – sequential index along the skeleton
    """
    x_coords = [c[0] - frame for c in coord_list]
    y_coords = [c[1] - frame for c in coord_list]
    i_coords = list(np.arange(len(x_coords)))
    return x_coords, y_coords, i_coords


def fit_bivariate_polynomials(i_coords, x_coords, y_coords, degree):
    """Fits separate polynomials x(i) and y(i) and evaluates them on i_coords.

    Parameters
    ----------
    i_coords : list or array
        Sequential index (0, 1, 2, …) along the skeleton.
    x_coords, y_coords : list or array
        Raw skeleton x and y pixel coordinates.
    degree : int
        Polynomial degree.  Higher values follow the raw skeleton more closely
        but are more susceptible to noise at the tips.

    Returns
    -------
    x_hat, y_hat : np.ndarray
        Polynomial-smoothed skeleton coordinates.
    """
    x_fit = np.polyfit(i_coords, x_coords, degree)
    y_fit = np.polyfit(i_coords, y_coords, degree)
    x_hat = np.polyval(x_fit, i_coords)
    y_hat = np.polyval(y_fit, i_coords)
    return x_hat, y_hat


def skeleton_assisted_bivariate_axis(cropped_mask, degree):
    """Computes a smooth medial axis for a rod-shaped cell.

    Skeletonises the binary mask, traces the skeleton from pole to pole,
    applies bivariate polynomial smoothing, and interpolates to 0.1-pixel
    resolution.

    Parameters
    ----------
    cropped_mask : 2-D bool np.ndarray
        Binary cell mask (True = cell pixel, False = background).
    degree : int
        Polynomial degree for bivariate fitting (50 is a common default for
        ~40-200 px long E. coli cells).

    Returns
    -------
    axis_df : pd.DataFrame
        Columns:
        - 'i_coords'  : skeleton index
        - 'x_coords'  : raw skeleton x (pixels, no padding)
        - 'y_coords'  : raw skeleton y (pixels, no padding)
        - 'x_hat'     : polynomial-fitted x
        - 'y_hat'     : polynomial-fitted y
        - 'cropped_x' : rolling-average-smoothed x (3-px window)
        - 'cropped_y' : rolling-average-smoothed y (3-px window)
        Reindexed and interpolated to 0.1-pixel steps along the skeleton.

    Raises
    ------
    ValueError
        If the skeleton's maximum neighbour count exceeds 4, indicating a
        cell crossing event (two cells touching or overlapping).
    """
    # Add 1-px padding to prevent boundary artefacts in skeletonisation
    cropped_mask = image_framing(cropped_mask, 1)
    skel, dist = skeletonize_mask(cropped_mask)
    skel = prune_skeleton_to_main_axis(skel)
    neigh = get_pixel_neighbors(skel)

    print(np.max(neigh))

    if np.max(neigh) <= 4:
        coord_list = scan_skeleton(skel)
        x_coords, y_coords, i_coords = unwrap_coordinates(coord_list, 1)
        x_hat, y_hat = fit_bivariate_polynomials(i_coords, x_coords, y_coords, degree)

        axis_df = pd.DataFrame()
        axis_df['i_coords'] = i_coords
        axis_df['x_coords'] = x_coords
        axis_df['y_coords'] = y_coords
        axis_df['x_hat'] = x_hat
        axis_df['y_hat'] = y_hat

        # Rolling average to smooth sub-pixel jitter from the discrete skeleton
        roll_df = axis_df.rolling(3, min_periods=1, center=True).mean()
        axis_df['cropped_x'] = roll_df.x_coords
        axis_df['cropped_y'] = roll_df.y_coords

        # Interpolate to 0.1-pixel resolution along the skeleton index
        axis_df = axis_df.reindex(np.arange(0, axis_df.shape[0] - 1 + 0.1, 0.1)).interpolate()

        return axis_df
    else:
        raise ValueError('Cell crossing event detected — skeleton has branch points.')


def get_arched_lengths(medial_axis_df):
    """Computes cumulative arc length along the medial axis.

    Integrates the Euclidean distance between consecutive smoothed medial-axis
    points (cropped_x, cropped_y) to obtain the arc length in pixels.  Also
    computes centred (origin at cell midpoint) and scaled (−1 to +1) versions.

    Parameters
    ----------
    medial_axis_df : pd.DataFrame
        Output of skeleton_assisted_bivariate_axis; must contain 'cropped_x'
        and 'cropped_y' columns.

    Returns
    -------
    pd.DataFrame
        Input dataframe with three new columns:
        - 'arch_length'         : cumulative arc length in **pixels** (0 at pole 1)
        - 'arch_length_centered': arc length centred on the cell midpoint (pixels)
        - 'arch_length_scaled'  : arch_length_centered / half_cell_length (−1 to +1)
    """
    x = np.array(medial_axis_df['cropped_x'])
    y = np.array(medial_axis_df['cropped_y'])

    # Step distances between consecutive points (px)
    delta_x_sqr = (x[1:] - x[:-1]) ** 2
    delta_y_sqr = (y[1:] - y[:-1]) ** 2
    disp_array = np.sqrt(delta_x_sqr + delta_y_sqr)

    disp_list = [0]
    for disp in disp_array:
        disp_list.append(disp_list[-1] + disp)

    medial_axis_df['arch_length'] = disp_list                                    # px
    medial_axis_df['arch_length_centered'] = disp_list - np.max(disp_list) / 2   # px, 0 at midpoint
    medial_axis_df['arch_length_scaled'] = (
        medial_axis_df['arch_length_centered'] /
        medial_axis_df['arch_length_centered'].max()
    )  # dimensionless, −1 to +1

    return medial_axis_df


def get_oned_coordinates(cell_mask, medial_axis_df):
    """Projects each cell pixel onto the nearest point on the medial axis.

    For every cell pixel the function finds the closest medial-axis point
    (minimum Euclidean distance), assigns the arc-length position, and
    computes the signed lateral distance (width coordinate).  The sign
    convention uses the cross-product of the local medial-axis tangent with
    the pixel–axis vector: positive = left side, negative = right side
    (arbitrary, consistent within a cell).

    Parameters
    ----------
    cell_mask : 2-D bool np.ndarray
        Cropped binary cell mask.
    medial_axis_df : pd.DataFrame
        Output of skeleton_assisted_bivariate_axis + get_arched_lengths;
        must contain 'cropped_x', 'cropped_y', 'arch_length_centered',
        'arch_length_scaled'.

    Returns
    -------
    cell_mask_df : pd.DataFrame
        One row per cell pixel with columns:
        - 'x', 'y'          : pixel coordinates (px)
        - 'arch_length'     : centred medial-axis position (px)
        - 'scaled_length'   : scaled medial-axis position (−1 to +1)
        - 'width'           : signed lateral distance from axis (px)
    """
    cell_mask_df = pd.DataFrame()
    cell_mask_df['x'] = np.nonzero(cell_mask)[1]
    cell_mask_df['y'] = np.nonzero(cell_mask)[0]

    def get_pixel_projection(pixel_x, pixel_y, medial_axis_df):
        """Returns (arch_length_centered, arch_length_scaled, signed_width) for one pixel."""
        medial_axis_df['pixel_distance'] = np.sqrt(
            (medial_axis_df.cropped_x - pixel_x) ** 2 +
            (medial_axis_df.cropped_y - pixel_y) ** 2
        )
        min_df = medial_axis_df[
            medial_axis_df.pixel_distance == medial_axis_df.pixel_distance.min()
        ]
        min_arch = min_df.arch_length_centered.values[0]       # px
        min_scaled = min_df.arch_length_scaled.values[0]       # dimensionless
        min_dist_abs = min_df.pixel_distance.values[0]         # px
        min_index = min_df.index.values[0]
        axis_coords = (min_df.cropped_x.values[0], min_df.cropped_y.values[0])

        def get_relative_distance(min_distance_abs, medial_axis_df, min_index,
                                  medial_axis_coords, pixel_x, pixel_y):
            """Computes signed lateral distance using cross-product of tangent and pixel vectors."""
            half_window = 5

            # Select a local window around the nearest axis point to estimate the tangent
            if min_index >= half_window and min_index < medial_axis_df.index.max() - half_window:
                idx_lo, idx_hi = min_index - half_window, min_index + half_window
            elif min_index < half_window:
                idx_lo, idx_hi = 0, min_index + half_window
            else:
                idx_lo, idx_hi = min_index - half_window, medial_axis_df.index.max()

            # Local tangent vector along the medial axis
            delta_x = medial_axis_df.iloc[idx_hi].cropped_x - medial_axis_df.iloc[idx_lo].cropped_x
            delta_y = medial_axis_df.iloc[idx_hi].cropped_y - medial_axis_df.iloc[idx_lo].cropped_y
            tangent = [delta_x, delta_y]

            # Vector from nearest axis point to the pixel
            px_vec = [pixel_x - medial_axis_coords[0], pixel_y - medial_axis_coords[1]]

            cross = np.cross(tangent, px_vec)
            if cross != 0:
                return np.sign(cross) * min_distance_abs
            else:
                return 0

        signed_width = get_relative_distance(
            min_dist_abs, medial_axis_df, min_index, axis_coords, pixel_x, pixel_y
        )
        return min_arch, min_scaled, signed_width

    cell_mask_df['oned_coords'] = cell_mask_df.apply(
        lambda x: get_pixel_projection(x.x, x.y, medial_axis_df), axis=1
    )
    cell_mask_df[['arch_length', 'scaled_length', 'width']] = pd.DataFrame(
        cell_mask_df.oned_coords.to_list(), index=cell_mask_df.index
    )
    cell_mask_df = cell_mask_df.drop(['oned_coords'], axis=1)

    return cell_mask_df
