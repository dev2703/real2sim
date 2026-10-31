"""Geometry-aware RGB targets must follow images and fail on missing evidence."""

import cv2
import numpy as np
import pytest

from real2sim.policy.visual_targets import camera_pose_task, locate_objects


def camera():
    intrinsic = np.array([[650., 0, 320], [0, 650., 240], [0, 0, 1.]])
    pose = np.eye(4)
    pose[:3, :3] = np.diag([1., -1., -1.])
    pose[:3, 3] = [0, 0, 1.]
    return intrinsic, pose


def scene(ball=(.18, .10), bowl=(-.16, -.10), *, hide_ball=False):
    K, pose = camera()
    image = np.full((480, 640, 3), 40, dtype=np.uint8)
    # Ray/sphere intersections render an exact perspective sphere silhouette,
    # including the off-axis distinction from a projected circle or centroid.
    yy, xx = np.indices(image.shape[:2])
    rays = np.stack([(xx - K[0, 2]) / K[0, 0], -(yy - K[1, 2]) / K[1, 1],
                     -np.ones_like(xx)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    center = np.r_[ball, .0335] - pose[:3, 3]
    if not hide_ball:
        mask = np.linalg.norm(np.cross(rays, center), axis=-1) <= .0335
        image[mask] = [200, 220, 40]
    # Solid grey disk represents the bowl's complete outer rim contour.
    points = np.c_[bowl[0] + .052 * np.cos(np.linspace(0, 2 * np.pi, 300)),
                   bowl[1] + .052 * np.sin(np.linspace(0, 2 * np.pi, 300)),
                   np.full(300, .03)]
    camera_points = (points - pose[:3, 3]) @ pose[:3, :3]
    pixels = camera_points @ K.T
    cv2.fillPoly(image, [np.rint(pixels[:, :2] / pixels[:, 2, None]).astype(np.int32)], (150, 150, 145))
    return image


def locate(image, **kwargs):
    K, pose = camera()
    return locate_objects(image, K, pose, ball_radius_m=.0335,
                          ball_center_height_m=.0335, bowl_rim_height_m=.03,
                          bowl_outer_radius_m=.052, workspace_xy=[[-.4, -.3], [.4, .3]], **kwargs)


def test_targets_follow_shifted_images_without_configured_object_centers():
    for ball, bowl in [((.18, .10), (-.16, -.10)), ((.09, -.11), (-.21, .14))]:
        result = locate(scene(ball, bowl))
        for name, expected in (("ball", ball), ("bowl", bowl)):
            assert result[name] is not None
            np.testing.assert_allclose(result[name].center_task[:2], expected, atol=.001)
            assert 0 < result[name].confidence <= 1


def test_missing_ball_has_no_position_fallback():
    result = locate(scene(hide_ball=True))
    assert result["ball"] is None
    assert result["bowl"] is not None
    assert locate(np.zeros((480, 640, 3), dtype=np.uint8)) == {"ball": None, "bowl": None}


def test_warm_wood_background_does_not_merge_ball_with_image_border():
    image = scene()
    image[np.all(image == 40, axis=2)] = [110, 83, 29]
    result = locate(image)
    assert result["ball"] is not None
    np.testing.assert_allclose(result["ball"].center_task[:2], [.18, .10], atol=.001)


def test_camera_frame_conversion_preserves_projection_axes():
    pose = camera_pose_task([.1, -.3, .9], np.eye(3), [.01, .02, .7])
    np.testing.assert_allclose(pose[:3, 3], [.09, -.32, .2])
    np.testing.assert_allclose(pose[:3, :3], np.diag([1, -1, -1]))


def test_occluded_ball_does_not_return_confident_center():
    image = scene()
    # Remove the right half of the ball, leaving an occlusion edge rather than
    # the sphere tangent cone; the metric fit/area checks must reject it.
    image[:240, 442:] = 40
    assert locate(image)["ball"] is None


def test_wrong_object_scale_and_invalid_input_are_rejected():
    K, pose = camera()
    result = locate_objects(scene(), K, pose, ball_radius_m=.06,
                            ball_center_height_m=.0335, bowl_rim_height_m=.03,
                            bowl_outer_radius_m=.10)
    assert result == {"ball": None, "bowl": None}
    with pytest.raises(ValueError, match="uint8"):
        locate(scene().astype(float) / 255)


def test_dim_connected_pixels_do_not_hide_a_complete_bright_rim():
    image = scene()
    # A dim neutral protrusion merges with the bowl at the original V>=80
    # threshold, although its brighter complete rim is still visible.
    overlay = image.copy()
    cv2.rectangle(overlay, (225, 290), (270, 320), (100, 100, 100), -1)
    background = np.all(image == 40, axis=2)
    image[background] = overlay[background]
    result = locate(image)
    assert result["bowl"] is not None
    np.testing.assert_allclose(result["bowl"].center_task[:2], [-.16, -.10], atol=.001)


def test_threshold_repeats_do_not_hide_two_independent_matching_bowls():
    image = scene()
    other = scene(bowl=(-.16, .18), hide_ball=True)
    grey = (other[:, :, 0] == 150) & (other[:, :, 1] == 150)
    image[grey] = other[grey]
    assert locate(image)["bowl"] is None
