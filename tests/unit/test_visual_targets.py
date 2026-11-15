"""Geometry-aware RGB targets must follow images and fail on missing evidence."""

import cv2
import numpy as np
import pytest

from real2sim.policy.visual_targets import camera_pose_task, locate_ball_3d, locate_objects


def camera():
    intrinsic = np.array([[650., 0, 320], [0, 650., 240], [0, 0, 1.]])
    pose = np.eye(4)
    pose[:3, :3] = np.diag([1., -1., -1.])
    pose[:3, 3] = [0, 0, 1.]
    return intrinsic, pose


def sphere_image(centers, *, scale=1.):
    K, pose = camera()
    K[:2] *= scale
    image = np.full((round(480 * scale), round(640 * scale), 3), 40, np.uint8)
    yy, xx = np.indices(image.shape[:2])
    rays = np.stack([(xx - K[0, 2]) / K[0, 0], -(yy - K[1, 2]) / K[1, 1],
                     -np.ones_like(xx)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    for center in centers:
        mask = np.linalg.norm(np.cross(rays, np.array(center) - pose[:3, 3]), axis=-1) <= .0335
        image[mask] = [200, 220, 40]
    return image


@pytest.mark.parametrize("center", [(.18, .10, .0335), (.08, -.10, .22), (-.12, .08, .4)])
def test_3d_sphere_observation_follows_height_without_table_or_tcp_prior(center):
    K, pose = camera()
    result = locate_ball_3d(sphere_image([center]), K, pose, ball_radius_m=.0335)
    assert result is not None
    np.testing.assert_allclose(result.center_task, center, atol=.004)


def test_3d_sphere_rejects_missing_occluded_clipped_and_ambiguous_images():
    K, pose = camera()
    images = [sphere_image([]), sphere_image([(.15, 0, .2), (-.15, 0, .2)]),
              sphere_image([(.38, 0, .2)])]
    occluded = sphere_image([(0, 0, .2)])
    occluded[:, 320:] = 40
    images.append(occluded)
    for image in images:
        assert locate_ball_3d(image, K, pose, ball_radius_m=.0335) is None


def test_shallow_edge_occlusion_cannot_create_a_false_3d_lift_baseline():
    K, pose = camera()
    image = sphere_image([(0, 0, .0335)])
    right = np.nonzero(np.any(image != 40, axis=2))[1].max()
    image[:, right - 3:] = 40
    assert locate_ball_3d(image, K, pose, ball_radius_m=.0335) is None


@pytest.mark.parametrize("center", [(0, 0, .0335), (.15, -.10, .22), (-.12, .08, .4)])
@pytest.mark.parametrize("scale", [.75, 1., 1.5])
def test_tangent_arcs_recover_partial_sphere_without_inventing_height(center, scale):
    K, pose = camera()
    K[:2] *= scale
    image = sphere_image([center], scale=scale)
    yy, xx = np.nonzero(np.any(image != 40, axis=2))
    # A jaw obscures a shallow edge, excluding that chord from the sphere fit.
    image[:yy.max() + 1, xx.max() - max(3, round(.15 * np.ptp(xx))):] = 40
    result = locate_ball_3d(image, K, pose, ball_radius_m=.0335, fit_mode="arcs")
    assert result is not None
    np.testing.assert_allclose(result.center_task, center, atol=.006)


def test_occlusion_reveal_does_not_become_a_false_lift_with_arc_fit():
    K, pose = camera()
    full = sphere_image([(0, 0, .0335)])
    partial = full.copy()
    right = np.nonzero(np.any(partial != 40, axis=2))[1].max()
    partial[:, right - 8:] = 40
    measurements = [locate_ball_3d(im, K, pose, ball_radius_m=.0335, fit_mode="arcs")
                    for im in [partial, full]]
    assert all(result is not None for result in measurements)
    assert abs(measurements[1].center_task[2] - measurements[0].center_task[2]) < .004


@pytest.mark.parametrize("kind", ["missing", "half", "clipped", "two", "square", "ellipse", "tiny"])
def test_arc_fit_rejects_weak_or_ambiguous_silhouettes(kind):
    K, pose = camera()
    image = sphere_image([(0, 0, .2)])
    if kind == "half":
        image[:, 320:] = 40
    elif kind == "clipped":
        image = sphere_image([(.38, 0, .2)])
    elif kind == "two":
        image = sphere_image([(.15, 0, .2), (-.15, 0, .2)])
    else:
        image[:] = 40
        if kind == "square":
            cv2.rectangle(image, (300, 220), (340, 260), (200, 220, 40), -1)
        elif kind == "ellipse":
            cv2.ellipse(image, (320, 240), (35, 18), 0, 0, 360, (200, 220, 40), -1)
        elif kind == "tiny":
            cv2.circle(image, (320, 240), 4, (200, 220, 40), -1)
    assert locate_ball_3d(image, K, pose, ball_radius_m=.0335, fit_mode="arcs") is None


def test_arc_fit_is_deterministic_and_enforces_workspace_and_mode():
    K, pose = camera()
    image = sphere_image([(.15, 0, .2)])
    first = locate_ball_3d(image, K, pose, ball_radius_m=.0335, fit_mode="arcs")
    second = locate_ball_3d(image, K, pose, ball_radius_m=.0335, fit_mode="arcs")
    np.testing.assert_array_equal(first.center_task, second.center_task)
    assert locate_ball_3d(image, K, pose, ball_radius_m=.0335, fit_mode="arcs",
                          workspace_xy=[[-.4, -.3], [0., .3]]) is None
    with pytest.raises(ValueError, match="fit mode"):
        locate_ball_3d(image, K, pose, ball_radius_m=.0335, fit_mode="unknown")


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
