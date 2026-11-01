"""실행: python -m unittest -v test_pose_utils"""
import math
import unittest

from pose_utils import angle_at, analyze_pose, line_direction, reliable_point


class AngleTests(unittest.TestCase):
    def test_known_angles(self):
        for a, c, expected in [((1, 0), (0, 1), 90), ((1, 0), (-1, 0), 180),
                               ((1, 0), (2, 0), 0), ((1, 0), (1, 1), 45)]:
            with self.subTest(expected=expected):
                self.assertAlmostEqual(angle_at(a, (0, 0), c), expected)

    def test_translation_scale_invariance(self):
        self.assertAlmostEqual(angle_at((12, 10), (10, 10), (10, 12)), 90)

    def test_degenerate_and_nonfinite(self):
        for a in (None, (0, 0), (float("nan"), 1), (float("inf"), 1)):
            self.assertIsNone(angle_at(a, (0, 0), (1, 0)))

    def test_missing_pose(self):
        self.assertTrue(all(v is None for v in analyze_pose(None).values()))

    def test_confidence_gating(self):
        points = [[10., 10., 0.9] for _ in range(17)]
        points[5], points[7], points[9] = [10, 10, .9], [10, 20, .9], [20, 20, .9]
        self.assertAlmostEqual(analyze_pose(points)["left_elbow"], 90)
        points[7][2] = .49
        self.assertIsNone(analyze_pose(points)["left_elbow"])
        points[7][2] = .5
        self.assertAlmostEqual(analyze_pose(points)["left_elbow"], 90)

    def test_invalid_keypoints(self):
        for point in ([0, 0, 1], [-1, 10, 1], [10, 10, math.nan], [10, math.inf, 1]):
            self.assertIsNone(reliable_point([point], 0))

    def test_torso_and_directions(self):
        points = [[0., 0., 0.] for _ in range(17)]
        points[5], points[6] = [10, 10, 1], [30, 10, 1]
        points[11], points[12] = [10, 30, 1], [30, 30, 1]
        angles = analyze_pose(points)
        self.assertEqual(angles["torso_lean"], 0)
        self.assertEqual(angles["shoulder_direction"], 0)
        self.assertEqual(angles["hip_direction"], 0)
        points[5][2] = .1
        self.assertIsNone(analyze_pose(points)["torso_lean"])
        self.assertIsNone(analyze_pose(points)["shoulder_direction"])

    def test_clockwise_image_direction(self):
        self.assertEqual(line_direction((1, 1), (1, 2)), 90)
        self.assertEqual(line_direction((1, 2), (1, 1)), -90)
        self.assertIsNone(line_direction((1, 1), (1, 1)))

    def test_all_four_joint_maps(self):
        points = [[0., 0., 0.] for _ in range(17)]
        for i, point in {5: (10, 10), 7: (10, 20), 9: (20, 20),
                         6: (30, 10), 8: (30, 20), 10: (40, 20),
                         11: (10, 30), 13: (10, 40), 15: (20, 40),
                         12: (30, 30), 14: (30, 40), 16: (40, 40)}.items():
            points[i] = [*point, 1.]
        for name in ("left_elbow", "right_elbow", "left_knee", "right_knee"):
            self.assertAlmostEqual(analyze_pose(points)[name], 90)
        points[16][2] = .1
        self.assertIsNone(analyze_pose(points)["right_knee"])
        self.assertAlmostEqual(analyze_pose(points)["left_knee"], 90)

    def test_signed_torso_lean(self):
        points = [[0., 0., 0.] for _ in range(17)]
        points[5], points[6] = [30, 10, 1], [50, 10, 1]
        points[11], points[12] = [10, 30, 1], [30, 30, 1]
        self.assertAlmostEqual(analyze_pose(points)["torso_lean"], 45)


if __name__ == "__main__":
    unittest.main()
