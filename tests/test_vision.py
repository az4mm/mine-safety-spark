import unittest

from mine_safety.tracking import (
    Detection, EpisodeFilter, IoUTracker, assign_helmet_status,
    intersection_over_union, normalize_polygon, point_in_polygon,
)
from mine_safety.vision import non_maximum_suppression
from mine_safety.vision_evaluation import match_detections


class GeometryAndSafetyTests(unittest.TestCase):
    def test_polygon_includes_edge_and_excludes_outside(self):
        polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
        self.assertTrue(point_in_polygon((5, 5), polygon))
        self.assertTrue(point_in_polygon((10, 4), polygon))
        self.assertFalse(point_in_polygon((10.1, 4), polygon))
        self.assertFalse(point_in_polygon((5, 5), None))

    def test_polygon_scale_and_invalid_input(self):
        self.assertEqual(normalize_polygon([[0, 0], [1, 0], [1, 1]], 100, 50),
                         [(0, 0), (100, 0), (100, 50)])
        with self.assertRaises(ValueError):
            normalize_polygon([[0, 0], [1, 0], [2, 1]], 100, 50)

    def test_missing_or_conflicting_helmet_evidence_stays_unknown(self):
        person = Detection((10, 10, 110, 210), .9, "Person")
        helmet = Detection((45, 10, 75, 45), .8, "Hardhat")
        bare = Detection((45, 10, 75, 45), .7, "NO-Hardhat")
        generic_head = Detection((45, 10, 75, 45), .9, "head")
        self.assertEqual(assign_helmet_status([person], [])[0][0], "unknown")
        self.assertEqual(assign_helmet_status([person], [generic_head])[0][0], "unknown")
        self.assertEqual(assign_helmet_status([person], [helmet])[0][0], "present")
        self.assertEqual(assign_helmet_status([person], [bare])[0][0], "absent")
        self.assertEqual(assign_helmet_status([person], [helmet, bare])[0][0], "unknown")

    def test_one_helmet_cannot_mark_two_overlapping_people_compliant(self):
        people = [Detection((0, 0, 100, 200), .9, "Person"),
                  Detection((40, 0, 140, 200), .9, "Person")]
        helmet = Detection((40, 2, 65, 35), .8, "Hardhat")
        statuses = assign_helmet_status(people, [helmet])
        self.assertEqual(sum(s[0] == "present" for s in statuses), 1)

    def test_helmet_carried_near_waist_does_not_count_as_worn(self):
        person = Detection((0, 0, 100, 200), .9, "Person")
        carried = Detection((20, 120, 60, 155), .95, "Hardhat")
        self.assertEqual(assign_helmet_status([person], [carried])[0][0], "unknown")

    def test_tracker_retains_identity_then_expires(self):
        tracker = IoUTracker(max_gap_s=1)
        p1 = Detection((0, 0, 100, 200), .9, "Person")
        p2 = Detection((5, 5, 105, 205), .9, "Person")
        first = tracker.update([p1], 0)[0][0]
        self.assertEqual(tracker.update([p2], .5)[0][0], first)
        self.assertNotEqual(tracker.update([p2], 2)[0][0], first)

    def test_episode_confirmation_gap_and_reentry(self):
        episodes = EpisodeFilter("cam:session", confirmation_s=1)
        self.assertIsNone(episodes.update("p1", "helmet", True, 0))
        self.assertIsNone(episodes.update("p1", "helmet", True, .5))
        first = episodes.update("p1", "helmet", True, 1)
        self.assertIsNotNone(first)
        self.assertEqual(episodes.update("p1", "helmet", True, 2), first)
        self.assertIsNone(episodes.update("p1", "helmet", None, 2.2))
        self.assertEqual(episodes.update("p1", "helmet", True, 2.5), first)
        episodes.update("p1", "helmet", False, 3)
        self.assertIsNone(episodes.update("p1", "helmet", True, 4))
        second = episodes.update("p1", "helmet", True, 5)
        self.assertNotEqual(first, second)

    def test_unknown_interrupts_unconfirmed_evidence(self):
        episodes = EpisodeFilter("cam:session", confirmation_s=1)
        episodes.update("p1", "helmet", True, 0)
        episodes.update("p1", "helmet", None, .5)
        self.assertIsNone(episodes.update("p1", "helmet", True, 1))
        self.assertIsNotNone(episodes.update("p1", "helmet", True, 2))

    def test_class_aware_nms(self):
        a = Detection((0, 0, 100, 100), .9, "Person")
        b = Detection((2, 2, 102, 102), .6, "Person")
        c = Detection((2, 2, 102, 102), .8, "Hardhat")
        self.assertGreater(intersection_over_union(a.box, b.box), .9)
        self.assertEqual(non_maximum_suppression([a, b, c]), [a, c])

    def test_evaluation_does_not_double_match_a_ground_truth_box(self):
        gt = [Detection((0, 0, 100, 100), 1, "hardhat")]
        predictions = [Detection((0, 0, 100, 100), .9, "hardhat"),
                       Detection((1, 1, 101, 101), .8, "hardhat")]
        self.assertEqual(match_detections(predictions, gt)["hardhat"], {"tp": 1, "fp": 1, "fn": 0})

    def test_evaluation_rejects_wrong_class_and_bad_localization(self):
        gt = [Detection((0, 0, 100, 100), 1, "hardhat")]
        predictions = [Detection((0, 0, 100, 100), .9, "no-hardhat")]
        counts = match_detections(predictions, gt)
        self.assertEqual(counts["hardhat"]["fn"], 1)
        self.assertEqual(counts["no-hardhat"]["fp"], 1)


if __name__ == "__main__":
    unittest.main()
