import unittest

from studio.output_axis import infer_output_axis, output_axis_from_dict


class OutputAxisMetadataTests(unittest.TestCase):
    def test_numeric_frequency_targets_preserve_order_and_unit(self):
        axis = infer_output_axis(
            ["frequency_1.0_GHz", "frequency_1.5_GHz", "frequency_2.0_GHz"]
        )

        self.assertEqual(axis.display_label, "Frequency (GHz)")
        self.assertEqual(axis.values, (1.0, 1.5, 2.0))
        self.assertEqual(axis.source, "target_columns")

    def test_unstructured_outputs_use_neutral_one_based_coordinates(self):
        axis = infer_output_axis(["gain_left", "gain_center", "gain_right"])

        self.assertEqual(axis.display_label, "Output coordinate")
        self.assertEqual(axis.values, (1.0, 2.0, 3.0))
        self.assertEqual(axis.source, "output_index")

    def test_saved_axis_must_match_output_count(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            output_axis_from_dict(
                {
                    "label": "Theta",
                    "unit": "deg",
                    "values": [-90.0, 0.0],
                    "source": "target_columns",
                },
                ["theta_-90", "theta_0", "theta_90"],
            )


if __name__ == "__main__":
    unittest.main()
