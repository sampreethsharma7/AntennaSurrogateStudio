import math
import unittest

from studio.scientific_plot import ScientificPlotState
from studio.theme import FONTS


class ReadabilityScaleTests(unittest.TestCase):
    def test_application_type_scale_is_at_least_twenty_percent_larger(self):
        previous_sizes = {
            "display": 32,
            "title": 25,
            "section": 19,
            "card_title": 16,
            "body": 14,
            "body_small": 13,
            "caption": 12,
            "button": 13,
            "mono": 12,
        }

        for style_name, previous_size in previous_sizes.items():
            with self.subTest(style=style_name):
                self.assertGreaterEqual(
                    FONTS[style_name][1],
                    math.ceil(previous_size * 1.2),
                )

    def test_scientific_plot_default_text_is_at_least_twenty_percent_larger(self):
        state = ScientificPlotState()

        self.assertGreaterEqual(state.plot_title_font_size, math.ceil(14 * 1.2))
        self.assertGreaterEqual(state.x_label_font_size, math.ceil(11 * 1.2))
        self.assertGreaterEqual(state.y_label_font_size, math.ceil(11 * 1.2))
        self.assertGreaterEqual(state.x_value_font_size, math.ceil(9 * 1.2))
        self.assertGreaterEqual(state.y_value_font_size, math.ceil(9 * 1.2))
        self.assertGreaterEqual(state.legend_font_size, math.ceil(9 * 1.2))


if __name__ == "__main__":
    unittest.main()
