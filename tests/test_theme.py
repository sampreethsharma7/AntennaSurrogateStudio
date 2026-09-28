import ast
import math
import unittest
from pathlib import Path

from studio.scientific_plot import ScientificPlotState
from studio.theme import FONTS


class ReadabilityScaleTests(unittest.TestCase):
    def test_smallest_theme_text_is_sixteen_points(self):
        self.assertGreaterEqual(
            min(font[1] for font in FONTS.values()),
            16,
        )

    def test_gui_sources_have_no_literal_font_below_theme_floor(self):
        studio_root = Path(__file__).resolve().parents[1] / "studio"
        floor = min(font[1] for font in FONTS.values())
        undersized: list[str] = []
        for source_path in studio_root.glob("*.py"):
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                for keyword in node.keywords:
                    if keyword.arg not in {"font", "dropdown_font"}:
                        continue
                    value = keyword.value
                    if not isinstance(value, ast.Tuple) or len(value.elts) < 2:
                        continue
                    size = value.elts[1]
                    if (
                        isinstance(size, ast.Constant)
                        and isinstance(size.value, int)
                        and size.value < floor
                    ):
                        undersized.append(
                            f"{source_path.name}:{node.lineno}={size.value}"
                        )
        self.assertEqual(undersized, [])

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
