import math
import unittest

from studio.scientific_plot import (
    ScientificPlotState,
    adaptive_major_interval_count,
    engineering_tick,
)


class ScientificPlotStateTests(unittest.TestCase):
    def setUp(self):
        self.state = ScientificPlotState()

    def _add_curve(self, *, replace=False, inputs=None):
        return self.state.add_curve(
            x_values=[1.0, 2.0, 3.0],
            y_values=[10.0, 20.0, 15.0],
            target_names=["theta_1", "theta_2", "theta_3"],
            inputs=inputs or {"P2": 1.2, "P3": 3.4},
            replace_selected=replace,
        )

    def test_multiple_curves_preserve_their_input_snapshots(self):
        first_inputs = {"P2": 1.2, "P3": 3.4}
        first = self._add_curve(inputs=first_inputs)
        first_inputs["P2"] = 99.0
        second = self.state.add_curve(
            x_values=[1.0, 2.0, 3.0],
            y_values=[11.0, 18.0, 16.0],
            target_names=["theta_1", "theta_2", "theta_3"],
            inputs={"P2": 2.5, "P3": 4.5},
        )

        self.assertEqual(len(self.state.curves), 2)
        self.assertEqual(first.inputs, {"P2": 1.2, "P3": 3.4})
        self.assertEqual(second.inputs, {"P2": 2.5, "P3": 4.5})
        self.assertEqual(self.state.selected_curve_id, second.curve_id)

    def test_replace_updates_only_selected_curve_and_its_inputs(self):
        first = self._add_curve()
        second = self.state.add_curve(
            x_values=[1.0, 2.0],
            y_values=[4.0, 5.0],
            target_names=["out_1", "out_2"],
            inputs={"P2": 7.0},
        )
        self.state.select_curve(first.curve_id)

        replaced = self.state.add_curve(
            x_values=[0.0, 1.0],
            y_values=[8.0, 9.0],
            target_names=["out_0", "out_1"],
            inputs={"P2": 8.0},
            replace_selected=True,
        )

        self.assertEqual(len(self.state.curves), 2)
        self.assertEqual(replaced.curve_id, first.curve_id)
        self.assertEqual(replaced.name, "Prediction 1")
        self.assertEqual(replaced.inputs, {"P2": 8.0})
        self.assertEqual(self.state.curve(second.curve_id).inputs, {"P2": 7.0})

    def test_curve_visibility_rename_and_delete(self):
        first = self._add_curve()
        second = self.state.add_curve(
            x_values=[1.0, 2.0],
            y_values=[2.0, 3.0],
            target_names=["a", "b"],
            inputs={},
        )

        self.state.rename_curve(first.curve_id, "Baseline")
        self.state.set_curve_visible(first.curve_id, False)
        self.assertEqual(first.name, "Baseline")
        self.assertFalse(first.visible)

        self.state.delete_curve(first.curve_id)
        self.assertEqual([curve.curve_id for curve in self.state.curves], [second.curve_id])
        self.assertEqual(self.state.selected_curve_id, second.curve_id)

    def test_annotations_are_associated_and_removed_with_curve(self):
        curve = self._add_curve()
        marker = self.state.add_annotation(
            2.0,
            20.0,
            label="Peak",
            curve_id=curve.curve_id,
        )
        self.assertEqual(marker.label, "Peak")
        self.assertEqual(marker.curve_id, curve.curve_id)

        self.state.delete_curve(curve.curve_id)
        self.assertEqual(self.state.annotations, [])

    def test_zoom_pan_reset_and_autoscale_are_deterministic(self):
        self._add_curve()
        original = self.state.view_limits

        self.state.zoom(0.5)
        zoomed = self.state.view_limits
        self.assertAlmostEqual(zoomed[1] - zoomed[0], (original[1] - original[0]) * 0.5)
        self.assertAlmostEqual(zoomed[3] - zoomed[2], (original[3] - original[2]) * 0.5)

        self.state.pan(0.1, -0.2)
        panned = self.state.view_limits
        self.assertNotEqual(panned, zoomed)
        self.assertAlmostEqual(panned[1] - panned[0], zoomed[1] - zoomed[0])

        self.assertEqual(self.state.reset_view(), original)
        self.assertEqual(self.state.autoscale(), original)

    def test_editable_axis_labels_and_limits_validate(self):
        self.state.set_axis_labels("Frequency", "S11")
        self.state.set_limits(
            1.0e9,
            2.0e9,
            -40.0,
            0.0,
            user_defined=True,
        )
        self.state.major_grid = False
        self.state.minor_grid = True

        self.assertEqual(self.state.x_label, "Frequency")
        self.assertEqual(self.state.y_label, "S11")
        self.assertEqual(self.state.view_limits, (1.0e9, 2.0e9, -40.0, 0.0))
        self.assertFalse(self.state.major_grid)
        self.assertTrue(self.state.minor_grid)
        self.assertTrue(self.state.axis_labels_user_defined)
        self.assertTrue(self.state.axis_limits_user_defined)
        with self.assertRaisesRegex(ValueError, "X-axis minimum"):
            self.state.set_limits(2.0, 1.0, -1.0, 1.0)
        with self.assertRaisesRegex(ValueError, "Both axis labels"):
            self.state.set_axis_labels("", "Y")

    def test_engineering_tick_labels(self):
        self.assertEqual(engineering_tick(0.0), "0")
        self.assertEqual(engineering_tick(2.4e9), "2.4G")
        self.assertEqual(engineering_tick(12.0e-3), "12m")
        self.assertEqual(engineering_tick(-3.0e-6), "-3µ")
        self.assertEqual(engineering_tick(math.inf), "—")

    def test_x_axis_major_ticks_adapt_to_compact_plot_width(self):
        self.assertEqual(adaptive_major_interval_count(210, 11), 3)
        self.assertEqual(adaptive_major_interval_count(320, 11), 5)
        self.assertEqual(adaptive_major_interval_count(560, 11), 5)
        self.assertEqual(adaptive_major_interval_count(60, 11), 2)

    def test_plot_settings_apply_title_scales_grid_and_legend(self):
        self._add_curve()
        self.state.configure_plot(
            plot_title="Gain Response",
            x_label="Frequency",
            y_label="Gain",
            limits=(1.0, 10.0, 1.0, 100.0),
            x_scale="Log",
            y_scale="Log",
            major_grid=True,
            minor_grid=False,
            legend_visible=False,
            legend_location="Lower left",
            plot_title_font_size=18,
            x_label_font_size=14,
            y_label_font_size=15,
            x_value_font_size=10,
            y_value_font_size=11,
            legend_font_size=12,
            legend_line_width=4,
        )

        self.assertEqual(self.state.plot_title, "Gain Response")
        self.assertEqual(self.state.x_scale, "Log")
        self.assertEqual(self.state.y_scale, "Log")
        self.assertFalse(self.state.minor_grid)
        self.assertFalse(self.state.legend_visible)
        self.assertEqual(self.state.legend_location, "Lower left")
        self.assertEqual(self.state.legend_position, (0.28, 0.72))
        self.assertEqual(self.state.plot_title_font_size, 18)
        self.assertEqual(self.state.x_label_font_size, 14)
        self.assertEqual(self.state.y_label_font_size, 15)
        self.assertEqual(self.state.x_value_font_size, 10)
        self.assertEqual(self.state.y_value_font_size, 11)
        self.assertEqual(self.state.legend_font_size, 12)
        self.assertEqual(self.state.legend_line_width, 4)

        original = self.state.view_limits
        self.state.zoom(0.5)
        self.assertGreater(self.state.view_limits[0], 0)
        self.assertGreater(self.state.view_limits[2], 0)
        self.state.pan(0.1, -0.1)
        self.assertGreater(self.state.view_limits[0], 0)
        self.assertNotEqual(self.state.view_limits, original)

    def test_plot_typography_rejects_unsafe_font_and_legend_width_values(self):
        original_title_size = self.state.plot_title_font_size
        with self.assertRaisesRegex(ValueError, "X-value font size"):
            self.state.configure_plot(
                plot_title="Response",
                x_label="X",
                y_label="Y",
                limits=(0.0, 1.0, 0.0, 1.0),
                x_scale="Linear",
                y_scale="Linear",
                major_grid=True,
                minor_grid=True,
                legend_visible=True,
                legend_location="Upper right",
                x_value_font_size=50,
            )
        self.assertEqual(self.state.plot_title_font_size, original_title_size)

        with self.assertRaisesRegex(ValueError, "Legend line width"):
            self.state.configure_plot(
                plot_title="Response",
                x_label="X",
                y_label="Y",
                limits=(0.0, 1.0, 0.0, 1.0),
                x_scale="Linear",
                y_scale="Linear",
                major_grid=True,
                minor_grid=True,
                legend_visible=True,
                legend_location="Upper right",
                legend_line_width=0.1,
            )

    def test_log_scale_rejects_nonpositive_visible_curve_values(self):
        self.state.add_curve(
            x_values=[0.0, 1.0],
            y_values=[-1.0, 2.0],
            target_names=["a", "b"],
            inputs={},
        )
        with self.assertRaisesRegex(ValueError, "Log X scale"):
            self.state.configure_plot(
                plot_title="Response",
                x_label="X",
                y_label="Y",
                limits=(0.1, 2.0, -2.0, 3.0),
                x_scale="Log",
                y_scale="Linear",
                major_grid=True,
                minor_grid=True,
                legend_visible=True,
                legend_location="Upper right",
            )

        positive = ScientificPlotState()
        positive.add_curve(
            x_values=[1.0, 2.0],
            y_values=[1.0, 2.0],
            target_names=["a", "b"],
            inputs={},
        )
        positive.configure_plot(
            plot_title="Log Response",
            x_label="X",
            y_label="Y",
            limits=(1.0, 2.0, 1.0, 2.0),
            x_scale="Log",
            y_scale="Log",
            major_grid=True,
            minor_grid=True,
            legend_visible=True,
            legend_location="Upper right",
        )
        with self.assertRaisesRegex(ValueError, "nonpositive Y"):
            positive.add_curve(
                x_values=[1.0, 2.0],
                y_values=[-1.0, 2.0],
                target_names=["a", "b"],
                inputs={},
            )

    def test_selected_curve_style_settings_are_isolated(self):
        first = self._add_curve()
        second = self.state.add_curve(
            x_values=[1.0, 2.0, 3.0],
            y_values=[11.0, 21.0, 16.0],
            target_names=["a", "b", "c"],
            inputs={},
        )
        self.state.select_curve(first.curve_id)

        styled = self.state.configure_selected_curve(
            line_width=4.5,
            line_style="Dashed",
            marker_style="Diamond",
            marker_size=6.0,
        )

        self.assertEqual(styled.line_width, 4.5)
        self.assertEqual(styled.line_style, "Dashed")
        self.assertEqual(styled.marker_style, "Diamond")
        self.assertEqual(styled.marker_size, 6.0)
        self.assertEqual(second.line_width, 2.0)
        self.assertEqual(second.line_style, "Solid")
        with self.assertRaisesRegex(ValueError, "line width"):
            self.state.configure_selected_curve(
                line_width=12.0,
                line_style="Solid",
                marker_style="Circle",
                marker_size=3.0,
            )

    def test_marker_only_style_and_clear_curves_preserve_plot_settings(self):
        curve = self._add_curve()
        self.state.plot_title = "Residual Analysis"
        self.state.x_label = "Predicted value"
        self.state.configure_selected_curve(
            line_width=2.0,
            line_style="None",
            marker_style="Circle",
            marker_size=4.0,
        )

        self.assertEqual(curve.line_style, "None")
        self.state.clear_curves()

        self.assertEqual(self.state.curves, [])
        self.assertEqual(self.state.plot_title, "Residual Analysis")
        self.assertEqual(self.state.x_label, "Predicted value")


if __name__ == "__main__":
    unittest.main()
