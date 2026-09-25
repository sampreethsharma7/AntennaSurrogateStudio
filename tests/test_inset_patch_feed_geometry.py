import unittest

from studio.antenna_agent import create_default_agent
from studio.antenna_design import evaluate_scalar, resolve_parameter_values
from studio.antenna_geometry import build_geometry_scene
from studio.antenna_recipes import MATERIALS, _microstrip_width_50_ohm
from studio.cst_antenna_adapter import CSTAdapter


class InsetPatchFeedGeometryTests(unittest.TestCase):
    def setUp(self):
        self.agent = create_default_agent()
        self.adapter = CSTAdapter()

    def _design(self, changes=None):
        design = self.agent.create_design("inset_patch")
        if changes:
            design = self.agent.update_parameters(design, changes).design
        return design

    def _assert_feed_reaches_board_edge(self, design):
        values = resolve_parameter_values(design)
        geometry = {item.object_id: item for item in design.geometry}
        feed = geometry["element_1_1_feed"]
        substrate = geometry["substrate"]
        feed_dimensions = feed.dimension_map()
        substrate_dimensions = substrate.dimension_map()

        feed_x_min = evaluate_scalar(feed_dimensions["x_min"], values)
        feed_x_max = evaluate_scalar(feed_dimensions["x_max"], values)
        feed_y_min = evaluate_scalar(feed_dimensions["y_min"], values)
        feed_y_max = evaluate_scalar(feed_dimensions["y_max"], values)
        substrate_x_min = evaluate_scalar(substrate_dimensions["x_min"], values)
        substrate_x_max = evaluate_scalar(substrate_dimensions["x_max"], values)
        substrate_y_min = evaluate_scalar(substrate_dimensions["y_min"], values)

        self.assertAlmostEqual(feed_y_min, substrate_y_min, places=9)
        self.assertAlmostEqual(
            feed_y_max,
            -design.patch_length_mm / 2 + design.inset_depth_mm,
            places=9,
        )
        self.assertAlmostEqual(feed_x_max - feed_x_min, design.feed_width_mm, places=9)
        self.assertGreaterEqual(feed_x_min, substrate_x_min)
        self.assertLessEqual(feed_x_max, substrate_x_max)

        port = design.ports[0]
        for point in (port.negative_point, port.positive_point):
            self.assertAlmostEqual(evaluate_scalar(point[1], values), substrate_y_min, places=9)
        self.assertAlmostEqual(evaluate_scalar(port.negative_point[0], values), 0.0, places=9)
        self.assertAlmostEqual(evaluate_scalar(port.positive_point[0], values), 0.0, places=9)

        scene = build_geometry_scene(design)
        substrate_solid = next(item for item in scene.solids if "substrate" in item.tags)
        radiator = next(item for item in scene.solids if "patch_element" in item.tags)
        self.assertAlmostEqual(radiator.bounds[2], substrate_solid.bounds[2], places=9)
        self.assertLessEqual(radiator.bounds[2], substrate_solid.bounds[3])
        self.assertIn("element_1_1_feed", radiator.source_ids)
        self.assertEqual(scene.evaluated_operations, ("element_1_1_conductor_union",))
        self.assertFalse(scene.warnings)

        operations = self.adapter.history_operations(design)
        feed_history = next(
            item for item in operations
            if item.name == "define brick: Antenna:element_1_1_feed"
        )
        port_history = next(item for item in operations if item.name == "define port: Port 1")
        self.assertEqual(feed_history.category, "primitive")
        self.assertIn('.Yrange "-BoardL/2", "0-PatchL/2+Inset"', feed_history.script)
        self.assertIn('.SetP1 "False", "0", "-BoardL/2", "0"', port_history.script)
        self.assertIn('.SetP2 "False", "0", "-BoardL/2", "SubH"', port_history.script)

        parameters = dict(self.adapter.parameter_values(design))
        self.assertEqual(
            parameters["BoardL"],
            "PatchL+(ArrayRows-1)*ElementSpacing+2*Margin",
        )
        self.assertNotIn("FeedStub", parameters)

    def test_default_245_ghz_fr4_feed_reaches_substrate_edge(self):
        self._assert_feed_reaches_board_edge(self._design())

    def test_feed_relationship_survives_relevant_parameter_changes(self):
        cases = {
            "patch length": {"patch_length_mm": 35.0},
            "substrate margin": {"board_margin_mm": 11.5},
            "inset": {"inset_depth_mm": 6.0},
            "feed width": {"feed_width_mm": 4.2},
            "frequency-derived dimensions": {"frequency_ghz": 5.2},
        }
        baseline = self._design()
        for label, changes in cases.items():
            with self.subTest(label=label):
                design = self.agent.update_parameters(baseline, changes).design
                self._assert_feed_reaches_board_edge(design)

    def test_margin_changes_feed_length_without_moving_patch_connection(self):
        narrow = self._design({"board_margin_mm": 4.0})
        wide = self._design({"board_margin_mm": 12.0})
        values_narrow = resolve_parameter_values(narrow)
        values_wide = resolve_parameter_values(wide)
        feed_narrow = next(item for item in narrow.geometry if item.object_id == "element_1_1_feed").dimension_map()
        feed_wide = next(item for item in wide.geometry if item.object_id == "element_1_1_feed").dimension_map()

        self.assertAlmostEqual(
            evaluate_scalar(feed_narrow["y_max"], values_narrow),
            evaluate_scalar(feed_wide["y_max"], values_wide),
            places=9,
        )
        self.assertAlmostEqual(
            evaluate_scalar(feed_narrow["y_min"], values_narrow)
            - evaluate_scalar(feed_wide["y_min"], values_wide),
            8.0,
            places=9,
        )

    def test_substrate_and_material_changes_rederive_50_ohm_feed_width(self):
        default = self._design()
        thin = self.agent.update_parameters(
            default, {"substrate_thickness_mm": 0.8}
        ).design
        self.assertAlmostEqual(
            thin.feed_width_mm,
            _microstrip_width_50_ohm(MATERIALS["FR4"].epsilon_r, 0.8),
            places=9,
        )
        self.assertNotAlmostEqual(thin.feed_width_mm, 3.0, places=3)

        rogers = self.agent.update_parameters(
            thin, {"material": "Rogers RT5880"}
        ).design
        self.assertAlmostEqual(
            rogers.feed_width_mm,
            _microstrip_width_50_ohm(MATERIALS["Rogers RT5880"].epsilon_r, 0.8),
            places=9,
        )

    def test_explicit_feed_width_is_preserved_when_substrate_changes_together(self):
        changed = self.agent.update_parameters(
            self._design(),
            {"substrate_thickness_mm": 0.8, "feed_width_mm": 2.2},
        ).design
        self.assertAlmostEqual(changed.feed_width_mm, 2.2, places=9)


if __name__ == "__main__":
    unittest.main()
