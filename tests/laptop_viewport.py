"""Hold the UI scale still at the laptop viewport the layout tests describe.

`responsive_window_layout` derives `ui_scaling` from the host's screen, and
deliberately shrinks the interface below the design scale when that screen is
narrower than `DESIGN_MIN_WIDTH`. Every physical widget size moves with that
factor, so a test that asserts design-pixel minima is really asserting the
host's monitor.

A real 1366x768 laptop lands between 1.00 and 1.08 across 100%-150% Windows
scaling, which is what those minima were written for. A headless 1024x768
session lands at 1024/1260 = 0.81, and there the same correct layout reports a
149px primary action against a 170px expectation, and docks SnowBuddy where a
laptop shows it in focus view. Neither is a product defect: at 0.81 the fonts
shrink with the boxes and the docked page still clears its own minima.

`pin_laptop_ui_scale` leaves the real screen geometry alone, so the window
still opens at a size that fits the host, and pins only the scale factor. The
tests then describe the laptop they name, on any machine.
"""

from __future__ import annotations

import dataclasses
from unittest import mock

import studio.ui as ui

LAPTOP_WIDTH = 1366
LAPTOP_HEIGHT = 768


def pin_laptop_ui_scale(test_class: type) -> None:
    """Pin the UI scale to the laptop viewport for the whole of `test_class`.

    The patch has to outlive app construction, because the Studio recomputes
    the layout from the live monitor whenever the window reports a DPI change,
    which a test that resizes the window triggers.
    """

    real = ui.responsive_window_layout

    def pinned(screen_width, screen_height, dpi_scaling):
        layout = real(screen_width, screen_height, dpi_scaling)
        dpi = max(1.0, float(dpi_scaling or 1.0))
        # The scale a 1366x768 physical viewport earns at this DPI, which is
        # what the laptop layout contract is written against.
        scale = real(LAPTOP_WIDTH / dpi, LAPTOP_HEIGHT / dpi, dpi).ui_scaling
        return dataclasses.replace(
            layout,
            ui_scaling=scale,
            widget_scaling_factor=scale / dpi,
        )

    patcher = mock.patch.object(ui, "responsive_window_layout", pinned)
    patcher.start()
    test_class.addClassCleanup(patcher.stop)
