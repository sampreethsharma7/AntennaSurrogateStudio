"""High-contrast light and dark instrument-lab palettes for the Studio."""

import tkinter as tk
from tkinter import font as tkfont
from typing import Iterable, TypeAlias


ColorValue: TypeAlias = tuple[str, str]

LIGHT_COLORS = {
    "app_bg": "#F3F6F8",
    "surface": "#FFFFFF",
    "surface_alt": "#EDF2F5",
    "surface_elevated": "#E5EDF1",
    "hero": "#E4F2F4",
    "sidebar": "#E8EFF3",
    "sidebar_hover": "#D9E5EB",
    "nav_active": "#CDECEF",
    "ink": "#172630",
    "muted": "#536977",
    "subtle": "#617783",
    "border": "#C8D5DC",
    "border_strong": "#9EB3BF",
    "primary": "#087D8E",
    "primary_hover": "#066675",
    "primary_soft": "#D5EDF0",
    "cyan": "#08798A",
    "violet": "#6650C8",
    "violet_soft": "#E8E3F7",
    "violet_hover": "#D9D0F2",
    "success": "#16734A",
    "success_soft": "#DDF2E6",
    "warning": "#8A5B00",
    "warning_soft": "#FCECC5",
    "danger": "#C33F3F",
    "chat_user": "#D8EFF2",
    "chat_assistant": "#EAF0F3",
    "control": "#F7FAFB",
    "control_hover": "#DCE7EC",
    "disabled": "#D9E2E7",
    "disabled_text": "#6E818D",
    "scrollbar": "#B5C5CE",
    "scrollbar_hover": "#8FA7B3",
    "on_accent": "#FFFFFF",
    "sidebar_ink": "#172630",
    "hero_ink": "#172630",
    "on_violet_soft": "#4A369E",
    "on_primary": "#FFFFFF",
}

DARK_COLORS = {
    "app_bg": "#070B12",
    "surface": "#101722",
    "surface_alt": "#151E2B",
    "surface_elevated": "#1A2635",
    "hero": "#0B1821",
    "sidebar": "#090D14",
    "sidebar_hover": "#151F2C",
    "nav_active": "#12333B",
    "ink": "#E8F0F7",
    "muted": "#A7B8C7",
    "subtle": "#7F93A5",
    "border": "#2A3A49",
    "border_strong": "#3B5366",
    "primary": "#0B7F91",
    "primary_hover": "#0A6877",
    "primary_soft": "#12333B",
    "cyan": "#35D6E7",
    "violet": "#8066E8",
    "violet_soft": "#292341",
    "violet_hover": "#3A315C",
    "success": "#39D98A",
    "success_soft": "#123527",
    "warning": "#F6C453",
    "warning_soft": "#3D3013",
    "danger": "#FF6B6B",
    "chat_user": "#123C4A",
    "chat_assistant": "#192534",
    "control": "#182433",
    "control_hover": "#223244",
    "disabled": "#202B38",
    "disabled_text": "#718598",
    "scrollbar": "#314455",
    "scrollbar_hover": "#456075",
    "on_accent": "#F4F8FB",
    "sidebar_ink": "#F4F8FB",
    "hero_ink": "#F4F8FB",
    "on_violet_soft": "#F4F8FB",
    "on_primary": "#FFFFFF",
}

COLORS: dict[str, ColorValue] = {
    name: (light_color, DARK_COLORS[name])
    for name, light_color in LIGHT_COLORS.items()
}

FONTS = {
    "display": ("Segoe UI Semibold", 40),
    "title": ("Segoe UI Semibold", 31),
    "section": ("Segoe UI Semibold", 24),
    "card_title": ("Segoe UI Semibold", 21),
    "body": ("Segoe UI", 18),
    "body_small": ("Segoe UI", 17),
    "caption": ("Segoe UI", 16),
    "button": ("Segoe UI Semibold", 17),
    "mono": ("Cascadia Mono", 16),
}


# Widths are measured once per font and reused.  CustomTkinter leaves tuple
# fonts unscaled, so a measurement taken at scaling 1.0 stays valid: widget
# scaling only ever widens the box around text that never grew.
_MEASUREMENT_FONTS: dict[str, tkfont.Font] = {}
_FALLBACK_PIXELS_PER_CHARACTER = 13


def _measurement_font(font_key: str) -> tkfont.Font | None:
    """Return a cached measuring font, or None while no Tk root exists."""

    cached = _MEASUREMENT_FONTS.get(font_key)
    if cached is not None:
        return cached
    family, size, *style = FONTS[font_key]
    try:
        font = tkfont.Font(
            family=family,
            size=size,
            weight=style[0] if style else "normal",
        )
    except (RuntimeError, tk.TclError):
        return None
    _MEASUREMENT_FONTS[font_key] = font
    return font


def text_width(font_key: str, text: str) -> int:
    """Rendered width of ``text`` in unscaled pixels for a named theme font.

    Falls back to a conservative per-character estimate only when Tk cannot
    measure, so callers never have to guess a character width themselves.
    """

    for _attempt in range(2):
        font = _measurement_font(font_key)
        if font is None:
            break
        try:
            return int(font.measure(text))
        except (RuntimeError, tk.TclError):
            # The root this font belonged to is gone; drop it and re-measure.
            _MEASUREMENT_FONTS.pop(font_key, None)
    return len(text) * _FALLBACK_PIXELS_PER_CHARACTER


def widest_text_width(
    font_key: str,
    texts: Iterable[str],
    *,
    padding: int = 0,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    """Width that fits every string in ``texts``, clamped to a sane range."""

    widest = max((text_width(font_key, str(text)) for text in texts), default=0)
    widest = max(widest + padding, minimum)
    if maximum is not None:
        widest = min(widest, maximum)
    return widest


def column_safe_width(widget, target: int) -> int:
    """Width to request so a widget renders no wider than ``target`` pixels.

    CustomTkinter multiplies a widget's ``width`` argument by the current
    widget scaling, while Tk's grid ``minsize`` stays in raw pixels.  Passing a
    raw target therefore overflows its own column at any scaling above 1.0,
    which widens that row alone and pulls the heading out of alignment.
    """

    from customtkinter import ScalingTracker

    try:
        scaling = float(ScalingTracker.get_widget_scaling(widget))
    except Exception:  # pragma: no cover - scaling is unavailable before Tk
        scaling = 1.0
    if scaling <= 0:
        scaling = 1.0
    return max(1, int(target / scaling))


def status_palette(status: str) -> tuple[ColorValue, ColorValue]:
    normalized = status.lower()
    if normalized in {"data ready", "ready", "prepared"}:
        return COLORS["success_soft"], COLORS["success"]
    if normalized in {"in progress", "discovered"}:
        return COLORS["warning_soft"], COLORS["warning"]
    return COLORS["primary_soft"], COLORS["cyan"]
