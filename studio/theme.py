"""High-contrast light and dark instrument-lab palettes for the Studio."""

from typing import TypeAlias


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
    "display": ("Segoe UI Semibold", 39),
    "title": ("Segoe UI Semibold", 30),
    "section": ("Segoe UI Semibold", 23),
    "card_title": ("Segoe UI Semibold", 20),
    "body": ("Segoe UI", 17),
    "body_small": ("Segoe UI", 16),
    "caption": ("Segoe UI", 15),
    "button": ("Segoe UI Semibold", 16),
    "mono": ("Cascadia Mono", 15),
}


def status_palette(status: str) -> tuple[ColorValue, ColorValue]:
    normalized = status.lower()
    if normalized in {"data ready", "ready", "prepared"}:
        return COLORS["success_soft"], COLORS["success"]
    if normalized in {"in progress", "discovered"}:
        return COLORS["warning_soft"], COLORS["warning"]
    return COLORS["primary_soft"], COLORS["cyan"]
