import platform

import customtkinter as ctk

_SYSTEM = platform.system()

if _SYSTEM == "Darwin":
    UI_FAMILY = "Helvetica Neue"
    MONO_FAMILY = "Menlo"
elif _SYSTEM == "Windows":
    UI_FAMILY = "Segoe UI"
    MONO_FAMILY = "Consolas"
else:
    UI_FAMILY = "DejaVu Sans"
    MONO_FAMILY = "DejaVu Sans Mono"

PALETTE = {
    "app_bg": "#f5f6fb",
    "surface": "#ffffff",
    "surface_alt": "#f8f9fd",
    "border": "#e4e7f1",
    "border_strong": "#c7cce0",
    "text": "#161a2b",
    "text_secondary": "#5b6072",
    "text_muted": "#9096a8",
    "accent": "#4f46e5",
    "accent_hover": "#4338ca",
    "accent_soft": "#eef0fd",
    "accent_text": "#ffffff",
    "success": "#15803d",
    "success_soft": "#eefbf1",
    "warning": "#b45309",
    "warning_soft": "#fef6e8",
    "danger": "#b91c1c",
    "danger_soft": "#fdecec",
    "disabled": "#c3c7d4",
    "sidebar_bg": "#15172b",
    "sidebar_bg_hover": "#1e2142",
    "sidebar_active": "#4f46e5",
    "sidebar_text": "#c7c9e0",
    "sidebar_text_active": "#ffffff",
    "sidebar_text_locked": "#4d5074",
    "sidebar_section_label": "#6b6f9c",
    "sidebar_border": "#262a4a",
    "sidebar_badge_bg": "#22254a",
    "sidebar_badge_done": "#22c55e",
}


def setup():
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")


def ui_font(size=13, bold=False):
    return ctk.CTkFont(family=UI_FAMILY, size=size, weight="bold" if bold else "normal")


def mono_font(size=12):
    return ctk.CTkFont(family=MONO_FAMILY, size=size)
