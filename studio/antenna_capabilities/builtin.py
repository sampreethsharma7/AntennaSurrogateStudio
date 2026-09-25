"""Built-in, validated antenna recipes."""

from studio.antenna_recipes import register_builtin_recipes
from studio.antenna_modifiers import register_builtin_modifiers


def register_capabilities(registry) -> None:
    register_builtin_recipes(registry)
    register_builtin_modifiers(registry)
