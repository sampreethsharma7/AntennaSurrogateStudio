"""Fail the CI job unless VTK binds Mesa's software OpenGL.

The hosted Windows runner has no GPU. Microsoft's System32 opengl32.dll is a
GDI stub capped at OpenGL 1.1, which VTK 9.7 cannot use: it fails to find a
usable pixel format, falls back to vtkOSOpenGLRenderWindow, finds no
osmesa.dll, and the process dies partway through the suite.

The workflow fixes that by putting Mesa's software implementation where the
loader will find it first. This script proves it worked before the suite runs,
so a provisioning change that silently restored the stub is reported here as
itself instead of as a confusing crash in an unrelated test.
"""

from __future__ import annotations

import sys

import vtkmodules.vtkRenderingOpenGL2  # noqa: F401  (registers the OpenGL2 backend)
from vtkmodules.vtkRenderingCore import vtkRenderWindow

WANTED = (
    "OpenGL vendor string",
    "OpenGL renderer string",
    "OpenGL version string",
)


def main() -> int:
    window = vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(300, 200)
    # A real draw, not just a context probe: an implementation can report
    # capabilities and still fail on the first render.
    window.Render()
    report = window.ReportCapabilities()
    window.Finalize()

    described = [
        line.strip()
        for line in report.splitlines()
        if line.startswith(WANTED)
    ]
    if not described:
        print("VTK reported no OpenGL capability strings at all.", file=sys.stderr)
        return 1
    print("\n".join(described))

    if not any("llvmpipe" in line or "Mesa" in line for line in described):
        print(
            "\nVTK did not bind Mesa's software OpenGL. The GUI and VTK tests "
            "would either crash or run against an untested implementation, so "
            "this job stops here rather than reporting on them.",
            file=sys.stderr,
        )
        return 1

    print("\nMesa software OpenGL is active; the VTK tests can render headlessly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
