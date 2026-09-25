"""Depth-buffered VTK visualization for evaluated antenna geometry scenes."""

from __future__ import annotations

import math
import tkinter as tk
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageTk

from studio.antenna_geometry import GeometryScene, GeometrySolid

try:
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkPolygon
    from vtkmodules.vtkFiltersCore import vtkFeatureEdges, vtkPolyDataNormals, vtkTubeFilter
    from vtkmodules.vtkFiltersSources import vtkConeSource, vtkLineSource
    from vtkmodules.vtkRenderingCore import (
        vtkActor,
        vtkBillboardTextActor3D,
        vtkPolyDataMapper,
        vtkRenderer,
        vtkRenderWindow,
        vtkTextActor,
        vtkWindowToImageFilter,
    )
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
except Exception as exc:  # pragma: no cover - exercised only on broken installations
    VTK_IMPORT_ERROR: Exception | None = exc
else:
    VTK_IMPORT_ERROR = None


def _rgb(color: str) -> tuple[float, float, float]:
    value = color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected a six-digit hex color, got {color!r}.")
    return tuple(int(value[index:index + 2], 16) / 255 for index in (0, 2, 4))


def polydata_from_solid(solid: GeometrySolid):
    """Convert one already-evaluated render mesh to VTK without geometry changes."""

    if VTK_IMPORT_ERROR is not None:
        raise RuntimeError("VTK is unavailable.") from VTK_IMPORT_ERROR
    points = vtkPoints()
    for vertex in solid.vertices:
        points.InsertNextPoint(*vertex)
    polygons = vtkCellArray()
    for face in solid.faces:
        polygon = vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(len(face))
        for offset, index in enumerate(face):
            polygon.GetPointIds().SetId(offset, index)
        polygons.InsertNextCell(polygon)
    data = vtkPolyData()
    data.SetPoints(points)
    data.SetPolys(polygons)
    return data


def polydata_matches_solid(solid: GeometrySolid, data: Any) -> bool:
    """Return whether VTK received the canonical mesh topology unchanged."""

    return (
        data.GetNumberOfPoints() == len(solid.vertices)
        and data.GetNumberOfPolys() == len(solid.faces)
    )


class VtkAntennaPreview(tk.Frame):
    """Tk-embedded VTK renderer consuming only an evaluated GeometryScene."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        background: str,
        ink: str,
        muted: str,
        danger: str,
    ) -> None:
        super().__init__(parent, background=background, highlightthickness=0, bd=0)
        self.scene: GeometryScene | None = None
        self._colors = {
            "background": background,
            "ink": ink,
            "muted": muted,
            "danger": danger,
        }
        self._mesh_actors: list[Any] = []
        self._mesh_data: list[Any] = []
        self._edge_actors: list[Any] = []
        self._port_actors: list[Any] = []
        self._port_labels: list[Any] = []
        self._view_initialized = False
        self._finalized = False
        self._render_after_id: str | None = None
        self._orbit_origin: tuple[int, int] | None = None
        self._pan_origin: tuple[int, int] | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._image_item: int | None = None
        self._available = VTK_IMPORT_ERROR is None
        if not self._available:
            self._error_label = tk.Label(
                self,
                text=(
                    "3D preview unavailable. VTK could not initialize.\n"
                    "Run setup_windows.bat, then restart the Studio."
                ),
                background=background,
                foreground=danger,
                font=("Segoe UI Semibold", 13),
                justify="center",
            )
            self._error_label.pack(fill="both", expand=True)
            return

        self._canvas = tk.Canvas(
            self,
            background=background,
            highlightthickness=0,
            bd=0,
        )
        self._canvas.pack(fill="both", expand=True)
        self._render_window = vtkRenderWindow()
        self._render_window.SetOffScreenRendering(1)
        self._render_window.SetShowWindow(False)
        self._render_window.SetNumberOfLayers(2)
        self._render_window.SetMultiSamples(8)

        self._renderer = vtkRenderer()
        self._renderer.SetLayer(0)
        self._renderer.SetBackground(*_rgb(background))
        self._renderer.SetUseFXAA(True)
        self._render_window.AddRenderer(self._renderer)

        self._annotation_renderer = vtkRenderer()
        self._annotation_renderer.SetLayer(1)
        self._annotation_renderer.SetInteractive(False)
        self._annotation_renderer.SetPreserveDepthBuffer(False)
        self._annotation_renderer.SetActiveCamera(self._renderer.GetActiveCamera())
        self._render_window.AddRenderer(self._annotation_renderer)

        self._canvas.bind("<Configure>", self._on_configure)
        self._canvas.bind("<ButtonPress-1>", self._start_orbit)
        self._canvas.bind("<B1-Motion>", self._orbit)
        self._canvas.bind("<ButtonPress-2>", self._start_pan)
        self._canvas.bind("<B2-Motion>", self._pan)
        self._canvas.bind("<ButtonPress-3>", self._start_pan)
        self._canvas.bind("<B3-Motion>", self._pan)
        self._canvas.bind("<MouseWheel>", self._wheel)
        self._canvas.bind("<Button-4>", lambda _event: self._zoom(1.12))
        self._canvas.bind("<Button-5>", lambda _event: self._zoom(1 / 1.12))
        self.bind("<Destroy>", self._on_destroy, add="+")

        self._title_actor = self._text_actor(16, bold=True, color=ink)
        self._summary_actor = self._text_actor(13, color=muted)
        self._warning_actor = self._text_actor(12, bold=True, color=danger)
        self._help_actor = self._text_actor(12, color=muted, right=True)
        for actor in (
            self._title_actor,
            self._summary_actor,
            self._warning_actor,
            self._help_actor,
        ):
            self._annotation_renderer.AddViewProp(actor)
        self._help_actor.SetInput(
            "Left-drag orbit · Right/middle-drag pan · Wheel zoom"
        )

    @property
    def available(self) -> bool:
        return self._available

    @property
    def mesh_actor_count(self) -> int:
        return len(self._mesh_actors)

    @property
    def vtk_mesh_counts(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (data.GetNumberOfPoints(), data.GetNumberOfPolys())
            for data in self._mesh_data
        )

    def _text_actor(
        self,
        size: int,
        *,
        bold: bool = False,
        color: str,
        right: bool = False,
    ):
        actor = vtkTextActor()
        prop = actor.GetTextProperty()
        prop.SetFontFamilyToArial()
        prop.SetFontSize(size)
        prop.SetBold(bold)
        prop.SetColor(*_rgb(color))
        if right:
            prop.SetJustificationToRight()
        return actor

    def _on_destroy(self, event: tk.Event) -> None:
        if event.widget is not self or self._finalized:
            return
        self._finalized = True
        try:
            self._render_window.Finalize()
        except Exception:
            pass

    def _on_configure(self, _event: tk.Event) -> None:
        self._schedule_render()

    def _start_orbit(self, event: tk.Event) -> None:
        self._orbit_origin = (event.x, event.y)

    def _orbit(self, event: tk.Event) -> None:
        if self._orbit_origin is None or not self._available:
            return
        x, y = self._orbit_origin
        self._orbit_origin = (event.x, event.y)
        camera = self._renderer.GetActiveCamera()
        camera.Azimuth(-(event.x - x) * 0.55)
        camera.Elevation((event.y - y) * 0.55)
        camera.OrthogonalizeViewUp()
        self._renderer.ResetCameraClippingRange()
        self._schedule_render(immediate=True)

    def _start_pan(self, event: tk.Event) -> None:
        self._pan_origin = (event.x, event.y)

    def _pan(self, event: tk.Event) -> None:
        if self._pan_origin is None or not self._available:
            return
        x, y = self._pan_origin
        self._pan_origin = (event.x, event.y)
        camera = self._renderer.GetActiveCamera()
        direction = np.asarray(camera.GetDirectionOfProjection(), dtype=float)
        up = np.asarray(camera.GetViewUp(), dtype=float)
        right = np.cross(direction, up)
        right_norm = np.linalg.norm(right)
        if right_norm <= 1e-12:
            return
        right /= right_norm
        up_norm = np.linalg.norm(up)
        if up_norm > 1e-12:
            up /= up_norm
        world_per_pixel = (
            2 * camera.GetParallelScale() / max(self._canvas.winfo_height(), 1)
        )
        shift = (
            right * (x - event.x) + up * (event.y - y)
        ) * world_per_pixel
        position = np.asarray(camera.GetPosition()) + shift
        focal = np.asarray(camera.GetFocalPoint()) + shift
        camera.SetPosition(*position)
        camera.SetFocalPoint(*focal)
        self._renderer.ResetCameraClippingRange()
        self._schedule_render(immediate=True)

    def _wheel(self, event: tk.Event) -> None:
        self._zoom(1.12 if event.delta > 0 else 1 / 1.12)

    def _zoom(self, factor: float) -> None:
        if not self._available:
            return
        camera = self._renderer.GetActiveCamera()
        camera.SetParallelScale(
            max(1e-6, min(1e9, camera.GetParallelScale() / factor))
        )
        self._renderer.ResetCameraClippingRange()
        self._schedule_render(immediate=True)

    def _schedule_render(self, *, immediate: bool = False) -> None:
        if not self._available or self._finalized:
            return
        if self._render_after_id is not None:
            self.after_cancel(self._render_after_id)
            self._render_after_id = None
        if immediate:
            self.render()
        else:
            self._render_after_id = self.after(35, self.render)

    def _position_overlays(self) -> None:
        if not self._available or not self.winfo_exists():
            return
        width = max(self._canvas.winfo_width(), 320)
        height = max(self._canvas.winfo_height(), 260)
        self._title_actor.SetDisplayPosition(18, height - 32)
        self._summary_actor.SetDisplayPosition(18, height - 56)
        self._warning_actor.SetDisplayPosition(18, height - 82)
        self._help_actor.SetDisplayPosition(width - 16, 14)

    def _solid_actor(self, solid: GeometrySolid):
        data = polydata_from_solid(solid)
        normals = vtkPolyDataNormals()
        normals.SetInputData(data)
        normals.SetFeatureAngle(40)
        normals.SplittingOn()
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOff()
        normals.ComputePointNormalsOn()
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())
        mapper.ScalarVisibilityOff()
        actor = vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(*_rgb(solid.color))
        prop.EdgeVisibilityOff()
        prop.SetInterpolationToPhong()
        if "substrate" in solid.tags:
            prop.SetAmbient(0.34)
            prop.SetDiffuse(0.62)
            prop.SetSpecular(0.06)
        else:
            prop.SetAmbient(0.28)
            prop.SetDiffuse(0.66)
            prop.SetSpecular(0.28)
            prop.SetSpecularPower(28)
        return actor, data

    def _edge_actor(self, data: Any):
        edges = vtkFeatureEdges()
        edges.SetInputData(data)
        edges.BoundaryEdgesOn()
        edges.FeatureEdgesOn()
        edges.NonManifoldEdgesOn()
        edges.ManifoldEdgesOff()
        edges.SetFeatureAngle(35)
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(edges.GetOutputPort())
        mapper.ScalarVisibilityOff()
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.12, 0.18, 0.22)
        actor.GetProperty().SetOpacity(0.62)
        actor.GetProperty().SetLineWidth(1.0)
        return actor

    def _add_port(
        self,
        number: int,
        negative: tuple[float, float, float],
        positive: tuple[float, float, float],
        scene_span: float,
    ) -> None:
        delta = tuple(positive[index] - negative[index] for index in range(3))
        length = math.sqrt(sum(value * value for value in delta))
        if length <= 1e-12:
            return
        direction = tuple(value / length for value in delta)
        radius = max(scene_span * 0.0025, 0.055)

        line = vtkLineSource()
        line.SetPoint1(*negative)
        line.SetPoint2(*positive)
        tube = vtkTubeFilter()
        tube.SetInputConnection(line.GetOutputPort())
        tube.SetRadius(radius)
        tube.SetNumberOfSides(16)
        tube.CappingOn()
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())
        shaft = vtkActor()
        shaft.SetMapper(mapper)
        shaft.GetProperty().SetColor(0.89, 0.16, 0.19)

        cone_height = max(scene_span * 0.025, radius * 4)
        cone = vtkConeSource()
        cone.SetResolution(24)
        cone.SetRadius(radius * 2.5)
        cone.SetHeight(cone_height)
        cone.SetDirection(*direction)
        cone.SetCenter(
            *(positive[index] - direction[index] * cone_height / 2 for index in range(3))
        )
        cone_mapper = vtkPolyDataMapper()
        cone_mapper.SetInputConnection(cone.GetOutputPort())
        tip = vtkActor()
        tip.SetMapper(cone_mapper)
        tip.GetProperty().SetColor(0.89, 0.16, 0.19)

        label = vtkBillboardTextActor3D()
        label.SetInput(f"P{number}")
        offset = max(scene_span * 0.009, radius * 3)
        label.SetPosition(
            positive[0] + offset,
            positive[1] + offset,
            positive[2] + offset,
        )
        text = label.GetTextProperty()
        text.SetFontFamilyToArial()
        text.SetFontSize(16)
        text.SetBold(True)
        text.SetColor(0.89, 0.16, 0.19)

        for actor in (shaft, tip):
            self._annotation_renderer.AddActor(actor)
            self._port_actors.append(actor)
        self._annotation_renderer.AddActor(label)
        self._port_labels.append(label)

    def set_scene(self, scene: GeometryScene) -> None:
        self.scene = scene
        if not self._available:
            return
        for actor in (*self._mesh_actors, *self._edge_actors):
            self._renderer.RemoveActor(actor)
        for actor in (*self._port_actors, *self._port_labels):
            self._annotation_renderer.RemoveActor(actor)
        self._mesh_actors.clear()
        self._mesh_data.clear()
        self._edge_actors.clear()
        self._port_actors.clear()
        self._port_labels.clear()

        for solid in scene.solids:
            actor, data = self._solid_actor(solid)
            self._renderer.AddActor(actor)
            self._mesh_actors.append(actor)
            self._mesh_data.append(data)
            edge_actor = self._edge_actor(data)
            self._renderer.AddActor(edge_actor)
            self._edge_actors.append(edge_actor)

        span = max(
            scene.bounds[1] - scene.bounds[0],
            scene.bounds[3] - scene.bounds[2],
            scene.bounds[5] - scene.bounds[4],
            1.0,
        )
        for number, (negative, positive) in enumerate(scene.port_segments, start=1):
            self._add_port(number, negative, positive, span)

        x_span = scene.bounds[1] - scene.bounds[0]
        y_span = scene.bounds[3] - scene.bounds[2]
        z_span = scene.bounds[5] - scene.bounds[4]
        self._title_actor.SetInput(scene.state.topology_label)
        self._summary_actor.SetInput(
            f"Envelope {x_span:.1f} × {y_span:.1f} × {z_span:.1f} mm  ·  "
            f"{len(scene.port_segments)} port{'s' if len(scene.port_segments) != 1 else ''}"
        )
        self._warning_actor.SetInput(
            "Preview warning: " + " ".join(scene.warnings) if scene.warnings else ""
        )
        if not self._view_initialized:
            self.reset_view()
            self._view_initialized = True
        else:
            self._renderer.ResetCameraClippingRange(scene.bounds)
            self.render()
        self.after_idle(self._position_overlays)

    def clear_scene(self) -> None:
        """Remove rendered geometry when the canonical project has no design."""

        self.scene = None
        if not self._available:
            return
        for actor in (*self._mesh_actors, *self._edge_actors):
            self._renderer.RemoveActor(actor)
        for actor in (*self._port_actors, *self._port_labels):
            self._annotation_renderer.RemoveActor(actor)
        self._mesh_actors.clear()
        self._mesh_data.clear()
        self._edge_actors.clear()
        self._port_actors.clear()
        self._port_labels.clear()
        self._title_actor.SetInput("")
        self._summary_actor.SetInput("")
        self._warning_actor.SetInput("")
        self._schedule_render(immediate=True)

    def reset_view(self) -> None:
        if not self._available or self.scene is None:
            return
        bounds = self.scene.bounds
        center = (
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2,
        )
        span = max(
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4],
            1.0,
        )
        camera = self._renderer.GetActiveCamera()
        camera.SetFocalPoint(*center)
        camera.SetPosition(
            center[0] + span * 1.25,
            center[1] - span * 1.45,
            center[2] + span * 1.05,
        )
        camera.SetViewUp(0, 0, 1)
        camera.ParallelProjectionOn()
        camera.SetParallelScale(span * 0.66)
        self._renderer.ResetCameraClippingRange(bounds)
        self.render()

    def refresh_theme(
        self,
        *,
        background: str | None = None,
        ink: str | None = None,
        muted: str | None = None,
        danger: str | None = None,
    ) -> None:
        updates = {
            "background": background,
            "ink": ink,
            "muted": muted,
            "danger": danger,
        }
        self._colors.update({key: value for key, value in updates.items() if value is not None})
        self.configure(background=self._colors["background"])
        if not self._available:
            self._error_label.configure(
                background=self._colors["background"],
                foreground=self._colors["danger"],
            )
            return
        self._canvas.configure(background=self._colors["background"])
        self._renderer.SetBackground(*_rgb(self._colors["background"]))
        for actor, key in (
            (self._title_actor, "ink"),
            (self._summary_actor, "muted"),
            (self._warning_actor, "danger"),
            (self._help_actor, "muted"),
        ):
            actor.GetTextProperty().SetColor(*_rgb(self._colors[key]))
        self.render()

    def render(self) -> None:
        self._render_after_id = None
        if not self._available or self._finalized or not self.winfo_exists():
            return
        width = max(self._canvas.winfo_width(), 320)
        height = max(self._canvas.winfo_height(), 260)
        self._position_overlays()
        self._render_window.SetSize(width, height)
        self._render_window.Render()
        capture = vtkWindowToImageFilter()
        capture.SetInput(self._render_window)
        capture.SetInputBufferTypeToRGBA()
        capture.ReadFrontBufferOff()
        capture.Update()
        output = capture.GetOutput()
        dimensions = output.GetDimensions()
        scalars = output.GetPointData().GetScalars()
        pixels = vtk_to_numpy(scalars).reshape(dimensions[1], dimensions[0], 4)
        image = Image.fromarray(np.flipud(pixels).copy(), mode="RGBA")
        self._photo = ImageTk.PhotoImage(image=image, master=self._canvas)
        if self._image_item is None:
            self._image_item = self._canvas.create_image(
                0, 0, anchor="nw", image=self._photo
            )
        else:
            self._canvas.itemconfigure(self._image_item, image=self._photo)

    def save_screenshot(self, path: str | Path) -> Path:
        """Capture the current VTK framebuffer for visual regression evidence."""

        if not self._available:
            raise RuntimeError("VTK preview is unavailable.") from VTK_IMPORT_ERROR
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.render()
        if self._photo is None:
            raise RuntimeError("VTK did not produce a framebuffer image.")
        capture = vtkWindowToImageFilter()
        capture.SetInput(self._render_window)
        capture.SetInputBufferTypeToRGBA()
        capture.ReadFrontBufferOff()
        capture.Update()
        output = capture.GetOutput()
        dimensions = output.GetDimensions()
        pixels = vtk_to_numpy(output.GetPointData().GetScalars()).reshape(
            dimensions[1], dimensions[0], 4
        )
        Image.fromarray(np.flipud(pixels).copy(), mode="RGBA").save(destination)
        return destination
