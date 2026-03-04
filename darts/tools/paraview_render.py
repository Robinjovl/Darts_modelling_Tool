import argparse
import json
import math
import os
import re
import shutil
import subprocess
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Snapshot:
    """
    Container describing one snapshot entry from a PVD collection.

    :param index: Zero-based snapshot index in the parsed ``.pvd`` file order.
    :type index: int
    :param timestep: Snapshot time value parsed from the PVD ``timestep`` attribute.
    :type timestep: float
    :param file_path: Absolute path to the corresponding VTK dataset file.
    :type file_path: pathlib.Path
    """

    index: int
    timestep: float
    file_path: Path


@dataclass
class FieldSpec:
    """
    Description of a scalar/vector array available in a VTK dataset.

    :param association: Data association name (``CELLS`` or ``POINTS``).
    :type association: str
    :param name: VTK array name used for coloring.
    :type name: str
    :param n_components: Number of array components.
    :type n_components: int
    """

    association: str  # CELLS or POINTS
    name: str
    n_components: int


class ParaViewMultiViewRenderer:
    """
    Headless-safe multi-view renderer for VTK snapshots referenced by a ``.pvd`` file.

    The default implementation uses VTK offscreen rendering so it can run on remote
    servers without X display. A compatibility ParaView probe backend is available
    explicitly, but final rendering is still delegated to the robust VTK pipeline.
    """

    def __init__(
        self,
        pvd_file: str,
        output_dir: str | None = None,
        *,
        backend: str = "auto",
        timestep_mode: str = "all",
        timestep_stride: int = 1,
        fields: Sequence[str] | None = None,
        field_association: str = "both",
        columns: int | None = None,
        image_width: int = 1920,
        image_height: int = 1080,
        image_prefix: str = "snapshot",
        image_compression_level: int = 5,
        overwrite_images: bool = True,
        color_range_mode: str = "global",
        field_ranges: dict[str, Sequence[float]] | None = None,
        colormap: str = "coolwarm",
        colormap_reverse: bool = False,
        show_scalar_bars: bool = True,
        scalar_bar_position: Sequence[float] = (0.78, 0.12),
        scalar_bar_width: float = 0.11,
        scalar_bar_height: float = 0.76,
        scalar_bar_bar_ratio: float = 0.2,
        scalar_bar_font_size: int | None = None,
        scalar_bar_text_color: Sequence[float] | None = None,
        show_cell_edges: bool = True,
        edge_line_width: float = 0.5,
        edge_color: Sequence[float] = (0.15, 0.15, 0.15),
        wells_vtk_file: str | None = None,
        show_wells: bool = True,
        wells_color: Sequence[float] = (0.15, 0.15, 0.15),
        wells_opacity: float = 1.0,
        wells_line_width: float = 1.0,
        show_orientation_axes: bool = True,
        orientation_axes_labels: Sequence[str] = ("X", "Y", "Z"),
        orientation_axes_position: Sequence[float] = (0.01, 0.01),
        orientation_axes_size: float = 0.30,
        orientation_axes_label_font_size: int | None = None,
        orientation_axes_text_color: Sequence[float] = (0.0, 0.0, 0.0),
        show_field_titles: bool = False,
        background_color: Sequence[float] = (1.0, 1.0, 1.0),
        camera: dict | None = None,
        annotate_time: bool = True,
        annotate_view_index: int = 0,
        annotate_template: str = "t = {time:.3f}",
        annotate_position: Sequence[float] = (0.02, 0.92),
        annotate_font_size: int = 34,
        annotate_color: Sequence[float] = (0.0, 0.0, 0.0),
        video_enabled: bool = True,
        video_filename: str = "multiview",
        video_format: str = "ogv",
        video_length_sec: float = 12.0,
        video_fps: int | None = None,
        video_quality: int = 80,
        ffmpeg_executable: str | None = None,
        pvpython_executable: str | None = None,
        verbose: bool = False,
    ):
        """
        Initialize a renderer instance and normalize all rendering options.

        :param pvd_file: Path to input ``.pvd`` file listing VTK snapshots.
        :type pvd_file: str
        :param output_dir: Directory where rendered images/video and manifest are saved.
            If ``None``, ``<pvd_dir>/paraview_renders`` is used.
        :type output_dir: str, optional
        :param backend: Rendering backend selection: ``auto``, ``vtk`` or ``paraview``.
            ``auto`` tries VTK first.
        :type backend: str
        :param timestep_mode: Snapshot selection mode: ``all``, ``latest`` or ``first``.
        :type timestep_mode: str
        :param timestep_stride: Keep every N-th snapshot when ``timestep_mode='all'``.
        :type timestep_stride: int
        :param fields: Optional list of field tokens to render. Supported token forms:
            ``field_name``, ``CELLS:field_name``, ``POINTS:field_name``.
            If ``None``, all available fields are rendered.
        :type fields: Sequence[str], optional
        :param field_association: Field source to scan in datasets:
            ``cell``, ``point`` or ``both``.
        :type field_association: str
        :param columns: Optional fixed number of columns in layout. If ``None``,
            layout columns are computed automatically.
        :type columns: int, optional
        :param image_width: Output image width in pixels.
        :type image_width: int
        :param image_height: Output image height in pixels.
        :type image_height: int
        :param image_prefix: Prefix used when naming frame images.
        :type image_prefix: str
        :param image_compression_level: PNG compression level in ``[0, 9]``.
        :type image_compression_level: int
        :param overwrite_images: Whether to overwrite existing frame images.
        :type overwrite_images: bool
        :param color_range_mode: Color range mode. ``global`` fixes ranges across
            selected snapshots; ``frame`` recomputes range per frame.
        :type color_range_mode: str
        :param field_ranges: Optional per-field fixed range overrides ``(min, max)``.
            Keys may be unqualified field names (if unambiguous) or qualified names
            ``CELLS:...`` / ``POINTS:...``.
        :type field_ranges: Dict[str, Sequence[float]], optional
        :param colormap: Colormap name understood by Matplotlib (default ``coolwarm``).
        :type colormap: str
        :param colormap_reverse: Whether to reverse colormap direction.
        :type colormap_reverse: bool
        :param show_scalar_bars: Whether to draw scalar bars in each view.
        :type show_scalar_bars: bool
        :param scalar_bar_position: Scalar bar position in normalized viewport coordinates.
        :type scalar_bar_position: Sequence[float]
        :param scalar_bar_width: Scalar bar width in normalized viewport units.
        :type scalar_bar_width: float
        :param scalar_bar_height: Scalar bar height in normalized viewport units.
        :type scalar_bar_height: float
        :param scalar_bar_bar_ratio: Fraction of scalar bar actor width used for
            the color strip.
        :type scalar_bar_bar_ratio: float
        :param scalar_bar_font_size: Scalar bar text font size. If ``None``, annotation
            font size is reused.
        :type scalar_bar_font_size: int, optional
        :param scalar_bar_text_color: Scalar bar text RGB color in ``[0, 1]``.
            If ``None``, annotation color is reused.
        :type scalar_bar_text_color: Sequence[float], optional
        :param show_cell_edges: Whether to draw grid/cell edges on top of surfaces.
        :type show_cell_edges: bool
        :param edge_line_width: Width of rendered cell edges.
        :type edge_line_width: float
        :param edge_color: Cell edge RGB color in ``[0, 1]``.
        :type edge_color: Sequence[float]
        :param wells_vtk_file: Optional VTK/VTU/VTP file containing wells geometry
            to overlay in every rendered view.
        :type wells_vtk_file: str, optional
        :param show_wells: Whether to render wells overlay when
            ``wells_vtk_file`` is provided.
        :type show_wells: bool
        :param wells_color: Wells actor RGB color in ``[0, 1]``.
        :type wells_color: Sequence[float]
        :param wells_opacity: Wells actor opacity in ``[0, 1]``.
        :type wells_opacity: float
        :param wells_line_width: Wells line width for line-based well geometry.
        :type wells_line_width: float
        :param show_orientation_axes: Whether to draw orientation axes in each view.
        :type show_orientation_axes: bool
        :param orientation_axes_labels: Axis labels for orientation marker in order
            ``(X, Y, Z)``.
        :type orientation_axes_labels: Sequence[str]
        :param orientation_axes_position: Lower-left marker position inside each view,
            in normalized per-view coordinates.
        :type orientation_axes_position: Sequence[float]
        :param orientation_axes_size: Marker size as fraction of per-view width/height.
        :type orientation_axes_size: float
        :param orientation_axes_label_font_size: Orientation axis label font size.
            If ``None``, ``annotate_font_size`` is used.
        :type orientation_axes_label_font_size: int, optional
        :param orientation_axes_text_color: RGB color for orientation axis labels in
            ``[0, 1]``.
        :type orientation_axes_text_color: Sequence[float]
        :param show_field_titles: Whether to draw ``CELLS:name`` / ``POINTS:name``
            labels inside each view.
        :type show_field_titles: bool
        :param background_color: Render background RGB color in ``[0, 1]``.
        :type background_color: Sequence[float]
        :param camera: Optional dictionary with shared camera controls. Supported keys:
            ``reset_to_data_bounds``, ``position``, ``focal_point``, ``view_up``,
            ``clipping_range``, ``parallel_projection``, ``parallel_scale``,
            ``view_angle``, ``azimuth``, ``elevation``, ``roll``, ``dolly``,
            ``zoom``, ``fit_margin``.
        :type camera: Dict, optional
        :param annotate_time: Whether to draw timestep annotation.
        :type annotate_time: bool
        :param annotate_view_index: View index where time annotation is shown.
        :type annotate_view_index: int
        :param annotate_template: Format template for annotation text. Available fields:
            ``time``, ``step``, ``frame``, ``file``.
        :type annotate_template: str
        :param annotate_position: Annotation position in normalized viewport coordinates.
        :type annotate_position: Sequence[float]
        :param annotate_font_size: Time annotation font size.
        :type annotate_font_size: int
        :param annotate_color: Time annotation RGB color in ``[0, 1]``.
        :type annotate_color: Sequence[float]
        :param video_enabled: Whether to assemble a video from rendered frames.
        :type video_enabled: bool
        :param video_filename: Output video base filename without extension.
        :type video_filename: str
        :param video_format: Video format: ``ogv``, ``mp4`` or ``gif``.
        :type video_format: str
        :param video_length_sec: Target video duration in seconds when fps is not set.
        :type video_length_sec: float
        :param video_fps: Explicit frames-per-second. If ``None``, inferred from
            ``video_length_sec``.
        :type video_fps: int, optional
        :param video_quality: Requested quality in ``[0, 100]`` used by encoders.
        :type video_quality: int
        :param ffmpeg_executable: Optional explicit path to ``ffmpeg`` for MP4 encoding.
        :type ffmpeg_executable: str, optional
        :param pvpython_executable: Optional explicit path to ``pvpython`` for
            compatibility probing with ``backend='paraview'``.
        :type pvpython_executable: str, optional
        :param verbose: Enable verbose logging to stdout.
        :type verbose: bool
        :raises FileNotFoundError: If ``pvd_file`` does not exist.
        """
        self.pvd_path = Path(pvd_file).expanduser().resolve()
        if not self.pvd_path.exists():
            raise FileNotFoundError(f"PVD file does not exist: {self.pvd_path}")

        self.output_dir = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else (self.pvd_path.parent / "paraview_renders").resolve()
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.backend = backend.lower().strip()
        self.timestep_mode = timestep_mode.lower().strip()
        self.timestep_stride = max(1, int(timestep_stride))
        self.requested_fields = list(fields) if fields is not None else None
        self.field_association = field_association.lower().strip()
        self.columns = columns
        self.image_width = int(max(320, image_width))
        self.image_height = int(max(240, image_height))
        self.image_prefix = image_prefix
        self.image_compression_level = int(min(9, max(0, image_compression_level)))
        self.overwrite_images = bool(overwrite_images)
        self.color_range_mode = color_range_mode.lower().strip()
        self.field_ranges = dict(field_ranges) if field_ranges is not None else {}
        self.colormap = str(colormap).strip()
        self.colormap_reverse = bool(colormap_reverse)

        self.show_scalar_bars = bool(show_scalar_bars)
        self.scalar_bar_position = (
            float(scalar_bar_position[0]),
            float(scalar_bar_position[1]),
        )
        self.scalar_bar_width = float(scalar_bar_width)
        self.scalar_bar_height = float(scalar_bar_height)
        self.scalar_bar_bar_ratio = float(scalar_bar_bar_ratio)
        self.show_cell_edges = bool(show_cell_edges)
        self.edge_line_width = float(max(0.1, edge_line_width))
        self.edge_color = self._validate_rgb(edge_color)
        wells_path_raw = (
            str(wells_vtk_file).strip() if wells_vtk_file is not None else ""
        )
        self.wells_vtk_path = (
            Path(wells_path_raw).expanduser().resolve() if wells_path_raw else None
        )
        self.show_wells = bool(show_wells)
        self.wells_color = self._validate_rgb(wells_color)
        self.wells_opacity = min(1.0, max(0.0, float(wells_opacity)))
        self.wells_line_width = float(max(0.1, wells_line_width))
        if self.wells_vtk_path is None:
            self.show_wells = False
        elif not self.wells_vtk_path.exists():
            warnings.warn(
                f"Wells VTK file does not exist and will be ignored: {self.wells_vtk_path}",
                stacklevel=2,
            )
            self.wells_vtk_path = None
            self.show_wells = False
        self.show_orientation_axes = bool(show_orientation_axes)
        if len(orientation_axes_labels) != 3:
            raise ValueError(
                "orientation_axes_labels must contain exactly 3 labels (X, Y, Z)."
            )
        self.orientation_axes_labels = tuple(str(v) for v in orientation_axes_labels)
        self.orientation_axes_position = (
            float(orientation_axes_position[0]),
            float(orientation_axes_position[1]),
        )
        self.orientation_axes_size = float(min(0.95, max(0.05, orientation_axes_size)))
        default_orientation_font_size = (
            annotate_font_size
            if orientation_axes_label_font_size is None
            else orientation_axes_label_font_size
        )
        self.orientation_axes_label_font_size = int(
            max(8, default_orientation_font_size)
        )
        self.orientation_axes_text_color = self._validate_rgb(
            orientation_axes_text_color
        )
        self.show_field_titles = bool(show_field_titles)
        self.background_color = self._validate_rgb(background_color)

        self.camera = {
            "reset_to_data_bounds": True,
            "position": None,
            "focal_point": None,
            "view_up": None,
            "clipping_range": None,
            "parallel_projection": None,
            "parallel_scale": None,
            "view_angle": None,
            "azimuth": -20.0,
            "elevation": -45.0,
            "roll": 0.0,
            "dolly": 1.0,
            "zoom": 1.2,
            "fit_margin": 0.02,
        }
        if camera:
            self.camera.update(camera)

        self.annotate_time = bool(annotate_time)
        self.annotate_view_index = int(annotate_view_index)
        self.annotate_template = annotate_template
        self.annotate_position = (
            float(annotate_position[0]),
            float(annotate_position[1]),
        )
        self.annotate_font_size = int(max(8, annotate_font_size))
        self.annotate_color = self._validate_rgb(annotate_color)
        self.scalar_bar_font_size = (
            int(max(8, scalar_bar_font_size))
            if scalar_bar_font_size is not None
            else int(max(8, self.annotate_font_size))
        )
        self.scalar_bar_text_color = (
            self._validate_rgb(scalar_bar_text_color)
            if scalar_bar_text_color is not None
            else self.annotate_color
        )

        self.video_enabled = bool(video_enabled)
        self.video_filename = video_filename
        self.video_format = video_format.lower().strip()
        self.video_length_sec = float(max(0.1, video_length_sec))
        self.video_fps = int(video_fps) if video_fps is not None else None
        self.video_quality = int(video_quality)
        self.ffmpeg_executable = ffmpeg_executable

        self.pvpython_executable = pvpython_executable
        self.verbose = bool(verbose)

    def render(self) -> dict:
        """
        Render selected snapshots and return render artifacts metadata.

        :returns: Rendering result dictionary containing output paths, selected
            fields, timesteps and manifest path.
        :rtype: Dict
        :raises RuntimeError: If no snapshots are found or all backends fail.
        :raises ValueError: If backend selection contains unsupported value.
        """
        snapshots = self._select_snapshots(self._parse_pvd_snapshots())
        if not snapshots:
            raise RuntimeError(f"No snapshots were found in '{self.pvd_path}'.")

        backend_order = self._resolve_backend_order()
        errors = []
        for backend in backend_order:
            try:
                if backend == "vtk":
                    return self._render_with_vtk(snapshots)
                if backend == "paraview":
                    return self._render_with_paraview_subprocess()
                raise ValueError(f"Unsupported backend '{backend}'.")
            except Exception as exc:
                errors.append((backend, str(exc)))
                if self.verbose:
                    print(f"Render backend '{backend}' failed: {exc}")

        raise RuntimeError(
            "All rendering backends failed: "
            + "; ".join([f"{b}: {e}" for b, e in errors])
        )

    @staticmethod
    def _validate_rgb(color: Sequence[float]) -> tuple[float, float, float]:
        """
        Validate and clamp an RGB triplet to ``[0, 1]``.

        :param color: RGB color sequence with exactly three components.
        :type color: Sequence[float]
        :returns: Normalized RGB tuple.
        :rtype: Tuple[float, float, float]
        :raises ValueError: If input length is not equal to three.
        """
        if len(color) != 3:
            raise ValueError(f"RGB color must have 3 elements, got {len(color)}")
        vals = [float(c) for c in color]
        return (
            min(1.0, max(0.0, vals[0])),
            min(1.0, max(0.0, vals[1])),
            min(1.0, max(0.0, vals[2])),
        )

    def _resolve_backend_order(self) -> list[str]:
        """
        Resolve backend execution order from current backend mode.

        :returns: Ordered backend names to try.
        :rtype: List[str]
        :raises ValueError: If backend option is unsupported.
        """
        if self.backend == "auto":
            # VTK backend is selected first because it is robust on no-X servers.
            return ["vtk", "paraview"]
        if self.backend in ("vtk", "paraview"):
            return [self.backend]
        raise ValueError(
            f"Unsupported backend '{self.backend}'. Use 'auto', 'vtk' or 'paraview'."
        )

    def _parse_pvd_snapshots(self) -> list[Snapshot]:
        """
        Parse snapshot entries from the input PVD file.

        :returns: Parsed snapshot descriptors in PVD order.
        :rtype: List[Snapshot]
        :raises FileNotFoundError: If a referenced VTK file does not exist.
        """
        tree = ET.parse(self.pvd_path)
        root = tree.getroot()

        snapshots: list[Snapshot] = []
        idx = 0
        for elem in root.iter():
            if not elem.tag.endswith("DataSet"):
                continue
            rel_file = elem.attrib.get("file")
            if not rel_file:
                continue
            timestep_str = elem.attrib.get("timestep", str(idx))
            try:
                timestep = float(timestep_str)
            except ValueError:
                timestep = float(idx)

            file_path = (self.pvd_path.parent / rel_file).resolve()
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Snapshot file referenced by PVD does not exist: {file_path}"
                )
            snapshots.append(
                Snapshot(index=idx, timestep=timestep, file_path=file_path)
            )
            idx += 1

        return snapshots

    def _select_snapshots(self, snapshots: list[Snapshot]) -> list[Snapshot]:
        """
        Select snapshots according to ``timestep_mode`` and ``timestep_stride``.

        :param snapshots: Parsed snapshots from :meth:`_parse_pvd_snapshots`.
        :type snapshots: List[Snapshot]
        :returns: Filtered snapshot list to render.
        :rtype: List[Snapshot]
        :raises ValueError: If ``timestep_mode`` is unsupported.
        """
        mode = self.timestep_mode
        if mode == "all":
            selected = snapshots
        elif mode == "latest":
            selected = snapshots[-1:]
        elif mode == "first":
            selected = snapshots[:1]
        else:
            raise ValueError(
                f"Unsupported timestep_mode='{mode}'. Supported: all, latest, first."
            )

        if self.timestep_stride > 1 and len(selected) > 1:
            selected = selected[:: self.timestep_stride]
            if selected[-1].index != snapshots[-1].index and mode == "all":
                selected.append(snapshots[-1])
        return selected

    @staticmethod
    def _sanitize_token(value: str) -> str:
        """
        Sanitize arbitrary text for safe inclusion in file names.

        :param value: Input token.
        :type value: str
        :returns: Token with non-safe characters replaced by underscores.
        :rtype: str
        """
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")

    def _snapshot_image_path(self, frame_idx: int, snapshot: Snapshot) -> Path:
        """
        Build output PNG path for a rendered snapshot frame.

        :param frame_idx: Zero-based frame index in rendering order.
        :type frame_idx: int
        :param snapshot: Snapshot metadata for naming.
        :type snapshot: Snapshot
        :returns: Absolute output path for the frame image.
        :rtype: pathlib.Path
        """
        t_token = self._sanitize_token(f"{snapshot.timestep:g}")
        name = f"{self.image_prefix}_{frame_idx:05d}_t{t_token}.png"
        return self.output_dir / name

    @staticmethod
    def _create_vtk_reader(file_path: Path):
        """
        Create a VTK reader instance based on dataset file extension.

        :param file_path: Snapshot dataset file path.
        :type file_path: pathlib.Path
        :returns: Configured VTK reader with file name set.
        :rtype: vtk.vtkAlgorithm
        """
        import vtk

        suffix = file_path.suffix.lower()
        if suffix == ".vts":
            reader = vtk.vtkXMLStructuredGridReader()
        elif suffix == ".vtu":
            reader = vtk.vtkXMLUnstructuredGridReader()
        elif suffix == ".vtr":
            reader = vtk.vtkXMLRectilinearGridReader()
        elif suffix == ".vti":
            reader = vtk.vtkXMLImageDataReader()
        elif suffix == ".vtp":
            reader = vtk.vtkXMLPolyDataReader()
        elif suffix == ".vtk":
            reader = vtk.vtkDataSetReader()
        else:
            reader = vtk.vtkXMLGenericDataObjectReader()
        reader.SetFileName(str(file_path))
        return reader

    @staticmethod
    def _data_object_from_reader(reader):
        """
        Execute reader pipeline and return produced VTK data object.

        :param reader: VTK reader instance.
        :type reader: vtk.vtkAlgorithm
        :returns: Reader output dataset object if available, otherwise ``None``.
        :rtype: vtk.vtkDataObject or None
        """
        reader.Update()
        if hasattr(reader, "GetOutputDataObject"):
            obj = reader.GetOutputDataObject(0)
            if obj is not None:
                return obj
        if hasattr(reader, "GetOutput"):
            return reader.GetOutput()
        return None

    @staticmethod
    def _collect_fields_from_dataset(dataset, association_mode: str) -> list[FieldSpec]:
        """
        Collect available data arrays from a VTK dataset.

        :param dataset: VTK dataset used as field probe source.
        :type dataset: vtk.vtkDataSet
        :param association_mode: Which associations to include:
            ``cell``, ``point`` or ``both``.
        :type association_mode: str
        :returns: Unique field specifications preserving discovery order.
        :rtype: List[FieldSpec]
        :raises ValueError: If ``association_mode`` is unsupported.
        """
        fields: list[FieldSpec] = []

        def _collect(data, assoc: str):
            if data is None:
                return
            for i in range(data.GetNumberOfArrays()):
                arr = data.GetArray(i)
                if arr is None:
                    continue
                name = arr.GetName()
                if not name:
                    name = f"{assoc}_{i}"
                fields.append(
                    FieldSpec(
                        association=assoc,
                        name=name,
                        n_components=int(arr.GetNumberOfComponents()),
                    )
                )

        mode = association_mode
        if mode not in ("cell", "point", "both"):
            raise ValueError(
                f"Unsupported field_association='{association_mode}'. Supported: cell, point, both."
            )

        if mode in ("cell", "both"):
            _collect(dataset.GetCellData(), "CELLS")
        if mode in ("point", "both"):
            _collect(dataset.GetPointData(), "POINTS")

        dedup: list[FieldSpec] = []
        seen = set()
        for f in fields:
            key = (f.association, f.name)
            if key not in seen:
                dedup.append(f)
                seen.add(key)
        return dedup

    @staticmethod
    def _parse_field_token(token: str) -> tuple[str | None, str]:
        """
        Parse user field token into optional association hint and field name.

        :param token: Field token such as ``pressure``, ``CELLS:pressure``,
            ``POINTS:temperature``.
        :type token: str
        :returns: Tuple ``(association_hint, field_name)`` where association hint
            is ``CELLS``, ``POINTS`` or ``None``.
        :rtype: Tuple[Optional[str], str]
        """
        if ":" in token:
            prefix, name = token.split(":", 1)
            prefix = prefix.strip().upper()
            if prefix in ("CELL", "CELLS"):
                return "CELLS", name.strip()
            if prefix in ("POINT", "POINTS"):
                return "POINTS", name.strip()
        return None, token.strip()

    def _resolve_fields(
        self, available_fields: list[FieldSpec], requested_fields: Sequence[str] | None
    ) -> list[FieldSpec]:
        """
        Resolve requested field tokens against discovered dataset arrays.

        :param available_fields: Fields available in probe dataset.
        :type available_fields: List[FieldSpec]
        :param requested_fields: Optional list of user-requested tokens.
            If ``None`` or empty, all available fields are returned.
        :type requested_fields: Sequence[str], optional
        :returns: Selected unique field specifications.
        :rtype: List[FieldSpec]
        :raises KeyError: If a requested field token does not match any field.
        """
        if not requested_fields:
            return available_fields

        selected: list[FieldSpec] = []
        for token in requested_fields:
            assoc_hint, field_name = self._parse_field_token(str(token))
            matches = []
            for field in available_fields:
                if field.name != field_name:
                    continue
                if assoc_hint is not None and field.association != assoc_hint:
                    continue
                matches.append(field)

            if not matches:
                raise KeyError(
                    f"Requested field '{token}' not found. Available fields: "
                    + ", ".join([f"{f.association}:{f.name}" for f in available_fields])
                )

            # If token omitted association and duplicates exist, use first match.
            selected.append(matches[0])

        dedup = []
        seen = set()
        for f in selected:
            key = (f.association, f.name)
            if key not in seen:
                dedup.append(f)
                seen.add(key)
        return dedup

    def _resolve_range_overrides(
        self, field_specs: list[FieldSpec]
    ) -> dict[tuple[str, str], tuple[float, float]]:
        """
        Normalize user-provided range overrides for rendered fields.

        Accepted keys are unqualified field names (only if unambiguous) and
        fully qualified names (``CELLS:field_name`` / ``POINTS:field_name``).

        :param field_specs: Rendered field definitions.
        :type field_specs: List[FieldSpec]
        :returns: Mapping ``(association, name) -> (min, max)``.
        :rtype: Dict[Tuple[str, str], Tuple[float, float]]
        :raises ValueError: If override range is malformed, non-finite, not strictly
            increasing, or ambiguous.
        :raises KeyError: If override key does not match rendered fields.
        """
        overrides: dict[tuple[str, str], tuple[float, float]] = {}
        if not self.field_ranges:
            return overrides

        by_name: dict[str, list[FieldSpec]] = {}
        for field in field_specs:
            by_name.setdefault(field.name, []).append(field)

        for raw_key, raw_range in self.field_ranges.items():
            assoc_hint, field_name = self._parse_field_token(str(raw_key))
            if raw_range is None or len(raw_range) != 2:
                raise ValueError(
                    f"Range override for '{raw_key}' must be a 2-item sequence (min, max)."
                )
            lo, hi = float(raw_range[0]), float(raw_range[1])
            if not math.isfinite(lo) or not math.isfinite(hi):
                raise ValueError(
                    f"Range override for '{raw_key}' contains non-finite values: {raw_range}."
                )
            if hi <= lo:
                raise ValueError(
                    f"Range override for '{raw_key}' is invalid: min={lo} must be < max={hi}."
                )

            matches = by_name.get(field_name, [])
            if assoc_hint is not None:
                matches = [m for m in matches if m.association == assoc_hint]

            if not matches:
                raise KeyError(
                    f"Range override field '{raw_key}' does not match rendered fields: "
                    + ", ".join([f"{f.association}:{f.name}" for f in field_specs])
                )
            if assoc_hint is None and len(matches) > 1:
                raise ValueError(
                    f"Range override key '{raw_key}' is ambiguous. "
                    f"Use 'CELLS:{field_name}' or 'POINTS:{field_name}'."
                )

            target = matches[0]
            overrides[(target.association, target.name)] = (lo, hi)

        return overrides

    @staticmethod
    def _get_data_array(dataset, field: FieldSpec):
        """
        Fetch data array from a dataset according to field association.

        :param dataset: VTK dataset containing field arrays.
        :type dataset: vtk.vtkDataSet
        :param field: Field descriptor to fetch.
        :type field: FieldSpec
        :returns: VTK data array or ``None`` if not found.
        :rtype: vtk.vtkDataArray or None
        """
        if field.association == "CELLS":
            return dataset.GetCellData().GetArray(field.name)
        return dataset.GetPointData().GetArray(field.name)

    @staticmethod
    def _get_array_range(dataset, field: FieldSpec) -> tuple[float, float]:
        """
        Compute robust numeric range for one dataset field.

        For multi-component arrays, VTK magnitude range is used.

        :param dataset: VTK dataset containing the field.
        :type dataset: vtk.vtkDataSet
        :param field: Field descriptor.
        :type field: FieldSpec
        :returns: Tuple ``(min_value, max_value)`` with finite strictly increasing
            bounds.
        :rtype: Tuple[float, float]
        :raises KeyError: If field array is absent in the dataset.
        """
        arr = ParaViewMultiViewRenderer._get_data_array(dataset, field)
        if arr is None:
            raise KeyError(
                f"Array '{field.association}:{field.name}' not found in dataset."
            )
        comp = -1 if int(arr.GetNumberOfComponents()) > 1 else 0
        r = arr.GetRange(comp)
        lo, hi = float(r[0]), float(r[1])
        if not math.isfinite(lo) or not math.isfinite(hi):
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1e-12
        return lo, hi

    def _compute_global_ranges(
        self, snapshots: list[Snapshot], fields: list[FieldSpec]
    ) -> dict[tuple[str, str], tuple[float, float]]:
        """
        Compute fixed color ranges across selected snapshots for each field.

        :param snapshots: Snapshots included in rendering.
        :type snapshots: List[Snapshot]
        :param fields: Selected fields to render.
        :type fields: List[FieldSpec]
        :returns: Mapping ``(association, name) -> (min, max)``.
        :rtype: Dict[Tuple[str, str], Tuple[float, float]]
        :raises RuntimeError: If any snapshot cannot be read.
        """
        ranges: dict[tuple[str, str], tuple[float, float]] = {
            (f.association, f.name): (math.inf, -math.inf) for f in fields
        }
        for snap in snapshots:
            reader = self._create_vtk_reader(snap.file_path)
            dataset = self._data_object_from_reader(reader)
            if dataset is None:
                raise RuntimeError(f"Failed to read dataset from '{snap.file_path}'.")
            for field in fields:
                lo, hi = self._get_array_range(dataset, field)
                cur_lo, cur_hi = ranges[(field.association, field.name)]
                ranges[(field.association, field.name)] = (
                    min(cur_lo, lo),
                    max(cur_hi, hi),
                )

        for key, (lo, hi) in list(ranges.items()):
            if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
                ranges[key] = (0.0, 1.0)
        return ranges

    def _apply_camera_options(self, camera, first_renderer) -> None:
        """
        Apply configured camera parameters to shared renderer camera.

        :param camera: Shared VTK camera object.
        :type camera: vtk.vtkCamera
        :param first_renderer: First view renderer used for optional camera reset.
        :type first_renderer: vtk.vtkRenderer
        :returns: None
        :rtype: None
        """
        if bool(self.camera.get("reset_to_data_bounds", True)):
            first_renderer.ResetCamera()

        if self.camera.get("position") is not None:
            camera.SetPosition(*self.camera["position"])
        if self.camera.get("focal_point") is not None:
            camera.SetFocalPoint(*self.camera["focal_point"])
        if self.camera.get("view_up") is not None:
            camera.SetViewUp(*self.camera["view_up"])
        if self.camera.get("clipping_range") is not None:
            camera.SetClippingRange(*self.camera["clipping_range"])

        if self.camera.get("parallel_projection") is not None:
            camera.SetParallelProjection(bool(self.camera["parallel_projection"]))
        if self.camera.get("parallel_scale") is not None:
            camera.SetParallelScale(float(self.camera["parallel_scale"]))
        if self.camera.get("view_angle") is not None:
            camera.SetViewAngle(float(self.camera["view_angle"]))

        azimuth = float(self.camera.get("azimuth", 0.0) or 0.0)
        elevation = float(self.camera.get("elevation", 0.0) or 0.0)
        roll = float(self.camera.get("roll", 0.0) or 0.0)
        dolly = float(self.camera.get("dolly", 1.0) or 1.0)
        zoom = float(self.camera.get("zoom", 1.0) or 1.0)
        fit_margin = float(self.camera.get("fit_margin", 0.0) or 0.0)

        if abs(azimuth) > 0:
            camera.Azimuth(azimuth)
        if abs(elevation) > 0:
            camera.Elevation(elevation)
        if abs(roll) > 0:
            camera.Roll(roll)
        if fit_margin > 0:
            # Keep small frame margin to avoid visually clipped corners.
            camera.Zoom(1.0 / (1.0 + fit_margin))
        if abs(dolly - 1.0) > 1e-12:
            camera.Dolly(dolly)
        if abs(zoom - 1.0) > 1e-12:
            camera.Zoom(zoom)

    def _make_lut(self):
        """
        Create and populate VTK lookup table from configured colormap.

        :returns: Built VTK lookup table.
        :rtype: vtk.vtkLookupTable
        """
        import vtk

        lut = vtk.vtkLookupTable()
        n_colors = 256
        lut.SetNumberOfTableValues(n_colors)

        cmap_name = self.colormap
        cmap_norm = cmap_name.lower().replace(" ", "").replace("-", "").replace("_", "")
        cmap_aliases = {
            "cooltowarm": "coolwarm",
        }
        cmap_name = cmap_aliases.get(cmap_norm, cmap_name)

        cmap = None
        try:
            import matplotlib

            cmap = matplotlib.colormaps.get_cmap(cmap_name)
        except Exception:
            cmap = None

        if cmap is None:
            warnings.warn(
                f"Colormap '{self.colormap}' is not available. Falling back to 'coolwarm'.",
                stacklevel=2,
            )
            try:
                import matplotlib

                cmap = matplotlib.colormaps.get_cmap("coolwarm")
            except Exception:
                cmap = None

        if cmap is not None:
            for i in range(n_colors):
                t = float(i) / float(max(1, n_colors - 1))
                if self.colormap_reverse:
                    t = 1.0 - t
                r, g, b, a = cmap(t)
                lut.SetTableValue(i, float(r), float(g), float(b), float(a))
        else:
            # Last-resort fallback without matplotlib.
            lut.SetHueRange(0.666, 0.0)
            lut.SetSaturationRange(1.0, 1.0)
            lut.SetValueRange(1.0, 1.0)
        lut.Build()
        return lut

    @staticmethod
    def _add_text_actor(
        renderer, text: str, position: tuple[float, float], color, font_size
    ):
        """
        Add a 2D text actor to a renderer in normalized viewport coordinates.

        :param renderer: Target VTK renderer.
        :type renderer: vtk.vtkRenderer
        :param text: Initial text value.
        :type text: str
        :param position: Text position ``(x, y)`` in normalized viewport coordinates.
        :type position: Tuple[float, float]
        :param color: RGB text color in ``[0, 1]``.
        :type color: Sequence[float]
        :param font_size: Font size in pixels.
        :type font_size: int
        :returns: Added VTK text actor.
        :rtype: vtk.vtkTextActor
        """
        import vtk

        actor = vtk.vtkTextActor()
        actor.SetInput(text)
        prop = actor.GetTextProperty()
        prop.SetColor(*color)
        prop.SetFontSize(int(font_size))
        prop.BoldOff()
        prop.ItalicOff()
        prop.ShadowOff()
        actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        actor.GetPositionCoordinate().SetValue(float(position[0]), float(position[1]))
        renderer.AddActor2D(actor)
        return actor

    def _build_viewports(
        self, n_views: int
    ) -> tuple[int, int, list[tuple[float, float, float, float]]]:
        """
        Build rectangular viewport layout for the requested number of views.

        The last row is packed using its actual number of views to avoid empty
        tiles and keep layout tight.

        :param n_views: Number of subviews in the render window.
        :type n_views: int
        :returns: Tuple with ``(columns, rows, viewports)`` where each viewport is
            ``(x0, y0, x1, y1)`` in normalized coordinates.
        :rtype: Tuple[int, int, List[Tuple[float, float, float, float]]]
        """
        columns = (
            int(self.columns)
            if self.columns is not None
            else int(math.ceil(math.sqrt(max(1, n_views))))
        )
        columns = max(1, columns)
        rows = int(math.ceil(float(n_views) / float(columns)))

        viewports: list[tuple[float, float, float, float]] = []
        idx = 0
        for row in range(rows):
            cells_in_row = min(columns, n_views - idx)
            cells_in_row = max(1, cells_in_row)
            y1 = 1.0 - row / float(rows)
            y0 = 1.0 - (row + 1) / float(rows)
            for col in range(cells_in_row):
                x0 = col / float(cells_in_row)
                x1 = (col + 1) / float(cells_in_row)
                viewports.append((x0, y0, x1, y1))
                idx += 1
                if idx >= n_views:
                    break
        return columns, rows, viewports

    def _build_orientation_viewport(self) -> tuple[float, float, float, float]:
        """
        Build orientation-marker viewport in renderer-local coordinates.

        ``vtkOrientationMarkerWidget`` interprets viewport coordinates in the local
        renderer coordinate system when ``SetCurrentRenderer(...)`` is used.

        :returns: Orientation marker viewport ``(x0, y0, x1, y1)`` in normalized
            coordinates relative to current renderer.
        :rtype: Tuple[float, float, float, float]
        """
        ox = min(0.95, max(0.0, self.orientation_axes_position[0]))
        oy = min(0.95, max(0.0, self.orientation_axes_position[1]))
        s = self.orientation_axes_size

        x0 = ox
        y0 = oy
        x1 = min(0.999, x0 + s)
        y1 = min(0.999, y0 + s)
        if x1 <= x0:
            x1 = min(0.999, x0 + 0.05)
        if y1 <= y0:
            y1 = min(0.999, y0 + 0.05)
        return (x0, y0, x1, y1)

    def _determine_fps(self, n_frames: int) -> int:
        """
        Determine output video FPS from explicit option or target length.

        :param n_frames: Number of rendered frames.
        :type n_frames: int
        :returns: Frames-per-second value (minimum 1).
        :rtype: int
        """
        if self.video_fps is not None:
            return max(1, int(self.video_fps))
        return max(1, int(round(float(n_frames) / self.video_length_sec)))

    @staticmethod
    def _quality_to_crf(video_quality: int) -> int:
        """
        Convert quality scale ``[0, 100]`` to x264 CRF scale ``[51, 0]``.

        :param video_quality: User quality value.
        :type video_quality: int
        :returns: CRF value for ffmpeg ``libx264`` encoder.
        :rtype: int
        """
        # 0..100 quality -> 51..0 CRF
        q = max(0, min(100, int(video_quality)))
        return int(round((100 - q) * 51.0 / 100.0))

    def _encode_with_ffmpeg(
        self, fps: int, frame_pattern: str, video_path: Path
    ) -> None:
        """
        Encode MP4 video from rendered frame sequence using ffmpeg.

        :param fps: Input frame rate.
        :type fps: int
        :param frame_pattern: ffmpeg-compatible frame pattern (e.g. ``frame_%05d.png``).
        :type frame_pattern: str
        :param video_path: Output video path.
        :type video_path: pathlib.Path
        :returns: None
        :rtype: None
        :raises RuntimeError: If ffmpeg is unavailable or returns non-zero status.
        """
        ffmpeg = self.ffmpeg_executable or shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(
                "ffmpeg is not available but is required for video_format='mp4'."
            )

        crf = self._quality_to_crf(self.video_quality)
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            frame_pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            str(video_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed with code {proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()}"
            )

    def _encode_gif(self, fps: int, frame_paths: list[Path], video_path: Path) -> None:
        """
        Encode GIF animation from rendered frames using Pillow.

        :param fps: Animation frame rate.
        :type fps: int
        :param frame_paths: Ordered list of frame image paths.
        :type frame_paths: List[pathlib.Path]
        :param video_path: Output GIF path.
        :type video_path: pathlib.Path
        :returns: None
        :rtype: None
        :raises RuntimeError: If no frames are provided.
        """
        from PIL import Image

        if not frame_paths:
            raise RuntimeError("No frames available for GIF encoding.")

        duration_ms = int(round(1000.0 / float(max(1, fps))))
        imgs = [Image.open(p) for p in frame_paths]
        try:
            imgs[0].save(
                video_path,
                save_all=True,
                append_images=imgs[1:],
                duration=duration_ms,
                loop=0,
                optimize=False,
            )
        finally:
            for im in imgs:
                im.close()

    def _render_with_vtk(self, snapshots: list[Snapshot]) -> dict:
        """
        Render snapshots using the headless VTK backend.

        Produces one PNG image per snapshot and optionally assembles video output.

        :param snapshots: Selected snapshots to render.
        :type snapshots: List[Snapshot]
        :returns: Rendering metadata with output paths and selected ranges.
        :rtype: Dict
        :raises RuntimeError: If dataset reading or rendering fails.
        :raises ValueError: If color range mode is unsupported.
        """
        import vtk

        # Resolve fields from first snapshot.
        probe_reader = self._create_vtk_reader(snapshots[0].file_path)
        probe_dataset = self._data_object_from_reader(probe_reader)
        if probe_dataset is None:
            raise RuntimeError(
                f"Failed to read dataset from '{snapshots[0].file_path}'."
            )

        available_fields = self._collect_fields_from_dataset(
            probe_dataset, self.field_association
        )
        field_specs = self._resolve_fields(available_fields, self.requested_fields)
        if not field_specs:
            raise RuntimeError("No fields selected for rendering.")

        range_overrides = self._resolve_range_overrides(field_specs)
        global_ranges = {}
        if self.color_range_mode == "global":
            global_ranges = self._compute_global_ranges(snapshots, field_specs)
        elif self.color_range_mode != "frame":
            raise ValueError(
                f"Unsupported color_range_mode='{self.color_range_mode}'. Use 'frame' or 'global'."
            )
        for key, val in range_overrides.items():
            global_ranges[key] = val

        n_views = len(field_specs)
        _, _, viewports = self._build_viewports(n_views)

        render_window = vtk.vtkRenderWindow()
        render_window.SetOffScreenRendering(1)
        render_window.SetSize(self.image_width, self.image_height)
        render_window.SetBorders(0)

        shared_camera = vtk.vtkCamera()
        wells_reader = None
        wells_rendered = False
        if self.show_wells and self.wells_vtk_path is not None:
            try:
                wells_reader = self._create_vtk_reader(self.wells_vtk_path)
                wells_dataset = self._data_object_from_reader(wells_reader)
                has_points = (
                    hasattr(wells_dataset, "GetNumberOfPoints")
                    and wells_dataset.GetNumberOfPoints() > 0
                )
                if wells_dataset is None or not has_points:
                    warnings.warn(
                        f"Wells VTK dataset is empty and will be ignored: {self.wells_vtk_path}",
                        stacklevel=2,
                    )
                    wells_reader = None
            except Exception as exc:
                warnings.warn(
                    f"Failed to read wells VTK file '{self.wells_vtk_path}': {exc}",
                    stacklevel=2,
                )
                wells_reader = None

        view_states = []
        for i, field in enumerate(field_specs):
            renderer = vtk.vtkRenderer()
            renderer.SetViewport(*viewports[i])
            renderer.SetBackground(*self.background_color)
            renderer.SetActiveCamera(shared_camera)

            reader = self._create_vtk_reader(snapshots[0].file_path)
            reader.Update()

            mapper = vtk.vtkDataSetMapper()
            mapper.SetInputConnection(reader.GetOutputPort())
            mapper.SetScalarVisibility(True)
            if field.association == "CELLS":
                mapper.SetScalarModeToUseCellFieldData()
            else:
                mapper.SetScalarModeToUsePointFieldData()
            mapper.SelectColorArray(field.name)
            mapper.SetColorModeToMapScalars()

            lut = self._make_lut()
            mapper.SetLookupTable(lut)

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            prop.SetRepresentationToSurface()
            if self.show_cell_edges:
                prop.EdgeVisibilityOn()
                prop.SetEdgeColor(*self.edge_color)
                prop.SetLineWidth(self.edge_line_width)
            else:
                prop.EdgeVisibilityOff()
            renderer.AddActor(actor)

            wells_actor = None
            if wells_reader is not None:
                wells_mapper = vtk.vtkDataSetMapper()
                wells_mapper.SetInputConnection(wells_reader.GetOutputPort())
                wells_mapper.SetScalarVisibility(False)

                wells_actor = vtk.vtkActor()
                wells_actor.SetMapper(wells_mapper)
                wells_prop = wells_actor.GetProperty()
                wells_prop.SetColor(*self.wells_color)
                wells_prop.SetOpacity(self.wells_opacity)
                wells_prop.SetLineWidth(self.wells_line_width)
                if hasattr(wells_prop, "SetRenderLinesAsTubes"):
                    wells_prop.SetRenderLinesAsTubes(1)
                renderer.AddActor(wells_actor)
                wells_rendered = True

            scalar_bar = None
            if self.show_scalar_bars:
                scalar_bar = vtk.vtkScalarBarActor()
                scalar_bar.SetLookupTable(lut)
                scalar_bar.SetTitle(field.name)
                scalar_bar.SetNumberOfLabels(4)
                if hasattr(scalar_bar, "UnconstrainedFontSizeOn"):
                    scalar_bar.UnconstrainedFontSizeOn()
                scalar_bar.SetDrawTickLabels(True)
                scalar_bar.SetDrawAnnotations(True)
                scalar_bar.SetTitleRatio(0.4)
                scalar_bar.SetTextPad(4)
                scalar_bar.SetVerticalTitleSeparation(4)
                scalar_bar.SetPosition(
                    float(self.scalar_bar_position[0]),
                    float(self.scalar_bar_position[1]),
                )
                scalar_bar.SetWidth(float(self.scalar_bar_width))
                scalar_bar.SetHeight(float(self.scalar_bar_height))
                scalar_bar.SetBarRatio(float(self.scalar_bar_bar_ratio))

                title_prop = scalar_bar.GetTitleTextProperty()
                label_prop = scalar_bar.GetLabelTextProperty()
                for prop in (title_prop, label_prop):
                    prop.SetColor(*self.scalar_bar_text_color)
                    prop.SetFontSize(self.scalar_bar_font_size)
                    prop.BoldOff()
                    prop.ItalicOff()
                    prop.ShadowOff()
                renderer.AddActor2D(scalar_bar)

            title_actor = None
            if self.show_field_titles:
                title_actor = self._add_text_actor(
                    renderer=renderer,
                    text=f"{field.association}:{field.name}",
                    position=(0.02, 0.02),
                    color=self.annotate_color,
                    font_size=max(10, self.annotate_font_size - 2),
                )

            render_window.AddRenderer(renderer)
            view_states.append(
                {
                    "field": field,
                    "reader": reader,
                    "mapper": mapper,
                    "lut": lut,
                    "renderer": renderer,
                    "scalar_bar": scalar_bar,
                    "title_actor": title_actor,
                    "wells_actor": wells_actor,
                }
            )

        orientation_widgets = []
        if self.show_orientation_axes:
            interactor = vtk.vtkRenderWindowInteractor()
            interactor.SetRenderWindow(render_window)
            for state in view_states:
                axes_actor = vtk.vtkAxesActor()
                axes_actor.SetXAxisLabelText(self.orientation_axes_labels[0])
                axes_actor.SetYAxisLabelText(self.orientation_axes_labels[1])
                axes_actor.SetZAxisLabelText(self.orientation_axes_labels[2])
                for caption_actor in (
                    axes_actor.GetXAxisCaptionActor2D(),
                    axes_actor.GetYAxisCaptionActor2D(),
                    axes_actor.GetZAxisCaptionActor2D(),
                ):
                    text_actor = (
                        caption_actor.GetTextActor()
                        if hasattr(caption_actor, "GetTextActor")
                        else None
                    )
                    if text_actor is not None:
                        if hasattr(text_actor, "SetTextScaleModeToNone"):
                            text_actor.SetTextScaleModeToNone()
                        elif hasattr(text_actor, "SetTextScaleMode"):
                            text_scale_mode_none = getattr(
                                vtk.vtkTextActor, "TEXT_SCALE_MODE_NONE", 0
                            )
                            text_actor.SetTextScaleMode(text_scale_mode_none)

                    caption_prop = caption_actor.GetCaptionTextProperty()
                    caption_prop.SetColor(*self.orientation_axes_text_color)
                    caption_prop.SetFontSize(self.orientation_axes_label_font_size)
                    caption_prop.BoldOff()
                    caption_prop.ItalicOff()
                    caption_prop.ShadowOff()
                    caption_actor.SetWidth(0.60)
                    caption_actor.SetHeight(0.22)

                widget = vtk.vtkOrientationMarkerWidget()
                widget.SetOrientationMarker(axes_actor)
                widget.SetInteractor(interactor)
                widget.SetCurrentRenderer(state["renderer"])
                widget.SetViewport(*self._build_orientation_viewport())
                widget.SetEnabled(1)
                widget.InteractiveOff()
                orientation_widgets.append((widget, axes_actor))

        time_actor = None
        if self.annotate_time:
            annotate_idx = min(max(0, self.annotate_view_index), len(view_states) - 1)
            time_actor = self._add_text_actor(
                renderer=view_states[annotate_idx]["renderer"],
                text="",
                position=self.annotate_position,
                color=self.annotate_color,
                font_size=self.annotate_font_size,
            )

        window_to_image = vtk.vtkWindowToImageFilter()
        window_to_image.SetInput(render_window)
        window_to_image.ReadFrontBufferOff()
        window_to_image.SetInputBufferTypeToRGB()

        png_writer = vtk.vtkPNGWriter()
        png_writer.SetCompressionLevel(self.image_compression_level)

        # Video writer (OGV) runs during rendering for robustness without ffmpeg.
        video_path = None
        ogg_writer = None
        fps = self._determine_fps(len(snapshots)) if self.video_enabled else None

        if self.video_enabled and self.video_format == "ogv":
            video_path = self.output_dir / f"{self.video_filename}.ogv"
            ogg_writer = vtk.vtkOggTheoraWriter()
            ogg_writer.SetInputConnection(window_to_image.GetOutputPort())
            ogg_writer.SetFileName(str(video_path))
            ogg_writer.SetRate(int(fps))

            qmin = int(ogg_writer.GetQualityMinValue())
            qmax = int(ogg_writer.GetQualityMaxValue())
            if qmin <= self.video_quality <= qmax:
                q = int(self.video_quality)
            else:
                # map 0..100 input to writer range
                q = int(
                    round(
                        qmin
                        + (qmax - qmin) * max(0, min(100, self.video_quality)) / 100.0
                    )
                )
            ogg_writer.SetQuality(q)
            ogg_writer.Start()

        image_paths: list[str] = []
        if self.verbose:
            print(
                f"Rendering {len(snapshots)} snapshot(s) with {len(field_specs)} view(s), "
                f"backend=vtk, size={self.image_width}x{self.image_height}."
            )

        for frame_idx, snap in enumerate(snapshots):
            for state in view_states:
                field = state["field"]
                reader = state["reader"]
                mapper = state["mapper"]
                lut = state["lut"]

                reader.SetFileName(str(snap.file_path))
                dataset = self._data_object_from_reader(reader)
                if dataset is None:
                    raise RuntimeError(f"Failed to read '{snap.file_path}'.")

                if (field.association, field.name) in range_overrides:
                    lo, hi = range_overrides[(field.association, field.name)]
                elif self.color_range_mode == "global":
                    lo, hi = global_ranges[(field.association, field.name)]
                else:
                    lo, hi = self._get_array_range(dataset, field)

                lut.SetRange(lo, hi)
                mapper.SetScalarRange(lo, hi)

            if frame_idx == 0:
                self._apply_camera_options(shared_camera, view_states[0]["renderer"])

            if time_actor is not None:
                try:
                    txt = self.annotate_template.format(
                        time=snap.timestep,
                        step=snap.index,
                        frame=frame_idx,
                        file=snap.file_path.name,
                    )
                except Exception:
                    txt = f"t = {snap.timestep:g}"
                time_actor.SetInput(txt)

            for state in view_states:
                state["renderer"].ResetCameraClippingRange()
            render_window.Render()

            image_path = self._snapshot_image_path(frame_idx, snap)
            if self.overwrite_images or not image_path.exists():
                window_to_image.Modified()
                window_to_image.Update()
                png_writer.SetFileName(str(image_path))
                png_writer.SetInputConnection(window_to_image.GetOutputPort())
                png_writer.Write()

            image_paths.append(str(image_path))

            if ogg_writer is not None:
                window_to_image.Modified()
                window_to_image.Update()
                ogg_writer.Write()

        if ogg_writer is not None:
            ogg_writer.End()

        # Optional ffmpeg/PIL video assembly from already rendered frames.
        if self.video_enabled and self.video_format in ("mp4", "gif"):
            video_path = self.output_dir / f"{self.video_filename}.{self.video_format}"
            if self.video_format == "mp4":
                # ffmpeg expects numerical pattern; build deterministic sequential links.
                seq_dir = self.output_dir / "_ffmpeg_seq"
                seq_dir.mkdir(parents=True, exist_ok=True)
                seq_paths = []
                try:
                    for i, p in enumerate([Path(x) for x in image_paths]):
                        link_path = seq_dir / f"frame_{i:05d}.png"
                        if link_path.exists() or link_path.is_symlink():
                            link_path.unlink()
                        try:
                            os.symlink(p, link_path)
                        except OSError:
                            shutil.copy2(p, link_path)
                        seq_paths.append(link_path)
                    self._encode_with_ffmpeg(
                        fps=int(fps),
                        frame_pattern=str((seq_dir / "frame_%05d.png").resolve()),
                        video_path=video_path,
                    )
                finally:
                    for p in seq_paths:
                        if p.exists() or p.is_symlink():
                            p.unlink()
                    if seq_dir.exists():
                        seq_dir.rmdir()
            else:
                self._encode_gif(int(fps), [Path(p) for p in image_paths], video_path)

        result = {
            "ok": True,
            "backend": "vtk",
            "pvd_file": str(self.pvd_path),
            "wells_vtk_file": str(self.wells_vtk_path)
            if self.wells_vtk_path is not None
            else None,
            "wells_rendered": bool(wells_rendered),
            "output_dir": str(self.output_dir),
            "fields": [f"{f.association}:{f.name}" for f in field_specs],
            "timesteps": [s.timestep for s in snapshots],
            "images": image_paths,
            "video": str(video_path)
            if video_path is not None and Path(video_path).exists()
            else None,
            "fps": int(fps) if fps is not None else None,
            "field_ranges": {
                f"{f.association}:{f.name}": list(
                    range_overrides[(f.association, f.name)]
                    if (f.association, f.name) in range_overrides
                    else (
                        global_ranges[(f.association, f.name)]
                        if self.color_range_mode == "global"
                        else self._get_array_range(probe_dataset, f)
                    )
                )
                for f in field_specs
            },
        }

        manifest_path = self.output_dir / "render_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as fp:
            json.dump(result, fp, indent=2)
        result["manifest"] = str(manifest_path)

        return result

    def _render_with_paraview_subprocess(self) -> dict:
        """
        Run pvpython availability probe and fallback to VTK rendering.

        This method is kept for compatibility with older workflows requesting
        ``backend='paraview'``.

        :returns: Rendering result dictionary from VTK backend.
        :rtype: Dict
        :raises RuntimeError: If ``pvpython`` is unavailable or probe fails.
        """
        # Kept for explicit compatibility only.
        pvpython = self.pvpython_executable or os.environ.get("DARTS_PARAVIEW_PVPYTHON")
        if not pvpython:
            pvpython = shutil.which("pvpython")
        if not pvpython:
            raise RuntimeError(
                "pvpython executable was not found for backend='paraview'."
            )

        cmd = [
            pvpython,
            "--force-offscreen-rendering",
            str(Path(__file__).resolve()),
            "--_internal-paraview-probe",
            "--pvd-file",
            str(self.pvd_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                "ParaView backend probe failed. "
                + (proc.stderr.strip() or proc.stdout.strip())
            )

        # If probe succeeded, still delegate rendering to robust vtk backend to ensure
        # functionality on headless servers.
        warnings.warn(
            "backend='paraview' probe passed, but rendering is executed with VTK backend "
            "for robustness on headless systems.",
            stacklevel=2,
        )
        return self._render_with_vtk(
            self._select_snapshots(self._parse_pvd_snapshots())
        )


def render_multiview_from_pvd(
    pvd_file: str,
    output_dir: str | None = None,
    timestep_mode: str = "all",
    image_width: int = 1920,
    image_height: int = 1080,
    columns: int | None = None,
    include_point_arrays: bool = False,
    pvpython_executable: str | None = None,
    timeout_sec: int = 600,
    force_offscreen: bool = True,
    use_mesa: bool = True,
    mesa_backend: str = "llvmpipe",
    dry_run: bool = False,
    **kwargs,
) -> dict:
    """
    Backward-compatible convenience wrapper around
    :class:`ParaViewMultiViewRenderer`.

    Deprecated legacy parameters are accepted for API compatibility; rendering is
    performed by the robust VTK offscreen backend.

    :param pvd_file: Path to input ``.pvd`` file.
    :type pvd_file: str
    :param output_dir: Output directory for rendered artifacts.
    :type output_dir: str, optional
    :param timestep_mode: Snapshot selection mode (``all``, ``latest``, ``first``).
    :type timestep_mode: str
    :param image_width: Output image width in pixels.
    :type image_width: int
    :param image_height: Output image height in pixels.
    :type image_height: int
    :param columns: Optional fixed number of subview columns.
    :type columns: int, optional
    :param include_point_arrays: Whether to include point arrays in field scan.
        If ``False``, only cell arrays are rendered.
    :type include_point_arrays: bool
    :param pvpython_executable: Optional explicit ``pvpython`` path (compatibility
        probe only).
    :type pvpython_executable: str, optional
    :param timeout_sec: Deprecated argument kept for compatibility; ignored.
    :type timeout_sec: int
    :param force_offscreen: Deprecated argument kept for compatibility; ignored.
    :type force_offscreen: bool
    :param use_mesa: Deprecated argument kept for compatibility; ignored.
    :type use_mesa: bool
    :param mesa_backend: Deprecated argument kept for compatibility; ignored.
    :type mesa_backend: str
    :param dry_run: Deprecated argument kept for compatibility; ignored.
    :type dry_run: bool
    :param kwargs: Additional keyword options forwarded to
        :class:`ParaViewMultiViewRenderer`.
    :type kwargs: dict
    :returns: Rendering result dictionary.
    :rtype: Dict
    """
    _ = (
        timeout_sec,
        force_offscreen,
        use_mesa,
        mesa_backend,
        dry_run,
    )

    field_association = "both" if include_point_arrays else "cell"
    renderer = ParaViewMultiViewRenderer(
        pvd_file=pvd_file,
        output_dir=output_dir,
        timestep_mode=timestep_mode,
        image_width=image_width,
        image_height=image_height,
        columns=columns,
        field_association=field_association,
        pvpython_executable=pvpython_executable,
        **kwargs,
    )
    return renderer.render()


def _main() -> int:
    """
    CLI entry point used for internal ParaView probe mode.

    :returns: Process exit code.
    :rtype: int
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--_internal-paraview-probe", action="store_true")
    parser.add_argument("--pvd-file", type=str, required=True)
    args = parser.parse_args()

    if args._internal_paraview_probe:
        # Keep this tiny so it can be used as compatibility probe from pvpython.
        from paraview.simple import PVDReader  # type: ignore

        r = PVDReader(FileName=args.pvd_file)
        r.UpdatePipeline()
        print("OK")
        return 0

    parser.error("No valid mode specified.")


if __name__ == "__main__":
    raise SystemExit(_main())
