"""Factory for generating RFC-7396 ``Patch*`` Pydantic models from ``Strict*``.

The JSON spec layer maintains two parallel hierarchies: ``Strict<Name>Spec`` for
full validation, and ``Patch<Name>Spec`` for partial merge-patch updates (every
field optional). Maintaining the patch variants by hand drifts from the strict
side whenever a new field is added.

:func:`make_patch_model` walks a strict class's fields, makes every required
field optional, optionally substitutes nested strict references with their
patch counterparts (so e.g. ``list[StrictPropertyRegionSpec]`` becomes
``list[PatchPropertyRegionSpec]``), and preserves ``Field`` metadata,
constraints, descriptions, and validators carried by the source class.

Designed so the generated class produces a JSON Schema equivalent to a
hand-written patch class modulo title/example differences — see
``tests/python/test_patch_factory.py`` for the equivalence proof per pair.
"""

from __future__ import annotations

import types
from copy import copy
from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, create_model
from pydantic.fields import FieldInfo


def _is_optional(annotation: Any) -> bool:
    """Return ``True`` iff ``annotation`` includes ``None`` as a union member.

    :param annotation: a type annotation (possibly an ``Annotated`` wrapper or
        a ``Union``/``X | Y`` form)
    :type annotation: Any
    :return: whether the annotation already accepts ``None``
    :rtype: bool
    """
    origin = get_origin(annotation)
    if origin is Annotated:
        return _is_optional(get_args(annotation)[0])
    if origin in (Union, types.UnionType):
        return type(None) in get_args(annotation)
    return False


def _substitute_types(annotation: Any, type_map: dict[type, type]) -> Any:
    """Return ``annotation`` with every key in ``type_map`` replaced by its value.

    Walks generic containers (``list``, ``dict``, ``tuple``), unions (PEP 604
    ``X | Y`` and ``typing.Union``), and ``Annotated`` wrappers, preserving
    structure and metadata. Leaves unknown leaf types untouched.

    :param annotation: source type annotation
    :type annotation: Any
    :param type_map: mapping from strict source classes to their patch
        replacements
    :type type_map: dict[type, type]
    :return: annotation with substitutions applied
    :rtype: Any
    """
    if annotation in type_map:
        return type_map[annotation]

    origin = get_origin(annotation)
    if origin is None:
        return annotation

    args = get_args(annotation)
    if origin is Annotated:
        head, *metadata = args
        new_head = _substitute_types(head, type_map)
        return Annotated[(new_head, *metadata)]

    new_args = tuple(_substitute_types(a, type_map) for a in args)

    if origin in (Union, types.UnionType):
        # Rebuild PEP-604-style union via reduce of `|`.
        result = new_args[0]
        for nxt in new_args[1:]:
            result = result | nxt
        return result

    # Generic container — rebuild with substituted args.
    try:
        return origin[new_args] if len(new_args) > 1 else origin[new_args[0]]
    except TypeError:
        return annotation


def _make_field_optional(
    annotation: Any, finfo: FieldInfo, type_map: dict[type, type]
) -> tuple[Any, FieldInfo]:
    """Produce the ``(annotation, FieldInfo)`` pair for an optional override.

    Adds ``| None`` to the annotation if absent, copies metadata from the
    source ``FieldInfo`` (description, constraints, examples, validators),
    and forces ``default=None``.
    """
    new_anno = _substitute_types(annotation, type_map)
    if not _is_optional(new_anno):
        new_anno = new_anno | None

    new_finfo = copy(finfo)
    new_finfo.default = None
    new_finfo.default_factory = None
    return new_anno, new_finfo


def make_patch_model(
    source_cls: type[BaseModel],
    *,
    name: str,
    type_map: dict[type, type] | None = None,
    title: str | None = None,
    json_schema_extra: dict[str, Any] | None = None,
) -> type[BaseModel]:
    """Generate a Patch model from a Strict source class.

    All required fields become optional with default ``None``; nested strict
    class references in field annotations are rewritten through ``type_map``;
    ``Field`` metadata (descriptions, constraints, examples) on the source
    side is preserved.

    The returned class inherits from ``source_cls`` so existing validators,
    methods, and ``model_config`` settings (notably ``extra="forbid"``) are
    inherited unchanged. Only the field annotations and defaults are
    overridden.

    :param source_cls: the strict source Pydantic model
    :type source_cls: type[pydantic.BaseModel]
    :param name: name for the generated class (e.g. ``"PatchPhysicsSpec"``)
    :type name: str
    :param type_map: substitutions for nested strict classes referenced in
        field annotations (e.g.
        ``{StrictPropertyRegionSpec: PatchPropertyRegionSpec}``)
    :type type_map: dict[type, type] | None
    :param title: JSON Schema title; defaults to ``name``
    :type title: str | None
    :param json_schema_extra: optional ``json_schema_extra`` dict to attach to
        the generated class's ``model_config``
    :type json_schema_extra: dict[str, Any] | None
    :return: a new Pydantic model class with every field optional
    :rtype: type[pydantic.BaseModel]
    """
    type_map = type_map or {}

    field_overrides: dict[str, tuple[Any, FieldInfo]] = {}
    for fname, finfo in source_cls.model_fields.items():
        new_anno, new_finfo = _make_field_optional(finfo.annotation, finfo, type_map)
        field_overrides[fname] = (new_anno, new_finfo)

    config_kwargs: dict[str, Any] = {
        "extra": "forbid",
        "title": title or name,
    }
    if json_schema_extra is not None:
        config_kwargs["json_schema_extra"] = json_schema_extra

    return create_model(
        name,
        __base__=source_cls,
        __doc__=f"Auto-generated patch variant of :class:`{source_cls.__name__}`.",
        __config__=ConfigDict(**config_kwargs),
        **field_overrides,
    )


__all__ = ["make_patch_model"]
