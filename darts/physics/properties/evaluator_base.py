"""Root contracts for DARTS property evaluators.

Every concrete evaluator is expected to:

1. Subclass a family ABC (``Density``, ``Viscosity``, ``RelPerm`` …) declaring
   its ``evaluate(...)`` signature.
2. Subclass :class:`EvaluatorBase` and implement ``to_config()`` +
   ``from_config()`` so the evaluator round-trips through Pydantic.
3. Register itself via :func:`register_evaluator` so dicts with a ``kind`` field
   can be dispatched to the right class by :func:`evaluator_field`.

The contract is split into two layers on purpose.  The family ABC encodes
runtime behaviour (what ``evaluate`` returns, what arguments it accepts);
``EvaluatorBase`` encodes serialization discipline.  A concrete evaluator
inherits from both.
"""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar

from pydantic import BaseModel, BeforeValidator, ConfigDict


class EvaluatorConfigBase(BaseModel):
    """Root of the evaluator-config discriminated union.

    Concrete evaluator Configs inherit from this and override ``kind`` with a
    ``Literal["..."]`` value so Pydantic can dispatch JSON payloads to the
    right Config class.
    """

    model_config = ConfigDict(extra="forbid")

    kind: str


_MISSING = object()


class EvaluatorBase(ABC):  # noqa: B024 — kept for ABCMeta so subclasses can declare abstractmethods
    """Mixin contract for JSON-serializable evaluators.

    Subclasses bind a Config class via :func:`register_evaluator` (which
    populates :attr:`_config_cls`).  The default :meth:`to_config` /
    :meth:`from_config` impls then handle the common case of "constructor
    args == stored attributes with the same names" with no boilerplate.

    Override either method explicitly when:

    - Constructor args are mutated before storage (preserve via ``_init_<name>``
      attribute, which the default :meth:`to_config` will pick up).
    - Construction needs context (``nc``, ``components``, ``Mw`` …) that isn't
      part of the Config — provide an explicit :meth:`from_config` reading
      those from ``**context``.
    - The Config holds nested evaluator Configs that need custom
      materialization.
    """

    _config_cls: ClassVar[type[EvaluatorConfigBase] | None] = None

    def to_config(self) -> EvaluatorConfigBase:
        """Build a Config from ``self`` by reading attributes named after
        ``self._config_cls`` fields. For each field, look up
        ``self.<field>`` first, then fall back to ``self._init_<field>``
        (the latter lets classes that mutate constructor args preserve the
        original value for round-tripping).
        """
        config_cls = type(self)._config_cls
        if config_cls is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no _config_cls bound; either "
                "register it via register_evaluator() or override to_config()"
            )
        values: dict[str, Any] = {}
        for fname in config_cls.model_fields:
            if fname == "kind":
                continue
            v = getattr(self, fname, _MISSING)
            if v is _MISSING:
                v = getattr(self, f"_init_{fname}", _MISSING)
            if v is _MISSING:
                continue
            if isinstance(v, EvaluatorBase):
                v = v.to_config()
            elif isinstance(v, list) and v and isinstance(v[0], EvaluatorBase):
                v = [x.to_config() for x in v]
            values[fname] = v
        return config_cls(**values)

    @classmethod
    def from_config(cls, config: EvaluatorConfigBase, **context: Any) -> EvaluatorBase:
        """Build an instance by passing Config fields (minus ``kind``) plus
        ``**context`` straight to ``cls(...)``. Override when the constructor
        signature diverges from the Config field names.
        """
        if cls._config_cls is None:
            raise NotImplementedError(
                f"{cls.__name__} has no _config_cls bound; either register it "
                "via register_evaluator() or override from_config()"
            )
        data = config.model_dump(exclude={"kind"})
        return cls(**data, **context)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_EVALUATOR_REGISTRY: dict[
    str, tuple[type[EvaluatorBase], type[EvaluatorConfigBase]]
] = {}


def register_evaluator(
    kind: str,
    cls: type[EvaluatorBase],
    config_cls: type[EvaluatorConfigBase],
) -> None:
    """Register a concrete evaluator under its Config ``kind`` literal.

    Called at module load time, immediately below the class definition.
    Idempotent: re-registering the same ``(kind, cls, config_cls)`` triple is a
    no-op, but registering a different class under an existing kind raises.
    """
    existing = _EVALUATOR_REGISTRY.get(kind)
    if existing is not None:
        if existing == (cls, config_cls):
            return
        raise RuntimeError(
            f"Evaluator kind '{kind}' is already registered to {existing!r}; "
            f"cannot re-register to ({cls!r}, {config_cls!r})"
        )
    _EVALUATOR_REGISTRY[kind] = (cls, config_cls)
    # Bind Config -> class so the default to_config/from_config impls work.
    # Set on cls.__dict__ directly so subclasses with their own Config don't
    # accidentally inherit a parent's binding.
    if cls.__dict__.get("_config_cls") is None:
        cls._config_cls = config_cls


def resolve_evaluator(
    kind: str,
) -> tuple[type[EvaluatorBase], type[EvaluatorConfigBase]]:
    """Look up the ``(evaluator_cls, config_cls)`` pair for a registered kind."""
    try:
        return _EVALUATOR_REGISTRY[kind]
    except KeyError:
        known = sorted(_EVALUATOR_REGISTRY)
        raise ValueError(
            f"No evaluator registered under kind '{kind}'. Known kinds: {known}"
        ) from None


def registered_kinds() -> list[str]:
    """Return all currently registered evaluator kinds (sorted)."""
    return sorted(_EVALUATOR_REGISTRY)


# ---------------------------------------------------------------------------
# Pydantic field helper
# ---------------------------------------------------------------------------


def evaluator_field(family: type) -> BeforeValidator:
    """Build a Pydantic ``BeforeValidator`` for a field that may hold either a
    live evaluator instance or an evaluator Config.

    The validator normalizes input to either the instance (Python path) or an
    :class:`EvaluatorConfigBase` instance (JSON / dict path).  Materialization
    of Config → instance is deferred to the owning container's ``from_config``
    because some evaluators require construction context (``nc``,
    ``components``, ``Mw``) that only the container knows.

    Usage::

        density_ev: Annotated[Any, evaluator_field(Density)] | None = None
    """

    def _normalize(v: Any) -> Any:
        if v is None:
            return v
        if isinstance(v, family):
            return v
        if isinstance(v, EvaluatorConfigBase):
            _check_family(v.kind, family)
            return v
        if isinstance(v, dict):
            kind = v.get("kind")
            if kind is None:
                raise TypeError(
                    f"Evaluator dict for {family.__name__} missing required "
                    f"'kind' discriminator; got keys {sorted(v)}"
                )
            _, config_cls = _check_family(kind, family)
            return config_cls.model_validate(v)
        raise TypeError(
            f"Expected {family.__name__} instance, {EvaluatorConfigBase.__name__}, "
            f"or dict with 'kind'; got {type(v).__name__}"
        )

    return BeforeValidator(_normalize)


def _check_family(
    kind: str, family: type
) -> tuple[type[EvaluatorBase], type[EvaluatorConfigBase]]:
    eval_cls, config_cls = resolve_evaluator(kind)
    if not issubclass(eval_cls, family):
        raise TypeError(
            f"Evaluator kind '{kind}' maps to {eval_cls.__name__}, which is "
            f"not a subclass of {family.__name__}"
        )
    return eval_cls, config_cls


def _dispatch_evaluator_config(value: Any) -> Any:
    """``BeforeValidator`` callable that dispatches a dict to the concrete
    Config class registered under its ``kind`` discriminator.

    Use this on a Config field typed as :class:`EvaluatorConfigBase` (or a
    subtype) when the family ABC is not yet importable at the field's
    declaration site.  The family-vs-kind compatibility check is deferred to
    :func:`materialize_evaluator`.

    :param value: raw input — instance, Config, or dict with ``kind``
    :type value: Any
    :return: typed Config instance (or pass-through for non-dict inputs)
    :rtype: Any
    """
    if isinstance(value, EvaluatorConfigBase):
        return value
    if isinstance(value, dict):
        kind = value.get("kind")
        if kind is None:
            raise TypeError(
                f"Evaluator dict missing required 'kind' discriminator; "
                f"got keys {sorted(value)}"
            )
        _, config_cls = resolve_evaluator(kind)
        return config_cls.model_validate(value)
    return value


dispatch_evaluator_config = BeforeValidator(_dispatch_evaluator_config)


def materialize_evaluator(value: Any, family: type, **context: Any) -> Any:
    """Resolve a normalized evaluator-field value to a live instance.

    Used by property-container-like classes in their ``from_config`` methods.
    ``value`` may be:

    - ``None`` — returned as-is.
    - An instance of ``family`` — returned as-is.
    - An :class:`EvaluatorConfigBase` — dispatched to the registered concrete
      class's ``from_config(config, **context)``.
    """
    if value is None:
        return None
    if isinstance(value, family):
        return value
    if isinstance(value, EvaluatorConfigBase):
        eval_cls, _ = _check_family(value.kind, family)
        return eval_cls.from_config(value, **context)
    raise TypeError(
        f"Cannot materialize {type(value).__name__} as {family.__name__}; "
        f"expected instance or Config"
    )
