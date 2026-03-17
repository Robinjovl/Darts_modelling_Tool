"""
Stateful adapters that wrap ModelBuilder with spec tracking and idempotency.

ModelAdapter is the abstract base providing cumulative spec state, merge-patch
application, and a single ``apply`` entry point.  JsonModelAdapter handles
batch workflows (full config from a JSON file), while MCPModelAdapter supports
interactive tool-by-tool construction with section-level updates and the
stepwise MCP server protocol.
"""

from typing import Any

from darts.api.builder import ModelBuilder
from darts.api.model_spec import (
    json_merge_patch,
    normalize_keys,
    validate_model_spec_dict,
)
from darts.api.schemas import ModelSpec, PatchModelSpec


class ModelAdapter:
    """Shared adapter for managing ModelSpec state and validation."""

    def __init__(
        self,
        model: Any,
        *,
        base_path: str | None = None,
        object_store: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.base_path = base_path
        self.object_store = object_store
        self._spec: dict[str, Any] = {}
        self._applied_keys: set[str] = set()

    def update_spec(
        self,
        patch: dict[str, Any],
        *,
        mode: str = "merge",
        validate_only: bool = False,
        idempotency_key: str | None = None,
        strict: bool = False,
    ) -> dict[str, Any]:
        base = {} if mode == "replace" else self._spec
        merged = (
            patch
            if mode == "replace"
            else json_merge_patch(base, normalize_keys(patch))
        )

        v = validate_model_spec_dict(merged, strict=strict)
        if not v.get("ok", False):
            return v

        if validate_only:
            return {"ok": True, "validated": True, "spec": merged}

        if idempotency_key:
            if idempotency_key in self._applied_keys:
                return {"ok": True, "spec": self._spec, "idempotent": True}
            self._applied_keys.add(idempotency_key)

        self._spec = merged
        return {"ok": True, "spec": merged}

    def to_dict(self) -> dict[str, Any]:
        return dict(self._spec)


class JsonModelAdapter(ModelAdapter):
    """Adapter for schema-guided generation (JSON input)."""

    def apply_spec(self, spec: ModelSpec) -> None:
        ModelBuilder.apply(
            spec,
            self.model,
            base_path=self.base_path,
            object_store=self.object_store,
        )

    def apply_spec_dict(self, spec_dict: dict[str, Any]) -> None:
        if hasattr(ModelSpec, "model_validate"):
            spec = ModelSpec.model_validate(spec_dict)
        else:
            spec = ModelSpec(**spec_dict)
        self.apply_spec(spec)


class MCPModelAdapter(ModelAdapter):
    """Adapter for tool-by-tool configuration (MCP)."""

    def build_model(self) -> None:
        if hasattr(ModelSpec, "model_validate"):
            spec = ModelSpec.model_validate(self._spec)
        else:
            spec = ModelSpec(**self._spec)
        ModelBuilder.apply(
            spec,
            self.model,
            base_path=self.base_path,
            object_store=self.object_store,
        )

    def validate_patch(self, patch: dict[str, Any]) -> dict[str, Any]:
        if hasattr(PatchModelSpec, "model_validate"):
            PatchModelSpec.model_validate(patch)
        else:
            PatchModelSpec(**patch)
        return {"ok": True}
