"""Production runtime capabilities; HTTP liveness is not business readiness."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_CAPABILITIES = (
    "database", "store", "agent_checkpointer", "agent_graph",
    "workflow_checkpointer", "productivity_workflow", "scheduler", "telegram_webhook",
)
_ROLES = {name: "REQUIRED" for name in _CAPABILITIES}
_ALLOWED_ROLES = {"REQUIRED", "DEGRADED_ALLOWED", "OPTIONAL_BACKGROUND"}


@dataclass
class RuntimeCapabilities:
    """Non-sensitive startup state for the complete configured Assistant product.

    State is frozen after startup until the process is restarted. It is not an
    authorization source: the webhook guard is only an ingress safety boundary.
    """

    components: dict[str, bool] = field(default_factory=lambda: {name: False for name in _CAPABILITIES})
    failures: dict[str, dict[str, str]] = field(default_factory=dict)
    roles: dict[str, str] = field(default_factory=lambda: dict(_ROLES))

    @property
    def ready(self) -> bool:
        return all(self.components.get(name, False) for name, role in self.roles.items()
                   if role == "REQUIRED")

    @property
    def full_ready(self) -> bool:
        return self.ready and all(self.components.values())

    def register(self, component: str, role: str) -> None:
        if (not component.isidentifier() or component in self.roles or
                role not in _ALLOWED_ROLES):
            raise ValueError("Invalid runtime capability registration")
        self.roles[component] = role
        self.components[component] = False

    def required_ready_except(self, component: str) -> bool:
        return all(self.components.get(name, False) for name, role in self.roles.items()
                   if role == "REQUIRED" and name != component)

    def mark_ready(self, component: str) -> None:
        self._check(component)
        self.components[component] = True
        self.failures.pop(component, None)
        logger.info("runtime component=%s status=ready", component)

    def mark_failed(self, component: str, stage: str, error: BaseException | str) -> None:
        self._check(component)
        self.components[component] = False
        category = type(error).__name__ if isinstance(error, BaseException) else error
        # Only fixed diagnostic categories are accepted; never render an
        # exception's message (which could contain a DSN or credentials).
        if not isinstance(category, str) or not category.isidentifier():
            category = "Unavailable"
        self.failures[component] = {"stage": stage, "error_category": category}
        logger.warning("runtime component=%s stage=%s status=unavailable category=%s",
                       component, stage, category)

    def snapshot(self) -> dict:
        return {"alive": True, "ready": self.ready,
                "components": dict(self.components), "failures": dict(self.failures),
                "capability_roles": dict(self.roles),
                "full_ready": self.full_ready,
                "recovery": "restart_required"}

    def _check(self, component: str) -> None:
        if component not in self.roles:
            raise ValueError("Unknown runtime capability")
