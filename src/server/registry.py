from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

try:
    from .config import OrchestratorSettings
    from .logging_utils import get_logger
    from .schemas import RegistryModel, RegistryServer, default_registry
except ImportError:
    from config import OrchestratorSettings
    from logging_utils import get_logger
    from schemas import RegistryModel, RegistryServer, default_registry

logger = get_logger("orchestrator.registry")


@contextmanager
def _lock_file(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        try:
            import fcntl

            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        except ImportError:
            pass
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except ImportError:
            pass
        os.close(lock_fd)


class RegistryStore:
    def __init__(self, settings: OrchestratorSettings):
        self.settings = settings
        self._cached_registry: RegistryModel | None = None
        self._cached_mtime: tuple[int, int] | None = None

    @property
    def registry_path(self) -> Path:
        return self.settings.registry_path

    @property
    def lock_path(self) -> Path:
        return self.registry_path.with_suffix(self.registry_path.suffix + ".lock")

    def _read_registry_from_disk(self) -> RegistryModel:
        if not self.registry_path.exists():
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            defaults = default_registry()
            self.registry_path.write_text(json.dumps(defaults.model_dump(mode="json"), indent=2), encoding="utf-8")
            return defaults

        raw = self.registry_path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        return RegistryModel.model_validate(parsed)

    def load_registry(self) -> RegistryModel:
        with _lock_file(self.lock_path):
            if self.settings.enable_hot_reload:
                if self.registry_path.exists():
                    stat = self.registry_path.stat()
                    current_mtime = (stat.st_mtime_ns, stat.st_size)
                else:
                    current_mtime = None
                if self._cached_registry is None or self._cached_mtime != current_mtime:
                    registry = self._read_registry_from_disk()
                    self._cached_registry = registry
                    self._cached_mtime = current_mtime
                return self._cached_registry

            registry = self._read_registry_from_disk()
            self._cached_registry = registry
            if self.registry_path.exists():
                stat = self.registry_path.stat()
                self._cached_mtime = (stat.st_mtime_ns, stat.st_size)
            else:
                self._cached_mtime = None
            return registry

    def save_registry(self, registry: RegistryModel) -> RegistryModel:
        with _lock_file(self.lock_path):
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            self.registry_path.write_text(json.dumps(registry.model_dump(mode="json"), indent=2), encoding="utf-8")
            self._cached_registry = registry
            stat = self.registry_path.stat()
            self._cached_mtime = (stat.st_mtime_ns, stat.st_size)
            return registry

    def update_registry(
        self,
        upsert: list[dict[str, object]],
        remove: list[str],
        provided_secret: str | None,
    ) -> RegistryModel:
        if not self.settings.update_registry_enabled:
            raise PermissionError("Registry update is disabled. Set ORCHESTRATOR_ENABLE_UPDATE_REGISTRY=true to enable it.")
        if self.settings.admin_secret and provided_secret != self.settings.admin_secret:
            raise PermissionError("Admin authentication failed for update_registry.")

        current = self.load_registry()
        by_id = {entry.id: entry for entry in current.mcp_servers}

        for server_id in remove:
            by_id.pop(server_id, None)

        for entry in upsert:
            validated = RegistryServer.model_validate(entry)
            by_id[validated.id] = validated

        updated = RegistryModel(
            version=current.version,
            mcp_servers=list(by_id.values()),
            category_aliases=current.category_aliases,
            routing_rules=current.routing_rules,
        )
        logger.info("Registry updated at %s", datetime.now(UTC).isoformat())
        return self.save_registry(updated)