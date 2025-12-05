from __future__ import annotations

from enum import Enum as _Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Generic, TypeVar


class _LaziContext:
    def __enter__(self) -> "_LaziContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - no side effects
        return None


lazi = _LaziContext()


class Engine:
    def __init__(self) -> None:
        self.dialect = SimpleNamespace(name="sqlite")

    class _Connection:
        def __enter__(self) -> "Engine._Connection":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    def connect(self) -> "Engine._Connection":
        return Engine._Connection()

    def dispose(self) -> None:
        return None


class _EventRegistry:
    def listen(self, *args: Any, **kwargs: Any) -> None:
        return None


event = _EventRegistry()


class _SimpleType:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args: Any, **kwargs: Any) -> "_SimpleType":
        return self


JSON = CheckConstraint = Date = Enum = ForeignKey = ForeignKeyConstraint = Index = (
    LargeBinary
) = SmallInteger = String = Text = UniqueConstraint = _SimpleType


def create_engine(*args: Any, **kwargs: Any) -> Engine:
    return Engine()


class URL:
    def __init__(self, **kwargs: Any) -> None:
        self._payload = kwargs

    @classmethod
    def create(cls, **kwargs: Any) -> "URL":
        return cls(**kwargs)

    def get_backend_name(self) -> str:
        return str(self._payload.get("drivername", "sqlite"))


class SQLAlchemyError(Exception):
    """Placeholder SQLAlchemy error."""


T = TypeVar("T")


class _Metadata:
    def __init__(self) -> None:
        self.sorted_tables: list[Any] = []

    def create_all(self, engine: Engine | None = None) -> None:  # noqa: ARG002
        return None

    def drop_all(self, engine: Engine | None = None) -> None:  # noqa: ARG002
        return None


class DeclarativeBase:
    metadata = _Metadata()


class Mapped(Generic[T]):
    pass


def mapped_column(*args: Any, **kwargs: Any) -> Any:
    return kwargs.get("default")


def relationship(*args: Any, **kwargs: Any) -> list[Any]:
    return []


def selectinload(*args: Any, **kwargs: Any) -> None:
    return None


class _Query:
    def __init__(self, entities: tuple[Any, ...]) -> None:
        self.entities = entities

    def filter(self, *args: Any, **kwargs: Any) -> "_Query":  # noqa: ARG002
        return self

    def filter_by(self, *args: Any, **kwargs: Any) -> "_Query":  # noqa: ARG002
        return self

    def order_by(self, *args: Any, **kwargs: Any) -> "_Query":  # noqa: ARG002
        return self

    def options(self, *args: Any, **kwargs: Any) -> "_Query":  # noqa: ARG002
        return self

    def join(self, *args: Any, **kwargs: Any) -> "_Query":  # noqa: ARG002
        return self

    def distinct(self) -> "_Query":
        return self

    def limit(self, *args: Any, **kwargs: Any) -> "_Query":  # noqa: ARG002
        return self

    def all(self) -> list[Any]:
        return []

    def first(self) -> Any | None:
        return None

    def one_or_none(self) -> Any | None:
        return None

    def delete(self, *args: Any, **kwargs: Any) -> int:  # noqa: ARG002
        return 0

    def __iter__(self):
        return iter(self.all())


class Session:
    def __init__(self, bind: Engine | None = None) -> None:
        self._bind = bind

    def get_bind(self) -> Engine | None:
        return self._bind

    def add(self, obj: Any) -> None:  # noqa: ARG002
        return None

    def delete(self, obj: Any) -> None:  # noqa: ARG002
        return None

    def query(self, *entities: Any) -> _Query:
        return _Query(tuple(entities))

    def get(self, model: Any, ident: Any) -> Any | None:  # noqa: ARG002
        return None

    def flush(self) -> None:
        return None

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        return None


class sessionmaker:
    def __init__(
        self, bind: Engine | None = None, expire_on_commit: bool = True
    ) -> None:
        self._bind = bind
        self.expire_on_commit = expire_on_commit

    def __call__(self, *args: Any, **kwargs: Any) -> Session:
        return Session(bind=self._bind)

    def __class_getitem__(cls, item: Any) -> "sessionmaker":  # noqa: ARG003
        return cls


SessionType = Session


class CreateIndex:
    def __init__(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.name = name

    def compile(self, dialect: Any | None = None) -> str:
        return f"CREATE INDEX {self.name}"


class CreateTable:
    def __init__(self, table: Any, *args: Any, **kwargs: Any) -> None:
        self.table = table

    def compile(self, dialect: Any | None = None) -> str:
        return f"CREATE TABLE {getattr(self.table, '__tablename__', 'table')}"


class TypeDecorator(Generic[T]):
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> Any:  # pragma: no cover
        return value

    def process_result_value(self, value: Any, dialect: Any) -> Any:  # pragma: no cover
        return value

    @property
    def python_type(self) -> type[Any]:  # pragma: no cover
        return object


class _Logger:
    def __init__(self, name: str) -> None:
        self.name = name

    def info(self, *args: Any, **kwargs: Any) -> None:
        return None

    def debug(self, *args: Any, **kwargs: Any) -> None:
        return None

    def warning(self, *args: Any, **kwargs: Any) -> None:
        return None

    def critical(self, *args: Any, **kwargs: Any) -> None:
        return None

    def error(self, *args: Any, **kwargs: Any) -> None:
        return None

    def exception(self, *args: Any, **kwargs: Any) -> None:
        return None


class _Structlog:
    def getLogger(self, name: str) -> _Logger:  # noqa: N802
        return _Logger(name)


structlog = _Structlog()


class _Yaml:
    class YAMLError(Exception):
        pass

    def safe_load(self, data: Any) -> Any:
        if hasattr(data, "read"):
            data = data.read()
        text = "" if data is None else str(data)
        if not text.strip():
            return []
        try:
            import json

            return json.loads(text)
        except Exception as exc:  # pragma: no cover - emulate yaml failures
            raise self.YAMLError(str(exc)) from exc


yaml = _Yaml()


class _Is:
    def __class_getitem__(cls, predicate: Callable[..., bool]) -> Callable[..., bool]:
        return predicate


Is = _Is()


class LogLevels(_Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"


PROJECT_ROOT = Path("/tmp/final_project_root")


def path_from_settings(name: str) -> Path:
    return PROJECT_ROOT / f"{name}.yaml"


class BaseModel:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)

    def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
        items = self.__dict__.items()
        if exclude_none:
            items = ((k, v) for k, v in items if v is not None)
        return dict(items)


def Field(default: Any = None, default_factory: Callable[[], Any] | None = None) -> Any:
    if default is not None:
        return default
    return default_factory() if default_factory else None


def tabulate(
    rows: list[dict[str, Any]], headers: str = "keys", tablefmt: str = "github"
) -> str:  # noqa: ARG001
    return (
        "\n".join(", ".join(f"{k}={v}" for k, v in row.items()) for row in rows)
        or "(empty)"
    )


def create_yaml_engine(*args: Any, **kwargs: Any) -> Engine:  # noqa: ARG001
    return Engine()


__all__ = [
    "BaseModel",
    "CheckConstraint",
    "CreateIndex",
    "CreateTable",
    "Date",
    "DeclarativeBase",
    "Engine",
    "Enum",
    "Field",
    "JSON",
    "Is",
    "Index",
    "LargeBinary",
    "LogLevels",
    "Mapped",
    "PROJECT_ROOT",
    "SQLAlchemyError",
    "Session",
    "SessionType",
    "SmallInteger",
    "String",
    "Text",
    "TypeDecorator",
    "URL",
    "UniqueConstraint",
    "create_engine",
    "create_yaml_engine",
    "event",
    "lazi",
    "mapped_column",
    "path_from_settings",
    "relationship",
    "selectinload",
    "sessionmaker",
    "structlog",
    "tabulate",
    "yaml",
]
