"""Contains methods for working with the database."""

from .stubs import (
    JSON,
    URL,
    CheckConstraint,
    CreateIndex,
    CreateTable,
    Date,
    DeclarativeBase,
    Engine,
    Enum,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    LargeBinary,
    Mapped,
    SessionType,
    SmallInteger,
    SQLAlchemyError,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
    create_engine,
    event,
    lazi,
    mapped_column,
    relationship,
    selectinload,
    sessionmaker,
)

with lazi:  # type: ignore[attr-defined] # lazi has incorrectly typed code
    import ast
    import atexit
    import json
    import sys
    import tomllib
    from collections.abc import Mapping, Sequence
    from datetime import date as dtdate
    from functools import cache
    from pathlib import Path
    from types import MappingProxyType
    from typing import (
        Annotated,
        Any,
        Literal,
        Protocol,
        TextIO,
        cast,
        runtime_checkable,
    )

    from .stubs import (
        PROJECT_ROOT,
        BaseModel,
        Field,
        Is,
        LogLevels,
        path_from_settings,
        structlog,
        tabulate,
        yaml,
    )

# ysaqml needs an eager import; lazi wrappers block naay.dumps resolution.
from .stubs import create_yaml_engine

logger = structlog.getLogger("final_project")

CONFIG_PATH = path_from_settings("config")
SAMPLE_NPC_PATH = path_from_settings("sample_npc")
SAMPLE_LOCATION_PATH = path_from_settings(
    "sample_locations",
)
SAMPLE_ENCOUNTER_PATH = path_from_settings(
    "sample_encounters",
)
CAMPAIGN_STATUSES: tuple[str, ...] = (
    "ACTIVE",
    "ONHOLD",
    "COMPLETED",
    "CANCELED",
)


class ISODate(TypeDecorator[dtdate]):
    """TypeDecorator that accepts ISO date strings or ``date`` objects."""

    impl = Date
    cache_ok = True
    _python_type = dtdate

    def process_bind_param(
        self,
        value: dtdate | str | None,
        dialect: Any,  # noqa: ARG002
    ) -> dtdate | None:
        """
        Normalize incoming values before storage.

        :param value: Date object, ISO-formatted string, or blank/``None`` sentinel.
        :param dialect: SQLAlchemy dialect object (only supplied by SQLAlchemy).
        :returns: ``datetime.date`` when a value is provided, otherwise ``None``.
        :raises ValueError: If the supplied text cannot be parsed as an ISO date.
        :raises TypeError: If the value type cannot be converted.
        """
        if value is None or isinstance(value, dtdate):
            return value
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return None
            try:
                return dtdate.fromisoformat(normalized)
            except ValueError as exc:  # pragma: no cover - configuration data issue
                msg = f"Invalid ISO date string: {value!r}"
                raise ValueError(msg) from exc
        msg = f"Unsupported date value type: {type(value)!r}"
        raise TypeError(msg)

    @property
    def python_type(self) -> type[dtdate]:
        """
        Return the native Python type for the decorator.

        :returns: The ``datetime.date`` type used for campaign dates.
        """
        return self._python_type


class CanonicalJSON(TypeDecorator[dict[str, Any]]):
    """TypeDecorator that normalizes JSON text into mappings."""

    impl = JSON
    cache_ok = True
    _python_type: type[Any] = dict

    def process_bind_param(
        self,
        value: Mapping[str, Any] | str | None,
        dialect: Any,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """
        Normalize incoming JSON values before storage.

        :param value: Mapping object, JSON string, or None.
        :param dialect: SQLAlchemy dialect object (only supplied by SQLAlchemy).
        :returns: Normalized dictionary when a value is provided, otherwise None.
        :raises TypeError: If the value type cannot be converted to JSON.
        :raises ValueError: If the supplied text cannot be parsed as JSON.
        """
        if value is None:
            return None
        if isinstance(value, Mapping):
            return self._normalize_mapping(value)
        if isinstance(value, str):
            parsed = self._deserialize_mapping(value)
            return None if parsed is None else self._normalize_mapping(parsed)
        msg = f"Unsupported JSON value type: {type(value)!r}"
        raise TypeError(msg)

    def process_result_value(
        self,
        value: Mapping[str, Any] | str | None,
        dialect: Any,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """
        Normalize JSON values when loading from the database.

        :param value: Mapping object, JSON string, or None from the database.
        :param dialect: SQLAlchemy dialect object (only supplied by SQLAlchemy).
        :returns: Normalized dictionary when a value is provided, otherwise None.
        """
        if value is None:
            return None
        if isinstance(value, Mapping):
            return self._normalize_mapping(value)
        if isinstance(value, str):
            parsed = self._deserialize_mapping(value)
            return None if parsed is None else self._normalize_mapping(parsed)
        return value

    @staticmethod
    def _normalize_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
        return {str(key): val for key, val in payload.items()}

    @staticmethod
    def _deserialize_mapping(text: str) -> Mapping[str, Any] | None:
        normalized = text.strip()
        if not normalized:
            return None
        try:
            parsed: Any = json.loads(normalized)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(normalized)
            except (ValueError, SyntaxError) as exc:
                msg = f"Invalid JSON text: {text!r}"
                raise ValueError(msg) from exc
        if parsed is None:
            return None
        if not isinstance(parsed, Mapping):
            msg = f"JSON field requires an object; received {type(parsed)!r}"
            raise TypeError(msg)
        return parsed

    @property
    def python_type(self) -> type[Any]:
        """
        Return the native Python type for the decorator.

        :returns: The dictionary type used for JSON storage.
        """
        return self._python_type


class EngineManager:
    """Manage the lifecycle of the shared SQLAlchemy engine."""

    def __init__(self) -> None:
        """Initialize the engine manager with default state."""
        self._engine: Engine | None = None
        self.yaml_storage_path: Path | None = None
        self.purge_requested = False
        self._dispose_registered = False

    def connect(self, loglevel: LogLevels = LogLevels.WARNING) -> Engine:
        """Connect to the db, caching the engine inside the manager."""
        if self._engine is not None:
            return self._engine

        config_data = dict(_read_config().get("DB", {}))
        if "drivername" not in config_data or "database" not in config_data:
            msg = "Database configuration incomplete; expected drivername and database"
            raise RuntimeError(msg)
        drivername = str(config_data.get("drivername"))
        database_value = str(config_data.get("database"))
        normalized_driver = drivername.lower()
        echo = loglevel == LogLevels.DEBUG

        if normalized_driver.startswith("ysaqml"):
            storage_path = _resolve_database_path(database_value)
            storage_path.mkdir(parents=True, exist_ok=True)
            if self.purge_requested:
                _purge_yaml_storage(storage_path)
                self.purge_requested = False
            engine = create_yaml_engine(
                Base.metadata,
                storage_path,
                echo=echo,
            )
            event.listen(engine, "connect", _enable_sqlite_foreign_keys)
            self.yaml_storage_path = storage_path
            if not self._dispose_registered:
                atexit.register(self.dispose)
                self._dispose_registered = True
        else:
            if normalized_driver.startswith("sqlite") and database_value != ":memory:":
                db_path = _resolve_database_path(database_value)
                db_path.parent.mkdir(parents=True, exist_ok=True)
                config_data["database"] = str(db_path)
            db_config = DBConfig(**config_data)
            db_url = URL.create(**db_config.model_dump(exclude_none=True))

            connect_args: dict[str, Any] = {}
            if db_url.get_backend_name() == "sqlite":
                connect_args["check_same_thread"] = False
            engine = create_engine(
                db_url,
                echo=echo,
                connect_args=connect_args,
            )  # echo=True for logging SQL statements
            if db_url.get_backend_name() == "sqlite":
                event.listen(engine, "connect", _enable_sqlite_foreign_keys)
            self.yaml_storage_path = None
        try:
            with engine.connect():
                logger.info("Successfully connected to the database!")
        except Exception as e:  # noqa: BLE001
            logger.critical("Error connecting to the database", error=e)
        self._engine = engine
        return engine

    def dispose(self) -> None:
        """Dispose of the cached engine and clear dependent caches."""
        if self._engine is None:
            return
        try:
            self._engine.dispose()
        finally:
            self._engine = None
            self.yaml_storage_path = None
            self.purge_requested = False
            self._dispose_registered = False
            cache_clear = getattr(_get_session_factory, "cache_clear", None)
            if callable(cache_clear):
                cache_clear()


engine_manager = EngineManager()

# beartype annotations
type Varchar256 = Annotated[str, Is[lambda s: isinstance(s, str) and len(s) <= 256]]  # pyright: ignore[reportUnknownLambdaType] # noqa: PLR2004
type SmallInt = Annotated[int, Is[lambda x: isinstance(x, int) and 0 <= x <= 65535]]  # pyright: ignore[reportUnknownLambdaType] # noqa: PLR2004
LongBlob = LargeBinary(length=(2**32) - 1)  # Max length for LONGBLOB


@cache
def _read_config() -> dict[str, Any]:
    with CONFIG_PATH.open("rb") as file:
        ret = tomllib.load(file)
    logger.debug("read config data", config=ret)
    return ret


@cache
def _load_sample_data(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.exists():
        logger.error("%s file missing", label, path=str(path))
        return []
    try:
        with path.open("r", encoding="utf-8") as file:
            raw_data: Any = yaml.safe_load(file)
    except yaml.YAMLError:
        logger.exception("failed to parse %s yaml", label, path=str(path))
        return []
    if raw_data is None:
        raw_data = []
    if not isinstance(raw_data, list):
        logger.error("%s data must be a list", label, path=str(path))
        return []
    samples: list[dict[str, Any]] = []
    entries = cast(list[Any], raw_data)  # type: ignore[redundant-cast] #pyright gets confused
    for entry in entries:
        if isinstance(entry, Mapping):
            entry = cast(Mapping[Any, Any], entry)  # type: ignore[redundant-cast] #pyright gets confused
            samples.append({str(k): v for k, v in dict(entry).items()})
        else:
            logger.warning("skipping malformed %s entry", label, entry=entry)
    return samples


def _read_image_bytes(path: Path | None) -> bytes | None:
    if path is None:
        return None
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    try:
        return resolved.read_bytes()
    except FileNotFoundError:
        logger.warning("sample npc image not found", path=str(resolved))
        return None


def _coerce_optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def _resolve_database_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _purge_yaml_storage(path: Path | None) -> None:
    storage = path
    if storage is None or not storage.exists():
        return
    for yaml_file in storage.glob("*.yaml"):
        try:
            yaml_file.unlink()
        except OSError:
            logger.warning("failed to delete yaml file", path=str(yaml_file))


def _attach_image_blob(owner: Any, image_blob: bytes | None) -> None:
    """Assign an ImageStore row to the owning instance when needed."""
    if image_blob is None or not hasattr(owner, "image"):
        return
    owner.image = ImageStore(image_blob=image_blob)


class DBConfig(BaseModel):
    """Pydantic model of the config for a db connection."""

    drivername: str
    database: str
    username: str | None = None
    password: str | None = None
    host: str | None = None
    port: int | None = None
    query: Mapping[str, Sequence[str] | str] = Field(default_factory=dict)


@runtime_checkable
class _Connector(Protocol):
    def __call__(self, loglevel: LogLevels = LogLevels.WARNING) -> Engine: ...


def _connector_factory() -> _Connector:
    return engine_manager.connect


def dispose_engine() -> None:
    """Dispose of the shared database engine and clear related caches."""
    engine_manager.dispose()


def _enable_sqlite_foreign_keys(dbapi_connection: Any, _connection_record: Any) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        cursor.close()


connect = _connector_factory()


@cache
def _get_session_factory() -> sessionmaker[SessionType]:
    """Return a cached session factory bound to the configured engine."""
    factory: sessionmaker[SessionType] = sessionmaker(
        bind=connect(),
        expire_on_commit=False,
    )
    return factory


def get_session() -> SessionType:
    """Create a new SQLAlchemy session using the shared factory."""
    factory = _get_session_factory()
    return factory()


class Base(DeclarativeBase):
    """
    SqlAlchemy base class.

    see https://docs.sqlalchemy.org/en/20/changelog/whatsnew_20.html#migrating-an-existing-mapping
    """

    pass


# --- Core Entities ---


class Campaign(Base):
    """Represents the campaign Table."""

    __tablename__ = "campaign"
    name: Mapped[Varchar256] = mapped_column(String(256), primary_key=True)
    start_date: Mapped[dtdate] = mapped_column(ISODate())
    status: Mapped[Literal["ACTIVE", "ONHOLD", "COMPLETED", "CANCELED"]] = (
        mapped_column(
            Enum(*CAMPAIGN_STATUSES, name="campaign_status"),
        )
    )

    locations = relationship(
        "Location",
        back_populates="campaign",
        cascade="all, delete-orphan",
        passive_deletes=True,
        foreign_keys="[CampaignRecord.campaign_name]",
    )
    encounters = relationship(
        "Encounter",
        back_populates="campaign",
        cascade="all, delete-orphan",
        passive_deletes=True,
        foreign_keys="[CampaignRecord.campaign_name]",
    )
    factions = relationship(
        "Faction",
        back_populates="campaign",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    npcs = relationship(
        "NPC",
        back_populates="campaign",
        cascade="all, delete-orphan",
        passive_deletes=True,
        foreign_keys="[CampaignRecord.campaign_name]",
    )


class CampaignRecord(Base):
    """Shared base for campaign-scoped records that may have images."""

    __tablename__ = "campaign_record"
    __table_args__ = (Index("ix_campaign_record_campaign", "campaign_name"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    campaign_name: Mapped[Varchar256] = mapped_column(
        String(256),
        ForeignKey(
            "campaign.name",
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
    )
    record_type: Mapped[str] = mapped_column(String(50))

    image = relationship(
        "ImageStore",
        back_populates="campaign_record",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
        lazy="joined",
    )

    __mapper_args__ = MappingProxyType(
        {
            "polymorphic_on": record_type,
            "polymorphic_identity": "campaign_record",
        },
    )


class Location(CampaignRecord):
    """Represents the location table."""

    __tablename__ = "location"
    __table_args__ = (
        UniqueConstraint(
            "campaign_name",
            "name",
            name="uq_location_campaign_name",
        ),
    )
    _campaign_name_mirror_field = "_campaign_name_copy"
    _campaign_name_copy: Mapped[Varchar256] = mapped_column(
        "campaign_name",
        String(256),
        ForeignKey(
            "campaign.name",
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    id: Mapped[int] = mapped_column(
        ForeignKey(
            "campaign_record.id",
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )
    name: Mapped[Varchar256] = mapped_column(String(256), index=True)
    type: Mapped[Literal["DUNGEON", "WILDERNESS", "TOWN", "INTERIOR"]] = mapped_column(
        Enum("DUNGEON", "WILDERNESS", "TOWN", "INTERIOR", name="location_type"),
    )
    description: Mapped[str] = mapped_column(Text)

    campaign = relationship(
        "Campaign",
        back_populates="locations",
        foreign_keys=[CampaignRecord.campaign_name],
    )
    encounters = relationship(
        "Encounter",
        back_populates="location",
        cascade="all, delete-orphan",
        passive_deletes=True,
        foreign_keys="Encounter.location_name",
    )
    __mapper_args__ = MappingProxyType({"polymorphic_identity": "location"})


class Encounter(CampaignRecord):
    """Represents the encounter table."""

    __tablename__ = "encounter"
    __table_args__ = (
        UniqueConstraint(
            "campaign_name",
            "location_name",
            "date",
            name="uq_encounter_campaign_location_date",
        ),
        ForeignKeyConstraint(
            ["campaign_name", "location_name"],
            ["location.campaign_name", "location.name"],
            onupdate="CASCADE",
            ondelete="CASCADE",
            name="fk_encounter_location",
        ),
    )
    _campaign_name_mirror_field = "_campaign_name_copy"
    _campaign_name_copy: Mapped[Varchar256] = mapped_column(
        "campaign_name",
        String(256),
        ForeignKey(
            "campaign.name",
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    id: Mapped[int] = mapped_column(
        ForeignKey(
            "campaign_record.id",
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )
    location_name: Mapped[Varchar256] = mapped_column(
        String(256),
    )
    date: Mapped[dtdate] = mapped_column(ISODate(), nullable=True)
    description: Mapped[str] = mapped_column(Text)

    campaign = relationship(
        "Campaign",
        back_populates="encounters",
        foreign_keys=[CampaignRecord.campaign_name],
    )
    location = relationship(
        "Location",
        back_populates="encounters",
        foreign_keys=[location_name],
    )
    participants = relationship(
        "EncounterParticipants",
        back_populates="encounter",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    __mapper_args__ = MappingProxyType({"polymorphic_identity": "encounter"})


class NPC(CampaignRecord):
    """Represents the NPC table."""

    __tablename__ = "npc"
    __table_args__ = (
        UniqueConstraint(
            "campaign_name",
            "name",
            name="uq_npc_campaign_name",
        ),
        CheckConstraint("age BETWEEN 0 AND 65535", name="ck_npc_age_range"),
    )
    _campaign_name_mirror_field = "_campaign_name_copy"
    _campaign_name_copy: Mapped[Varchar256] = mapped_column(
        "campaign_name",
        String(256),
        ForeignKey(
            "campaign.name",
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    id: Mapped[int] = mapped_column(
        ForeignKey(
            "campaign_record.id",
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )
    name: Mapped[Varchar256] = mapped_column(String(256), index=True)
    age: Mapped[SmallInt] = mapped_column(SmallInteger)
    gender: Mapped[Literal["FEMALE", "MALE", "NONBINARY", "UNSPECIFIED"]] = (
        mapped_column(
            Enum(
                "FEMALE",
                "MALE",
                "NONBINARY",
                "UNSPECIFIED",
                name="gender_enum",
            ),
            nullable=False,
            default="UNSPECIFIED",
            server_default="UNSPECIFIED",
        )
    )
    alignment_name: Mapped[
        Literal[
            "LAWFUL GOOD",
            "LAWFUL NEUTRAL",
            "LAWFUL EVIL",
            "NEUTRAL GOOD",
            "TRUE NEUTRAL",
            "NEUTRAL EVIL",
            "CHAOTIC GOOD",
            "CHAOTIC NEUTRAL",
            "CHAOTIC EVIL",
        ]
    ] = mapped_column(
        Enum(
            "LAWFUL GOOD",
            "LAWFUL NEUTRAL",
            "LAWFUL EVIL",
            "NEUTRAL GOOD",
            "TRUE NEUTRAL",
            "NEUTRAL EVIL",
            "CHAOTIC GOOD",
            "CHAOTIC NEUTRAL",
            "CHAOTIC EVIL",
            name="alignment_enum",
        ),
    )
    description: Mapped[str] = mapped_column(Text)
    species_name: Mapped[Varchar256] = mapped_column(
        String(256),
        ForeignKey(
            "species.name",
            onupdate="CASCADE",
            ondelete="RESTRICT",
        ),
    )
    abilities_json: Mapped[dict[str, Any]] = mapped_column(
        CanonicalJSON(),
        nullable=True,
    )

    campaign = relationship(
        "Campaign",
        back_populates="npcs",
        foreign_keys=[CampaignRecord.campaign_name],
    )
    species = relationship("Species", back_populates="npcs")
    factions = relationship(
        "FactionMembers",
        back_populates="npc",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    encounters = relationship(
        "EncounterParticipants",
        back_populates="npc",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    relationships = relationship(
        "Relationship",
        foreign_keys="[Relationship.npc_id_1]",
        back_populates="origin",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    related_to = relationship(
        "Relationship",
        foreign_keys="[Relationship.npc_id_2]",
        back_populates="target",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    __mapper_args__ = MappingProxyType({"polymorphic_identity": "npc"})


class ImageStore(Base):
    """Store image blobs linked to exactly one owner."""

    __tablename__ = "image_store"
    campaign_record_id: Mapped[int] = mapped_column(
        ForeignKey(
            "campaign_record.id",
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )
    image_blob: Mapped[bytes] = mapped_column(LongBlob, nullable=False)

    campaign_record = relationship(
        "CampaignRecord",
        back_populates="image",
        uselist=False,
    )


class Species(Base):
    """Represents the Species table."""

    __tablename__ = "species"
    name: Mapped[Varchar256] = mapped_column(String(256), primary_key=True)
    traits_json: Mapped[str] = mapped_column(Text)

    npcs = relationship("NPC", back_populates="species")


class Faction(Base):
    """Represents the faction table."""

    __tablename__ = "faction"
    __table_args__ = (Index("ix_faction_campaign", "campaign_name"),)
    name: Mapped[Varchar256] = mapped_column(String(256), primary_key=True)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    campaign_name: Mapped[Varchar256] = mapped_column(
        String(256),
        ForeignKey(
            "campaign.name",
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        nullable=True,
    )

    campaign = relationship("Campaign", back_populates="factions")
    members = relationship(
        "FactionMembers",
        back_populates="faction",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


# --- Join Tables ---


class FactionMembers(Base):
    """Represents join table that holds which faction has which members."""

    __tablename__ = "faction_members"
    __table_args__ = (UniqueConstraint("npc_id", name="uq_faction_members_npc"),)
    faction_name: Mapped[Varchar256] = mapped_column(
        String(256),
        ForeignKey(
            "faction.name",
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )
    npc_id: Mapped[int] = mapped_column(
        ForeignKey(
            "npc.id",
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )
    notes: Mapped[str] = mapped_column(Text)

    faction = relationship("Faction", back_populates="members", passive_deletes=True)
    npc = relationship("NPC", back_populates="factions", passive_deletes=True)


class EncounterParticipants(Base):
    """Represents join table that holds which encounter has which members."""

    __tablename__ = "encounter_participants"
    __table_args__ = (Index("ix_encounter_participants_encounter", "encounter_id"),)
    npc_id: Mapped[int] = mapped_column(
        ForeignKey(
            "npc.id",
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )
    encounter_id: Mapped[int] = mapped_column(
        ForeignKey(
            "encounter.id",
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )
    notes: Mapped[str] = mapped_column(Text)

    npc = relationship(
        "NPC",
        back_populates="encounters",
        passive_deletes=True,
    )
    encounter = relationship(
        "Encounter",
        back_populates="participants",
        passive_deletes=True,
    )


class Relationship(Base):
    """Represents relationships between NPCs."""

    __tablename__ = "relationship"
    npc_id_1: Mapped[int] = mapped_column(
        ForeignKey(
            "npc.id",
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )
    npc_id_2: Mapped[int] = mapped_column(
        ForeignKey(
            "npc.id",
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )
    name: Mapped[Varchar256] = mapped_column(String(256))

    origin = relationship(
        "NPC",
        foreign_keys=[npc_id_1],
        back_populates="relationships",
    )
    target = relationship(
        "NPC",
        foreign_keys=[npc_id_2],
        back_populates="related_to",
        passive_deletes=True,
    )


def _mirror_campaign_name_column(model: type) -> None:
    attr_name = getattr(model, "_campaign_name_mirror_field", None)
    if not attr_name:
        return

    def _synchronize(_mapper: Any, _connection: Any, target: Any) -> None:
        setattr(target, attr_name, target.campaign_name)

    event.listen(model, "before_insert", _synchronize)
    event.listen(model, "before_update", _synchronize)


for _model in (Location, NPC, Encounter):
    _mirror_campaign_name_column(_model)


def _entry_name(entry: Mapping[str, Any]) -> str:
    return str(entry.get("name", "")).strip()


def _campaign_from_data(
    session: SessionType,
    campaign_data: Mapping[str, Any],
) -> Campaign:
    campaign_name = str(campaign_data["name"])
    campaign = session.get(Campaign, campaign_name)
    if campaign is not None:
        return campaign
    start_date = dtdate.fromisoformat(str(campaign_data["start_date"]))
    status = str(campaign_data["status"]).upper()
    campaign = Campaign(
        name=campaign_name,
        start_date=start_date,
        status=status,
    )
    session.add(campaign)
    return campaign


def _species_from_data(
    session: SessionType,
    species_data: Mapping[str, Any],
) -> Species:
    species_name = str(species_data["name"])
    species = session.get(Species, species_name)
    if species is not None:
        return species
    traits_source = species_data.get("traits", {})
    if isinstance(traits_source, str):
        traits_text = traits_source
    else:
        traits_text = json.dumps(traits_source or {}, indent=2)
    species = Species(
        name=species_name,
        traits_json=traits_text,
    )
    session.add(species)
    return species


def _location_from_data(
    session: SessionType,
    location_data: Mapping[str, Any],
    default_campaign: Campaign,
) -> Location:
    location_name = str(location_data["name"])
    location = (
        session.query(Location).filter(Location.name == location_name).one_or_none()
    )
    if location is not None:
        return location
    campaign = default_campaign
    override_campaign = location_data.get("campaign")
    if isinstance(override_campaign, Mapping):
        campaign = _campaign_from_data(
            session,
            cast(Mapping[str, Any], override_campaign),
        )
    image_path = _coerce_optional_path(location_data.get("image_path"))
    image_blob = _read_image_bytes(image_path)
    location = Location(
        name=location_name,
        type=str(location_data["type"]).upper(),
        description=str(location_data["description"]),
        campaign=campaign,
    )
    _attach_image_blob(location, image_blob)
    session.add(location)
    return location


def _load_all_sample_npcs(session: SessionType) -> int:
    samples = _load_sample_data(SAMPLE_NPC_PATH, "sample npc")
    created = 0
    for sample in samples:
        npc_name = _entry_name(sample)
        if not npc_name:
            continue
        exists = session.query(NPC).filter(NPC.name == npc_name).one_or_none()
        if exists is not None:
            continue
        alignment = str(sample["alignment_name"]).upper()
        abilities_source = dict(sample.get("abilities", {}))
        abilities = {str(k): v for k, v in abilities_source.items()}
        image_path = _coerce_optional_path(sample.get("image_path"))
        image_blob = _read_image_bytes(image_path)
        description = str(sample["description"])
        age = int(sample["age"])
        gender_value = (
            str(sample.get("gender", "UNSPECIFIED")).strip().upper() or "UNSPECIFIED"
        )
        campaign_data = cast(Mapping[str, Any], sample["campaign"])
        species_data = cast(Mapping[str, Any], sample["species"])
        campaign = _campaign_from_data(session, campaign_data)
        species = _species_from_data(session, species_data)
        npc = NPC(
            name=npc_name,
            age=age,
            gender=gender_value,
            alignment_name=alignment,
            description=description,
            species=species,
            campaign=campaign,
            abilities_json=abilities,
        )
        _attach_image_blob(npc, image_blob)
        session.add(npc)
        created += 1
    return created


def _load_all_sample_locations(session: SessionType) -> int:
    samples = _load_sample_data(SAMPLE_LOCATION_PATH, "sample location")
    created = 0
    for sample in samples:
        location_name = _entry_name(sample)
        if not location_name:
            continue
        exists = (
            session.query(Location).filter(Location.name == location_name).one_or_none()
        )
        if exists is not None:
            continue
        campaign = _campaign_from_data(
            session,
            cast(Mapping[str, Any], sample["campaign"]),
        )
        _location_from_data(session, sample, campaign)
        created += 1
    return created


def _load_all_sample_encounters(session: SessionType) -> int:
    samples = _load_sample_data(SAMPLE_ENCOUNTER_PATH, "sample encounter")
    created = 0
    for sample in samples:
        description = str(sample["description"])
        date_value = dtdate.fromisoformat(str(sample["date"]))
        exists = (
            session.query(Encounter)
            .filter(Encounter.description == description)
            .filter(Encounter.date == date_value)
            .one_or_none()
        )
        if exists is not None:
            continue
        image_path = _coerce_optional_path(sample.get("image_path"))
        image_blob = _read_image_bytes(image_path)
        campaign = _campaign_from_data(
            session,
            cast(Mapping[str, Any], sample["campaign"]),
        )
        location_payload = cast(Mapping[str, Any], sample["location"])
        location = _location_from_data(session, location_payload, campaign)
        encounter = Encounter(
            campaign=campaign,
            location=location,
            date=date_value,
            description=description,
        )
        _attach_image_blob(encounter, image_blob)
        session.add(encounter)
        created += 1
    return created


def load_all_sample_data() -> dict[str, int]:
    """Load every bundled sample NPC, location, and encounter definition."""
    session = get_session()
    results = {"campaigns": 0, "locations": 0, "npcs": 0, "encounters": 0}
    try:
        engine = session.get_bind() or connect()
        Base.metadata.create_all(engine)
        existing_campaigns = {name for (name,) in session.query(Campaign.name).all()}
        results["locations"] = _load_all_sample_locations(session)
        results["npcs"] = _load_all_sample_npcs(session)
        results["encounters"] = _load_all_sample_encounters(session)
        session.flush()
        current_campaigns = {name for (name,) in session.query(Campaign.name).all()}
        results["campaigns"] = len(current_campaigns - existing_campaigns)
    except Exception:
        session.rollback()
        raise
    else:
        session.commit()
        return results
    finally:
        session.close()


def _format_faction_entry(membership: FactionMembers) -> str:
    faction_name = (
        membership.faction.name if membership.faction is not None else "Unknown"
    )
    notes = (membership.notes or "").strip()
    return f"{faction_name} ({notes})" if notes else faction_name


def _format_relationships(npc: NPC) -> str:
    relationships: list[str] = []
    origin_relationships = sorted(
        npc.relationships,
        key=lambda rel: (
            rel.name,
            rel.target.name if rel.target is not None else "",
        ),
    )
    for rel in origin_relationships:
        target_name = rel.target.name if rel.target is not None else "Unknown"
        relationships.append(f"{rel.name} -> {target_name}")

    inbound_relationships = sorted(
        npc.related_to,
        key=lambda rel: (
            rel.name,
            rel.origin.name if rel.origin is not None else "",
        ),
    )
    for rel in inbound_relationships:
        origin_name = rel.origin.name if rel.origin is not None else "Unknown"
        relationships.append(f"{rel.name} <- {origin_name}")

    return "; ".join(relationships) if relationships else "None"


def list_all_npcs(session: SessionType) -> None:
    """Return all NPCs currently stored in the database."""
    try:
        npcs = (
            session.query(NPC)
            .options(
                selectinload(NPC.campaign),
                selectinload(NPC.factions).selectinload(FactionMembers.faction),
                selectinload(NPC.relationships).selectinload(Relationship.target),
                selectinload(NPC.related_to).selectinload(Relationship.origin),
            )
            .order_by(NPC.campaign_name, NPC.name)
            .all()
        )
        if not npcs:
            print("No NPCs found.")
            return

        rows: list[dict[str, str]] = []
        for npc in npcs:
            campaign_name = (
                npc.campaign.name
                if npc.campaign is not None
                else (npc.campaign_name or "Unknown")
            )
            faction_labels = [
                _format_faction_entry(membership)
                for membership in sorted(
                    npc.factions,
                    key=lambda member: (
                        member.faction.name if member.faction is not None else "",
                        member.npc_id,
                    ),
                )
                if membership.faction is not None
            ]
            relationships = _format_relationships(npc)
            rows.append(
                {
                    "NPC": npc.name,
                    "Campaign": campaign_name,
                    "Faction": ", ".join(faction_labels) if faction_labels else "None",
                    "Relationships": relationships,
                },
            )

        print(tabulate(rows, headers="keys", tablefmt="github"))
    finally:
        session.close()


def get_campaigns() -> list[str]:
    """Return a list of campaign names from the database."""
    session = SessionType(bind=connect())
    try:
        campaigns = session.query(Campaign).all()
        return [campaign.name for campaign in campaigns]
    finally:
        session.close()


def _get_campaign(session: SessionType, name: str) -> Campaign:
    """Return the Campaign with the given name, or raise ValueError if not found."""
    campaign = session.get(Campaign, name)
    if campaign is None:
        msg = f"Campaign '{name}' does not exist."
        raise ValueError(msg)
    return campaign


def delete_campaign(name: str) -> None:
    """Remove the campaign and all related domain data."""
    normalized = name.strip()
    if not normalized or normalized in {"No Campaigns", "New Campaign"}:
        msg = "Select a campaign before attempting to delete it."
        raise ValueError(msg)
    session = get_session()
    try:
        campaign = _get_campaign(session, normalized)
        session.delete(campaign)
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        logger.exception("failed to delete campaign", campaign=normalized)
        msg = "Unable to delete the campaign. Check logs for details."
        raise RuntimeError(msg) from exc
    except ValueError:
        session.rollback()
        raise
    finally:
        session.close()


def create_campaign(
    name: str,
    start_date: dtdate | str,
    status: str,
) -> Campaign:
    """Create a new campaign with the provided metadata."""
    normalized_name = name.strip()
    if not normalized_name:
        msg = "Campaign name cannot be empty."
        raise ValueError(msg)
    if isinstance(start_date, str):
        try:
            date_value = dtdate.fromisoformat(start_date)
        except ValueError as exc:
            msg = "Campaign start date must be in YYYY-MM-DD format."
            raise ValueError(msg) from exc
    else:
        date_value = start_date
    status_value = status.strip().upper()
    if status_value not in CAMPAIGN_STATUSES:
        allowed = ", ".join(CAMPAIGN_STATUSES)
        msg = f"Campaign status must be one of: {allowed}."
        raise ValueError(msg)
    session = get_session()
    try:
        existing = session.get(Campaign, normalized_name)
        if existing is not None:
            msg = f"Campaign '{normalized_name}' already exists."
            raise ValueError(msg)
        campaign = Campaign(
            name=normalized_name,
            start_date=date_value,
            status=status_value,
        )
        session.add(campaign)
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        logger.exception("failed to create campaign", campaign=normalized_name)
        msg = "Unable to create the campaign. Check logs for details."
        raise RuntimeError(msg) from exc
    else:
        return campaign
    finally:
        session.close()


def get_npcs(campaign: str | None = None) -> list[str]:
    """Return a list of NPC names from the database, optionally filtered by campaign."""
    session = SessionType(bind=connect())
    try:
        query = session.query(NPC)
        if campaign:
            query = query.filter(NPC.campaign_name == campaign)
        npcs = query.order_by(NPC.name).all()
        return [npc.name for npc in npcs]
    finally:
        session.close()


def get_npc_identity_rows(
    campaign: str | None = None,
) -> list[tuple[int, str, str]]:
    """Return (id, name, campaign_name) rows for NPCs, optionally filtered."""
    session = SessionType(bind=connect())
    try:
        query = session.query(NPC.id, NPC.name, NPC.campaign_name)
        if campaign:
            query = query.filter(NPC.campaign_name == campaign)
        rows = query.order_by(NPC.name).all()
        return [(npc_id, name, campaign_name) for npc_id, name, campaign_name in rows]
    finally:
        session.close()


def get_species(campaign: str | None = None) -> list[str]:
    """Return species names, optionally restricted to a single campaign."""
    session = SessionType(bind=connect())
    try:
        if campaign:
            query = (
                session.query(Species.name)
                .join(NPC, NPC.species_name == Species.name)
                .filter(NPC.campaign_name == campaign)
                .distinct()
                .order_by(Species.name)
            )
            return [name for (name,) in query.all()]
        species_list = session.query(Species).order_by(Species.name).all()
        return [species.name for species in species_list]
    finally:
        session.close()


def get_locations(campaign: str | None = None) -> list[str]:
    """Return a list of location names, optionally filtered by campaign."""
    session = SessionType(bind=connect())
    try:
        query = session.query(Location)
        if campaign:
            query = query.filter(Location.campaign_name == campaign)
        locations = query.order_by(Location.name).all()
        return [location.name for location in locations]
    finally:
        session.close()


def core_tables_empty() -> bool:
    """Return True when the NPC, location, and encounter tables have no rows."""
    session = get_session()
    try:
        npc_missing = session.query(NPC.id).limit(1).first() is None
        location_missing = session.query(Location.id).limit(1).first() is None
        encounter_missing = session.query(Encounter.id).limit(1).first() is None
        return npc_missing and location_missing and encounter_missing
    finally:
        session.close()


def get_factions(campaign: str | None = None) -> list[str]:
    """Return faction names, optionally filtered by campaign."""
    session = SessionType(bind=connect())
    try:
        query = session.query(Faction)
        if campaign:
            query = query.filter(Faction.campaign_name == campaign)
        factions = query.order_by(Faction.name).all()
        return [faction.name for faction in factions]
    finally:
        session.close()


def get_faction_details(name: str) -> tuple[str, str] | None:
    """Return (description, campaign_name) for a faction."""
    session = SessionType(bind=connect())
    try:
        faction = session.query(Faction).filter(Faction.name == name).one_or_none()
        if faction is None:
            return None
        return faction.description, faction.campaign_name
    finally:
        session.close()


def get_faction_membership(npc_id: int) -> tuple[str, str] | None:
    """Return the first faction membership (name, notes) for the NPC id."""
    session = SessionType(bind=connect())
    try:
        membership = (
            session.query(FactionMembers)
            .filter(FactionMembers.npc_id == npc_id)
            .order_by(FactionMembers.faction_name)
            .first()
        )
        if membership is None:
            return None
        return membership.faction_name, membership.notes
    finally:
        session.close()


def upsert_faction(name: str, description: str, campaign_name: str) -> None:
    """Create or update a faction definition."""
    session = get_session()
    try:
        faction = session.query(Faction).filter(Faction.name == name).one_or_none()
        if faction is None:
            faction = Faction(
                name=name,
                description=description,
                campaign_name=campaign_name,
            )
            session.add(faction)
        else:
            faction.description = description
            faction.campaign_name = campaign_name
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        logger.exception("failed to save faction", faction=name)
        msg = "Unable to save the faction. Check logs for details."
        raise RuntimeError(msg) from exc
    finally:
        session.close()


def assign_faction_member(npc_id: int, faction_name: str, notes: str) -> None:
    """Assign an NPC to a faction, replacing previous memberships."""
    session = get_session()
    try:
        npc = session.get(NPC, npc_id)
        if npc is None:
            session.rollback()
            msg = "Select a valid NPC before assigning a faction."
            raise ValueError(msg)
        (
            session.query(FactionMembers)
            .filter(FactionMembers.npc_id == npc_id)
            .delete(synchronize_session=False)
        )
        member = FactionMembers(
            faction_name=faction_name,
            npc_id=npc_id,
            notes=notes,
        )
        session.add(member)
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        logger.exception(
            "failed to assign faction membership",
            npc_id=npc_id,
            faction=faction_name,
        )
        msg = "Unable to update the faction membership. Check logs for details."
        raise RuntimeError(msg) from exc
    finally:
        session.close()


def clear_faction_membership(npc_id: int) -> None:
    """Remove any faction memberships for the specified NPC."""
    session = get_session()
    try:
        npc = session.get(NPC, npc_id)
        if npc is None:
            return
        (
            session.query(FactionMembers)
            .filter(FactionMembers.npc_id == npc_id)
            .delete(synchronize_session=False)
        )
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        logger.exception("failed to clear faction membership", npc_id=npc_id)
        msg = "Unable to clear the faction membership. Check logs for details."
        raise RuntimeError(msg) from exc
    finally:
        session.close()


def get_encounter_participants(encounter_id: int) -> list[tuple[int, str, str | None]]:
    """Return (npc_id, npc_name, notes) rows for the encounter."""
    session = get_session()
    try:
        rows = (
            session.query(NPC.id, NPC.name, EncounterParticipants.notes)
            .join(NPC, NPC.id == EncounterParticipants.npc_id)
            .filter(EncounterParticipants.encounter_id == encounter_id)
            .order_by(NPC.name)
            .all()
        )
        return [(npc_id, npc_name, notes) for npc_id, npc_name, notes in rows]
    except SQLAlchemyError:
        logger.exception(
            "failed to load encounter participants",
            encounter=encounter_id,
        )
        return []
    finally:
        session.close()


def upsert_encounter_participant(
    encounter_id: int,
    npc_id: int,
    notes: str,
) -> None:
    """Insert or update a participant row for the encounter."""
    session = get_session()
    try:
        encounter = session.get(Encounter, encounter_id)
        if encounter is None:
            session.rollback()
            msg = "Save the encounter before editing participants."
            raise ValueError(msg)
        npc = session.get(NPC, npc_id)
        if npc is None:
            session.rollback()
            msg = "Select a valid NPC before adding them to the encounter."
            raise ValueError(msg)
        participant = (
            session.query(EncounterParticipants)
            .filter(
                EncounterParticipants.encounter_id == encounter_id,
                EncounterParticipants.npc_id == npc.id,
            )
            .one_or_none()
        )
        if participant is None:
            participant = EncounterParticipants(
                encounter_id=encounter_id,
                npc_id=npc.id,
                notes=notes,
            )
            session.add(participant)
        else:
            participant.notes = notes
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        logger.exception(
            "failed to update encounter participant",
            encounter=encounter_id,
            npc_id=npc_id,
        )
        msg = "Unable to update encounter participants. Check logs for details."
        raise RuntimeError(msg) from exc
    finally:
        session.close()


def delete_encounter_participant(encounter_id: int, npc_id: int) -> None:
    """Remove the NPC from the encounter participants list."""
    session = get_session()
    try:
        (
            session.query(EncounterParticipants)
            .filter(
                EncounterParticipants.encounter_id == encounter_id,
                EncounterParticipants.npc_id == npc_id,
            )
            .delete(synchronize_session=False)
        )
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        logger.exception(
            "failed to delete encounter participant",
            encounter=encounter_id,
            npc_id=npc_id,
        )
        msg = "Unable to remove the encounter participant. Check logs for details."
        raise RuntimeError(msg) from exc
    finally:
        session.close()


def is_text_column(column: Any) -> bool:
    """Return True when the provided SQLAlchemy column stores text."""
    return isinstance(getattr(column, "type", None), Text)


def get_relationship_rows(source_id: int) -> list[tuple[int, str, str]]:
    """Return (target_id, target_name, relation_name) rows for the NPC."""
    session = get_session()
    try:
        rows = (
            session.query(NPC.id, NPC.name, Relationship.name)
            .join(NPC, NPC.id == Relationship.npc_id_2)
            .filter(Relationship.npc_id_1 == source_id)
            .order_by(NPC.name)
            .all()
        )
        return [
            (target_id, target_name, relation)
            for target_id, target_name, relation in rows
        ]
    except SQLAlchemyError:
        logger.exception("failed to load relationships", npc_id=source_id)
        return []
    finally:
        session.close()


def save_relationship(
    source_id: int,
    target_id: int,
    relation_name: str,
) -> None:
    """Create or update an NPC relationship."""
    if source_id == target_id:
        msg = "Select a different NPC for the relationship."
        raise ValueError(msg)
    session = get_session()
    try:
        source = session.get(NPC, source_id)
        if source is None:
            msg = "Save the NPC before adding relationships."
            raise ValueError(msg)
        target = session.get(NPC, target_id)
        if target is None:
            msg = "Select a valid related NPC."
            raise ValueError(msg)
        relation = (
            session.query(Relationship)
            .filter(
                Relationship.npc_id_1 == source.id,
                Relationship.npc_id_2 == target.id,
            )
            .one_or_none()
        )
        if relation is None:
            relation = Relationship(
                npc_id_1=source.id,
                npc_id_2=target.id,
                name=relation_name,
            )
            session.add(relation)
        else:
            relation.name = relation_name
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        logger.exception("failed to save relationship", npc_id=source_id)
        msg = "Unable to save the relationship. Check logs for details."
        raise RuntimeError(msg) from exc
    finally:
        session.close()


def delete_relationship(source_id: int, target_id: int) -> None:
    """Remove a relationship between two NPCs if it exists."""
    session = get_session()
    try:
        relation = (
            session.query(Relationship)
            .filter(
                Relationship.npc_id_1 == source_id,
                Relationship.npc_id_2 == target_id,
            )
            .one_or_none()
        )
        if relation is None:
            return
        session.delete(relation)
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        logger.exception("failed to delete relationship", npc_id=source_id)
        msg = "Unable to delete the relationship. Check logs for details."
        raise RuntimeError(msg) from exc
    finally:
        session.close()


def get_types() -> list[str]:
    """Return a list of types."""
    return ["NPC", "Location", "Encounter"]


def setup_database(
    *,
    rebuild: bool = False,
    loglevel: LogLevels = LogLevels.INFO,
) -> sessionmaker[SessionType]:
    """Ensure schema exists (optionally rebuilding) and return a session factory."""
    engine_manager.purge_requested = rebuild
    engine = connect(loglevel)
    logger.debug("Create tables if missing.")
    if rebuild:
        logger.info("Dropping all tables and rebuilding schema.")
        _purge_yaml_storage(engine_manager.yaml_storage_path)
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def export_database_ddl(stream: TextIO | None = None) -> None:
    """Write CREATE TABLE/INDEX statements for the schema to a stream."""
    target = stream or sys.stdout
    engine = connect()
    table_statements: list[str] = []
    try:
        dialect = engine.dialect
        for table in Base.metadata.sorted_tables:
            table_sql = str(CreateTable(table).compile(dialect=dialect))
            table_statements.append(table_sql)
            index_statements = [
                str(CreateIndex(index).compile(dialect=dialect))
                for index in table.indexes
            ]
            table_statements.extend(index_statements)
    finally:
        engine.dispose()
    if not table_statements:
        target.write("-- No tables defined.\n")
        return
    formatted = [f"{statement.rstrip()};" for statement in table_statements]
    output = "-- Generated schema --\n" + "\n\n".join(formatted) + "\n"
    target.write(output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Manage the RPG NPC database.",
    )
    parser.add_argument(
        "--export-ddl",
        action="store_true",
        help="Exports the database DDL to stdout.",
    )
    args = parser.parse_args()

    if args.export_ddl:
        export_database_ddl()
    else:
        parser.print_help()
