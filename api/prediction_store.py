"""Enregistrement des prédictions servies par l'API (suivi en production).

Une ligne par appel à /predict ou /recommend, dans une base dédiée (Neon `agritech_api`,
séparée de la base MLflow). Aucune donnée personnelle : ni adresse IP, ni identifiant.

Principes :
- désactivé si ``PREDICTION_DB_URL`` est absente (CI, docker compose, tests par défaut) ;
- **jamais bloquant** : l'écriture se fait après la réponse (tâche de fond) et toute erreur
  de base (Neon en veille, panne, réseau) est journalisée sans faire échouer la prédiction ;
- table créée au démarrage si elle n'existe pas (pas d'outil de migration pour une table).
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from datetime import UTC, datetime
from statistics import median, quantiles

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    func,
    insert,
    inspect,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger("agritech.api")

ENV_VAR = "PREDICTION_DB_URL"
SUMMARY_WINDOW = 10_000  # nombre d'appels récents pris en compte par la synthèse

metadata = MetaData()
predictions = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False, index=True),
    Column("endpoint", String(20), nullable=False, index=True),
    Column("crop", String(40)),  # culture demandée (/predict)
    Column("top_crop", String(40)),  # culture retenue : demandée (/predict) ou 1re (/recommend)
    Column("yield_t_ha", Float),  # rendement prédit de cette culture
    Column("revenue_usd_ha", Float),  # chiffre d'affaires estimé de cette culture (v1.3.0)
    Column("country", String(80)),  # pays choisi pour les prix (v1.3.0), jamais pour le modèle
    Column("rank_by", String(10)),  # /recommend : "revenue" ou "yield" (v1.3.0)
    Column("avg_temp_c", Float, nullable=False),
    Column("rainfall_mm", Float, nullable=False),
    Column("pesticides_tonnes", Float, nullable=False),
    Column("fertilizer_used", Boolean),
    Column("irrigation_used", Boolean),
    Column("n_warnings", Integer, nullable=False),
    Column("warnings", JSON),
    Column("ranking", JSON),  # /recommend : [[culture, t/ha], ...] dans l'ordre
    Column("model_version", String(120), nullable=False),
    Column("api_version", String(20), nullable=False),
    Column("latency_ms", Float, nullable=False),
)


class PredictionStore:
    def __init__(self, url: str):
        self.engine: Engine = create_engine(url, pool_pre_ping=True)
        metadata.create_all(self.engine)
        self._add_missing_columns()

    def _add_missing_columns(self) -> None:
        """Migration douce : une table créée par une version antérieure (ex. sur Neon en
        v1.2.0) reçoit les colonnes ajoutées depuis, toutes facultatives (NULL autorisé)."""
        existing = {c["name"] for c in inspect(self.engine).get_columns(predictions.name)}
        missing = [c for c in predictions.columns if c.name not in existing]
        with self.engine.begin() as conn:
            for column in missing:
                type_sql = column.type.compile(dialect=self.engine.dialect)
                conn.execute(
                    text(f"ALTER TABLE {predictions.name} ADD COLUMN {column.name} {type_sql}")
                )
                logger.info("Colonne ajoutée à la table %s : %s", predictions.name, column.name)

    @classmethod
    def from_env(cls) -> PredictionStore | None:
        """Store configuré par l'environnement, ou None (enregistrement désactivé)."""
        url = os.getenv(ENV_VAR, "").strip()
        if not url:
            logger.info("Enregistrement des prédictions désactivé (%s absente)", ENV_VAR)
            return None
        try:
            store = cls(url)
        except SQLAlchemyError as exc:
            # Base injoignable au démarrage : l'API démarre quand même, sans enregistrement
            logger.error("Enregistrement des prédictions désactivé : base injoignable (%s)", exc)
            return None
        url_info = store.engine.url
        target = f"{url_info.get_backend_name()} {url_info.host or ''}/{url_info.database}"
        logger.info("Enregistrement des prédictions activé (%s)", target)
        return store

    def record(self, event: dict) -> bool:
        """Insère un appel. Ne lève jamais d'exception : retourne False en cas d'échec."""
        row = {key: event.get(key) for key in predictions.columns.keys() if key != "id"}
        row["created_at"] = row.get("created_at") or datetime.now(UTC)
        try:
            with self.engine.begin() as conn:
                conn.execute(insert(predictions), row)
            return True
        except SQLAlchemyError as exc:
            logger.warning("Prédiction non enregistrée (%s) : %s", type(exc).__name__, exc)
            return False

    def summary(self) -> dict:
        """Synthèse agrégée des appels récents (aucune ligne brute renvoyée)."""
        cols = predictions.c
        query = (
            select(
                cols.created_at,
                cols.endpoint,
                cols.crop,
                cols.top_crop,
                cols.n_warnings,
                cols.latency_ms,
            )
            .order_by(cols.created_at.desc())
            .limit(SUMMARY_WINDOW)
        )
        with self.engine.connect() as conn:
            rows = conn.execute(query).all()
            total = conn.execute(select(func.count()).select_from(predictions)).scalar_one()
        if not rows:
            return {"total_calls": total, "window": 0}
        latencies = [r.latency_ms for r in rows]
        p95 = quantiles(latencies, n=20)[-1] if len(latencies) >= 2 else latencies[0]
        return {
            "total_calls": total,
            "window": len(rows),
            "first_call": min(r.created_at for r in rows),
            "last_call": max(r.created_at for r in rows),
            "calls_by_endpoint": dict(Counter(r.endpoint for r in rows)),
            "latency_ms_median": round(median(latencies), 1),
            "latency_ms_p95": round(p95, 1),
            "share_with_warnings": round(sum(r.n_warnings > 0 for r in rows) / len(rows), 3),
            "top_requested_crops": dict(
                Counter(r.crop for r in rows if r.endpoint == "predict").most_common(5)
            ),
            "top_recommended_crops": dict(
                Counter(r.top_crop for r in rows if r.endpoint == "recommend").most_common(5)
            ),
        }
