"""Pipeline MLOps de bout en bout : prétraitement -> entraînement -> évaluation -> garde-fou.

Étapes :
1. **Prétraitement** :
   - ``--from-raw`` (local, données brutes présentes) : reconstruit le dataset consolidé
     depuis data/raw et vérifie qu'il est identique au fichier versionné (reproductibilité) ;
   - sinon (CI : pas de données brutes) : valide le dataset consolidé versionné
     (clé unique, pas de valeur manquante, rendements > 0, gains dans [0, 1]).
2. **Entraînement** : sélection (MLflow ``01-selection``) puis optimisation (``02-optimisation``),
   avec les pays de test figés (``configs/test_countries.json``).
3. **Évaluation** : métriques sur les pays de test (calculées par l'optimisation).
4. **Garde-fou** : comparaison avec le modèle de référence (``models/model_card.json``,
   celui que sert l'API). Échec (code de sortie 1) si le RMSE de test se dégrade de plus de
   ``--tolerance`` (5 % par défaut) ou si le classement des cultures recule de plus de 0,02.

Le modèle produit est écrit dans ``--output-dir`` (par défaut un dossier de travail, pour ne
**jamais** écraser le modèle servi sans relecture). Pour le remplacer volontairement :
``--output-dir models``.

Usage :
    python -m agritech_crop_reco.pipeline                       # CI / contrôle
    python -m agritech_crop_reco.pipeline --from-raw            # local, depuis les données brutes
    python -m agritech_crop_reco.pipeline --output-dir models   # réentraîne le modèle servi
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from agritech_crop_reco.config import PROJECT_ROOT, Settings, load_settings
from agritech_crop_reco.data.merge import (
    CONSOLIDATED_FILE,
    build_consolidated,
    validate_consolidated,
)
from agritech_crop_reco.modeling.train import run_optimisation, run_selection
from agritech_crop_reco.serving import MODEL_CARD_FILE

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "pipeline"
RMSE_TOLERANCE = 0.05  # dégradation relative maximale du RMSE de test
RANKING_TOLERANCE = 0.02  # baisse absolue maximale du classement des cultures


@dataclass(frozen=True)
class GateResult:
    passed: bool
    rows: list[dict]

    def markdown(self) -> str:
        lines = [
            "| Métrique (pays de test) | Référence | Nouveau | Écart | Seuil | Statut |",
            "|---|---|---|---|---|---|",
        ]
        for r in self.rows:
            status = "✅" if r["ok"] else "❌"
            lines.append(
                f"| {r['metric']} | {r['reference']:.3f} | {r['new']:.3f} | {r['delta']} "
                f"| {r['threshold']} | {status} |"
            )
        verdict = "**Garde-fou : OK**" if self.passed else "**Garde-fou : ÉCHEC**"
        return "\n".join([verdict, "", *lines])


def quality_gate(
    reference: dict,
    new: dict,
    rmse_tolerance: float = RMSE_TOLERANCE,
    ranking_tolerance: float = RANKING_TOLERANCE,
) -> GateResult:
    """Compare les métriques de test du nouveau modèle à celles du modèle servi."""
    rmse_change = new["rmse"] / reference["rmse"] - 1
    ranking_change = new["ranking_spearman"] - reference["ranking_spearman"]
    rows = [
        {
            "metric": "RMSE (t/ha)",
            "reference": reference["rmse"],
            "new": new["rmse"],
            "delta": f"{rmse_change:+.1%}",
            "threshold": f"≤ +{rmse_tolerance:.0%}",
            "ok": rmse_change <= rmse_tolerance,
        },
        {
            "metric": "Classement des cultures (Spearman)",
            "reference": reference["ranking_spearman"],
            "new": new["ranking_spearman"],
            "delta": f"{ranking_change:+.3f}",
            "threshold": f"≥ −{ranking_tolerance}",
            "ok": ranking_change >= -ranking_tolerance,
        },
    ]
    return GateResult(passed=all(r["ok"] for r in rows), rows=rows)


def preprocess(settings: Settings, from_raw: bool) -> str:
    path = settings.processed_dir / CONSOLIDATED_FILE
    versioned = pd.read_csv(path)
    if from_raw:
        rebuilt = build_consolidated(settings.raw_dir)
        pd.testing.assert_frame_equal(rebuilt, versioned, check_dtype=False)
        return (
            f"dataset reconstruit depuis data/raw : identique au fichier versionné "
            f"({len(rebuilt):,} lignes)"
        )
    validate_consolidated(versioned)
    return f"dataset consolidé versionné validé ({len(versioned):,} lignes)"


def _write_summary(text: str) -> None:
    """Résumé lisible dans l'onglet du run GitHub Actions (sans effet en local)."""
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(text + "\n")


def run(
    from_raw: bool = False,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    log_artifacts: bool = True,
    rmse_tolerance: float = RMSE_TOLERANCE,
) -> GateResult:
    settings = load_settings()
    reference = json.loads((settings.models_dir / MODEL_CARD_FILE).read_text())["test_metrics"]
    timings = {}

    start = time.perf_counter()
    print("1/4 Prétraitement :", preprocess(settings, from_raw))
    timings["prétraitement"] = time.perf_counter() - start

    out_settings = replace(settings, models_dir=Path(output_dir))
    start = time.perf_counter()
    print("2/4 Entraînement : sélection")
    run_selection(out_settings)
    print("2/4 Entraînement : optimisation")
    card = run_optimisation(out_settings, log_artifacts=log_artifacts)
    timings["entraînement"] = time.perf_counter() - start

    print("3/4 Évaluation (pays de test) :", card["test_metrics"])
    gate = quality_gate(reference, card["test_metrics"], rmse_tolerance)
    print("4/4", gate.markdown())

    _write_summary(
        "\n".join(
            [
                "## Pipeline d'entraînement",
                f"- Modèle : `{card['model']}` "
                f"(cible {card['target']}, pesticides {card['pesticides']})",
                f"- RMSE validation croisée : {card['cv_rmse_tuned']:.3f} t/ha",
                f"- Suivi MLflow : `{settings.mlflow_tracking_uri.split('@')[-1]}`",
                "- Durées : " + ", ".join(f"{k} {v:.0f} s" for k, v in timings.items()),
                "",
                gate.markdown(),
            ]
        )
    )
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--from-raw", action="store_true", help="reconstruire depuis data/raw")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-artifacts", action="store_true", help="MLflow : métadonnées seules")
    parser.add_argument("--tolerance", type=float, default=RMSE_TOLERANCE)
    args = parser.parse_args()
    gate = run(args.from_raw, args.output_dir, not args.no_artifacts, args.tolerance)
    sys.exit(0 if gate.passed else 1)


if __name__ == "__main__":
    main()
