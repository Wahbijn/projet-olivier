"""
One-time admin tool: copy the runs of a LOCAL MLflow database to the shared DagsHub MLflow server.

Copies, for every run: params, full metric history, tags, run name and times.
Adds the tags `author`, `migrated_from=local` and `source_run_id`.
Does NOT copy artifacts (models are versioned with DVC; the Kaggle artifacts are lost anyway).
Safe to run twice: runs already migrated (same source_run_id) are skipped.

Usage (project root, venv active, .env filled with the DagsHub values):
    python scripts/migrate_mlflow_to_dagshub.py --author rayen --experiment olive-detection
    python scripts/migrate_mlflow_to_dagshub.py --author kaggle-notebook --experiment olive-disease --dry-run

`--author` is applied only when a run has no author or "unknown" (also fixes "unknown-" run names).
"""
import argparse
import os

from dotenv import load_dotenv
from mlflow.entities import Param
from mlflow.tracking import MlflowClient

LOT = 1000   # MLflow API limit per log_batch call


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--author", required=True, help="your first name, added as tag `author`")
    p.add_argument("--source", default="sqlite:///mlflow.db", help="local tracking URI")
    p.add_argument("--experiment", default=None, help="migrate only this experiment (default: all)")
    p.add_argument("--dry-run", action="store_true", help="only list what would be copied")
    a = p.parse_args()

    load_dotenv()
    cible_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if not cible_uri.startswith("https://dagshub.com"):
        raise SystemExit("MLFLOW_TRACKING_URI in .env must point to DagsHub (https://dagshub.com/...mlflow)")

    src, dst = MlflowClient(tracking_uri=a.source), MlflowClient(tracking_uri=cible_uri)
    copies = ignores = 0
    for exp in src.search_experiments():
        if a.experiment and exp.name != a.experiment:
            continue
        runs = src.search_runs([exp.experiment_id], max_results=10000)
        if not runs:
            continue
        cible = dst.get_experiment_by_name(exp.name)
        exp_id = cible.experiment_id if cible else (None if a.dry_run else dst.create_experiment(exp.name))
        deja = set()
        if exp_id is not None:
            deja = {r.data.tags.get("source_run_id") for r in
                    dst.search_runs([exp_id], filter_string="tags.migrated_from = 'local'", max_results=10000)}

        for run in runs:
            rid_src = run.info.run_id
            nom = run.info.run_name or rid_src[:8]
            if nom.startswith("unknown-"):
                nom = f"{a.author}-" + nom[len("unknown-"):]
            if rid_src in deja:
                ignores += 1
                print(f"  = already migrated: {exp.name} / {nom}")
                continue
            auteur = run.data.tags.get("author")
            auteur = a.author if auteur in (None, "", "unknown") else auteur
            print(f"  + {exp.name} / {nom}  author={auteur}  ({len(run.data.params)} params, {len(run.data.metrics)} metrics)")
            if a.dry_run:
                continue

            tags = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}
            if tags.get("author") in (None, "", "unknown"):
                tags["author"] = a.author
            tags.update({"migrated_from": "local", "source_run_id": rid_src, "mlflow.runName": nom})
            nouveau = dst.create_run(exp_id, start_time=run.info.start_time, tags=tags)
            rid = nouveau.info.run_id

            params = [Param(k, str(v)[:500]) for k, v in run.data.params.items()]
            for i in range(0, len(params), 100):
                dst.log_batch(rid, params=params[i:i + 100])
            metriques = [m for cle in run.data.metrics for m in src.get_metric_history(rid_src, cle)]
            for i in range(0, len(metriques), LOT):
                dst.log_batch(rid, metrics=metriques[i:i + LOT])
            dst.set_terminated(rid, status=run.info.status, end_time=run.info.end_time)
            copies += 1

    print(f"\nDone: {copies} runs copied, {ignores} already there."
          + ("  (dry run: nothing written)" if a.dry_run else ""))


if __name__ == "__main__":
    main()