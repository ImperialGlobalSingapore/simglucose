"""
Rerun only the failed configurations from a previous T1DM OpenAPS parameter
tuning run.

Reads `patient_map.json` and `results.json` from a source run directory,
filters for `status == "Failed_After_Retry"`, and re-executes just those
configurations using the existing T1DMOpenAPSParameterTuning machinery.
Output is written to a fresh sibling timestamped directory under the same
parent (the original source run is left untouched).
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tests.controllers.t1dm_patient.t1dm_oref_bolus_parameter_tuning import (
    T1DMOpenAPSParameterTuning,
    result_dir,
)


logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


class T1DMOpenAPSParameterTuningRerun(T1DMOpenAPSParameterTuning):
    """Rerun the Failed_After_Retry configurations from a prior tuning run."""

    def __init__(
        self,
        source_run_dir: Path,
        output_dir: Path,
        max_workers: int | None = None,
    ):
        super().__init__(output_dir=output_dir, max_workers=max_workers)
        self._source_run_dir = source_run_dir

    def generate_patient_map_and_simulation_config(
        self,
        run_dir: Path,
    ) -> Tuple[Dict[str, Tuple[str, Dict[str, Any], int]], List[Tuple[str, Path]]]:
        source_pm_path = self._source_run_dir / "patient_map.json"
        source_results_path = self._source_run_dir / "results.json"

        with open(source_pm_path) as f:
            raw_pm = json.load(f)
        with open(source_results_path) as f:
            results = json.load(f)

        failed_vids = sorted(
            {
                r["virtual_patient_id"]
                for r in results
                if r.get("status") == "Failed_After_Retry"
            }
        )
        logger.info(
            f"Found {len(failed_vids)} unique failed configurations in {source_results_path}"
        )

        patient_map: Dict[str, Tuple[str, Dict[str, Any], int]] = {}
        simulation_configs: List[Tuple[str, Path]] = []
        missing = 0
        for vid in failed_vids:
            entry = raw_pm.get(vid)
            if entry is None:
                missing += 1
                continue
            patient_name, profile, carb_amount = entry
            patient_map[vid] = (patient_name, profile, carb_amount)

            patient_folder = patient_name.replace("#", "_")
            patient_dir = run_dir / patient_folder
            patient_dir.mkdir(exist_ok=True, parents=True)
            simulation_configs.append((vid, patient_dir))

        if missing:
            logger.warning(
                f"{missing} failed virtual_patient_ids had no entry in patient_map.json and were skipped"
            )

        logger.info(f"Prepared {len(simulation_configs)} configs for rerun")
        return patient_map, simulation_configs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run",
        type=str,
        default="20260430_083831",
        help="Timestamp folder name (relative to imgs/oref0_parameter_tuning/) of the run to retry.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel ProcessPoolExecutor workers.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    source_run_dir = result_dir / args.source_run
    if not source_run_dir.is_dir():
        raise SystemExit(f"Source run directory not found: {source_run_dir}")

    experiment = T1DMOpenAPSParameterTuningRerun(
        source_run_dir=source_run_dir,
        output_dir=result_dir,
        max_workers=args.max_workers,
    )
    results = experiment.run()
    logger.info(f"Rerun completed with {len(results)} results")
