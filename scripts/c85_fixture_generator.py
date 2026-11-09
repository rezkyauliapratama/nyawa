#!/usr/bin/env python3
"""Re-key the still-unopened C8.4 synthetic split for local cross-encoder evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from c81_fixture_generator import SCHEMA_C85, validate_dataset
from c84_fixture_generator import build_dataset as build_c84_dataset


def _rekey(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("C8.4", "C8.5").replace("C84", "C85").replace("c84", "c85")
    if isinstance(value, list):
        return [_rekey(item) for item in value]
    if isinstance(value, dict):
        return {key: _rekey(item) for key, item in value.items()}
    return value


def build_dataset() -> dict[str, Any]:
    """Use C8.4's unqueried holdout exactly once for the new scorer experiment.

    The calibration labels/cases are the already-observed C8.4 calibration set;
    holdout cases have never been retrieved or scored. Re-keying keeps this
    protocol explicit without pretending this is a new independent corpus.
    """
    dataset = _rekey(build_c84_dataset())
    dataset["schema_version"] = SCHEMA_C85
    dataset["description"] = (
        "C8.5 local cross-encoder follow-up. Re-keyed C8.4 synthetic data; "
        "calibration is reusable, holdout remains unqueried and one-shot."
    )
    dataset["gates"]["require_cross_encoder_scores"] = True
    validate_dataset(dataset)
    return dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = build_dataset()
    args.output.write_text(json.dumps(dataset, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "fixture": str(args.output), "schema_version": dataset["schema_version"],
        "records": len(dataset["records"]), "cases": len(dataset["cases"]),
        "calibration": sum(row["split"] == "calibration" for row in dataset["cases"]),
        "holdout": sum(row["split"] == "holdout" for row in dataset["cases"]),
        "holdout_source": "unqueried C8.4 split",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
