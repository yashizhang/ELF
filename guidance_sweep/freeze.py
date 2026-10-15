"""Freeze all four selection groups after both screening jobs have succeeded."""
from datetime import datetime, timezone
from pathlib import Path

from t5_adaptation.common import atomic_json, file_hash, read_json
from .common import select_confirmation
from .submit import statuses


def main():
    inputs = read_json("guidance_sweep/inputs.json")
    root = Path(inputs["artifact_root"])
    scheduler = statuses(root, read_json(root / "submission.json"))
    result = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "inputs_sha256": file_hash("guidance_sweep/inputs.json"),
              "plan_sha256": file_hash("guidance_sweep/plan.json"), "arms": {},
              "confirmation_started": False, "rule": "Frozen from screening only; no further search after confirmation."}
    for arm in ("C", "D"):
        assert any(j["arm"] == arm and j["phase"] == "screening" and j["state"] == "SUCCEEDED" for j in scheduler["jobs"])
        assert not any(j["phase"] == "confirmation" for j in scheduler["jobs"]), "Confirmation already submitted"
        assert read_json(root / arm / "screening_job_completed.json")["state"] == "completed"
        assert read_json(root / arm / "guidance_audit.json")["ordinary_cfg_inactive"]
        result["arms"][arm] = {}
        candidates = {file_hash(path): read_json(path) for path in (root / arm).glob("*/*/sc_*/seed_10/metrics.json")}
        for sampler in ("primary", "secondary"):
            selected = read_json(root / arm / f"selection-{sampler}.json")
            rows = [candidates[digest] for digest in selected["screening_metrics_sha256"]]
            assert select_confirmation(rows) == selected["selection"]
            assert all(r["stage"] in ("screening", "extrapolation", "refinement") and r["seed"] == 10 for r in rows)
            for row in rows:
                for name, digest in row["artifact_hashes"].items():
                    assert file_hash(Path(row["samples_file"]).parent / name) == digest
            result["arms"][arm][sampler] = selected
    output = Path("guidance_sweep/selection.json")
    assert not output.exists(), "Frozen selection already exists; do not overwrite"
    atomic_json(output, result)
    print({arm: {s: [v["sc_cfg"] for v in r["selection"]["selected"]] for s, r in samplers.items()} for arm, samplers in result["arms"].items()})


if __name__ == "__main__":
    main()
