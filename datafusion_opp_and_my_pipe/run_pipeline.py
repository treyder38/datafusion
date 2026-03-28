"""Unified pipeline orchestrator for datafusion_opp_and_my_pipe.

Runs the full training pipeline in dependency order, skips completed steps
by default, and can resume from existing artifacts with a single command.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
N_FOLDS = 4


@dataclass(frozen=True)
class Step:
    key: str
    label: str
    script: str
    outputs: tuple[str, ...]
    depends_on: tuple[str, ...] = ()

    def output_paths(self) -> list[Path]:
        return [ROOT / rel_path for rel_path in self.outputs]

    def script_path(self) -> Path:
        return ROOT / self.script


STEPS = [
    Step(
        key="fe",
        label="Feature engineering",
        script="01_feature_engineering.py",
        outputs=(
            "features/train_features.parquet",
            "features/test_features.parquet",
            "features/targets.parquet",
            "features/meta.json",
        ),
    ),
    Step(
        key="feat_sel",
        label="Per-target feature selection",
        script="01b_select_features.py",
        outputs=("features/selected_features/summary.json",),
        depends_on=("fe",),
    ),
    Step(
        key="oof_feats",
        label="Cross-target OOF features",
        script="01c_add_oof_features.py",
        outputs=("features/oof_features_train.parquet", "features/oof_features_test.parquet"),
        depends_on=("nn", "lgbm", "xgboost", "pyboost", "catboost"),
    ),
    Step(
        key="nn",
        label="Neural network",
        script="02_train_nn.py",
        outputs=tuple(f"checkpoints_nn/fold_{i}.npz" for i in range(N_FOLDS)),
        depends_on=("fe",),
    ),
    Step(
        key="knn_feats",
        label="kNN features",
        script="01d_knn_features.py",
        outputs=("features/knn_features_train.parquet", "features/knn_features_test.parquet"),
        depends_on=("nn",),
    ),
    Step(
        key="lgbm",
        label="LightGBM",
        script="03_train_lgbm.py",
        outputs=("checkpoints_lgbm/lgbm_predictions.npz",),
        depends_on=("fe",),
    ),
    Step(
        key="lgbm_diverse",
        label="LightGBM diverse variants",
        script="03b_train_lgbm_diverse.py",
        outputs=("checkpoints_lgbm_diverse/lgbm_diverse_predictions.npz",),
        depends_on=("fe",),
    ),
    Step(
        key="xgboost",
        label="XGBoost",
        script="04_train_xgboost.py",
        outputs=("checkpoints_xgboost/xgb_predictions.npz",),
        depends_on=("fe",),
    ),
    Step(
        key="pyboost",
        label="PyBoost",
        script="05_train_pyboost.py",
        outputs=("checkpoints_pyboost/pyboost_predictions.npz",),
        depends_on=("fe",),
    ),
    Step(
        key="catboost",
        label="CatBoost",
        script="06_train_catboost.py",
        outputs=("checkpoints_catboost/cb_predictions.npz",),
        depends_on=("fe", "feat_sel"),
    ),
    Step(
        key="lgbm_meta",
        label="LGBM meta",
        script="07_train_lgbm_meta.py",
        outputs=("checkpoints_lgbm_meta/lgbm_predictions.npz",),
        depends_on=("fe", "nn", "lgbm", "xgboost", "pyboost", "catboost"),
    ),
    Step(
        key="blend",
        label="Blend",
        script="08_blend.py",
        outputs=("blend_artifacts/blend_data.npz", "submissions/blend.parquet"),
        depends_on=("nn", "lgbm", "lgbm_diverse", "xgboost", "pyboost", "catboost", "lgbm_meta"),
    ),
    Step(
        key="stacking",
        label="Stacking",
        script="09_stacking.py",
        outputs=("submissions/stacking.parquet",),
        depends_on=("blend",),
    ),
    Step(
        key="pseudo_label",
        label="Pseudo-labeling",
        script="10_pseudo_label.py",
        outputs=("submissions/pseudo_labeled.parquet",),
        depends_on=("stacking",),
    ),
]

STEP_BY_KEY = {step.key: step for step in STEPS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full datafusion pipeline with one command.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch step scripts.",
    )
    parser.add_argument(
        "--from-step",
        dest="from_step",
        choices=[step.key for step in STEPS],
        help="Start from this step key.",
    )
    parser.add_argument(
        "--to-step",
        dest="to_step",
        choices=[step.key for step in STEPS],
        help="Stop after this step key.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=[step.key for step in STEPS],
        help="Run only the listed steps, in pipeline order.",
    )
    parser.add_argument(
        "--rerun",
        nargs="*",
        choices=[step.key for step in STEPS] + ["all"],
        help="Force rerun for selected steps even if outputs already exist.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print available step keys and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without actually launching scripts.",
    )
    return parser.parse_args()


def get_selected_steps(args: argparse.Namespace) -> list[Step]:
    if args.steps:
        selected = {key for key in args.steps}
        return [step for step in STEPS if step.key in selected]

    start_idx = 0
    end_idx = len(STEPS) - 1
    if args.from_step:
        start_idx = next(i for i, step in enumerate(STEPS) if step.key == args.from_step)
    if args.to_step:
        end_idx = next(i for i, step in enumerate(STEPS) if step.key == args.to_step)
    if start_idx > end_idx:
        raise SystemExit("--from-step must not be after --to-step")
    return STEPS[start_idx:end_idx + 1]


def get_rerun_keys(args: argparse.Namespace) -> set[str]:
    if not args.rerun:
        return set()
    if "all" in args.rerun:
        return {step.key for step in STEPS}
    return set(args.rerun)


def outputs_exist(step: Step) -> bool:
    return all(path.exists() for path in step.output_paths())


def ensure_dependencies(step: Step, selected_keys: set[str], rerun_keys: set[str]) -> None:
    missing = []
    for dep_key in step.depends_on:
        dep = STEP_BY_KEY[dep_key]
        dep_selected = dep_key in selected_keys
        dep_ready = outputs_exist(dep)
        dep_forced = dep_key in rerun_keys
        if dep_forced and not dep_selected:
            missing.append(f"{dep_key} (marked for rerun but not selected)")
        elif not dep_ready and not dep_selected:
            missing.append(dep_key)
    if missing:
        formatted = ", ".join(missing)
        raise SystemExit(
            f"Cannot run step '{step.key}': missing required upstream artifacts for {formatted}."
        )


def run_step(step: Step, python_exe: str, dry_run: bool) -> None:
    cmd = [python_exe, step.script]
    print(f"\n==> {step.label} [{step.key}]")
    print(f"    Command: {' '.join(cmd)}")
    if dry_run:
        return
    started = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - started
    if result.returncode != 0:
        raise SystemExit(f"Step '{step.key}' failed with exit code {result.returncode}")
    print(f"    Completed in {elapsed / 60:.1f} min")


def main() -> None:
    args = parse_args()

    if args.list:
        print("Available steps:")
        for step in STEPS:
            print(f"  {step.key:<10} {step.label}")
        return

    selected_steps = get_selected_steps(args)
    selected_keys = {step.key for step in selected_steps}
    rerun_keys = get_rerun_keys(args)

    print("=" * 72)
    print("DataFusion Pipeline Orchestrator")
    print("=" * 72)
    print(f"Root: {ROOT}")
    print(f"Python: {args.python}")
    print(f"Mode: {'dry-run' if args.dry_run else 'execute'}")
    print("")

    total_started = time.time()
    for step in selected_steps:
        ensure_dependencies(step, selected_keys, rerun_keys)
        if step.key not in rerun_keys and outputs_exist(step):
            print(f"==> {step.label} [{step.key}]")
            print("    Skipped: outputs already exist")
            continue
        run_step(step, args.python, args.dry_run)

    total_elapsed = time.time() - total_started
    print(f"\nPipeline finished in {total_elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
