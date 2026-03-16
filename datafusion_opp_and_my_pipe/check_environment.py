"""Environment checker for datafusion_opp_and_my_pipe.

Validates required Python packages, reports versions, and estimates which
device each training stage will use on the current machine.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from utils import detect_lgbm_device, get_device

try:
    from packaging.requirements import Requirement
except Exception:  # pragma: no cover - best effort fallback
    Requirement = None


ROOT = Path(__file__).resolve().parent
REQ_FILE = ROOT / "requirements.txt"


@dataclass
class CheckResult:
    name: str
    requirement: str
    installed: str | None
    ok: bool
    note: str = ""


PACKAGE_NAME_OVERRIDES = {
    "scikit-learn": "sklearn",
    "iterative-stratification": "iterstrat",
    "cupy-cuda12x": "cupy",
}


def parse_requirements() -> list[str]:
    lines = []
    for line in REQ_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def requirement_to_import_name(req_name: str) -> str:
    return PACKAGE_NAME_OVERRIDES.get(req_name, req_name.replace("-", "_"))


def check_requirement(req_line: str) -> CheckResult:
    if Requirement is None:
        return CheckResult(req_line, req_line, None, False, "packaging is not installed")

    req = Requirement(req_line)
    try:
        installed_version = version(req.name)
    except PackageNotFoundError:
        return CheckResult(req.name, req_line, None, False, "not installed")

    ok = installed_version in req.specifier
    note = "" if ok else f"expected {req.specifier or 'any version'}"
    return CheckResult(req.name, req_line, installed_version, ok, note)


def check_optional_package(dist_name: str, import_name: str | None = None) -> tuple[bool, str]:
    import_name = import_name or requirement_to_import_name(dist_name)
    try:
        mod = import_module(import_name)
        installed_version = getattr(mod, "__version__", "unknown")
        return True, installed_version
    except Exception as exc:
        return False, str(exc)


def detect_catboost_device() -> tuple[str, str]:
    try:
        from catboost import CatBoostClassifier
    except Exception as exc:
        return "UNAVAILABLE", str(exc)

    try:
        cb = CatBoostClassifier(iterations=1, task_type="GPU", devices="0", verbose=0)
        cb.fit([[0, 0], [1, 1]], [0, 1])
        return "GPU", "CatBoost GPU check passed"
    except Exception as exc:
        return "CPU", str(exc)


def detect_pyboost_device() -> tuple[str, str]:
    try:
        import torch
    except Exception as exc:
        return "UNAVAILABLE", f"torch import failed: {exc}"

    if not torch.cuda.is_available():
        return "UNAVAILABLE", "torch.cuda.is_available() == False"

    try:
        import cupy  # noqa: F401
    except Exception as exc:
        return "UNAVAILABLE", f"cupy import failed: {exc}"

    try:
        from py_boost import SketchBoost  # noqa: F401
    except Exception as exc:
        return "UNAVAILABLE", f"py_boost import failed: {exc}"

    return "CUDA", "torch + cupy + py_boost are available"


def detect_nn_device() -> tuple[str, str]:
    try:
        return str(get_device()).upper(), "torch device detection succeeded"
    except Exception as exc:
        return "UNAVAILABLE", str(exc)


def detect_safe_lgbm_device() -> tuple[str, str]:
    try:
        _, device_name, _ = detect_lgbm_device()
        return device_name, "LightGBM device detection succeeded"
    except Exception as exc:
        return "UNAVAILABLE", str(exc)


def print_requirements_report() -> bool:
    print("=" * 72)
    print("Package Checks")
    print("=" * 72)
    all_ok = True
    parsed_requirements = parse_requirements()
    required_dist_names = set()
    for req_line in parsed_requirements:
        result = check_requirement(req_line)
        required_dist_names.add(result.name)
        installed = result.installed or "-"
        status = "OK" if result.ok else "FAIL"
        print(f"{status:>5}  {result.name:<26} installed={installed:<12} required={result.requirement}")
        if result.note:
            print(f"       note: {result.note}")
        all_ok &= result.ok

    print("")
    for dist_name in ["py-boost", "cupy-cuda12x"]:
        if dist_name in required_dist_names:
            continue
        ok, detail = check_optional_package(dist_name)
        status = "OK" if ok else "MISS"
        print(f"{status:>5}  {dist_name:<26} {detail}")
    return all_ok


def print_system_report() -> None:
    print("\n" + "=" * 72)
    print("Runtime Checks")
    print("=" * 72)
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {sys.platform}")

    try:
        import torch

        print(f"torch version: {torch.__version__}")
        print(f"torch cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"torch cuda device count: {torch.cuda.device_count()}")
            print(f"torch current device: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"torch check failed: {exc}")


def print_training_plan() -> None:
    nn_device, nn_note = detect_nn_device()
    lgbm_device, lgbm_note = detect_safe_lgbm_device()
    pyboost_device, pyboost_note = detect_pyboost_device()
    catboost_device, catboost_note = detect_catboost_device()

    print("\n" + "=" * 72)
    print("Training Device Plan")
    print("=" * 72)
    print(f"01_feature_engineering.py   -> CPU")
    print(f"02_train_nn.py             -> {nn_device}")
    print(f"03_train_lgbm.py           -> {lgbm_device}")
    print(f"04_train_pyboost.py        -> {pyboost_device}")
    print(f"05_train_catboost.py       -> {catboost_device}")
    print(f"06_train_lgbm_meta.py      -> {lgbm_device}")
    print(f"07_blend.py                -> CPU")
    print(f"08_stacking.py             -> Ridge=CPU, LGBM meta={lgbm_device}")

    print("\nNotes:")
    print(f"- NN: {nn_note}")
    print(f"- LightGBM: {lgbm_note}")
    print(f"- PyBoost: {pyboost_note}")
    print(f"- CatBoost: {catboost_note}")


def main() -> None:
    all_required_ok = print_requirements_report()
    print_system_report()
    print_training_plan()

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    if all_required_ok:
        print("Required Python packages look OK.")
    else:
        print("Some required Python packages are missing or do not satisfy requirements.")
    print("Use this report to decide whether the full pipeline can run as-is.")


if __name__ == "__main__":
    main()
