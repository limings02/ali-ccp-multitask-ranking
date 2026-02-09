import subprocess
import sys
from pathlib import Path


CONFIGS = [
    "configs/experiments/more_adapt/more_1_ple.yaml",
    "configs/experiments/more_adapt/final_ple.yaml",
    "configs/experiments/more_adapt/final_mmoe.yaml",
]


def run_train(config_path: str) -> int:
    cmd = [
        sys.executable,
        "-m",
        "src.cli.main",
        "train",
        "--config",
        config_path,
    ]
    print("\n=== Running:", " ".join(cmd), "===", flush=True)
    return subprocess.call(cmd)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    results = []
    for cfg in CONFIGS:
        cfg_path = str((repo_root / cfg).as_posix())
        code = run_train(cfg_path)
        results.append((cfg, code))

    print("\n=== Summary ===")
    exit_code = 0
    for cfg, code in results:
        status = "OK" if code == 0 else f"FAIL({code})"
        print(f"{cfg}: {status}")
        if code != 0:
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
