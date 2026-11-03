"""Run pinned author GRID quantizers on shared ES1/ES2 inputs.

Train and infer in the author environment, then import the full-catalog SID mapping.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def es1_resource_overrides(grid_python, env):
    # Probe the author interpreter, honoring CUDA_VISIBLE_DEVICES.
    available = int(
        subprocess.check_output(
            [str(grid_python), "-c", "import torch; print(torch.cuda.device_count())"],
            env=env,
            text=True,
        ).strip()
    )
    if available < 1:
        raise RuntimeError("ES1 SID fit requires at least one visible CUDA GPU")
    devices = 8 if available >= 8 else 1
    batch = 16384 // devices
    print(
        f"ES1 SID fit: {available} visible GPU(s), using {devices}; "
        f"batch/GPU={batch}, global batch=16384, updates=3000",
        file=sys.stderr,
    )
    overrides = [
        f"trainer.devices={devices}",
        f"data_loading.datamodule.train_dataloader_config.batch_size_per_device={batch}",
    ]
    if devices == 1:
        # Large batches must not be transferred through worker shared memory.
        prefix = "data_loading.datamodule.train_dataloader_config"
        overrides.extend(
            [
                f"{prefix}.num_workers=0",
                f"{prefix}.timeout=0",
                f"{prefix}.pin_memory=false",
                f"{prefix}.persistent_workers=false",
            ]
        )
    return overrides


def fit(experiment, raw, data, output, grid, grid_python, category, method, seed):
    from model_datasets.amazon2014_grid.grid_sids import import_grid_sids

    raw, data, output, grid = (p.resolve() for p in (raw, data, output, grid))
    # Resolving the interpreter symlink would escape its virtual environment.
    grid_python = grid_python.absolute()
    source = read(data / "stage01/runtime/source-inputs.json")
    if source["category"] != category:
        raise ValueError("Prepared data belongs to a different category")
    imported = output.parent / "sids" if experiment == "es1" else output / "imported"
    for destination in (output, imported):
        if destination.exists():
            raise FileExistsError(destination)
    env = {**os.environ, "PYTHONPATH": str(grid), "NVIDIA_TF32_OVERRIDE": "0"}
    train_overrides = (
        es1_resource_overrides(grid_python, env) if experiment == "es1" else []
    )
    # Partial outputs are retained after failure. A retry must use a new directory.
    (output / "runtime").mkdir(parents=True)
    metadata = read(data / "basis/representations/metadata.json")
    recipe = "reproduce_rkmeans" if experiment == "es1" else f"es2/{method}"
    common = [
        f"data_dir={raw}",
        f"embedding_path={data}/stage01/grid/pickle/merged_predictions_tensor.pt",
        f"seed={seed}",
    ]
    for phase, module in (("train", "src.train"), ("inference", "src.inference")):
        command = [
            str(grid_python),
            "-m",
            module,
            f"experiment={recipe}_{phase}",
            *common,
            f"hydra.run.dir={output / phase}",
        ]
        if phase == "inference":
            command.append(f"ckpt_path={output}/train/checkpoints/last.ckpt")
        else:
            command.extend(train_overrides)
        subprocess.run(command, cwd=grid, env=env, check=True)
    cfg = yaml.safe_load((output / "train/.hydra/config.yaml").read_text())
    devices = cfg["trainer"]["devices"]
    device_count = len(devices) if isinstance(devices, list) else devices
    batch_size = cfg["data_loading"]["datamodule"]["train_dataloader_config"][
        "batch_size_per_device"
    ]
    # Import validates the author prediction pair against these recorded hashes.
    predictions = output / "inference/pickle"
    hashes = {
        str(predictions / name): digest(predictions / name)
        for name in ("merged_predictions_tensor.pt", "merged_predictions.pkl")
    }
    report = {
        **source,
        "method": method,
        "seed": seed,
        "run_mode": f"{experiment}_from_source",
        "fit_scope": "full_author_catalog",
        "items": metadata["catalog_items"],
        "codebook_size": [256, 256, 256],
        "sid_length": 4,
        "train_devices": device_count,
        "batch_size_per_device": batch_size,
        "total_steps": cfg["trainer"]["max_steps"],
        "normalize_residuals": cfg["model"]["normalize_residuals"],
        "sha256": hashes,
    }
    report["sid_fit_diagnostics"] = {
        **source["sid_fit_diagnostics"],
        "sid_fitted": True,
    }
    write(output / "runtime/sid-validation.json", report)
    import_grid_sids(
        sid_dir=output,
        catalog=data / "prepared/catalog/items.pq",
        source_report=data / "prepared/report.json",
        output_dir=imported,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    training = commands.add_parser("fit")
    training.add_argument("experiment", choices=("es1", "es2"))
    for name in ("raw", "data", "output", "grid", "grid_python"):
        training.add_argument(name, type=Path)
    training.add_argument("category", choices=("beauty", "toys", "sports"))
    training.add_argument("method", choices=("rkmeans", "rvq", "rqvae"))
    training.add_argument("seed", type=int, choices=range(42, 47))
    args = vars(parser.parse_args())
    args.pop("action")
    fit(**args)


if __name__ == "__main__":
    main()
