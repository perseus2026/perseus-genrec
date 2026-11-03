"""Dataset experiment plans executed by the shell entrypoints."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("tecd-small", "yambda-50m", "tecd-full", "yambda-500m")
FULL_DOMAINS = ("retail", "marketplace", "offers")
FULL_MODES = ("positives", "all-events")
FULL_SHARDS = 10
TECD_REVISION = "c55530a7a179f06330ff626d35634d20c8a56f43"
YAMBDA_REVISION = "dd6f3a19eef5866e346c3270e098baa641a44948"
ENV = {
    "PYTHONUNBUFFERED": "1",
    "OMP_NUM_THREADS": "1",
    "NVIDIA_TF32_OVERRIDE": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "HF_HUB_DISABLE_XET": "1",
    "HF_HUB_DOWNLOAD_TIMEOUT": "600",
}
logger = logging.getLogger(__name__)


def models(dataset):
    """The complete approved matrix; each row selects one shared preparation."""
    if dataset == "tecd-small":
        return [
            {"name": f"{model}-{domain}-{mode}", "prepared": f"{domain}-{source}"}
            for domain in ("retail", "marketplace")
            for model, mode, source in (
                ("onerec", "rq1", "rq1"),
                ("hstu", "rq1", "rq1"),
                ("onerec", "cross-global", "cross-global"),
                ("hstu", "md-to-sd", "cross-global"),
                ("grid", "rq1", "rq1"),
            )
        ]
    if dataset == "tecd-full":
        return [
            {
                "name": f"onerec-{domain}-{mode}-{setup}",
                "prepared": f"{domain}-{mode}-{setup}",
                "test_config": f"onerec-{domain}-{mode}-{setup}-test",
            }
            for domain in FULL_DOMAINS
            for mode in FULL_MODES
            for setup in ("sd", "md")
        ]
    if dataset in {"yambda-50m", "yambda-500m"}:
        return [
            {"name": f"{model}-{mode}", "prepared": source}
            for model, mode, source in (
                ("onerec", "like", "like-features"),
                ("hstu", "like", "like-features"),
                ("onerec", "like-features", "like-features"),
                ("hstu", "like-features", "like-features"),
                ("onerec", "all-features", "all-features"),
                ("hstu", "all-features", "all-features"),
                ("grid", "like", "like-features"),
            )
        ]
    raise ValueError(f"Unknown dataset: {dataset}")


def plan(dataset):
    """Describe the native stages in execution order."""
    steps = []

    def add(
        task,
        action,
        parents,
        *,
        gpus=0,
        threads=8,
        **values,
    ):
        steps.append(
            {
                "task": task,
                "action": action,
                "parents": list(parents),
                "gpus": gpus,
                "threads": threads,
                **values,
            }
        )

    if dataset == "tecd-full":
        _full_preparation(add)
    elif dataset == "tecd-small":
        for domain in ("retail", "marketplace"):
            add(
                f"build_{domain}",
                "build",
                [],
                domain=domain,
            )
            add(
                f"sid_{domain}",
                "fit_sid",
                [f"build_{domain}"],
                domain=domain,
                gpus=1,
            )
        add(
            "build_cross",
            "build_cross",
            [],
            threads=16,
        )
        for domain in ("retail", "marketplace"):
            other = "marketplace" if domain == "retail" else "retail"
            for mode in ("rq1", "cross-global"):
                parents = [f"build_{domain}", f"sid_{domain}"]
                if mode == "cross-global":
                    parents += [f"build_{other}", "build_cross"]
                source = f"{domain}-{mode}"
                add(
                    f"prepare_{source.replace('-', '_')}",
                    "prepare",
                    parents,
                    source=source,
                    config=f"onerec-{source}",
                    domain=domain,
                    mode=mode,
                )
    elif dataset in {"yambda-50m", "yambda-500m"}:
        add(
            "build",
            "build",
            [],
            threads=16,
        )
        add(
            "fit_sid",
            "fit_sid",
            ["build"],
            gpus=1,
        )
        add(
            "apply_sid",
            "apply_sid",
            ["build", "fit_sid"],
        )
        for source in ("like-features", "all-features"):
            add(
                f"prepare_{source.replace('-', '_')}",
                "prepare",
                ["apply_sid"],
                source=source,
                config=f"onerec-{source}",
                threads=16,
            )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    for model in models(dataset):
        name, source = model["name"], model["prepared"]
        suffix = name.replace("-", "_")
        add(
            f"train_{suffix}",
            "train",
            [f"prepare_{source.replace('-', '_')}"],
            model=model,
            gpus=8,
            threads=4,
        )
        add(
            f"inference_{suffix}",
            "inference",
            [f"train_{suffix}"],
            model=model,
            gpus=4,
            threads=4,
        )
        add(
            f"evaluate_{suffix}",
            "evaluate",
            [f"inference_{suffix}"],
            model=model,
            threads=4,
        )
    add(
        "report",
        "report",
        [f"evaluate_{model['name'].replace('-', '_')}" for model in models(dataset)],
        threads=1,
    )
    return steps


def _full_preparation(add):
    """Keep the existing Full protocol and its bounded ten-shard preparation."""
    for domain in FULL_DOMAINS:
        add(
            f"build_{domain}",
            "build_full",
            [],
            domain=domain,
            threads=8 if domain == "retail" else 16,
        )
        add(
            f"sid_{domain}",
            "fit_sid",
            [f"build_{domain}"],
            domain=domain,
            gpus=1,
            threads=16,
            omp_threads=8,
        )
        add(
            f"apply_sid_{domain}",
            "apply_sid",
            [f"sid_{domain}"],
            domain=domain,
            gpus=1,
            threads=16,
        )
        for mode in FULL_MODES:
            suffix = f"{domain}_{mode.replace('-', '_')}"
            source = f"{domain}-{mode}"
            add(
                f"partition_{suffix}",
                "partition_history",
                [f"build_{domain}"],
                history=f"data/basis/{domain}/{mode}",
                source=source,
                threads=2,
            )
            add(
                f"materialize_{suffix}",
                "materialize_full",
                [f"apply_sid_{domain}"],
                domain=domain,
                mode=mode,
                source=source,
            )
    add(
        "build_payments",
        "build_payments",
        [f"build_{domain}" for domain in FULL_DOMAINS],
        threads=16,
    )
    add(
        "partition_payments",
        "partition_history",
        ["build_payments"],
        history="data/payments",
        source="payments",
        threads=2,
    )
    encode_threads = {
        "retail": (4, 4, 4, 8),
        "marketplace": (4, 8, 8, 8),
        "offers": (4, 8, 8, 8),
    }
    for domain in FULL_DOMAINS:
        for mode_index, mode in enumerate(FULL_MODES):
            for setup_index, setup in enumerate(("sd", "md")):
                source = f"{domain}-{mode}-{setup}"
                suffix = source.replace("-", "_")
                materialized = f"run/materialized/{domain}-{mode}"
                groups = (
                    [f"{domain}-{mode}"]
                    if setup == "sd"
                    else [*[f"{other}-{mode}" for other in FULL_DOMAINS], "payments"]
                )
                read_threads = 16 if setup == "md" else 4 if mode == "positives" else 8
                threads = encode_threads[domain][2 * mode_index + setup_index]
                add(
                    f"plan_{suffix}",
                    "plan_full",
                    [f"materialize_{domain}_{mode.replace('-', '_')}"],
                    source=source,
                    materialized=materialized,
                )
                for shard in range(FULL_SHARDS):
                    add(
                        f"read_{suffix}_{shard:02d}",
                        "read_full",
                        [
                            f"plan_{suffix}",
                            *[
                                "partition_" + group.replace("-", "_")
                                for group in groups
                            ],
                        ],
                        source=source,
                        shard=shard,
                        histories=groups,
                        threads=read_threads,
                    )
                add(
                    f"fit_{suffix}",
                    "fit_full",
                    [f"read_{suffix}_{shard:02d}" for shard in range(FULL_SHARDS)],
                    source=source,
                    materialized=materialized,
                )
                for shard in range(FULL_SHARDS):
                    add(
                        f"encode_{suffix}_{shard:02d}",
                        "encode_full",
                        [f"fit_{suffix}"],
                        source=source,
                        shard=shard,
                        threads=threads,
                    )
                add(
                    f"prepare_{suffix}",
                    "assemble_full",
                    [f"encode_{suffix}_{shard:02d}" for shard in range(FULL_SHARDS)],
                    source=source,
                    threads=4,
                )


def _run(module, arguments, env, *, ranks=1):
    command = [sys.executable]
    if ranks > 1:
        command += [
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={ranks}",
            "--module",
            module,
        ]
    else:
        command += ["-m", module]
    command += [str(value) for value in arguments]
    logger.info("Run %s", shlex.join(command))
    subprocess.run(command, env=env, check=True)


def _model_links(workdir, prepared, *, samples=False):
    for name in ("dataset", "samples") if samples else ("dataset",):
        path = workdir / name
        if not path.is_symlink():
            path.symlink_to((prepared / name).resolve(), target_is_directory=True)


def _config_path(dataset, name, runs):
    """Persist the effective recipe with inheritance resolved."""
    from genrecsys.core import Config

    source = ROOT / "configs" / dataset / (name + ".yaml")
    path = runs / "configs" / (name + ".yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    Config.load(source).save(path)
    return path


def run_stage(dataset, task, *, data_dir, run_dir):
    """Execute one native stage."""
    from genrecsys.core import Config, DataContract
    from genrecsys.utils.files import code_provenance, file_sha256

    step = next(value for value in plan(dataset) if value["task"] == task)
    data, runs = Path(data_dir).resolve(), Path(run_dir).resolve()
    env = {
        **os.environ,
        **ENV,
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}",
        "POLARS_MAX_THREADS": str(step["threads"]),
        "RAYON_NUM_THREADS": str(step["threads"]),
        "OMP_NUM_THREADS": str(step.get("omp_threads", 1)),
    }
    timings = runs / "timings"
    timings.mkdir(parents=True, exist_ok=True)
    env["GENRECSYS_TIMINGS_PATH"] = str(timings / f"{task}.json")
    started = time.monotonic()
    logger.info("Start %s / %s", dataset, task)
    action = step["action"]
    if action == "build_full":
        _run(
            "model_datasets.tecd.full",
            [
                "download-and-build-full-table6",
                f"--domain={step['domain']}",
                f"--output-dir={data / 'basis' / step['domain']}",
                f"--revision={TECD_REVISION}",
                "--max-workers=8",
            ],
            env,
        )
    elif action == "build_payments":
        _run(
            "model_datasets.tecd.payments",
            [
                "download-and-build-full-payments-history",
                f"--samples-root={data / 'basis'}",
                f"--output-dir={data / 'payments'}",
                f"--revision={TECD_REVISION}",
                "--start-day=1092",
                "--end-day=1307",
                "--file-workers=4",
            ],
            env,
        )
    elif action == "partition_history":
        _run(
            "genrecsys",
            [
                "prepare-shards",
                "partition-histories",
                f"--history-dir={data / Path(step['history']).relative_to('data')}",
                f"--output-dir={data / 'partitioned' / step['source']}",
                f"--shards={FULL_SHARDS}",
            ],
            env,
        )
    elif action == "materialize_full":
        domain, mode = step["domain"], step["mode"]
        _run(
            "model_datasets.tecd.full",
            [
                "materialize-full-table6-basis",
                f"--config={_config_path(dataset, f'onerec-{domain}-{mode}-sd', runs)}",
                f"--basis={data / 'basis' / domain / mode}",
                f"--sid-mapping={data / 'sid' / domain / 'catalog/item_to_sid.parquet'}",
                f"--workdir={runs / 'materialized' / step['source']}",
            ],
            env,
        )
    elif action in {
        "plan_full",
        "read_full",
        "fit_full",
        "encode_full",
        "assemble_full",
    }:
        source = step["source"]
        work = runs / "shards" / source
        prepared = runs / "prepared" / source
        if action == "plan_full":
            _run(
                "genrecsys",
                [
                    "prepare-shards",
                    "plan",
                    f"--workdir={runs / Path(step['materialized']).relative_to('run')}",
                    f"--config={_config_path(dataset, f'onerec-{source}', runs)}",
                    f"--output-dir={work / 'plan'}",
                    f"--shards={FULL_SHARDS}",
                ],
                env,
            )
        elif action == "read_full":
            shard = step["shard"]
            archives = work / "archives" / f"{shard:02d}"
            archives.mkdir(parents=True)
            for group in step["histories"]:
                (archives / f"{group}.tar").symlink_to(
                    data / "partitioned" / group / f"{shard:02d}.tar"
                )
            _run(
                "genrecsys",
                [
                    "prepare-shards",
                    "read",
                    f"--input-dir={work / 'plan' / f'{shard:02d}'}",
                    f"--history-archive={archives}",
                    f"--output-dir={work / 'selected' / f'{shard:02d}'}",
                    "--delete-inputs",
                ],
                env,
            )
        elif action == "fit_full":
            _run(
                "genrecsys",
                [
                    "prepare-shards",
                    "fit",
                    f"--workdir={runs / Path(step['materialized']).relative_to('run')}",
                    f"--plan-dir={work / 'plan'}",
                    f"--shards-dir={work / 'selected'}",
                    f"--output-dir={work / 'fitted'}",
                ],
                env,
            )
        elif action == "encode_full":
            shard = step["shard"]
            output = work / "encoded" / f"{shard:02d}"
            _run(
                "genrecsys",
                [
                    "prepare-shards",
                    "encode",
                    f"--input-dir={work / 'selected' / f'{shard:02d}'}",
                    f"--fitted-dir={work / 'fitted'}",
                    f"--output-dir={output}",
                    "--delete-inputs",
                ],
                env,
            )
            events = prepared / "dataset/events"
            events.mkdir(parents=True, exist_ok=True)
            shutil.move(
                output / f"data-{shard:02d}.arrow", events / f"data-{shard:02d}.arrow"
            )
        else:
            # Arrow payloads are already at their final paths; assemble metadata.
            _run(
                "genrecsys",
                [
                    "prepare-shards",
                    "assemble",
                    f"--fitted-dir={work / 'fitted'}",
                    f"--encoded-dir={work / 'encoded'}",
                    f"--output-dir={work / 'assembled'}",
                ],
                env,
            )
            shutil.copytree(work / "assembled", prepared, dirs_exist_ok=True)
            _run(
                "genrecsys",
                [
                    "inference",
                    "distribute-samples",
                    f"--workdir={prepared}",
                    "--history-source=event_store",
                ],
                env,
            )
    elif action == "build":
        if dataset == "tecd-small":
            domain = step["domain"]
            raw, basis = data / "raw" / domain, data / "basis" / domain
            _run(
                "model_datasets.tecd",
                [
                    f"download-small-{domain}",
                    f"--output-dir={raw}",
                    f"--revision={TECD_REVISION}",
                ],
                env,
            )
            _run(
                "model_datasets.tecd",
                [
                    f"build-small-{domain}",
                    f"--source-dir={raw}",
                    f"--output-dir={basis}",
                    f"--source-revision={TECD_REVISION}",
                    "--history-gap-hours=0",
                    "--no-include-views",
                    "--train-ratio=0.75",
                ],
                env,
            )
        else:
            _run(
                "model_datasets.yambda",
                [
                    "download-source",
                    f"--output-dir={data / 'raw'}",
                    f"--revision={YAMBDA_REVISION}",
                    f"--dataset-size={dataset.removeprefix('yambda-')}",
                ],
                env,
            )
            _run(
                "model_datasets.yambda",
                [
                    "build-basis",
                    f"--source-dir={data / 'raw'}",
                    f"--output-dir={data / 'basis'}",
                    f"--source-revision={YAMBDA_REVISION}",
                    f"--dataset-size={dataset.removeprefix('yambda-')}",
                ],
                env,
            )
    elif action == "fit_sid":
        basis = (
            data / "basis" / step["domain"]
            if dataset.startswith("tecd-")
            else data / "basis"
        )
        output = (
            data / "sid" / step["domain"]
            if dataset.startswith("tecd-")
            else data / "sid-mapper"
        )
        if dataset == "tecd-full":
            output = output / "mapper"
        sizes = [256, 256] if dataset.startswith("tecd-") else [128, 128, 128]
        _run(
            "item_representations.semantic_ids",
            [
                "fit-sid-mapper",
                f"--items={basis / 'representations/train/items.pq'}",
                f"--embeddings={basis / 'representations/train/embeddings.npy'}",
                f"--output-dir={output}",
                "--method=rq_kmeans",
                *[f"--codebook-size={size}" for size in sizes],
                "--device=cuda",
                "--fit-batch-size=65536",
                "--eval-batch-size=8192",
                "--max-iter=100",
                "--n-init=3",
                "--mapper-only",
            ],
            env,
        )
    elif action == "apply_sid":
        domain = step.get("domain")
        checkpoint = (
            data / "sid" / domain / "mapper/checkpoint.pkl"
            if domain
            else data / "sid-mapper/checkpoint.pkl"
        )
        representations = (
            data / "basis" / domain / "representations/all"
            if domain
            else data / "basis/representations/all"
        )
        output = data / "sid" / domain / "catalog" if domain else data / "sids"
        _run(
            "item_representations.semantic_ids",
            [
                "apply-sid-mapper",
                f"--checkpoint={checkpoint}",
                f"--items={representations / 'items.pq'}",
                f"--embeddings={representations / 'embeddings.npy'}",
                f"--output-dir={output}",
                "--device=cuda" if domain else "--device=cpu",
                "--eval-batch-size=25000" if domain else "--eval-batch-size=8192",
            ],
            env,
        )
    elif action == "build_cross":
        _run(
            "model_datasets.tecd",
            [
                "download-and-build-cross-history",
                f"--output-dir={data / 'cross-history'}",
                f"--revision={TECD_REVISION}",
            ],
            env,
        )
        _run(
            "model_datasets.tecd",
            ["extract-eventhub", f"--basis={data / 'cross-history'}"],
            env,
        )
    elif action == "prepare":
        config = _config_path(dataset, step["config"], runs)
        workdir = runs / "prepared" / step["source"]
        if dataset == "tecd-small":
            domain = step["domain"]
            basis = data / "basis" / domain
            env["INTERNAL_STORAGE_TECD"] = str(basis / "eventhub")
            args = [
                "materialize-basis",
                f"--config={config}",
                f"--basis={basis}",
                f"--sid-checkpoint={data / 'sid' / domain / 'checkpoint.pkl'}",
                f"--embedding-items={basis / 'representations/all/items.pq'}",
                f"--embeddings={basis / 'representations/all/embeddings.npy'}",
                f"--workdir={workdir}",
            ]
            if step["mode"] == "cross-global":
                other = "marketplace" if domain == "retail" else "retail"
                cross = data / "cross" / domain
                cross.mkdir(parents=True)
                for folder in (
                    data / "basis" / other / "eventhub",
                    data / "cross-history/eventhub",
                ):
                    for path in folder.iterdir():
                        (cross / path.name).symlink_to(path, target_is_directory=True)
                env["INTERNAL_STORAGE_TECD_CROSS"] = str(cross)
                args.append(
                    f"--auxiliary-item-features={data / 'basis' / other / 'item_features.pq'}"
                )
            _run("model_datasets.tecd", args, env)
        else:
            env["INTERNAL_STORAGE_YAMBDA"] = str(data / "basis/eventhub")
            _run(
                "model_datasets.yambda",
                [
                    "materialize-basis",
                    f"--basis={data / 'basis'}",
                    f"--sid-mapping={data / 'sids/item_to_sid.parquet'}",
                    f"--workdir={workdir}",
                    f"--config={config}",
                ],
                env,
            )
        _run("genrecsys", ["train", "prepare-dataset", f"--workdir={workdir}"], env)
        _run(
            "genrecsys",
            [
                "inference",
                "distribute-samples",
                f"--workdir={workdir}",
                "--history-source=event_store",
            ],
            env,
        )
    elif action in {"train", "inference", "evaluate"}:
        model = step["model"]
        config_path = _config_path(dataset, model["name"], runs)
        prepared = runs / "prepared" / model["prepared"]
        workdir = runs / "models" / model["name"]
        if action == "train":
            config = Config.load(config_path)
            contract = DataContract.load(prepared / "checkpoint/data_contract.json")
            contract.require_compatible(config)
            shutil.copytree(prepared / "checkpoint", workdir / "checkpoint")
            _model_links(workdir, prepared)
            _run(
                "genrecsys",
                [
                    "train",
                    "fit-model",
                    f"--workdir={workdir}",
                    f"--config={config_path}",
                    "--memory-map-samples",
                ],
                env,
                ranks=step["gpus"],
            )
            (workdir / "checkpoint/experiment-input.json").write_text(
                json.dumps(
                    {
                        "prepared": model["prepared"],
                        "data_contract": contract.sha256,
                        "sid_mapping": file_sha256(
                            prepared
                            / "checkpoint/encoders/preprocessors/item/mapping.pq"
                        ),
                    },
                    indent=2,
                )
                + "\n"
            )
        else:
            if action == "inference" and "test_config" in model:
                test_config = Config.load(
                    _config_path(dataset, model["test_config"], runs)
                )
                DataContract.load(
                    workdir / "checkpoint/data_contract.json"
                ).require_compatible(test_config)
                test_config.save(workdir / "checkpoint/config.yaml")
            _model_links(workdir, prepared, samples=action == "inference")
            args = [
                "inference",
                "make-predictions" if action == "inference" else "evaluate",
                f"--workdir={workdir}",
            ]
            if action == "inference":
                args += [
                    "--history-source=event_store",
                    "--input-batch-size=1024",
                    "--prediction-write-batch-size=1024",
                ]
            _run("genrecsys", args, env, ranks=step["gpus"] or 1)
    elif action == "report":
        records, sid_mappings = [], {}
        for model in models(dataset):
            workdir = runs / "models" / model["name"]
            metrics = json.loads((workdir / "metrics.json").read_text())
            if not metrics["population_verified"]:
                raise ValueError(f"Incomplete test predictions: {model['name']}")
            identity = json.loads(
                (workdir / "checkpoint/experiment-input.json").read_text()
            )
            group = (
                model["prepared"].split("-", 1)[0]
                if dataset.startswith("tecd-")
                else dataset
            )
            sid_mappings.setdefault(group, set()).add(identity["sid_mapping"])
            records.append(
                {
                    "model": model["name"],
                    "prepared": identity["prepared"],
                    "data_contract": identity["data_contract"],
                    "sid_mapping": identity["sid_mapping"],
                    "samples": metrics["samples"],
                    **metrics["metrics"],
                }
            )
        if any(len(values) != 1 for values in sid_mappings.values()):
            raise ValueError(
                "Experiments do not share one frozen SID mapping per domain"
            )
        output = runs / "summary"
        output.mkdir(parents=True)
        (output / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "effective_configs": {
                        path.name: file_sha256(path)
                        for path in sorted((runs / "configs").glob("*.yaml"))
                    },
                    "models": models(dataset),
                    "stages": plan(dataset),
                },
                indent=2,
            )
            + "\n"
        )
        files = [
            *(ROOT / "src").rglob("*.py"),
            ROOT / "scripts/__init__.py",
            ROOT / "scripts/experiments.py",
            ROOT / "scripts" / dataset / "run_all.sh",
            *(ROOT / "configs" / dataset).glob("*.yaml"),
        ]
        (output / "code-provenance.json").write_text(
            json.dumps(code_provenance(ROOT, files), indent=2) + "\n"
        )
        with (output / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(json.dumps(records, indent=2), flush=True)
    else:
        raise ValueError(f"Unknown experiment action: {action}")
    elapsed = time.monotonic() - started
    (timings / f"{task}-stage.json").write_text(
        json.dumps({"task": task, "elapsed_seconds": elapsed}, indent=2) + "\n"
    )
    logger.info("Completed %s / %s elapsed=%.1fs", dataset, task, elapsed)


def run_all(dataset, *, data_dir, run_dir):
    """Run every stage sequentially on one machine."""
    for step in plan(dataset):
        run_stage(dataset, step["task"], data_dir=data_dir, run_dir=run_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run-all", "run-stage"))
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--task")
    parser.add_argument("--data-dir", type=Path, default=Path("work/data"))
    parser.add_argument("--run-dir", type=Path, default=Path("work/run"))
    args = parser.parse_args()
    if args.action == "run-all":
        run_all(args.dataset, data_dir=args.data_dir, run_dir=args.run_dir)
    else:
        if args.task is None:
            parser.error("run-stage requires --task")
        run_stage(args.dataset, args.task, data_dir=args.data_dir, run_dir=args.run_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    main()
