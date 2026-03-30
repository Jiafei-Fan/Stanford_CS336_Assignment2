from __future__ import annotations

import argparse
import os
import statistics
import timeit
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.cuda.nvtx as nvtx

from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import clip_gradient, cross_entropy
from cs336_basics.optimizer import AdamW, get_cosine_lr

try:
    import wandb
except ImportError:
    wandb = None

TimingPass = Literal["forward", "forward_backward"]

# -------------------------------
# Hardcoded config
# -------------------------------
DATA_DIR = Path("/home/fjf/Jiafei/Stanford_CS336_Assignment1/data")
TRAIN_DATA = DATA_DIR / "TinyStoriesV2-GPT4-train.npy"
CHECKPOINT_DIR = Path("/home/fjf/Jiafei/Stanford_CS336_Assignment2/checkpoints")
RESULTS_DIR = Path(__file__).resolve().parent
BENCHMARK_HISTORY_CSV = RESULTS_DIR / "benchmark_history.csv"
MEMORY_SNAPSHOT_DIR = RESULTS_DIR / "memory_snapshots"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

VOCAB_SIZE = 10000
ROPE_THETA = 10000.0
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8
MAX_GRAD_NORM = 1.0
BATCH_SIZE = 4
CONTEXT_LENGTH = 256
TOTAL_STEPS = 10000
VALIDATION_INTERVAL = 10
CHECKPOINT_INTERVAL = 1000
ENABLE_WANDB = True
WANDB_PROJECT = "cs336-systems"
WANDB_ENTITY = None
WANDB_RUN_NAME = None

# Model shape defaults (can still be overridden by CLI args below).
D_MODEL = 512
D_FF = 1344
NUM_LAYERS = 4
NUM_HEADS = 16


# Benchmark controls (hardcoded)
TIMING_PASS: TimingPass = "forward_backward"  # "forward" or "forward_backward"
WARMUP_STEPS = 5
MEASUREMENT_STEPS = 10
MEMORY_PROFILE_STEPS = 10


# Keep only these arguments as requested.
parser = argparse.ArgumentParser(description="transformer")
parser.add_argument("--d_model", type=int, default=D_MODEL, help="The model dimension.")
parser.add_argument("--d_ff", type=int, default=D_FF, help="The feedforward dimension.")
parser.add_argument("--num_layers", type=int, default=NUM_LAYERS, help="The number of layers in the Transformer.")
parser.add_argument("--num_heads", type=int, default=NUM_HEADS, help="The number of attention heads.")
parser.add_argument("--context_length", type=int, default=CONTEXT_LENGTH, help="The context length.")
parser.add_argument(
    "--model_size",
    type=str,
    default="custom",
    help="Model size tag used in checkpoint filenames (e.g. small, medium, 2.7B).",
)
parser.add_argument(
    "--run_only_partial",
    action="store_true",
    help="Run only benchmark steps, then exit.",
)
parser.add_argument(
    "--run_with_warmup",
    action="store_true",
    help="Run benchmark warmup steps before the timed measurement.",
)
parser.add_argument(
    "--run_feedfoward",
    action="store_true",
    help="Benchmark forward pass only.",
)
parser.add_argument(
    "--run_both_feedfoward_backward",
    action="store_true",
    help="Benchmark both forward and backward pass.",
)
parser.add_argument(
    "--mixed_precision",
    action="store_true",
    help="Use CUDA BF16 autocast for model forward passes.",
)
parser.add_argument(
    "--write_benchmark_csv",
    action="store_true",
    help="Append benchmark results to benchmark_history.csv.",
)
parser.add_argument(
    "--dump_memory_snapshot",
    action="store_true",
    help="Dump a CUDA memory snapshot pickle after benchmarking.",
)
args = parser.parse_args()

if args.run_feedfoward and args.run_both_feedfoward_backward:
    parser.error("Use only one of --run_feedfoward or --run_both_feedfoward_backward.")

timing_pass: TimingPass = TIMING_PASS
if args.run_feedfoward:
    timing_pass = "forward"
elif args.run_both_feedfoward_backward:
    timing_pass = "forward_backward"


@dataclass
class TrainConfig:
    device: torch.device
    dtype: torch.dtype
    mixed_precision: bool
    train_data: Path
    checkpoint_dir: Path | None
    vocab_size: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    model_size: str
    rope_theta: float
    learning_rate: float
    min_learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    eps: float
    max_gradient_norm: float
    batch_size: int
    context_length: int
    total_steps: int
    validation_interval: int
    checkpoint_interval: int
    run_only_partial: bool
    timing_pass: TimingPass
    warmup_steps: int
    measurement_steps: int
    write_benchmark_csv: bool
    dump_memory_snapshot: bool
    enable_wandb: bool
    wandb_project: str
    wandb_entity: str | None
    wandb_run_name: str | None


@dataclass
class BenchmarkResult:
    mean_time: float
    std_time: float
    precision: str


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    checkpoint_path: str | os.PathLike[str],
) -> None:
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )


def checkpoint_size_tag(model_size: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in model_size)


def bf16_autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def normalize_model_size(model_size: str) -> str:
    if model_size in {"2.7B", "2_7B"}:
        return "2_7B"
    return model_size


def append_benchmark_history_row(row: dict[str, object]) -> None:
    history_df = pd.DataFrame([row])
    history_df.to_csv(
        BENCHMARK_HISTORY_CSV,
        mode="a",
        header=not BENCHMARK_HISTORY_CSV.exists(),
        index=False,
    )
    print(f"[timing] appended history csv: {BENCHMARK_HISTORY_CSV}")


def save_benchmark_result(config: TrainConfig, result: BenchmarkResult) -> None:
    if not config.write_benchmark_csv:
        print("[timing] skipping benchmark csv write because write_benchmark_csv is disabled.")
        return
    model_size = normalize_model_size(config.model_size)
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_size": model_size,
        "context_length": config.context_length,
        "with_warmup": int(config.warmup_steps > 0),
        "feedfoward": int(config.timing_pass == "forward"),
        "both_feedfoward_backward": int(config.timing_pass == "forward_backward"),
        "mixed_precision": int(result.precision == "bf16_autocast"),
        "mean_step_time": result.mean_time,
        "std_step_time": result.std_time,
        "precision": result.precision,
        "status": "success",
        "error_message": "",
    }
    append_benchmark_history_row(row)


def save_benchmark_failure(config: TrainConfig, error_message: str) -> None:
    if not config.write_benchmark_csv:
        print("[timing] skipping benchmark csv write for failure because write_benchmark_csv is disabled.")
        return
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_size": normalize_model_size(config.model_size),
        "context_length": config.context_length,
        "with_warmup": int(config.warmup_steps > 0),
        "feedfoward": int(config.timing_pass == "forward"),
        "both_feedfoward_backward": int(config.timing_pass == "forward_backward"),
        "mixed_precision": int(config.mixed_precision),
        "mean_step_time": None,
        "std_step_time": None,
        "precision": "",
        "status": "oom",
        "error_message": error_message,
    }
    append_benchmark_history_row(row)


def benchmark_pass_tag(timing_pass: TimingPass) -> str:
    return "f" if timing_pass == "forward" else "fb"


def precision_tag(mixed_precision: bool) -> str:
    return "mp" if mixed_precision else "fp32"


def stop_memory_history(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.memory._record_memory_history(enabled=None)


def dump_memory_snapshot(config: TrainConfig, device: torch.device) -> None:
    if not config.dump_memory_snapshot:
        return
    if device.type != "cuda":
        print("[memory] skipping snapshot because CUDA is unavailable.")
        return

    snapshot_dir = MEMORY_SNAPSHOT_DIR
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / (
        f"memory_snapshot_{normalize_model_size(config.model_size)}_"
        f"ctx{config.context_length}_{benchmark_pass_tag(config.timing_pass)}_"
        f"{precision_tag(config.mixed_precision)}.pickle"
    )

    torch.cuda.synchronize()
    torch.cuda.memory._dump_snapshot(str(snapshot_path))
    stop_memory_history(device)
    print(f"[memory] dumped snapshot: {snapshot_path}")


def is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        message = str(exc).lower()
        return "out of memory" in message or "cuda error: out of memory" in message
    return False


def benchmark_steps(
    model: BasicsTransformerLM,
    token_ids: np.ndarray,
    config: TrainConfig,
    device: torch.device,
    timing_pass: TimingPass,
    warmup_steps: int,
    measurement_steps: int,
) -> BenchmarkResult:
    mixed_precision_enabled = config.mixed_precision and device.type == "cuda"

    @nvtx.range("run_one_step")
    def run_one_step() -> None:
        inputs, targets = get_batch(
            token_ids,
            config.batch_size,
            config.context_length,
            device=str(device),
        )

        if timing_pass == "forward":
            with torch.no_grad():
                with bf16_autocast_context(device, mixed_precision_enabled):
                    _ = model(inputs)
            return
        # For "forward_backward", include backward but not the optimizer update.
        model.zero_grad(set_to_none=True)
        with bf16_autocast_context(device, mixed_precision_enabled):
            logits = model(inputs)
        loss = cross_entropy(logits.float(), targets) if mixed_precision_enabled else cross_entropy(logits, targets)
        loss.backward()
    
    with nvtx.range("warm_up"):
        for _ in range(warmup_steps):
            run_one_step()
        if device.type == "cuda":
            torch.cuda.synchronize()

    timings: list[float] = []
    for _ in range(measurement_steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = timeit.default_timer()
        run_one_step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(timeit.default_timer() - start)

    mean_time = statistics.mean(timings)
    std_time = statistics.stdev(timings) if len(timings) > 1 else 0.0
    print(f"[timing] pass={timing_pass}")
    print(f"[timing] precision={'bf16_autocast' if mixed_precision_enabled else 'fp32'}")
    print(f"[timing] warmup_steps={warmup_steps}, measurement_steps={measurement_steps}")
    print(f"[timing] mean_step_time={mean_time:.6f}s")
    print(f"[timing] std_step_time={std_time:.6f}s")
    return BenchmarkResult(
        mean_time=mean_time,
        std_time=std_time,
        precision="bf16_autocast" if mixed_precision_enabled else "fp32",
    )


def profile_memory_steps(
    model: BasicsTransformerLM,
    optimizer: AdamW,
    token_ids: np.ndarray,
    config: TrainConfig,
    device: torch.device,
) -> None:
    if not config.dump_memory_snapshot:
        return
    if device.type != "cuda":
        print("[memory] skipping snapshot because CUDA is unavailable.")
        return

    mixed_precision_enabled = config.mixed_precision and device.type == "cuda"
    torch.cuda.memory._record_memory_history(max_entries=1000000)

    try:
        for profile_step in range(MEMORY_PROFILE_STEPS):
            inputs, targets = get_batch(
                token_ids,
                config.batch_size,
                config.context_length,
                device=str(device),
            )

            if config.timing_pass == "forward":
                with torch.no_grad():
                    with bf16_autocast_context(device, mixed_precision_enabled):
                        _ = model(inputs)
                continue

            optimizer.zero_grad()
            with bf16_autocast_context(device, mixed_precision_enabled):
                logits = model(inputs)
            loss = (
                cross_entropy(logits.float(), targets)
                if mixed_precision_enabled
                else cross_entropy(logits, targets)
            )
            loss.backward()
            clip_gradient(model.parameters(), config.max_gradient_norm)

            lr = get_cosine_lr(
                profile_step,
                config.learning_rate,
                config.min_learning_rate,
                config.total_steps // 100,
                config.total_steps,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            optimizer.step()

        dump_memory_snapshot(config, device)
    except Exception:
        stop_memory_history(device)
        raise


def train(config: TrainConfig) -> None:
    device = config.device
    wandb_run = None
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")
    mixed_precision_enabled = config.mixed_precision and device.type == "cuda"
    if config.mixed_precision and not mixed_precision_enabled:
        print("[warn] mixed_precision requested, but CUDA is unavailable; running in full precision.")
    elif mixed_precision_enabled:
        print("[info] Using CUDA BF16 autocast for model forward passes.")

    try:
        transformer_lm = BasicsTransformerLM(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            rope_theta=config.rope_theta,
        ).to(device=device, dtype=config.dtype)

        adamw = AdamW(
            transformer_lm.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
        )

        token_ids: np.ndarray = np.load(config.train_data, mmap_mode="r")
        size_tag = checkpoint_size_tag(config.model_size)

        if config.enable_wandb and not config.run_only_partial:
            if wandb is None:
                raise RuntimeError("wandb is not installed. Install dependencies and try again.")
            wandb_run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.wandb_run_name,
                config={
                    "batch_size": config.batch_size,
                    "context_length": config.context_length,
                    "total_steps": config.total_steps,
                    "learning_rate": config.learning_rate,
                    "min_learning_rate": config.min_learning_rate,
                    "weight_decay": config.weight_decay,
                    "d_model": config.d_model,
                    "d_ff": config.d_ff,
                    "num_layers": config.num_layers,
                    "num_heads": config.num_heads,
                    "model_size": config.model_size,
                },
            )

        # Run benchmark once before the full training loop, with optional warmup.
        benchmark_result = benchmark_steps(
            model=transformer_lm,
            token_ids=token_ids,
            config=config,
            device=device,
            timing_pass=config.timing_pass,
            warmup_steps=config.warmup_steps,
            measurement_steps=config.measurement_steps,
        )
        save_benchmark_result(config, benchmark_result)
        if config.dump_memory_snapshot:
            print(f"[memory] profiling {MEMORY_PROFILE_STEPS} step(s) before {'exiting' if config.run_only_partial else 'training continues'}.")
        profile_memory_steps(transformer_lm, adamw, token_ids, config, device)

        if config.run_only_partial:
            print("[info] run_only_partial is enabled; skipping full training loop.")
            if wandb_run is not None:
                wandb.finish()
            return

        print(f"Total training steps: {config.total_steps}")
        for step in range(config.total_steps):
            step_start = timeit.default_timer()

            inputs, targets = get_batch(
                token_ids,
                config.batch_size,
                config.context_length,
                device=str(device),
            )
            with bf16_autocast_context(device, mixed_precision_enabled):
                logits = transformer_lm(inputs)
            loss = cross_entropy(logits.float(), targets) if mixed_precision_enabled else cross_entropy(logits, targets)

            adamw.zero_grad()
            loss.backward()
            clip_gradient(transformer_lm.parameters(), config.max_gradient_norm)

            lr = get_cosine_lr(
                step,
                config.learning_rate,
                config.min_learning_rate,
                config.total_steps // 100,
                config.total_steps,
            )
            for param_group in adamw.param_groups:
                param_group["lr"] = lr
            adamw.step()

            step_time = timeit.default_timer() - step_start
            if (step + 1) % config.validation_interval == 0:
                print(
                    f"Step {step + 1}/{config.total_steps}, "
                    f"Loss: {loss.item():.4f}, "
                    f"LR: {lr:.6e}, "
                    f"Step Time: {step_time:.4f}s"
                )
                if wandb_run is not None:
                    wandb.log(
                        {
                            "train/loss": float(loss.item()),
                            "train/lr": float(lr),
                            "train/step_time_s": float(step_time),
                        },
                        step=step + 1,
                    )

            if config.checkpoint_dir is not None and (step + 1) % config.checkpoint_interval == 0:
                os.makedirs(config.checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_{size_tag}_step_{step + 1}.pt")
                save_checkpoint(transformer_lm, adamw, step + 1, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path} in step {step + 1}")

        if config.checkpoint_dir is not None:
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            final_checkpoint_path = os.path.join(config.checkpoint_dir, f"final_checkpoint_{size_tag}.pt")
            save_checkpoint(transformer_lm, adamw, config.total_steps, final_checkpoint_path)
            print(f"Saved final checkpoint to {final_checkpoint_path} after training")

        if wandb_run is not None:
            wandb.finish()
    except Exception as exc:
        if wandb_run is not None:
            wandb.finish()
        if not is_oom_error(exc):
            raise
        print(f"[warn] OOM for model_size={config.model_size}, context_length={config.context_length}: {exc}")
        if config.dump_memory_snapshot:
            stop_memory_history(device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[warn] Cleared CUDA cache after OOM.")
        save_benchmark_failure(config, str(exc))
        print("[warn] Skipping this configuration and continuing with the remaining runs.")
        return


tinystories_config = TrainConfig(
    device=DEVICE,
    dtype=DTYPE,
    mixed_precision=args.mixed_precision,
    train_data=TRAIN_DATA,
    checkpoint_dir=CHECKPOINT_DIR,
    vocab_size=VOCAB_SIZE,
    d_model=args.d_model,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    d_ff=args.d_ff,
    model_size=args.model_size,
    rope_theta=ROPE_THETA,
    learning_rate=LEARNING_RATE,
    min_learning_rate=MIN_LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    beta1=BETA1,
    beta2=BETA2,
    eps=EPS,
    max_gradient_norm=MAX_GRAD_NORM,
    batch_size=BATCH_SIZE,
    context_length=args.context_length,
    total_steps=TOTAL_STEPS,
    validation_interval=VALIDATION_INTERVAL,
    checkpoint_interval=CHECKPOINT_INTERVAL,
    run_only_partial=args.run_only_partial,
    timing_pass=timing_pass,
    warmup_steps=WARMUP_STEPS if args.run_with_warmup else 0,
    measurement_steps=MEASUREMENT_STEPS,
    write_benchmark_csv=args.write_benchmark_csv,
    dump_memory_snapshot=args.dump_memory_snapshot,
    enable_wandb=ENABLE_WANDB,
    wandb_project=WANDB_PROJECT,
    wandb_entity=WANDB_ENTITY,
    wandb_run_name=WANDB_RUN_NAME,
)


if __name__ == "__main__":
    train(tinystories_config)
