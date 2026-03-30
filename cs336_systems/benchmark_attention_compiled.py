from __future__ import annotations

import argparse
import gc
import statistics
import timeit

import torch

from cs336_basics.model import scaled_dot_product_attention


@torch.compile
def compiled_scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return scaled_dot_product_attention(Q=Q, K=K, V=V, mask=mask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark compiled scaled dot-product attention.")
    parser.add_argument("--batch_size", type=int, default=8, help="Fixed batch size for the benchmark.")
    parser.add_argument("--d_model", type=int, required=True, help="Embedding dimension without a head dimension.")
    parser.add_argument("--context_length", type=int, required=True, help="Sequence length.")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Warmup iterations before timing.")
    parser.add_argument("--forward_steps", type=int, default=100, help="Number of timed forward passes.")
    parser.add_argument("--backward_steps", type=int, default=100, help="Number of timed backward passes.")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        return "out of memory" in str(exc).lower()
    return False


def memory_allocated_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.memory_allocated(device) / (1024**2)


def peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def make_mask(context_length: int, device: torch.device) -> torch.Tensor:
    return torch.tril(torch.ones((1, context_length, context_length), dtype=torch.bool, device=device))


def benchmark_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    warmup_steps: int,
    forward_steps: int,
    device: torch.device,
) -> tuple[float, float]:
    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = compiled_scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
            synchronize(device)

        timings: list[float] = []
        for _ in range(forward_steps):
            synchronize(device)
            start = timeit.default_timer()
            _ = compiled_scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
            synchronize(device)
            timings.append(timeit.default_timer() - start)

    mean_time = statistics.mean(timings)
    std_time = statistics.stdev(timings) if len(timings) > 1 else 0.0
    return mean_time, std_time


def benchmark_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    warmup_steps: int,
    backward_steps: int,
    device: torch.device,
) -> tuple[float, float, float]:
    def zero_grads() -> None:
        q.grad = None
        k.grad = None
        v.grad = None

    for _ in range(warmup_steps):
        zero_grads()
        output = compiled_scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
        loss = output.sum()
        synchronize(device)
        loss.backward()
        synchronize(device)

    timings: list[float] = []
    pre_backward_memory_mb: list[float] = []
    for _ in range(backward_steps):
        zero_grads()
        output = compiled_scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
        loss = output.sum()
        synchronize(device)
        pre_backward_memory_mb.append(memory_allocated_mb(device))
        start = timeit.default_timer()
        loss.backward()
        synchronize(device)
        timings.append(timeit.default_timer() - start)

    mean_time = statistics.mean(timings)
    std_time = statistics.stdev(timings) if len(timings) > 1 else 0.0
    mean_pre_backward_memory = statistics.mean(pre_backward_memory_mb) if pre_backward_memory_mb else 0.0
    return mean_time, std_time, mean_pre_backward_memory


def main() -> int:
    args = parse_args()
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    mask = None
    q_forward = None
    k_forward = None
    v_forward = None
    q_backward = None
    k_backward = None
    v_backward = None

    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        mask = make_mask(args.context_length, device)

        q_forward = torch.randn(args.batch_size, args.context_length, args.d_model, device=device, dtype=dtype)
        k_forward = torch.randn_like(q_forward)
        v_forward = torch.randn_like(q_forward)

        forward_mean, forward_std = benchmark_forward(
            q=q_forward,
            k=k_forward,
            v=v_forward,
            mask=mask,
            warmup_steps=args.warmup_steps,
            forward_steps=args.forward_steps,
            device=device,
        )

        del q_forward, k_forward, v_forward
        q_forward = k_forward = v_forward = None
        cleanup(device)

        q_backward = torch.randn(
            args.batch_size,
            args.context_length,
            args.d_model,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        k_backward = torch.randn_like(q_backward, requires_grad=True)
        v_backward = torch.randn_like(q_backward, requires_grad=True)

        backward_mean, backward_std, mean_pre_backward_memory = benchmark_backward(
            q=q_backward,
            k=k_backward,
            v=v_backward,
            mask=mask,
            warmup_steps=args.warmup_steps,
            backward_steps=args.backward_steps,
            device=device,
        )

        print(
            "status=success "
            f"device={device.type} "
            f"batch_size={args.batch_size} "
            f"context_length={args.context_length} "
            f"d_model={args.d_model} "
            f"forward_mean_s={forward_mean:.6f} "
            f"forward_std_s={forward_std:.6f} "
            f"backward_mean_s={backward_mean:.6f} "
            f"backward_std_s={backward_std:.6f} "
            f"memory_before_backward_mb={mean_pre_backward_memory:.2f} "
            f"peak_memory_mb={peak_memory_mb(device):.2f}"
        )
        return 0
    except Exception as exc:
        if not is_oom_error(exc):
            raise
        print(
            "status=oom "
            f"device={device.type} "
            f"batch_size={args.batch_size} "
            f"context_length={args.context_length} "
            f"d_model={args.d_model} "
            f'error="{str(exc).replace(chr(10), " ")}"'
        )
        return 0
    finally:
        del mask, q_forward, k_forward, v_forward, q_backward, k_backward, v_backward
        cleanup(device)


if __name__ == "__main__":
    raise SystemExit(main())
