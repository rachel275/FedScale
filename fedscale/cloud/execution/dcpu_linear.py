import torch
import torch.nn as nn


class _DcpuBaseLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        # Base weight is frozen, but is required to propagate dL/dx.
        ctx.save_for_backward(weight)

        return torch.ops.torch_dcpu.linear_cpu(
            x,
            weight,
            bias,
        )

    @staticmethod
    def backward(ctx, grad_output):
        (weight,) = ctx.saved_tensors

        grad_input = None

        if ctx.needs_input_grad[0]:
           grad_input = torch.ops.torch_dcpu.mm_cpu(
            grad_output.contiguous(),
            weight,
           )
        # Frozen base weight and bias: no gradients.
        return grad_input, None, None


class DcpuRoutedLinear(nn.Module):
    """
    Dense CPU Linear whose frozen base weight is stored directly in the
    shared DGEMM frontend arena.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()

        if not isinstance(linear, nn.Linear):
            raise TypeError(
                f"DcpuRoutedLinear expects nn.Linear, got {type(linear)}"
            )

        if linear.weight.requires_grad:
            raise RuntimeError(
                "DcpuRoutedLinear expected frozen PEFT base weight, "
                "but weight.requires_grad=True"
            )

        weight = linear.weight.detach()

        if torch.ops.torch_dcpu.ptr_in_arena(weight):
            arena_weight = weight
        else:
            arena_weight = torch.ops.torch_dcpu.arena_clone_cpu(weight)

        self.weight = nn.Parameter(
            arena_weight,
            requires_grad=False,
        )

        self.bias = linear.bias
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape

        if x.dim() < 2:
            raise RuntimeError(
                f"DcpuRoutedLinear expected >=2D input, got {tuple(x.shape)}"
            )

        x2 = x.reshape(-1, original_shape[-1])


        y2 = _DcpuBaseLinearFunction.apply(
            x2,
            self.weight,
            self.bias,
        )

        return y2.reshape(
            *original_shape[:-1],
            self.out_features,
        )


class _DcpuBaseLinear4bitFunction(torch.autograd.Function):
    """
    Execute a frozen bitsandbytes 4-bit base Linear through DCPU.

    For each invocation we temporarily dequantize the weight to BF16
    and execute the dense GEMM through torch_dcpu. The packed 4-bit
    weight is retained for autograd and rematerialized to BF16 during
    backward to compute dL/dx, avoiding retention of dense base weights
    across the forward/backward boundary.

    No gradient is produced for the frozen base weight.
    """

    @staticmethod
    def forward(ctx, x, packed_weight, bias):
        import bitsandbytes as bnb

        quant_state = packed_weight.quant_state
        if quant_state is None:
            raise RuntimeError(
                "QLoRA Params4bit has no quant_state; "
                "cannot dequantize base weight"
            )

        # Dequantize the packed NF4/FP4 representation.
        weight = bnb.functional.dequantize_4bit(
            packed_weight.data,
            quant_state=quant_state,
        )

        # DCPU distributed GEMM currently consumes dense BF16/FP32.
        weight = weight.to(
            device="cpu",
            dtype=torch.bfloat16,
        ).contiguous()

        # Ensure x uses the same supported dtype.
        x = x.to(
            device="cpu",
            dtype=torch.bfloat16,
        ).contiguous()

        # IMPORTANT: retain only the packed 4-bit weight for backward.
        # Keeping the dense BF16 materialization here would keep every
        # routed layer's dense weight alive until its backward executes,
        # causing very high arena memory consumption for large models.
        ctx.save_for_backward(packed_weight)

        return torch.ops.torch_dcpu.linear_cpu(
            x,
            weight,
            bias,
        )

    @staticmethod
    def backward(ctx, grad_output):
        import bitsandbytes as bnb

        (packed_weight,) = ctx.saved_tensors

        grad_input = None

        if ctx.needs_input_grad[0]:
            quant_state = packed_weight.quant_state
            if quant_state is None:
                raise RuntimeError(
                    "QLoRA Params4bit has no quant_state during backward"
                )

            # Rematerialize the frozen base weight only when needed for
            # dL/dx instead of retaining its dense BF16 representation
            # throughout the complete forward pass.
            weight = bnb.functional.dequantize_4bit(
                packed_weight.data,
                quant_state=quant_state,
            )

            weight = weight.to(
                device="cpu",
                dtype=torch.bfloat16,
            ).contiguous()

            grad_input = torch.mm(
                grad_output.to(
                    dtype=weight.dtype,
                    device="cpu",
                ).contiguous(),
                weight,
            )

        # packed_weight and frozen bias receive no gradients.
        return grad_input, None, None


class DcpuRoutedLinear4bit(nn.Module):
    """
    DCPU wrapper for a frozen bitsandbytes Linear4bit base layer.

    The packed Params4bit remains packed between invocations. Only the
    currently executing layer is materialized as a dense BF16 weight.
    """

    def __init__(self, linear):
        super().__init__()

        import bitsandbytes as bnb

        if not isinstance(linear, bnb.nn.Linear4bit):
            raise TypeError(
                "DcpuRoutedLinear4bit expects bitsandbytes.nn.Linear4bit, "
                f"got {type(linear)}"
            )

        self.weight = linear.weight
        self.bias = linear.bias

        self.in_features = linear.in_features
        self.out_features = linear.out_features

        if not hasattr(self.weight, "quant_state"):
            raise RuntimeError(
                "Linear4bit weight does not contain quant_state"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape

        if x.dim() < 2:
            raise RuntimeError(
                f"DcpuRoutedLinear4bit expected >=2D input, "
                f"got {tuple(x.shape)}"
            )

        x2 = x.reshape(-1, original_shape[-1])

        y2 = _DcpuBaseLinear4bitFunction.apply(
            x2,
            self.weight,
            self.bias,
        )

        return y2.reshape(
            *original_shape[:-1],
            self.out_features,
        )

def route_peft_base_linears(model):
    """
    Route frozen model Linear layers through DCPU.

    PEFT-wrapped base layers:
        nn.Linear              -> DcpuRoutedLinear
        bitsandbytes Linear4bit -> DcpuRoutedLinear4bit

    Remaining raw QLoRA Linear4bit layers:
        bitsandbytes Linear4bit -> DcpuRoutedLinear4bit

    LoRA A/B modules remain ordinary PyTorch operations.
    """

    import bitsandbytes as bnb

    routed_dense = 0
    routed_4bit = 0
    arena_bytes = 0

    # ---------------------------------------------------------
    # Pass 1: replace base layers inside PEFT LoRA modules.
    # ---------------------------------------------------------
    modules = list(model.modules())

    for module in modules:
        if not hasattr(module, "base_layer"):
            continue

        base = module.base_layer

        if isinstance(base, bnb.nn.Linear4bit):
            module.base_layer = DcpuRoutedLinear4bit(base)
            routed_4bit += 1

            print(
                f"[dcpu] routed PEFT 4-bit base linear "
                f"{routed_4bit}: "
                f"{base.in_features} -> {base.out_features}"
            )

            continue

        if isinstance(base, nn.Linear):
            module.base_layer = DcpuRoutedLinear(base)
            routed_dense += 1

            arena_bytes += (
                base.weight.numel()
                * base.weight.element_size()
            )

            print(
                f"[dcpu] routed dense base linear "
                f"{routed_dense}: "
                f"{base.in_features} -> {base.out_features}, "
                f"arena total="
                f"{arena_bytes / (1024**3):.3f} GiB"
            )
        # ---------------------------------------------------------
    # Pass 2: route remaining large raw QLoRA projections.
    #
    # Q and V are already routed above because they are PEFT
    # LoRA targets. Route the remaining transformer projections
    # through the same BF16 DCPU path.
    # ---------------------------------------------------------
    raw_dcpu_targets = {
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }

    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if (
                name in raw_dcpu_targets
                and isinstance(child, bnb.nn.Linear4bit)
            ):
                setattr(
                    parent,
                    name,
                    DcpuRoutedLinear4bit(child),
                )

                routed_4bit += 1

                print(
                    f"[dcpu] routed raw 4-bit {name} "
                    f"{routed_4bit}: "
                    f"{child.in_features} -> "
                    f"{child.out_features}"
                )


    print(
        f"[dcpu] routing complete: "
        f"dense={routed_dense}, "
        f"4bit={routed_4bit}, "
        f"persistent dense arena weights="
        f"{arena_bytes / (1024**3):.3f} GiB"
    )

    return routed_dense + routed_4bit


