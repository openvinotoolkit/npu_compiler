# [IE] Guard ConvertFCToConv against zero-sized channel dimensions

## Summary

`ConvertFCToConvPass::safeRunOnFunc()` unconditionally marks `IE::FullyConnectedOp` as illegal via `target.addIllegalOp`, forcing every FC op through conversion to `IE::ConvolutionOp`. The `FullyConnectedOpConverter::matchAndRewrite()` blindly reshapes 2-D FC operands to 4-D `{N, C, 1, 1}` without validating that the channel dimension is positive. When per-group INT4 quantization decomposition (e.g. `group_size=128`) produces `IE::FullyConnectedOp` nodes whose channel dimension is zero, the reshape yields `tensor<Nx0x1x1>`, which causes `IE::ConvolutionOp` type inference to abort with:

```
LLVM ERROR: Failed to infer result type(s).
```

at location `as_convolution`. The SIGABRT kills the process immediately and cannot be caught by the caller.

This PR replaces `addIllegalOp<IE::FullyConnectedOp>()` with `addDynamicallyLegalOp<IE::FullyConnectedOp>()` using a predicate that exempts zero-dim and non-rank-2 FC ops from the illegality constraint. Exempt FC ops are considered "legal" and survive the pass unchanged. Valid FC ops remain "illegal" and are converted to convolutions as before.

A defense-in-depth guard in `matchAndRewrite()` is also retained as a belt-and-suspenders safety net.

## Reproduction

Observed when compiling Qwen3-0.6B (28 layers, INT4 grouped quantization, `openvino-int4-npu` format) for the NPU as a speculative decoding draft model in OpenVINO GenAI heterogeneous mode (`GPU target + NPU draft`):

```python
import openvino_genai as ov_genai

pipeline = ov_genai.LLMPipeline(
    "models/qwen3-14b/openvino-int4-gpu/",
    "GPU",
    draft_model=ov_genai.draft_model("models/qwen3-0.6b/openvino-int4-npu/", "NPU"),
    scheduler_config=scheduler,
)
```

The VPUX compiler aborts during model compilation (before inference) at the `self_attn.v_proj` linear layer in transformer block 0:

```
[ERROR] [vpux-compiler] Got Diagnostic at
  loc(fused<{name = "__module.model.layers.0.self_attn.v_proj/ov_ext::linear/MatMul",
  type = "MatMul"}>["...", "fc_decomposed", "matmul_0", "as_convolution"]) :
  Channels count of input tensor shape and filter shape must be the same: 0 != 8

LLVM ERROR: Failed to infer result type(s):
"IE.Convolution"(...) {} : (tensor<1x0x1x1xf16>, tensor<1x8x1x1xf16>) -> ( ??? )
```

**Environment:** Intel Core Ultra 7 258V (Lunar Lake), NPU driver 32.0.100.4514, OpenVINO 2026.0, Windows 11 Pro.

## Root Cause

The location trail `["fc_decomposed", "matmul_0", "as_convolution"]` reveals a multi-pass interaction:

1. **`GroupWisePatternRewriter`** (decompose multi-ZP quantization) decomposes per-group INT4 FCs into main/correction branches, producing `IE::FullyConnectedOp` with `fc_decomposed` tag.
2. **`UnrollFullyConnected`** unrolls the decomposed FC, producing sub-FCs with `matmul_0` tag — one of which has a zero-dim input.
3. **`ConvertFCToConv`** attempts to convert the zero-dim FC to `IE::ConvolutionOp`, hitting the SIGABRT.

The zero-dim FC is a valid intermediate IR artifact from the multi-pass interaction. `ConvertFCToConv` must tolerate it.

## Changes

**`src/vpux_compiler/src/dialect/IE/transforms/passes/convert_fc_to_conv.cpp`**:
- **`safeRunOnFunc()`**: Replaced `target.addIllegalOp<IE::FullyConnectedOp>()` with `target.addDynamicallyLegalOp<IE::FullyConnectedOp>(predicate)`. The predicate returns `true` (legal/exempt) for FC ops with non-rank-2 shapes or any dimension ≤ 0, and `false` (illegal/must-convert) for valid 2-D FC ops. This follows the existing pattern in `AdjustNCEOpsWithI32InputsPass::safeRunOnFunc()`.
- **`matchAndRewrite()`**: Retained the dimension guard as defense-in-depth (unreachable under normal conditions due to the predicate, but protects against future changes).

**`tests/lit/NPU/dialect/IE/passes/convert_fc_to_conv_zero_dim_guard.mlir`**:
- Changed from negative test (`not vpux-opt`, checking `failed to legalize`) to positive test: the zero-dim `IE.FullyConnected` now **survives** the pass unchanged.
- `CHECK-LABEL: @PreserveZeroDimFC` / `CHECK: IE.FullyConnected` confirms the op is preserved.

## Testing

- LIT test `@PreserveZeroDimFC`: `vpux-opt` exits with code 0, `FileCheck` confirms `IE.FullyConnected` survives in the output IR.
- Existing `convert_fc_to_conv.mlir` positive test is unaffected — valid FC ops are still converted to convolutions.
- No behavioral change for any model that only contains valid-shaped FC ops.

## Related

- OpenVINO issue: https://github.com/openvinotoolkit/openvino/issues/34450
- OpenVINO GenAI duplicate: https://github.com/openvinotoolkit/openvino.genai/issues/3429
