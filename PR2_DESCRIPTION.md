# [IE] Defense-in-depth: reject zero-dim FC ops in UnrollFullyConnected

## Summary

`UnrollFullyConnected::matchAndRewrite()` does not guard against zero or negative dimensions in `IE::FullyConnectedOp` operands before entering the `splitLeftInput` / `reshapeTo2d` unrolling logic. When per-group INT4 quantization decomposition (e.g. `GroupWisePatternRewriter` with `group_size=128`) produces intermediate FC ops with degenerate shapes, the unrolling pass propagates zero-dim tensors into unrolled sub-FCs. These poisoned sub-FCs then reach downstream passes like `ConvertFCToConv`, triggering a process-killing SIGABRT in `IE::ConvolutionOp` type inference.

This PR adds a defense-in-depth guard in `matchAndRewrite()` that rejects FC ops with zero or negative batch dimension or weight dimensions. The existing `inputChannels < numChunks` guard already catches some zero-dim edge cases on the LHS channel axis; this addition covers the batch dimension and all weight dimensions.

## Context

This is a companion to the `ConvertFCToConv` fix (#265). That fix prevents the crash at the `ConvertFCToConv` layer; this fix prevents degenerate shapes from propagating through `UnrollFullyConnected` in the first place.

The `ConvertFCToConv` fix is the primary crash prevention mechanism. This `UnrollFullyConnected` guard is strictly defense-in-depth — it hardens an earlier pass in the pipeline to reject shapes that should never have reached it.

## Root Cause

The location trail from the original crash (`["fc_decomposed", "matmul_0", "as_convolution"]`) shows that `UnrollFullyConnected` runs between `GroupWisePatternRewriter` and `ConvertFCToConv`. When the decomposition produces a zero-dim FC, `UnrollFullyConnected` blindly enters `splitLeftInput`, which slices the LHS tensor into `numChunks` equal blocks. If the batch dimension is zero, the sliced sub-FCs inherit the zero-dim shape and propagate it downstream.

## Changes

**`src/vpux_compiler/src/dialect/IE/transforms/passes/unroll_fully_connected.cpp`**:
- Added +18 lines after the existing `inputChannels < numChunks` guard (line ~475).
- **Batch dimension check**: `lhsShape[Dim(0)] <= 0` → return `mlir::failure()` with debug log.
- **Weight dimensions check**: Iterates all weight dimensions via `irange(wShape.size())`; any dimension `<= 0` → return `mlir::failure()` with debug log.
- Block-scoped `wShape` variable to avoid shadowing outer variables.
- References `openvinotoolkit/openvino#34450` in comment.

## Testing

- **LIT test added**: `tests/lit/NPU/dialect/IE/passes/unroll_fully_connected_zero_dim_guard.mlir` — constructs a `FakeQuantize→Concat→AffineReshape→Transpose→FullyConnected` pattern with a zero batch dimension (`tensor<0x3072xf32>`). Verifies the guard catches `lhsShape[Dim(0)] <= 0` and preserves the FC unchanged.
- Existing `unroll_fully_connected.mlir` regression test passes on all three architectures (NPU37XX, NPU40XX, NPU50XX) — no behavioral change for valid-shaped FC ops.

## Related

- OpenVINO issue: https://github.com/openvinotoolkit/openvino/issues/34450
- Companion PR: #265
