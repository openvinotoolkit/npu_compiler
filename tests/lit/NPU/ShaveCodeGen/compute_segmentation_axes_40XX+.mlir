//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --compute-segmentation-axes %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// CHECK: module @BroadcastH
// CHECK: func.func nested @generated_0(
// CHECK-SAME: kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @generated_info_0,
// CHECK-SAME: tilingAxes = [1, 2, 3],
// CHECK-SAME: numSlicedInputs = 2 : i64,
// CHECK-SAME: numTensorInputs = 2 : i64,
// CHECK-SAME: numAuxBuffers = 0 : i64,
// CHECK-SAME: inputSegmentationAxes = [array<i64: 1, 1>, array<i64: 2, -1>, array<i64: 3, -1>]>

module @BroadcastH {
  module @VPU.SW {
    func.func @generated_info_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> (index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index) {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      return
        %c1, %arg0, %arg1, %arg2,
          %c0, %arg3, %arg4, %arg5,
        %c1, %arg0, %c1, %c1,
          %c0, %arg3, %c0, %c0 : index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index
    }
    func.func nested @generated_0(%arg0: tensor<1x?x?x?xf16>, %arg1: tensor<1x?x1x1xf16>, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index) -> tensor<1x?x?x?xf16> attributes {
      kernelInfo = #VPU.KernelInfo<
        tilingInfoFunc = @generated_info_0,
        tilingAxes = [1, 2, 3],
        numSlicedInputs = 2 : i64,
        numTensorInputs = 2 : i64,
        numAuxBuffers = 0 : i64,
        inputSegmentationAxes = []>
    }
  }
}

// -----

// Both inputs are fully segmented on every tiling axis.
// Every axis should be valid; each input segments on the same dim as the axis.

// CHECK: module @AllSegmented
// CHECK: func.func nested @all_segmented(
// CHECK-SAME: kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @all_segmented_info,
// CHECK-SAME: tilingAxes = [1, 2, 3],
// CHECK-SAME: numSlicedInputs = 2 : i64,
// CHECK-SAME: numTensorInputs = 2 : i64,
// CHECK-SAME: numAuxBuffers = 0 : i64,
// CHECK-SAME: inputSegmentationAxes = [array<i64: 1, 1>, array<i64: 2, 2>, array<i64: 3, 3>]>
module @AllSegmented {
  module @VPU.SW {
    func.func @all_segmented_info(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> (index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index) {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      return
        %c1, %arg0, %arg1, %arg2,
          %c0, %arg3, %arg4, %arg5,
        %c1, %arg0, %arg1, %arg2,
          %c0, %arg3, %arg4, %arg5 : index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index
    }
    func.func nested @all_segmented(%arg0: tensor<1x?x?x?xf16>, %arg1: tensor<1x?x?x?xf16>, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index) -> tensor<1x?x?x?xf16> attributes {
      kernelInfo = #VPU.KernelInfo<
        tilingInfoFunc = @all_segmented_info,
        tilingAxes = [1, 2, 3],
        numSlicedInputs = 2 : i64,
        numTensorInputs = 2 : i64,
        numAuxBuffers = 0 : i64,
        inputSegmentationAxes = []>
    }
  }
}

// -----

// CHECK: module @SegmentedOnDifferentDim
// CHECK: func.func nested @diff_dim(
// CHECK-SAME: kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @diff_dim_info,
// CHECK-SAME: tilingAxes = [1, 2],
// CHECK-SAME: numSlicedInputs = 2 : i64,
// CHECK-SAME: numTensorInputs = 2 : i64,
// CHECK-SAME: numAuxBuffers = 0 : i64,
// CHECK-SAME: inputSegmentationAxes = [array<i64: 1, 1>, array<i64: 2, 3>]>
module @SegmentedOnDifferentDim {
  module @VPU.SW {
    func.func @diff_dim_info(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index) {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      return
        %c1, %arg0, %arg1, %c1,
          %c0, %arg2, %arg3, %c0,
        %c1, %arg0, %c1, %arg1,
          %c0, %arg2, %c0, %arg3 : index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index
    }
    func.func nested @diff_dim(%arg0: tensor<1x?x?x1xf16>, %arg1: tensor<1x?x1x?xf16>, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> tensor<1x?x?x?xf16> attributes {
      kernelInfo = #VPU.KernelInfo<
        tilingInfoFunc = @diff_dim_info,
        tilingAxes = [1, 2],
        numSlicedInputs = 2 : i64,
        numTensorInputs = 2 : i64,
        numAuxBuffers = 0 : i64,
        inputSegmentationAxes = []>
    }
  }
}

// -----

// CHECK: module @PartialAxesValid
// CHECK: func.func nested @partial(
// CHECK-SAME: kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @partial_info,
// CHECK-SAME: tilingAxes = [1, 2],
// CHECK-SAME: numSlicedInputs = 1 : i64,
// CHECK-SAME: numTensorInputs = 1 : i64,
// CHECK-SAME: numAuxBuffers = 0 : i64,
// CHECK-SAME: inputSegmentationAxes = [array<i64: 1>, array<i64: -2>]>
module @PartialAxesValid {
  module @VPU.SW {
    func.func @partial_info(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index, index, index, index, index, index) {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %derived = arith.muli %arg1, %c2 : index
      return
        %c1, %arg0, %arg1, %derived,
          %c0, %arg2, %arg3, %c0 : index, index, index, index, index, index, index, index
    }
    func.func nested @partial(%arg0: tensor<1x?x?x?xf16>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> tensor<1x?x?x?xf16> attributes {
      kernelInfo = #VPU.KernelInfo<
        tilingInfoFunc = @partial_info,
        tilingAxes = [1, 2],
        numSlicedInputs = 1 : i64,
        numTensorInputs = 1 : i64,
        numAuxBuffers = 0 : i64,
        inputSegmentationAxes = []>
    }
  }
}

// -----

// CHECK: module @NonSlicedTensorInput
// CHECK: func.func nested @non_sliced(
// CHECK-SAME: kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @non_sliced_info,
// CHECK-SAME: tilingAxes = [1, 2],
// CHECK-SAME: numSlicedInputs = 1 : i64,
// CHECK-SAME: numTensorInputs = 2 : i64,
// CHECK-SAME: numAuxBuffers = 0 : i64,
// CHECK-SAME: inputSegmentationAxes = [array<i64: 1, -1>, array<i64: 2, -1>]>
module @NonSlicedTensorInput {
  module @VPU.SW {
    func.func @non_sliced_info(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index, index, index, index, index, index) {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      return %c1, %arg0, %arg1, %c1, %c0, %arg2, %arg3, %c0 : index, index, index, index, index, index, index, index
    }
    func.func nested @non_sliced(%arg0: tensor<1x?x?x1xf16>, %arg1: tensor<1x?x?x?xf16>, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> tensor<1x?x?x?xf16> attributes {
      kernelInfo = #VPU.KernelInfo<
        tilingInfoFunc = @non_sliced_info,
        tilingAxes = [1, 2],
        numSlicedInputs = 1 : i64,
        numTensorInputs = 2 : i64,
        numAuxBuffers = 0 : i64,
        inputSegmentationAxes = []>
    }
  }
}

// -----

// The input cannot be classified on any tiling axis.

// CHECK: module @NoValidAxes
// CHECK: func.func nested @no_valid(
// CHECK-SAME: kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @no_valid_info,
// CHECK-SAME: tilingAxes = [1, 2],
// CHECK-SAME: numSlicedInputs = 1 : i64,
// CHECK-SAME: numTensorInputs = 1 : i64,
// CHECK-SAME: numAuxBuffers = 0 : i64,
// CHECK-SAME: inputSegmentationAxes = [array<i64: -2>, array<i64: -2>]>
module @NoValidAxes {
  module @VPU.SW {
    func.func @no_valid_info(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index, index, index, index) {
      %c2 = arith.constant 2 : index
      %derived = arith.muli %arg1, %c2 : index
      return
        %arg0, %derived, %arg0,
          %arg2, %arg3, %arg2 : index, index, index, index, index, index
    }
    func.func nested @no_valid(%arg0: tensor<1x?x?xf16>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> tensor<1x?x?xf16> attributes {
      kernelInfo = #VPU.KernelInfo<
        tilingInfoFunc = @no_valid_info,
        tilingAxes = [1, 2],
        numSlicedInputs = 1 : i64,
        numTensorInputs = 1 : i64,
        numAuxBuffers = 0 : i64,
        inputSegmentationAxes = []>
    }
  }
}

// -----

// CHECK: module @TwoDimsMatch
// CHECK: func.func nested @two_dims(
// CHECK-SAME: kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @two_dims_info,
// CHECK-SAME: tilingAxes = [1, 2],
// CHECK-SAME: numSlicedInputs = 1 : i64,
// CHECK-SAME: numTensorInputs = 1 : i64,
// CHECK-SAME: numAuxBuffers = 0 : i64,
// CHECK-SAME: inputSegmentationAxes = [array<i64: -2>, array<i64: 2>]>
module @TwoDimsMatch {
  module @VPU.SW {
    func.func @two_dims_info(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index, index, index, index, index, index) {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      return
        %c1, %arg0, %arg1, %arg0,
          %c0, %arg2, %arg3, %arg2 : index, index, index, index, index, index, index, index
    }
    func.func nested @two_dims(%arg0: tensor<1x?x?x?xf16>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> tensor<1x?x?x?xf16> attributes {
      kernelInfo = #VPU.KernelInfo<
        tilingInfoFunc = @two_dims_info,
        tilingAxes = [1, 2],
        numSlicedInputs = 1 : i64,
        numTensorInputs = 1 : i64,
        numAuxBuffers = 0 : i64,
        inputSegmentationAxes = []>
    }
  }
}

// -----

// CHECK: module @IndirectDependency
// CHECK: func.func nested @indirect_dep(
// CHECK-SAME: kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @indirect_dep_info,
// CHECK-SAME: tilingAxes = [1, 2],
// CHECK-SAME: numSlicedInputs = 1 : i64,
// CHECK-SAME: numTensorInputs = 1 : i64,
// CHECK-SAME: numAuxBuffers = 0 : i64,
// CHECK-SAME: inputSegmentationAxes = [array<i64: 1>, array<i64: -2>]>
module @IndirectDependency {
  module @VPU.SW {
    func.func @indirect_dep_info(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index, index, index, index, index, index) {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %derived = arith.addi %arg3, %c1 : index
      return
        %c1, %arg0, %arg1, %derived,
          %c0, %arg2, %arg3, %c0 : index, index, index, index, index, index, index, index
    }
    func.func nested @indirect_dep(%arg0: tensor<1x?x?x?xf16>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> tensor<1x?x?x?xf16> attributes {
      kernelInfo = #VPU.KernelInfo<
        tilingInfoFunc = @indirect_dep_info,
        tilingAxes = [1, 2],
        numSlicedInputs = 1 : i64,
        numTensorInputs = 1 : i64,
        numAuxBuffers = 0 : i64,
        inputSegmentationAxes = []>
    }
  }
}

// -----

// CHECK: module @SingleTilingAxis
// CHECK: func.func nested @single_axis(
// CHECK-SAME: kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @single_axis_info,
// CHECK-SAME: tilingAxes = [2],
// CHECK-SAME: numSlicedInputs = 1 : i64,
// CHECK-SAME: numTensorInputs = 1 : i64,
// CHECK-SAME: numAuxBuffers = 0 : i64,
// CHECK-SAME: inputSegmentationAxes = [array<i64: 2>]>
module @SingleTilingAxis {
  module @VPU.SW {
    func.func @single_axis_info(%arg0: index, %arg1: index) -> (index, index, index, index, index, index, index, index) {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      return
        %c1, %c1, %arg0, %c1,
        %c0, %c0, %arg1, %c0 : index, index, index, index, index, index, index, index
    }
    func.func nested @single_axis(%arg0: tensor<1x1x?x1xf16>, %arg1: index, %arg2: index) -> tensor<1x?x?x?xf16> attributes {
      kernelInfo = #VPU.KernelInfo<
        tilingInfoFunc = @single_axis_info,
        tilingAxes = [2],
        numSlicedInputs = 1 : i64,
        numTensorInputs = 1 : i64,
        numAuxBuffers = 0 : i64,
        inputSegmentationAxes = []>
    }
  }
}

// -----

// CHECK: module @AuxBuffer
// CHECK: func.func nested @aux_buf(
// CHECK-SAME: kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @aux_buf_info,
// CHECK-SAME: tilingAxes = [1, 2],
// CHECK-SAME: numSlicedInputs = 1 : i64,
// CHECK-SAME: numTensorInputs = 1 : i64,
// CHECK-SAME: numAuxBuffers = 1 : i64,
// CHECK-SAME: inputSegmentationAxes = [array<i64: 1, -1>, array<i64: 2, 2>]>
module @AuxBuffer {
  module @VPU.SW {
    func.func @aux_buf_info(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index) {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      return
        %c1, %arg0, %arg1, %c1,
          %c0, %arg2, %arg3, %c0,
        %c1, %c1, %arg1, %c1,
          %c0, %c0, %arg3, %c0 : index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index
    }
    func.func nested @aux_buf(%arg0: tensor<1x?x?x1xf16>, %arg1: tensor<1x1x?x1xf16>, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> tensor<1x?x?x?xf16> attributes {
      kernelInfo = #VPU.KernelInfo<
        tilingInfoFunc = @aux_buf_info,
        tilingAxes = [1, 2],
        numSlicedInputs = 1 : i64,
        numTensorInputs = 1 : i64,
        numAuxBuffers = 1 : i64,
        inputSegmentationAxes = []>
    }
  }
}

// -----

// CHECK: module @NonSlicedInputWithAuxBuffer
// CHECK: func.func nested @non_sliced_aux(
// CHECK-SAME: kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @non_sliced_aux_info,
// CHECK-SAME: tilingAxes = [2],
// CHECK-SAME: numSlicedInputs = 1 : i64,
// CHECK-SAME: numTensorInputs = 2 : i64,
// CHECK-SAME: numAuxBuffers = 1 : i64,
// CHECK-SAME: inputSegmentationAxes = [array<i64: 2, -1, -1>]>
module @NonSlicedInputWithAuxBuffer {
  module @VPU.SW {
    func.func @non_sliced_aux_info(%arg0: index, %arg1: index) -> (index, index, index, index, index, index, index, index, index, index, index, index) {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      return
        %c1, %c1, %arg0, %c1,
          %c0, %c0, %arg1, %c0,
        %c1, %c1,
          %c0, %c0 : index, index, index, index, index, index, index, index, index, index, index, index
    }
    func.func nested @non_sliced_aux(%arg0: tensor<1x1x?x1xf16>, %arg1: tensor<1x1x?x?xf16>, %arg2: tensor<1x1xf16>, %arg3: index, %arg4: index) -> tensor<1x1x?x?xf16> attributes {
      kernelInfo = #VPU.KernelInfo<
        tilingInfoFunc = @non_sliced_aux_info,
        tilingAxes = [2],
        numSlicedInputs = 1 : i64,
        numTensorInputs = 2 : i64,
        numAuxBuffers = 1 : i64,
        inputSegmentationAxes = []>
    }
  }
}

// -----

// CHECK: module @AuxBufferUnknownOnAxis
// CHECK: func.func nested @aux_unknown(
// CHECK-SAME: kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @aux_unknown_info,
// CHECK-SAME: tilingAxes = [1, 2],
// CHECK-SAME: numSlicedInputs = 1 : i64,
// CHECK-SAME: numTensorInputs = 1 : i64,
// CHECK-SAME: numAuxBuffers = 1 : i64,
// CHECK-SAME: inputSegmentationAxes = [array<i64: 1, -1>, array<i64: 2, -2>]>
module @AuxBufferUnknownOnAxis {
  module @VPU.SW {
    func.func @aux_unknown_info(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index) {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %derived = arith.muli %arg1, %c2 : index
      return
        %c1, %arg0, %arg1, %c1,
          %c0, %arg2, %arg3, %c0,
        %c1, %c1, %arg1, %derived,
          %c0, %c0, %arg3, %c0 : index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index
    }
    func.func nested @aux_unknown(%arg0: tensor<1x?x?x1xf16>, %arg1: tensor<1x1x?x?xf16>, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> tensor<1x?x?x?xf16> attributes {
      kernelInfo = #VPU.KernelInfo<
        tilingInfoFunc = @aux_unknown_info,
        tilingAxes = [1, 2],
        numSlicedInputs = 1 : i64,
        numTensorInputs = 1 : i64,
        numAuxBuffers = 1 : i64,
        inputSegmentationAxes = []>
    }
  }
}

// -----

// CHECK: module @DeclarationOnlyTilingInfoFunc
// CHECK: func.func nested @decl_only(
// CHECK-SAME: kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @decl_only_info,
// CHECK-SAME: tilingAxes = [2],
// CHECK-SAME: numSlicedInputs = 1 : i64,
// CHECK-SAME: numTensorInputs = 1 : i64,
// CHECK-SAME: numAuxBuffers = 0 : i64,
// CHECK-SAME: inputSegmentationAxes = []>
module @DeclarationOnlyTilingInfoFunc {
  module @VPU.SW {
    func.func private @decl_only_info(index, index) -> (index, index, index, index, index, index, index, index)
    func.func nested @decl_only(%arg0: tensor<1x1x?x1xf16>, %arg1: index, %arg2: index) -> tensor<1x?x?x?xf16> attributes {
      kernelInfo = #VPU.KernelInfo<
        tilingInfoFunc = @decl_only_info,
        tilingAxes = [2],
        numSlicedInputs = 1 : i64,
        numTensorInputs = 1 : i64,
        numAuxBuffers = 0 : i64,
        inputSegmentationAxes = []>
    }
  }
}
