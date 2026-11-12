//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --deduplicate-sw-kernels %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010 || platform-NPU6010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Two identical tiled kernels. Deduplicating the slice info functions makes the
// kernels equivalent.

module @DedupIdenticalTiledKernels {
  module @VPU.SW {
    func.func @generated_info_1(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> (index, index, index, index, index, index, index, index) {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      return %c1, %arg0, %arg1, %arg2, %c0, %arg3, %arg4, %arg5 : index, index, index, index, index, index, index, index
    }
    func.func @generated_1(%arg0: tensor<1x?x?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index) -> tensor<1x?x?x?xf32> attributes {kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @generated_info_1, tilingAxes = [1, 2, 3], numSlicedInputs = 1 : i64, numTensorInputs = 1 : i64, numAuxBuffers = 0 : i64, inputSegmentationAxes = []>} {
      %0 = tensor.empty(%arg1, %arg2, %arg3) : tensor<1x?x?x?xf32>
      %1 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x?x?x?xf32>) outs(%0 : tensor<1x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %2 = math.exp %in fastmath<afn> : f32
        linalg.yield %2 : f32
      } -> tensor<1x?x?x?xf32>
      return %1 : tensor<1x?x?x?xf32>
    }
    func.func @generated_info_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> (index, index, index, index, index, index, index, index) {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      return %c1, %arg0, %arg1, %arg2, %c0, %arg3, %arg4, %arg5 : index, index, index, index, index, index, index, index
    }
    func.func @generated_0(%arg0: tensor<1x?x?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index) -> tensor<1x?x?x?xf32> attributes {kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @generated_info_0, tilingAxes = [1, 2, 3], numSlicedInputs = 1 : i64, numTensorInputs = 1 : i64, numAuxBuffers = 0 : i64, inputSegmentationAxes = []>} {
      %0 = tensor.empty(%arg1, %arg2, %arg3) : tensor<1x?x?x?xf32>
      %1 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x?x?x?xf32>) outs(%0 : tensor<1x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %2 = math.exp %in fastmath<afn> : f32
        linalg.yield %2 : f32
      } -> tensor<1x?x?x?xf32>
      return %1 : tensor<1x?x?x?xf32>
    }
  }

  func.func @main(%arg0: tensor<1x16x4000x200xf32>, %arg1: tensor<1x16x4000x200xf32>) -> (tensor<1x16x4000x200xf32>, tensor<1x16x4000x200xf32>) {
    %0 = VPU.GenericSwLayer(%arg0 : tensor<1x16x4000x200xf32>) @VPU.SW::@generated_1 tiling(sizes = [16, 4000, 200], offsets = [0, 0, 0]) -> tensor<1x16x4000x200xf32>
    %1 = VPU.GenericSwLayer(%arg1 : tensor<1x16x4000x200xf32>) @VPU.SW::@generated_0 tiling(sizes = [16, 4000, 200], offsets = [0, 0, 0]) -> tensor<1x16x4000x200xf32>
    return %0, %1 : tensor<1x16x4000x200xf32>, tensor<1x16x4000x200xf32>
  }
}

// CHECK-LABEL: module @DedupIdenticalTiledKernels
// CHECK:   module @VPU.SW
// CHECK:     func.func @generated_info_1
// CHECK:     func.func @generated_1
// CHECK-SAME:  kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @generated_info_1
// CHECK-NOT: func.func @generated_info_0
// CHECK-NOT: func.func @generated_0

// CHECK:   func.func @main([[ARG0:%.+]]: tensor<1x16x4000x200xf32>, [[ARG1:%.+]]: tensor<1x16x4000x200xf32>)
// CHECK:     VPU.GenericSwLayer([[ARG0]] : tensor<1x16x4000x200xf32>) @VPU.SW::@generated_1
// CHECK:     VPU.GenericSwLayer([[ARG1]] : tensor<1x16x4000x200xf32>) @VPU.SW::@generated_1

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Same as above, but each kernel is placed before the info function it references.
// A single sweep would not perform the full deduplication.

module @DedupTiledKernelsDefinedBeforeInfoFuncs {
  module @VPU.SW {
    func.func @generated_1(%arg0: tensor<1x?x?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index) -> tensor<1x?x?x?xf32> attributes {kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @generated_info_1, tilingAxes = [1, 2, 3], numSlicedInputs = 1 : i64, numTensorInputs = 1 : i64, numAuxBuffers = 0 : i64, inputSegmentationAxes = []>} {
      %0 = tensor.empty(%arg1, %arg2, %arg3) : tensor<1x?x?x?xf32>
      %1 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x?x?x?xf32>) outs(%0 : tensor<1x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %2 = math.exp %in fastmath<afn> : f32
        linalg.yield %2 : f32
      } -> tensor<1x?x?x?xf32>
      return %1 : tensor<1x?x?x?xf32>
    }
    func.func @generated_info_1(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> (index, index, index, index, index, index, index, index) {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      return %c1, %arg0, %arg1, %arg2, %c0, %arg3, %arg4, %arg5 : index, index, index, index, index, index, index, index
    }
    func.func @generated_0(%arg0: tensor<1x?x?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index) -> tensor<1x?x?x?xf32> attributes {kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @generated_info_0, tilingAxes = [1, 2, 3], numSlicedInputs = 1 : i64, numTensorInputs = 1 : i64, numAuxBuffers = 0 : i64, inputSegmentationAxes = []>} {
      %0 = tensor.empty(%arg1, %arg2, %arg3) : tensor<1x?x?x?xf32>
      %1 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x?x?x?xf32>) outs(%0 : tensor<1x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %2 = math.exp %in fastmath<afn> : f32
        linalg.yield %2 : f32
      } -> tensor<1x?x?x?xf32>
      return %1 : tensor<1x?x?x?xf32>
    }
    func.func @generated_info_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> (index, index, index, index, index, index, index, index) {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      return %c1, %arg0, %arg1, %arg2, %c0, %arg3, %arg4, %arg5 : index, index, index, index, index, index, index, index
    }
  }

  func.func @main(%arg0: tensor<1x16x4000x200xf32>, %arg1: tensor<1x16x4000x200xf32>) -> (tensor<1x16x4000x200xf32>, tensor<1x16x4000x200xf32>) {
    %0 = VPU.GenericSwLayer(%arg0 : tensor<1x16x4000x200xf32>) @VPU.SW::@generated_1 tiling(sizes = [16, 4000, 200], offsets = [0, 0, 0]) -> tensor<1x16x4000x200xf32>
    %1 = VPU.GenericSwLayer(%arg1 : tensor<1x16x4000x200xf32>) @VPU.SW::@generated_0 tiling(sizes = [16, 4000, 200], offsets = [0, 0, 0]) -> tensor<1x16x4000x200xf32>
    return %0, %1 : tensor<1x16x4000x200xf32>, tensor<1x16x4000x200xf32>
  }
}

// CHECK-LABEL: module @DedupTiledKernelsDefinedBeforeInfoFuncs
// CHECK:   module @VPU.SW
// CHECK:     func.func @generated_1
// CHECK-SAME:  kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @generated_info_1
// CHECK:     func.func @generated_info_1
// CHECK-NOT: func.func @generated_0
// CHECK-NOT: func.func @generated_info_0

// CHECK:   func.func @main([[ARG0:%.+]]: tensor<1x16x4000x200xf32>, [[ARG1:%.+]]: tensor<1x16x4000x200xf32>)
// CHECK:     VPU.GenericSwLayer([[ARG0]] : tensor<1x16x4000x200xf32>) @VPU.SW::@generated_1
// CHECK:     VPU.GenericSwLayer([[ARG1]] : tensor<1x16x4000x200xf32>) @VPU.SW::@generated_1

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// The two kernels compute different functions, so only their identical info functions may
// be merged.

module @KeepDistinctTiledKernels {
  module @VPU.SW {
    func.func @generated_info_1(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> (index, index, index, index, index, index, index, index) {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      return %c1, %arg0, %arg1, %arg2, %c0, %arg3, %arg4, %arg5 : index, index, index, index, index, index, index, index
    }
    func.func @generated_1(%arg0: tensor<1x?x?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index) -> tensor<1x?x?x?xf32> attributes {kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @generated_info_1, tilingAxes = [1, 2, 3], numSlicedInputs = 1 : i64, numTensorInputs = 1 : i64, numAuxBuffers = 0 : i64, inputSegmentationAxes = []>} {
      %0 = tensor.empty(%arg1, %arg2, %arg3) : tensor<1x?x?x?xf32>
      %1 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x?x?x?xf32>) outs(%0 : tensor<1x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %2 = math.exp %in fastmath<afn> : f32
        linalg.yield %2 : f32
      } -> tensor<1x?x?x?xf32>
      return %1 : tensor<1x?x?x?xf32>
    }
    func.func @generated_info_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> (index, index, index, index, index, index, index, index) {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      return %c1, %arg0, %arg1, %arg2, %c0, %arg3, %arg4, %arg5 : index, index, index, index, index, index, index, index
    }
    func.func @generated_0(%arg0: tensor<1x?x?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index) -> tensor<1x?x?x?xf32> attributes {kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @generated_info_0, tilingAxes = [1, 2, 3], numSlicedInputs = 1 : i64, numTensorInputs = 1 : i64, numAuxBuffers = 0 : i64, inputSegmentationAxes = []>} {
      %0 = tensor.empty(%arg1, %arg2, %arg3) : tensor<1x?x?x?xf32>
      %1 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x?x?x?xf32>) outs(%0 : tensor<1x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %2 = math.sqrt %in fastmath<afn> : f32
        linalg.yield %2 : f32
      } -> tensor<1x?x?x?xf32>
      return %1 : tensor<1x?x?x?xf32>
    }
  }

  func.func @main(%arg0: tensor<1x16x4000x200xf32>, %arg1: tensor<1x16x4000x200xf32>) -> (tensor<1x16x4000x200xf32>, tensor<1x16x4000x200xf32>) {
    %0 = VPU.GenericSwLayer(%arg0 : tensor<1x16x4000x200xf32>) @VPU.SW::@generated_1 tiling(sizes = [16, 4000, 200], offsets = [0, 0, 0]) -> tensor<1x16x4000x200xf32>
    %1 = VPU.GenericSwLayer(%arg1 : tensor<1x16x4000x200xf32>) @VPU.SW::@generated_0 tiling(sizes = [16, 4000, 200], offsets = [0, 0, 0]) -> tensor<1x16x4000x200xf32>
    return %0, %1 : tensor<1x16x4000x200xf32>, tensor<1x16x4000x200xf32>
  }
}

// CHECK-LABEL: module @KeepDistinctTiledKernels
// CHECK:   module @VPU.SW
// CHECK:     func.func @generated_info_1
// CHECK:     func.func @generated_1
// CHECK-SAME:  kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @generated_info_1
// CHECK:       math.exp
// CHECK-NOT: func.func @generated_info_0
// CHECK:     func.func @generated_0
// CHECK-SAME:  kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @generated_info_1
// CHECK:       math.sqrt

// CHECK:   func.func @main([[ARG0:%.+]]: tensor<1x16x4000x200xf32>, [[ARG1:%.+]]: tensor<1x16x4000x200xf32>)
// CHECK:     VPU.GenericSwLayer([[ARG0]] : tensor<1x16x4000x200xf32>) @VPU.SW::@generated_1
// CHECK:     VPU.GenericSwLayer([[ARG1]] : tensor<1x16x4000x200xf32>) @VPU.SW::@generated_0

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @DedupIdenticalNonTiledKernels {
  module @VPU.SW {
    func.func @generated_1(%arg0: tensor<1x16x4x8xf32>) -> tensor<1x16x4x8xf32> {
      %0 = tensor.empty() : tensor<1x16x4x8xf32>
      %1 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x16x4x8xf32>) outs(%0 : tensor<1x16x4x8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %2 = math.exp %in fastmath<afn> : f32
        linalg.yield %2 : f32
      } -> tensor<1x16x4x8xf32>
      return %1 : tensor<1x16x4x8xf32>
    }
    func.func @generated_0(%arg0: tensor<1x16x4x8xf32>) -> tensor<1x16x4x8xf32> {
      %0 = tensor.empty() : tensor<1x16x4x8xf32>
      %1 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x16x4x8xf32>) outs(%0 : tensor<1x16x4x8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %2 = math.exp %in fastmath<afn> : f32
        linalg.yield %2 : f32
      } -> tensor<1x16x4x8xf32>
      return %1 : tensor<1x16x4x8xf32>
    }
  }

  func.func @main(%arg0: tensor<1x16x4x8xf32>, %arg1: tensor<1x16x4x8xf32>) -> (tensor<1x16x4x8xf32>, tensor<1x16x4x8xf32>) {
    %0 = VPU.GenericSwLayer(%arg0 : tensor<1x16x4x8xf32>) @VPU.SW::@generated_1 -> tensor<1x16x4x8xf32>
    %1 = VPU.GenericSwLayer(%arg1 : tensor<1x16x4x8xf32>) @VPU.SW::@generated_0 -> tensor<1x16x4x8xf32>
    return %0, %1 : tensor<1x16x4x8xf32>, tensor<1x16x4x8xf32>
  }
}

// CHECK-LABEL: module @DedupIdenticalNonTiledKernels
// CHECK:   module @VPU.SW
// CHECK:     func.func @generated_1
// CHECK-NOT: func.func @generated_0

// CHECK:   func.func @main([[ARG0:%.+]]: tensor<1x16x4x8xf32>, [[ARG1:%.+]]: tensor<1x16x4x8xf32>)
// CHECK:     VPU.GenericSwLayer([[ARG0]] : tensor<1x16x4x8xf32>) @VPU.SW::@generated_1
// CHECK:     VPU.GenericSwLayer([[ARG1]] : tensor<1x16x4x8xf32>) @VPU.SW::@generated_1

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @KeepDistinctNonTiledKernels {
  module @VPU.SW {
    func.func @generated_1(%arg0: tensor<1x16x4x8xf32>) -> tensor<1x16x4x8xf32> {
      %0 = tensor.empty() : tensor<1x16x4x8xf32>
      %1 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x16x4x8xf32>) outs(%0 : tensor<1x16x4x8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %2 = math.exp %in fastmath<afn> : f32
        linalg.yield %2 : f32
      } -> tensor<1x16x4x8xf32>
      return %1 : tensor<1x16x4x8xf32>
    }
    func.func @generated_0(%arg0: tensor<1x16x4x8xf32>) -> tensor<1x16x4x8xf32> {
      %0 = tensor.empty() : tensor<1x16x4x8xf32>
      %1 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x16x4x8xf32>) outs(%0 : tensor<1x16x4x8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %2 = math.sqrt %in fastmath<afn> : f32
        linalg.yield %2 : f32
      } -> tensor<1x16x4x8xf32>
      return %1 : tensor<1x16x4x8xf32>
    }
  }

  func.func @main(%arg0: tensor<1x16x4x8xf32>, %arg1: tensor<1x16x4x8xf32>) -> (tensor<1x16x4x8xf32>, tensor<1x16x4x8xf32>) {
    %0 = VPU.GenericSwLayer(%arg0 : tensor<1x16x4x8xf32>) @VPU.SW::@generated_1 -> tensor<1x16x4x8xf32>
    %1 = VPU.GenericSwLayer(%arg1 : tensor<1x16x4x8xf32>) @VPU.SW::@generated_0 -> tensor<1x16x4x8xf32>
    return %0, %1 : tensor<1x16x4x8xf32>, tensor<1x16x4x8xf32>
  }
}

// CHECK-LABEL: module @KeepDistinctNonTiledKernels
// CHECK:   module @VPU.SW
// CHECK:     func.func @generated_1
// CHECK:       math.exp
// CHECK:     func.func @generated_0
// CHECK:       math.sqrt

// CHECK:   func.func @main([[ARG0:%.+]]: tensor<1x16x4x8xf32>, [[ARG1:%.+]]: tensor<1x16x4x8xf32>)
// CHECK:     VPU.GenericSwLayer([[ARG0]] : tensor<1x16x4x8xf32>) @VPU.SW::@generated_1
// CHECK:     VPU.GenericSwLayer([[ARG1]] : tensor<1x16x4x8xf32>) @VPU.SW::@generated_0

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Tiled and non-tiled kernels in the same VPU.SW module.

module @DedupMixedTiledAndNonTiledKernels {
  module @VPU.SW {
    func.func @generated_info_3(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> (index, index, index, index, index, index, index, index) {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      return %c1, %arg0, %arg1, %arg2, %c0, %arg3, %arg4, %arg5 : index, index, index, index, index, index, index, index
    }
    func.func @generated_3(%arg0: tensor<1x?x?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index) -> tensor<1x?x?x?xf32> attributes {kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @generated_info_3, tilingAxes = [1, 2, 3], numSlicedInputs = 1 : i64, numTensorInputs = 1 : i64, numAuxBuffers = 0 : i64, inputSegmentationAxes = []>} {
      %0 = tensor.empty(%arg1, %arg2, %arg3) : tensor<1x?x?x?xf32>
      %1 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x?x?x?xf32>) outs(%0 : tensor<1x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %2 = math.exp %in fastmath<afn> : f32
        linalg.yield %2 : f32
      } -> tensor<1x?x?x?xf32>
      return %1 : tensor<1x?x?x?xf32>
    }
    func.func @generated_info_2(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> (index, index, index, index, index, index, index, index) {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      return %c1, %arg0, %arg1, %arg2, %c0, %arg3, %arg4, %arg5 : index, index, index, index, index, index, index, index
    }
    func.func @generated_2(%arg0: tensor<1x?x?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index) -> tensor<1x?x?x?xf32> attributes {kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @generated_info_2, tilingAxes = [1, 2, 3], numSlicedInputs = 1 : i64, numTensorInputs = 1 : i64, numAuxBuffers = 0 : i64, inputSegmentationAxes = []>} {
      %0 = tensor.empty(%arg1, %arg2, %arg3) : tensor<1x?x?x?xf32>
      %1 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x?x?x?xf32>) outs(%0 : tensor<1x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %2 = math.exp %in fastmath<afn> : f32
        linalg.yield %2 : f32
      } -> tensor<1x?x?x?xf32>
      return %1 : tensor<1x?x?x?xf32>
    }
    func.func @generated_1(%arg0: tensor<1x16x4x8xf32>) -> tensor<1x16x4x8xf32> {
      %0 = tensor.empty() : tensor<1x16x4x8xf32>
      %1 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x16x4x8xf32>) outs(%0 : tensor<1x16x4x8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %2 = math.exp %in fastmath<afn> : f32
        linalg.yield %2 : f32
      } -> tensor<1x16x4x8xf32>
      return %1 : tensor<1x16x4x8xf32>
    }
    func.func @generated_0(%arg0: tensor<1x16x4x8xf32>) -> tensor<1x16x4x8xf32> {
      %0 = tensor.empty() : tensor<1x16x4x8xf32>
      %1 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x16x4x8xf32>) outs(%0 : tensor<1x16x4x8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %2 = math.exp %in fastmath<afn> : f32
        linalg.yield %2 : f32
      } -> tensor<1x16x4x8xf32>
      return %1 : tensor<1x16x4x8xf32>
    }
  }

  func.func @main(%arg0: tensor<1x16x4000x200xf32>, %arg1: tensor<1x16x4000x200xf32>, %arg2: tensor<1x16x4x8xf32>, %arg3: tensor<1x16x4x8xf32>)
          -> (tensor<1x16x4000x200xf32>, tensor<1x16x4000x200xf32>, tensor<1x16x4x8xf32>, tensor<1x16x4x8xf32>) {
    %0 = VPU.GenericSwLayer(%arg0 : tensor<1x16x4000x200xf32>) @VPU.SW::@generated_3 tiling(sizes = [16, 4000, 200], offsets = [0, 0, 0]) -> tensor<1x16x4000x200xf32>
    %1 = VPU.GenericSwLayer(%arg1 : tensor<1x16x4000x200xf32>) @VPU.SW::@generated_2 tiling(sizes = [16, 4000, 200], offsets = [0, 0, 0]) -> tensor<1x16x4000x200xf32>
    %2 = VPU.GenericSwLayer(%arg2 : tensor<1x16x4x8xf32>) @VPU.SW::@generated_1 -> tensor<1x16x4x8xf32>
    %3 = VPU.GenericSwLayer(%arg3 : tensor<1x16x4x8xf32>) @VPU.SW::@generated_0 -> tensor<1x16x4x8xf32>
    return %0, %1, %2, %3 : tensor<1x16x4000x200xf32>, tensor<1x16x4000x200xf32>, tensor<1x16x4x8xf32>, tensor<1x16x4x8xf32>
  }
}

// CHECK-LABEL: module @DedupMixedTiledAndNonTiledKernels
// CHECK:   module @VPU.SW
// CHECK:     func.func @generated_info_3
// CHECK:     func.func @generated_3
// CHECK-SAME:  kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @generated_info_3
// CHECK-NOT: func.func @generated_info_2
// CHECK-NOT: func.func @generated_2
// CHECK:     func.func @generated_1(
// CHECK-SAME:  tensor<1x16x4x8xf32>
// CHECK-NOT: func.func @generated_0

// CHECK:   func.func @main([[ARG0:%.+]]: tensor<1x16x4000x200xf32>, [[ARG1:%.+]]: tensor<1x16x4000x200xf32>, [[ARG2:%.+]]: tensor<1x16x4x8xf32>, [[ARG3:%.+]]: tensor<1x16x4x8xf32>)
// CHECK:     VPU.GenericSwLayer([[ARG0]] : tensor<1x16x4000x200xf32>) @VPU.SW::@generated_3
// CHECK:     VPU.GenericSwLayer([[ARG1]] : tensor<1x16x4000x200xf32>) @VPU.SW::@generated_3
// CHECK:     VPU.GenericSwLayer([[ARG2]] : tensor<1x16x4x8xf32>) @VPU.SW::@generated_1
// CHECK:     VPU.GenericSwLayer([[ARG3]] : tensor<1x16x4x8xf32>) @VPU.SW::@generated_1
