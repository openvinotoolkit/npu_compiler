//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --optimize-copies %s | FileCheck %s
// REQUIRES: arch-NPU40XX
IE.TileResource 6 of @NCE at 1.700000e+03 MHz {
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: func.func @NotMoveTilingCopyBeforeSubviewByNotFixCMX
// CHECK-SAME:      [[ARG0:%.+]]: memref<11008x128x1x1
func.func @NotMoveTilingCopyBeforeSubviewByNotFixCMX(%arg0: memref<11008x128x1x1x!quant.uniform<i4:f16, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>, %arg1: memref<1x128x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>) {
    %cst0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<800x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, compute_shapes = [[208, 1, 1, 4], [208, 1, 1, 4], [192, 1, 1, 4], [192, 1, 1, 4]], compute_offsets = [[0, 0, 0, 0], [208, 0, 0, 0], [416, 0, 0, 0], [608, 0, 0, 0]], memory_shapes = [[208, 1, 1, 4], [208, 1, 1, 4], [192, 1, 1, 4], [192, 1, 1, 4]], memory_offsets = [[0, 0, 0, 0], [208, 0, 0, 0], [416, 0, 0, 0], [608, 0, 0, 0]]}>
    %cst1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<704x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, compute_shapes = [[176, 1, 1, 4], [176, 1, 1, 4], [176, 1, 1, 4], [176, 1, 1, 4]], compute_offsets = [[0, 0, 0, 0], [176, 0, 0, 0], [352, 0, 0, 0], [528, 0, 0, 0]], memory_shapes = [[176, 1, 1, 4], [176, 1, 1, 4], [176, 1, 1, 4], [176, 1, 1, 4]], memory_offsets = [[0, 0, 0, 0], [176, 0, 0, 0], [352, 0, 0, 0], [528, 0, 0, 0]]}>

    // two branches is enough to test issue
    %weights0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [800, 128, 1, 1] : memref<11008x128x1x1x!quant.uniform<i4:f16, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR> to memref<800x128x1x1x!quant.uniform<i4:f16, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
    // omit middle branches
    %weights1 = VPUIP.SubView %arg0 [4800, 0, 0, 0] [704, 128, 1, 1] : memref<11008x128x1x1x!quant.uniform<i4:f16, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR> to memref<704x128x1x1x!quant.uniform<i4:f16, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>

    %weights0_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<800x128x1x1x!quant.uniform<i4:f16, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, compute_shapes = [[208, 128, 1, 1], [208, 128, 1, 1], [192, 128, 1, 1], [192, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [208, 0, 0, 0], [416, 0, 0, 0], [608, 0, 0, 0]], memory_shapes = [[208, 128, 1, 1], [208, 128, 1, 1], [192, 128, 1, 1], [192, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [208, 0, 0, 0], [416, 0, 0, 0], [608, 0, 0, 0]]}>
    %weights0_copy = VPUIP.NCEClusterTiling inputs(%weights0 as %arg72: memref<800x128x1x1x!quant.uniform<i4:f16, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>) outputs(%weights0_cmx as %arg73: memref<800x128x1x1x!quant.uniform<i4:f16, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>) -> !VPUIP.DistributedBuffer<800x128x1x1x!quant.uniform<i4:f16, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, compute_shapes = [[208, 128, 1, 1], [208, 128, 1, 1], [192, 128, 1, 1], [192, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [208, 0, 0, 0], [416, 0, 0, 0], [608, 0, 0, 0]], memory_shapes = [[208, 128, 1, 1], [208, 128, 1, 1], [192, 128, 1, 1], [192, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [208, 0, 0, 0], [416, 0, 0, 0], [608, 0, 0, 0]]}> {
      %15858 = VPUIP.Copy inputs(%arg72 : memref<800x128x1x1x!quant.uniform<i4:f16, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>) outputs(%arg73 : memref<800x128x1x1x!quant.uniform<i4:f16, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>) -> memref<800x128x1x1x!quant.uniform<i4:f16, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>
    }

    %weights1_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<704x128x1x1x!quant.uniform<i4:f16, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, compute_shapes = [[176, 128, 1, 1], [176, 128, 1, 1], [176, 128, 1, 1], [176, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [176, 0, 0, 0], [352, 0, 0, 0], [528, 0, 0, 0]], memory_shapes = [[176, 128, 1, 1], [176, 128, 1, 1], [176, 128, 1, 1], [176, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [176, 0, 0, 0], [352, 0, 0, 0], [528, 0, 0, 0]]}>
    %weights1_copy = VPUIP.NCEClusterTiling inputs(%weights1 as %arg72: memref<704x128x1x1x!quant.uniform<i4:f16, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>) outputs(%weights1_cmx as %arg73: memref<704x128x1x1x!quant.uniform<i4:f16, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>) -> !VPUIP.DistributedBuffer<704x128x1x1x!quant.uniform<i4:f16, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, compute_shapes = [[176, 128, 1, 1], [176, 128, 1, 1], [176, 128, 1, 1], [176, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [176, 0, 0, 0], [352, 0, 0, 0], [528, 0, 0, 0]], memory_shapes = [[176, 128, 1, 1], [176, 128, 1, 1], [176, 128, 1, 1], [176, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [176, 0, 0, 0], [352, 0, 0, 0], [528, 0, 0, 0]]}> {
      %15858 = VPUIP.Copy inputs(%arg72 : memref<704x128x1x1x!quant.uniform<i4:f16, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>) outputs(%arg73 : memref<704x128x1x1x!quant.uniform<i4:f16, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>) -> memref<704x128x1x1x!quant.uniform<i4:f16, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>
    }

    %input0_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments, compute_shapes = [[1, 128, 288, 4], [1, 128, 288, 4], [1, 128, 288, 4], [1, 128, 288, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 128, 288, 4], [1, 128, 288, 4], [1, 128, 288, 4], [1, 128, 288, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    %input0 = VPUIP.NCEClusterTiling inputs(%arg1 as %arg72: memref<1x128x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>) outputs(%input0_cmx as %arg73: memref<1x128x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments, compute_shapes = [[1, 128, 288, 4], [1, 128, 288, 4], [1, 128, 288, 4], [1, 128, 288, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 128, 288, 4], [1, 128, 288, 4], [1, 128, 288, 4], [1, 128, 288, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
      %15858 = VPUIP.Copy inputs(%arg72 : memref<1x128x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>) outputs(%arg73 : memref<1x128x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>) -> memref<1x128x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>
    }

    %input1_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments, compute_shapes = [[1, 128, 288, 4], [1, 128, 288, 4], [1, 128, 288, 4], [1, 128, 288, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 128, 288, 4], [1, 128, 288, 4], [1, 128, 288, 4], [1, 128, 288, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    %input1 = VPUIP.NCEClusterTiling inputs(%arg1 as %arg72: memref<1x128x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>) outputs(%input1_cmx as %arg73: memref<1x128x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments, compute_shapes = [[1, 128, 288, 4], [1, 128, 288, 4], [1, 128, 288, 4], [1, 128, 288, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 128, 288, 4], [1, 128, 288, 4], [1, 128, 288, 4], [1, 128, 288, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
      %15858 = VPUIP.Copy inputs(%arg72 : memref<1x128x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>) outputs(%arg73 : memref<1x128x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>) -> memref<1x128x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>
    }

    %output0_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x800x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments, compute_shapes = [[1, 208, 288, 4], [1, 208, 288, 4], [1, 192, 288, 4], [1, 192, 288, 4]], compute_offsets = [[0, 0, 0, 0], [0, 208, 0, 0], [0, 416, 0, 0], [0, 608, 0, 0]], memory_shapes = [[1, 208, 288, 4], [1, 208, 288, 4], [1, 192, 288, 4], [1, 192, 288, 4]], memory_offsets = [[0, 0, 0, 0], [0, 208, 0, 0], [0, 416, 0, 0], [0, 608, 0, 0]]}>
    %ouput0 = VPUIP.NCEClusterTiling inputs(%input0 as %arg72: memref<1x128x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>, %weights0 as %arg73: memref<800x128x1x1x!quant.uniform<i4:f16, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>, %cst0 as %arg74: memref<800x1x1x4xsi32, @CMX_NN>) outputs(%output0_cmx as %arg75: memref<1x800x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x800x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments, compute_shapes = [[1, 208, 288, 4], [1, 208, 288, 4], [1, 192, 288, 4], [1, 192, 288, 4]], compute_offsets = [[0, 0, 0, 0], [0, 208, 0, 0], [0, 416, 0, 0], [0, 608, 0, 0]], memory_shapes = [[1, 208, 288, 4], [1, 208, 288, 4], [1, 192, 288, 4], [1, 192, 288, 4]], memory_offsets = [[0, 0, 0, 0], [0, 208, 0, 0], [0, 416, 0, 0], [0, 608, 0, 0]]}> {
      %15858 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 21715 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%arg72 : memref<1x128x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>) weights(%arg73 : memref<800x128x1x1x!quant.uniform<i4:f16, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>) weight_table(%arg74 : memref<800x1x1x4xsi32, @CMX_NN>) parent_input(%arg72 : memref<1x128x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>) parent_output(%arg75 : memref<1x800x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>) outputs(%arg75 : memref<1x800x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>) -> memref<1x800x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [3, 287, 127], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 287, 207], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [3, 287, 127], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 287, 207], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 2 : i64, inEnd = [3, 287, 127], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 287, 191], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 3 : i64, inEnd = [3, 287, 127], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 287, 191], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
      }
    }

    %output1_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x704x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments, compute_shapes = [[1, 176, 288, 4], [1, 176, 288, 4], [1, 176, 288, 4], [1, 176, 288, 4]], compute_offsets = [[0, 0, 0, 0], [0, 176, 0, 0], [0, 352, 0, 0], [0, 528, 0, 0]], memory_shapes = [[1, 176, 288, 4], [1, 176, 288, 4], [1, 176, 288, 4], [1, 176, 288, 4]], memory_offsets = [[0, 0, 0, 0], [0, 176, 0, 0], [0, 352, 0, 0], [0, 528, 0, 0]]}>
    %output1 = VPUIP.NCEClusterTiling inputs(%input1 as %arg72: memref<1x128x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>, %weights1 as %arg73: memref<704x128x1x1x!quant.uniform<i4:f16, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>, %cst1 as %arg74: memref<704x1x1x4xsi32, @CMX_NN>) outputs(%output1_cmx as %arg75: memref<1x704x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x704x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments, compute_shapes = [[1, 176, 288, 4], [1, 176, 288, 4], [1, 176, 288, 4], [1, 176, 288, 4]], compute_offsets = [[0, 0, 0, 0], [0, 176, 0, 0], [0, 352, 0, 0], [0, 528, 0, 0]], memory_shapes = [[1, 176, 288, 4], [1, 176, 288, 4], [1, 176, 288, 4], [1, 176, 288, 4]], memory_offsets = [[0, 0, 0, 0], [0, 176, 0, 0], [0, 352, 0, 0], [0, 528, 0, 0]]}> {
      %15858 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 20293 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%arg72 : memref<1x128x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>) weights(%arg73 : memref<704x128x1x1x!quant.uniform<i4:f16, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>) weight_table(%arg74 : memref<704x1x1x4xsi32, @CMX_NN>) parent_input(%arg72 : memref<1x128x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>) parent_output(%arg75 : memref<1x704x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>) outputs(%arg75 : memref<1x704x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>) -> memref<1x704x288x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [3, 287, 127], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 287, 175], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [3, 287, 127], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 287, 175], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 2 : i64, inEnd = [3, 287, 127], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 287, 175], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 3 : i64, inEnd = [3, 287, 127], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 287, 175], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
      }
    }

    return

    // CHECK:      [[WEIGHTS0:%.+]] = VPUIP.SubView [[ARG0]] [0, 0, 0, 0] [800, 128, 1, 1]
    // CHECK-NEXT: [[WEIGHTS1:%.+]] = VPUIP.SubView [[ARG0]] [4800, 0, 0, 0] [704, 128, 1, 1]
    // CHECK-NEXT: [[WEIGHTS0_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<800x128x1x1
    // CHECK-NEXT: [[COPY0:%.+]] = VPUIP.NCEClusterTiling inputs([[WEIGHTS0]]
    // CHECK-SAME:   outputs([[WEIGHTS0_CMX]]
    // CHECK:      [[WEIGHTS1_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<704x128x1x1
    // CHECK-NEXT: [[COPY1:%.+]] = VPUIP.NCEClusterTiling inputs([[WEIGHTS1]]
    // CHECK-SAME:   outputs([[WEIGHTS1_CMX]]
}

// -----

IE.TileResource 6 of @NCE at 1.700000e+03 MHz {
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!InputDistributedType = !VPUIP.DistributedBuffer<
    1x30x120x120xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!InputStub_CMX = memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>
!SpilledOutput_DDR = memref<1x30x120x120xf16, #NHWC, @DDR>

func.func @NotFuseCMXCopyToTheFrontOfTillingCopyDueToCMXSizeLimitation() -> !InputStub_CMX {
  %0 = VPURT.AllocDistributed -> !InputDistributedType
  %1 = memref.alloc() : !SpilledOutput_DDR
  %2 = VPUIP.Copy inputs(%0: !InputDistributedType) outputs(%1: !SpilledOutput_DDR) -> !SpilledOutput_DDR

  %3 = memref.alloc() : !InputStub_CMX
  %4 = VPUIP.Copy inputs(%2 : !SpilledOutput_DDR) outputs(%3 : !InputStub_CMX) -> !InputStub_CMX

  return %4 : !InputStub_CMX

  // CHECK:   [[BUF_0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x30x120x120xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
  // CHECK:   [[BUF_1:%.*]] = memref.alloc() : memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>
  // CHECK:   [[COPY_0:%.*]] = VPUIP.Copy
  // CHECK-SAME:  inputs([[BUF_0]] : !VPUIP.DistributedBuffer<1x30x120x120xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>)
  // CHECK-SAME:  outputs([[BUF_1]] : memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>
  // CHECK:   return [[COPY_0]] : memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

IE.TileResource 6 of @NCE at 1.700000e+03 MHz {
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
}
// CHECK-LABEL: func.func @NotEraseCMX2CMXCopyAfterSubviewDueToCMXSizeLimitation
// CHECK-SAME:      [[DATA:%.+]]: memref<8000x32xf16>
// CHECK-SAME:      [[INDICES:%.+]]: memref<64000x1xi64, [@CMX_NN, 0]>
func.func @NotEraseCMX2CMXCopyAfterSubviewDueToCMXSizeLimitation(%data : memref<8000x32xf16>, %indices : memref<64000x1xi64, [@CMX_NN, 0]>)
                              -> memref<16000x32xf16, [@CMX_NN, 0]>
{
  %subview_indices = VPUIP.SubView %indices [0, 0] [16000, 1] : memref<64000x1xi64, [@CMX_NN, 0]> to memref<16000x1xi64, [@CMX_NN, 0]>
  %alloc_subview_indices = memref.alloc() : memref<16000x1xi64, [@CMX_NN, 0]>
  %subview_indices_in = VPUIP.Copy inputs(%subview_indices : memref<16000x1xi64, [@CMX_NN, 0]>) outputs(%alloc_subview_indices : memref<16000x1xi64, [@CMX_NN, 0]>) -> memref<16000x1xi64, [@CMX_NN, 0]>
  %alloc_gather = memref.alloc() : memref<16000x32xf16, [@CMX_NN, 0]>
  %gather_out = VPUIP.GatherDMA {channelType = 0 : i64, elementSize = 0 : i64, padding = 0 : i64, port = 0 : i64} inputs(%data : memref<8000x32xf16>) indices(%subview_indices_in : memref<16000x1xi64, [@CMX_NN, 0]>) outputs(%alloc_gather : memref<16000x32xf16, [@CMX_NN, 0]>) -> memref<16000x32xf16, [@CMX_NN, 0]>
  return %gather_out : memref<16000x32xf16, [@CMX_NN, 0]>

  // CHECK:   [[INDICES_IN:%.+]] = VPUIP.ViewOp [[INDICES]] : memref<64000x1xi64, [@CMX_NN, 0]> to memref<16000x1xi64, [@CMX_NN, 0]>
  // CHECK-NOT:   memref.alloc()
  // CHECK-NOT:   VPUIP.Copy
  // CHECK:   [[ALLOC_GATHER:%.+]] = memref.alloc() : memref<16000x32xf16, [@CMX_NN, 0]>
  // CHECK:   [[GATHER_OUT:%.+]] = VPUIP.GatherDMA {channelType = 0 : i64, elementSize = 0 : i64, padding = 0 : i64, port = 0 : i64} inputs([[DATA]] : memref<8000x32xf16>) indices([[INDICES_IN]] : memref<16000x1xi64, [@CMX_NN, 0]>) outputs([[ALLOC_GATHER]] : memref<16000x32xf16, [@CMX_NN, 0]>) -> memref<16000x32xf16, [@CMX_NN, 0]>
  // CHECK:   return [[GATHER_OUT]] : memref<16000x32xf16, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DistributedBufferType = !VPUIP.DistributedBuffer<1x2x4x121xf16, #NCHW, @CMX_NN,
    {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
     compute_shapes = [[1, 2, 1, 121], [1, 2, 1, 121], [1, 2, 1, 121], [1, 2, 1, 121]],
     compute_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]],
     memory_shapes = [[1, 2, 1, 121], [1, 2, 1, 121], [1, 2, 1, 121], [1, 2, 1, 121]],
     memory_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]]}>

IE.TileResource 6 of @NCE at 1.700000e+03 MHz {
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @NotFuseClusterTilingCopiesThroughReshapeDueToNoAxisMapping
// CHECK-SAME:  [[INPUT:%.+]]: memref<2x2x1025x121xf16, @DDR>
func.func @NotFuseClusterTilingCopiesThroughReshapeDueToNoAxisMapping(%input : memref<2x2x1025x121xf16, @DDR>) -> !DistributedBufferType {
    %0 = VPUIP.SubView %input [0, 0, 0, 0] [2, 2, 2, 121] : memref<2x2x1025x121xf16, @DDR> to memref<2x2x2x121xf16, {order = #NCHW, strides = [248050, 124025, 121, 1]}, @DDR>
    %1 = memref.alloc() : memref<2x2x2x121xf16, @DDR>
    %2 = VPUIP.Copy inputs(%0 : memref<2x2x2x121xf16, {order = #NCHW, strides = [248050, 124025, 121, 1]}, @DDR>) outputs(%1 : memref<2x2x2x121xf16, @DDR>) -> memref<2x2x2x121xf16, @DDR>
    %3 = VPUIP.GenericReshape inputs(%2 : memref<2x2x2x121xf16, @DDR>) -> memref<1x2x4x121xf16, @DDR>
    %4 = VPURT.AllocDistributed -> !DistributedBufferType
    %5 = VPUIP.NCEClusterTiling inputs(%3 as %arg2: memref<1x2x4x121xf16>) outputs(%4 as %arg3: memref<1x2x4x121xf16, @CMX_NN>) -> !DistributedBufferType {
      %copy = VPUIP.Copy inputs(%arg2 : memref<1x2x4x121xf16>) outputs(%arg3 : memref<1x2x4x121xf16, @CMX_NN>) -> memref<1x2x4x121xf16, @CMX_NN>
    }
    return %5 : !DistributedBufferType

    // CHECK: [[SUBVIEW:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [2, 2, 2, 121] : memref<2x2x1025x121xf16, @DDR> to memref<2x2x2x121xf16, {order = #NCHW, strides = [248050, 124025, 121, 1]}, @DDR>
    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<2x2x2x121xf16, @DDR>
    // CHECK: [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:        inputs([[SUBVIEW]] : memref<2x2x2x121xf16, {order = #NCHW, strides = [248050, 124025, 121, 1]}, @DDR>)
    // CHECK-SAME:        outputs([[ALLOC]] : memref<2x2x2x121xf16, @DDR>) -> memref<2x2x2x121xf16, @DDR>
    // CHECK: [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[COPY]] : memref<2x2x2x121xf16, @DDR>) -> memref<1x2x4x121xf16, @DDR>
    // CHECK: [[ALLOC_DIST:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME{LITERAL}: -> !VPUIP.DistributedBuffer<1x2x4x121xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1],
    // CHECK-SAME:  num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 2, 1, 121], [1, 2, 1, 121], [1, 2, 1, 121], [1, 2, 1, 121]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 2, 1, 121], [1, 2, 1, 121], [1, 2, 1, 121], [1, 2, 1, 121]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]]

    // CHECK: [[COPY_DIST:%.+]] = VPUIP.NCEClusterTiling inputs([[RESHAPE]] as [[INNER_ARG0:[^:]+]]: memref<1x2x4x121xf16>)
    // CHECK:     outputs([[ALLOC_DIST]] as [[INNER_ARG1:[^:]+]]: memref<1x2x4x121xf16, @CMX_NN>)
    // CHECK-SAME{LITERAL}: -> !VPUIP.DistributedBuffer<1x2x4x121xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1],
    // CHECK-SAME:          num_clusters = 4 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 2, 1, 121], [1, 2, 1, 121], [1, 2, 1, 121], [1, 2, 1, 121]]
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]]
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 2, 1, 121], [1, 2, 1, 121], [1, 2, 1, 121], [1, 2, 1, 121]]
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]]
    // CHECK:     VPUIP.Copy inputs([[INNER_ARG0]] : memref<1x2x4x121xf16>) outputs([[INNER_ARG1]] : memref<1x2x4x121xf16, @CMX_NN>) -> memref<1x2x4x121xf16, @CMX_NN>

    // CHECK: return [[COPY_DIST]]
    // CHECK-SAME{LITERAL}:  !VPUIP.DistributedBuffer<1x2x4x121xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1],
    // CHECK-SAME:  num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 2, 1, 121], [1, 2, 1, 121], [1, 2, 1, 121], [1, 2, 1, 121]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 2, 1, 121], [1, 2, 1, 121], [1, 2, 1, 121], [1, 2, 1, 121]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]]
}
