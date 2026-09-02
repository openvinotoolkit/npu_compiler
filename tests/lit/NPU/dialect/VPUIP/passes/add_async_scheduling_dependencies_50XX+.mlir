//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --add-async-scheduling-dependencies %s | FileCheck %s
// REQUIRES: platform-NPU5010 || platform-NPU5020

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
    func.func nested @builtin_DynamicDequantize(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "dynamic_dequantize.cpp", VPU.kernel_entry = "dynamic_dequantize"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @AddDependency
func.func @AddDependency(
        %input: memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>,
        %weights: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>,
        %convOutput: memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>,
        %scale: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>,
        %dequantOutput: memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) {
    %convToken = async.execute {
        %conv = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }>
            input(%input : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weights(%weights : memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_input(%input : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%convOutput : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%convOutput : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants : {
                DPUTask {inEnd = [3, 3, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 3, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            }
            PPE : {
            }
        async.yield
    }

    %dequantToken = async.execute {
        %dequant = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DynamicDequantize
                inputs(%input as %dequantInput: memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                       %scale as %dequantScale: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                outputs(%dequantOutput as %dequantResult: memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                on tile 0 -> memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]> {
            VPUIP.SW.Kernel.run(%dequantInput, %dequantScale, %dequantResult) :
                    memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
        }
        async.yield
    }

    %nextDequantToken = async.execute {
        %dequant = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DynamicDequantize
                inputs(%input as %dequantInput: memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                       %scale as %dequantScale: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                outputs(%dequantOutput as %dequantResult: memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                on tile 0 -> memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]> {
            VPUIP.SW.Kernel.run(%dequantInput, %dequantScale, %dequantResult) :
                    memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
        }
        async.yield
    }

    %consumerToken = async.execute [%convToken, %dequantToken] {
        %consumer = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }>
            input(%convOutput : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weights(%dequantOutput : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_input(%convOutput : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%convOutput : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%convOutput : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants : {
                DPUTask {inEnd = [3, 3, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 3, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            }
            PPE : {
            }
        async.yield
    }

    %nextConsumerToken = async.execute [%consumerToken, %nextDequantToken] {
        %consumer = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }>
            input(%input : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weights(%dequantOutput : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_input(%input : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%convOutput : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%convOutput : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants : {
                DPUTask {inEnd = [3, 3, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 3, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            }
            PPE : {
            }
        async.yield
    }

    return

    // CHECK: [[CONV_TOKEN:%.+]] = async.execute
    // CHECK: [[DEQUANT_TOKEN:%.+]] = async.execute attributes {
    // CHECK: VPUIP.SW.Kernel {{.*}} @VPU.SW::@builtin_DynamicDequantize
    // CHECK: [[CONSUMER_TOKEN:%.+]] = async.execute [{{.*}}[[CONV_TOKEN]]{{.*}}[[DEQUANT_TOKEN]]{{.*}}]
    // CHECK: [[NEXT_DEQUANT_TOKEN:%.+]] = async.execute [{{ *}}[[CONSUMER_TOKEN]]{{ *}}]
    // CHECK: VPUIP.SW.Kernel {{.*}} @VPU.SW::@builtin_DynamicDequantize
    // CHECK: [[NEXT_CONSUMER_TOKEN:%.+]] = async.execute [{{.*}}[[NEXT_DEQUANT_TOKEN]]{{.*}}[[CONSUMER_TOKEN]]{{.*}}]
}

// CHECK-LABEL: @DoNotAddDependencyForUnrelatedTasks
func.func @DoNotAddDependencyForUnrelatedTasks(
        %input: memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>,
        %weights: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>,
        %convOutput: memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>,
        %scale: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>,
        %dequantOutput: memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) {
    %convToken = async.execute {
        %conv = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }>
            input(%input : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weights(%weights : memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_input(%input : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%convOutput : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%convOutput : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants : {
                DPUTask {inEnd = [3, 3, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 3, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            }
            PPE : {
            }
        async.yield
    }

    %dequantToken = async.execute {
        %dequant = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DynamicDequantize
                inputs(%input as %dequantInput: memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                       %scale as %dequantScale: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                outputs(%dequantOutput as %dequantResult: memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                on tile 0 -> memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]> {
            VPUIP.SW.Kernel.run(%dequantInput, %dequantScale, %dequantResult) :
                    memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                    memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
        }
        async.yield
    }

    return

    // CHECK: [[CONV_TOKEN:%.+]] = async.execute
    // CHECK: [[DEQUANT_TOKEN:%.+]] = async.execute{{[^\[]*}}
    // CHECK: VPUIP.SW.Kernel {{.*}} @VPU.SW::@builtin_DynamicDequantize
}
