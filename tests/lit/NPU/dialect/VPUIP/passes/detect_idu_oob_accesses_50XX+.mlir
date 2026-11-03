//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: env OV_NPU_LOG_LEVEL=LOG_TRACE env IE_NPU_LOG_FILTER="detect-idu-oob-accesses" vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --detect-idu-oob-accesses %s  -o /dev/null 2>&1 | FileCheck %s
// REQUIRES: dev-build && (platform-NPU5010 || platform-NPU5020)

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DataType = memref<1x3x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
!WeightsType = memref<3x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>

module {
    config.Resources 4 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 2096128 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }
    func.func @UnalignedWeightsWithinCMXBounds(%input: !DataType, %output: !DataType) -> !DataType {
        %weights = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !WeightsType
        %nce = VPUIP.NCEClusterTask <{
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [1, 1],
                    kernel_strides = [1, 1],
                    resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
                    task_type = #VPUIP.nce_task_type<CONV>
                }>
                input(%input : !DataType)
                weights(%weights : !WeightsType)
                parent_input(%input : !DataType)
                parent_output(%output : !DataType)
                outputs(%output : !DataType) -> !DataType
                variants : {
            DPUTask {inEnd = [15, 15, 2], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 2], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        } PPE : {
        }
        return %nce : !DataType
    }
    // CHECK:      No out-of-bounds accesses detected
    // CHECK-NOT:  Weights of NCE operation {{.*}} are near the CMX boundary
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DataType = memref<1x3x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
!WeightsType = memref<3x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>

module {
    config.Resources 4 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 2096128 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }
    func.func @UnalignedWeightsWithinCMXBoundsNearBoundary(%input: !DataType, %output: !DataType) -> !DataType {
        // Offset 2095600 = 2096128 - 16*16*sizeof(fp16) - 16
        %weights = VPURT.DeclareBuffer <CMX_NN> [0] <2095600> -> !WeightsType
        %nce = VPUIP.NCEClusterTask <{
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [1, 1],
                    kernel_strides = [1, 1],
                    resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
                    task_type = #VPUIP.nce_task_type<CONV>
                }>
                input(%input : !DataType)
                weights(%weights : !WeightsType)
                parent_input(%input : !DataType)
                parent_output(%output : !DataType)
                outputs(%output : !DataType) -> !DataType
                variants : {
            DPUTask {inEnd = [15, 15, 2], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 2], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        } PPE : {
        }
        return %nce : !DataType
    }
    // CHECK:  No out-of-bounds accesses detected
    // CHECK:  Weights of NCE operation {{.*}} are near the CMX boundary: 16 bytes left
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DataType = memref<1x3x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
!WeightsType = memref<3x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>

module {
    config.Resources 4 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 2096128 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }
    func.func @UnalignedWeightsAtCMXEnd(%input: !DataType, %output: !DataType) -> !DataType {
        // Offset 2096032 = 2096128 - 3*16*sizeof(fp16). The weights buffer fits in CMX. However, they are read by IDU as if there were 16 output channels, which leads to reads outside the CMX boundary
        %weights = VPURT.DeclareBuffer <CMX_NN> [0] <2096032> -> !WeightsType
        %nce = VPUIP.NCEClusterTask <{
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [1, 1],
                    kernel_strides = [1, 1],
                    resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
                    task_type = #VPUIP.nce_task_type<CONV>
                }>
                input(%input : !DataType)
                weights(%weights : !WeightsType)
                parent_input(%input : !DataType)
                parent_output(%output : !DataType)
                outputs(%output : !DataType) -> !DataType
                variants : {
            DPUTask {inEnd = [15, 15, 2], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 2], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        } PPE : {
        }
        return %nce : !DataType
    }
    // CHECK:  Weights of NCE operation '{{.*}}' need out-of-bounds read accesses. Address range [2096032; 2096544), CMX end boundary 2096128. Weights OC dimension: 3, aligned OC dimension: 16
}
