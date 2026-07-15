//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --fuse-last-copy %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IDataCMXType = memref<1x16x4x4xf16, {order = #NHWC}, @CMX_NN>
!ISMCMXType = memref<1x16x4x4xi1, {order = #NHWC}, @CMX_NN>

// CHECK-LABEL: @NoChangesDifferentMemSpaceSparse
// CHECK-SAME: ([[IN_DATA:%.+]]: memref<1x16x4x4xf16, {order = #NHWC}>, [[IN_SM:%.+]]: memref<1x16x4x4xi1, {order = #NHWC}>,
// CHECK-SAME: {{%.+}}: memref<16x1x1x4xsi32, @CMX_NN>, {{%.+}}: memref<16x1x1x16xui8, @CMX_NN>,
// CHECK-SAME: [[OUT_DATA:%.+]]: memref<1x16x4x4xf16, {order = #NHWC}>, [[OUT_SM:%.+]]: memref<1x16x4x4xi1, {order = #NHWC}>)
func.func @NoChangesDifferentMemSpaceSparse(
        %arg0data: memref<1x16x4x4xf16, {order = #NHWC}>, %arg0sm: memref<1x16x4x4xi1, {order = #NHWC}>,
        %arg1 : memref<16x1x1x4xsi32, @CMX_NN>,
        %arg2 : memref<16x1x1x16xui8, @CMX_NN>,
        %arg3data: memref<1x16x4x4xf16, {order = #NHWC}>, %arg3sm: memref<1x16x4x4xi1, {order = #NHWC}>)
        -> (memref<1x16x4x4xf16, {order = #NHWC}>, memref<1x16x4x4xi1, {order = #NHWC}>) {
    %data_buff = memref.alloc() : !IDataCMXType
    %sm_buff = memref.alloc() : !ISMCMXType
    %in_data_0 = VPUIP.Copy
        inputs(%arg0data: memref<1x16x4x4xf16, {order = #NHWC}>)
        outputs(%data_buff: !IDataCMXType)
        -> !IDataCMXType
    %in_sm_0 = VPUIP.Copy
        inputs(%arg0sm: memref<1x16x4x4xi1, {order = #NHWC}>)
        outputs(%sm_buff: !ISMCMXType)
        -> !ISMCMXType

    %out_data_0 = memref.alloc() : !IDataCMXType
    %out_sm_0 = memref.alloc() : !ISMCMXType
    %mp:2 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>} <{
            kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>,
            kernel_size = [2, 2],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }>
        input(%in_data_0 : !IDataCMXType)
        input_sparsity_map(%in_sm_0 : !ISMCMXType)
        weight_table(%arg1 : memref<16x1x1x4xsi32, @CMX_NN>)
        parent_input(%in_data_0 : !IDataCMXType)
        parent_input_sparsity_map(%in_sm_0 : !ISMCMXType)
        parent_output(%out_data_0 : !IDataCMXType)
        parent_output_sparsity_map(%out_sm_0 : !ISMCMXType)
        outputs(%out_data_0 : !IDataCMXType)
        output_sparsity_map(%out_sm_0 : !ISMCMXType) -> !IDataCMXType, !ISMCMXType
        variants :
        {
            DPUTask { outEnd = [16, 2, 2], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
        }
        PPE : {
        }

    %3 = VPUIP.Copy inputs(%mp#0 : !IDataCMXType) outputs(%arg3data : memref<1x16x4x4xf16, {order = #NHWC}>)
        -> memref<1x16x4x4xf16, {order = #NHWC}>
    %4 = VPUIP.Copy inputs(%mp#1 : !ISMCMXType) outputs(%arg3sm : memref<1x16x4x4xi1, {order = #NHWC}>)
        -> memref<1x16x4x4xi1, {order = #NHWC}>
    return %3, %4 : memref<1x16x4x4xf16, {order = #NHWC}>, memref<1x16x4x4xi1, {order = #NHWC}>

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x16x4x4xf16, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x16x4x4xi1, {order = #NHWC}, @CMX_NN>

    // CHECK:       [[IN_COPY_DATA:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[IN_DATA]]
    // CHECK-SAME:      outputs([[BUFF_0_DATA]]
    // CHECK:       [[IN_COPY_SM:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[IN_SM]]
    // CHECK-SAME:      outputs([[BUFF_0_SM]]

    // CHECK:       [[NCE_OUT_DATA:%.+]] = memref.alloc() : memref<1x16x4x4xf16, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[NCE_OUT_SM:%.+]] = memref.alloc() : memref<1x16x4x4xi1, {order = #NHWC}, @CMX_NN>

    // CHECK:       [[NCE_0:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:          input([[IN_COPY_DATA]]
    // CHECK-SAME:          input_sparsity_map([[IN_COPY_SM]]
    // CHECK-SAME:          outputs([[NCE_OUT_DATA]]
    // CHECK-SAME:          output_sparsity_map([[NCE_OUT_SM]]

    // CHECK:       [[COPY_DATA:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[NCE_0]]#0
    // CHECK-SAME:      outputs([[OUT_DATA]]

    // CHECK:       [[COPY_SM:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[NCE_0]]#1
    // CHECK-SAME:      outputs([[OUT_SM]]

    // CHECK:       return [[COPY_DATA]], [[COPY_SM]]
}
