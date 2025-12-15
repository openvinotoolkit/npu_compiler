//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --vpu-arch=%arch% --default-hw-mode="enable-se-ptrs-operations=true enable-activation-sparsity=false enable-weights-sparsity=false" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX


// CHECK-LABEL: @Interpolate
module @Interpolate {

net.NetworkInfo
    entryPoint : @sepInterpolate
    inputsInfo : {
        DataInfo "input" : tensor<1x512x8x8xf16>
    }
    outputsInfo : {
        DataInfo "sepInterpolate" : tensor<1x512x16x16xf16>
    }
func.func @sepInterpolate(%arg0: tensor<1x512x8x8xf16>) -> tensor<1x512x16x16xf16> {
   %0 = IE.Interpolate(%arg0) {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>, nearest_mode = <FLOOR>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [16, 16]
         } : tensor<1x512x8x8xf16> -> tensor<1x512x16x16xf16>

    return %0 : tensor<1x512x16x16xf16>

// CHECK:         VPUIP.NCEClusterTask {input_se_size = 512 : i64,
// CHECK-SAME:        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
// CHECK-SAME:        task_type = #VPUIP.nce_task_type<CONV>}
// CHECK-SAME:        input_storage_element_table
// CHECK-SAME:        parent_input_storage_element_table

}
}

