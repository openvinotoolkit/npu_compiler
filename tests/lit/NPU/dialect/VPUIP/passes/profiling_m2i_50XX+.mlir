//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --m2i-profiling %s | FileCheck %s
// REQUIRES: arch-NPU50XX

// CHECK-LABEL: @M2IProfiling
module @M2IProfiling {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x768x512x1xui8>
    } outputsInfo : {
        DataInfo "output" : tensor<1x512x512x3xui8>
    } profilingOutputsInfo : {
    }

    func.func @main(%arg0: memref<1x768x512x1xui8, @DDR>, %arg1: memref<1x512x512x3xui8, @DDR>) -> memref<1x512x512x3xui8, @DDR> {
      %alloc = memref.alloc() : memref<1x768x512x1xui8, [@CMX_NN, 0]>
      %0 = VPUIP.Copy inputs(%arg0 : memref<1x768x512x1xui8, @DDR>) outputs(%alloc : memref<1x768x512x1xui8, [@CMX_NN, 0]>) -> memref<1x768x512x1xui8, [@CMX_NN, 0]>
      %alloc_0 = memref.alloc() : memref<1x512x512x3xui8, [@CMX_NN, 0]>
      %output = VPUIP.M2ITask {chroma_out_reverse_channels, do_csc = true, do_norm = false, inFmt = #VPU.m2i_color_fmt<SP_NV12_8>, outFmt = #VPU.m2i_color_fmt<IL_RGB888>, scale_factor_x = 131072 : ui32, scale_factor_y = 131072 : ui32} inputs(%0 : memref<1x768x512x1xui8, [@CMX_NN, 0]>) outputs(%alloc_0 : memref<1x512x512x3xui8, [@CMX_NN, 0]>) -> memref<1x512x512x3xui8, [@CMX_NN, 0]>
      %1 = VPUIP.NNDMA inputs(%output : memref<1x512x512x3xui8, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x512x512x3xui8, @DDR>) -> memref<1x512x512x3xui8, @DDR>
      return %1 : memref<1x512x512x3xui8, @DDR>
    }

    // CHECK:        profilingOutputsInfo : {
    // CHECK-NEXT:     DataInfo "m2i" : tensor<64xui8>
    // CHECK-NEXT:   }
    // CHECK:        func.func @main(%arg0: memref<1x768x512x1xui8, @DDR>, %arg1: memref<1x512x512x3xui8, @DDR>, %arg2: memref<64xui8>) -> (memref<1x512x512x3xui8, @DDR>, memref<64xui8>)
    // CHECK:        [[PROF_BUF:%.+]] = memref.alloc() : memref<64xui8, [@CMX_NN, 0]>
    // CHECK:        [[PROF_OUT0:%.+]] = VPUIP.SubView [[PROF_BUF]] [0] [64] : memref<64xui8, [@CMX_NN, 0]> to memref<64xui8, [@CMX_NN, 0]>
    // CHECK-NEXT:   [[OP_RESULT_1:%.*]], [[OP_RESULT_PROF_1:%.*]] = VPUIP.M2ITask
    // CHECK-SAME:   profilingMetadata = #VPUIP.M2IProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64>
    // CHECK-SAME:   profiling_data([[PROF_OUT0]] : memref<64xui8, [@CMX_NN, 0]>)

    // CHECK:        [[PROF_OUTPUT:%.+]] = VPUIP.SubView %arg2 [0] [64] : memref<64xui8> to memref<64xui8>
    // CHECK:        [[CONCAT_PROF_RES:%.+]] = VPUIP.ConcatView inputs([[OP_RESULT_PROF_1]] : memref<64xui8, [@CMX_NN, 0]>) outputs([[PROF_BUF]] : memref<64xui8, [@CMX_NN, 0]>) -> memref<64xui8, [@CMX_NN, 0]>

    // CHECK:        [[PROF_BUF_COPY:%.+]] = VPUIP.NNDMA {profiling_buffer_mgmt} inputs([[CONCAT_PROF_RES]] : memref<64xui8, [@CMX_NN, 0]>) outputs([[PROF_OUTPUT]] : memref<64xui8>) -> memref<64xui8>
    // CHECK:        [[CONCAT_PROF_RES_FULL:%.+]] = VPUIP.ConcatView inputs([[PROF_BUF_COPY]] : memref<64xui8>) outputs(%arg2 : memref<64xui8>) -> memref<64xui8>

    // CHECK:        return [[R1:%.+]], [[CONCAT_PROF_RES_FULL]] : memref<1x512x512x3xui8, @DDR>, memref<64xui8>
}


// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @M2IProfilingMultitile
module @M2IProfilingMultitile {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x768x512x1xui8>
    } outputsInfo : {
        DataInfo "output" : tensor<2x512x512x3xui8>
    } profilingOutputsInfo : {
    }

    func.func @main(%arg0: memref<1x768x512x1xui8, @DDR>, %arg1: memref<2x512x512x3xui8, @DDR>) -> memref<2x512x512x3xui8, @DDR> {
        %alloc = memref.alloc() : memref<1x768x512x1xui8, [@CMX_NN, 0]>
        %0 = VPUIP.NNDMA inputs(%arg0 : memref<1x768x512x1xui8, @DDR>) outputs(%alloc : memref<1x768x512x1xui8, [@CMX_NN, 0]>) -> memref<1x768x512x1xui8, [@CMX_NN, 0]>

        // M2I Task #1
        %alloc_0 = memref.alloc() : memref<1x512x512x3xui8, [@CMX_NN, 0]>
        %output = VPUIP.M2ITask {chroma_out_reverse_channels, do_csc = true, do_norm = false, inFmt = #VPU.m2i_color_fmt<SP_NV12_8>, outFmt = #VPU.m2i_color_fmt<IL_RGB888>, scale_factor_x = 131072 : ui32, scale_factor_y = 131072 : ui32} inputs(%0 : memref<1x768x512x1xui8, [@CMX_NN, 0]>) outputs(%alloc_0 : memref<1x512x512x3xui8, [@CMX_NN, 0]>) -> memref<1x512x512x3xui8, [@CMX_NN, 0]>
        %alloc_1 = memref.alloc() : memref<1x768x512x1xui8, [@CMX_NN, 0]>
        %1 = VPUIP.NNDMA inputs(%arg0 : memref<1x768x512x1xui8, @DDR>) outputs(%alloc_1 : memref<1x768x512x1xui8, [@CMX_NN, 0]>) -> memref<1x768x512x1xui8, [@CMX_NN, 0]>

        // M2I Task #2
        %alloc_2 = memref.alloc() : memref<1x512x512x3xui8, [@CMX_NN, 0]>
        %output_3 = VPUIP.M2ITask {chroma_out_reverse_channels, do_csc = true, do_norm = false, inFmt = #VPU.m2i_color_fmt<SP_NV12_8>, outFmt = #VPU.m2i_color_fmt<IL_RGB888>, scale_factor_x = 131072 : ui32, scale_factor_y = 131072 : ui32} inputs(%1 : memref<1x768x512x1xui8, [@CMX_NN, 0]>) outputs(%alloc_2 : memref<1x512x512x3xui8, [@CMX_NN, 0]>) -> memref<1x512x512x3xui8, [@CMX_NN, 0]>

	// Combine results together for result
        %alloc_4 = memref.alloc() : memref<2x512x512x3xui8, @DDR>
        %2 = VPUIP.SubView %alloc_4 [0, 0, 0, 0] [1, 512, 512, 3] : memref<2x512x512x3xui8, @DDR> to memref<1x512x512x3xui8, @DDR>
        %3 = VPUIP.NNDMA inputs(%output : memref<1x512x512x3xui8, [@CMX_NN, 0]>) outputs(%2 : memref<1x512x512x3xui8, @DDR>) -> memref<1x512x512x3xui8, @DDR>
        %4 = VPUIP.SubView %alloc_4 [1, 0, 0, 0] [1, 512, 512, 3] : memref<2x512x512x3xui8, @DDR> to memref<1x512x512x3xui8, @DDR>
        %5 = VPUIP.NNDMA inputs(%output_3 : memref<1x512x512x3xui8, [@CMX_NN, 0]>) outputs(%4 : memref<1x512x512x3xui8, @DDR>) -> memref<1x512x512x3xui8, @DDR>
        %result = VPUIP.ConcatView inputs(%3, %5 : memref<1x512x512x3xui8, @DDR>, memref<1x512x512x3xui8, @DDR>) outputs(%arg1 : memref<2x512x512x3xui8, @DDR>) -> memref<2x512x512x3xui8, @DDR>
        return %result : memref<2x512x512x3xui8, @DDR>
    }

    // CHECK:        profilingOutputsInfo : {
    // CHECK-NEXT:     DataInfo "m2i" : tensor<128xui8>
    // CHECK-NEXT:   }
    // CHECK:        func.func @main(%arg0: memref<1x768x512x1xui8, @DDR>, %arg1: memref<2x512x512x3xui8, @DDR>, %arg2: memref<128xui8>)
    // CHECK:        [[PROF_BUF:%.+]] = memref.alloc() : memref<128xui8, [@CMX_NN, 0]>

    // CHECK:        [[PROF_BUF_SLOT_1:%.+]] = VPUIP.SubView [[PROF_BUF]] [0] [64] : memref<128xui8, [@CMX_NN, 0]> to memref<64xui8, [@CMX_NN, 0]>
    // CHECK-NEXT:   [[OP_RESULT_1:%.*]], [[OP_RESULT_PROF_1:%.*]] = VPUIP.M2ITask
    // CHECK-SAME:   profilingMetadata = #VPUIP.M2IProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64>
    // CHECK-SAME:   profiling_data([[PROF_BUF_SLOT_1]] : memref<64xui8, [@CMX_NN, 0]>)

    // CHECK:        [[PROF_BUF_SLOT_2:%.+]] = VPUIP.SubView [[PROF_BUF]] [64] [64] : memref<128xui8, [@CMX_NN, 0]> to memref<64xui8, [@CMX_NN, 0]>
    // CHECK-NEXT:   [[OP_RESULT_2:%.*]], [[OP_RESULT_PROF_2:%.*]] = VPUIP.M2ITask
    // CHECK-SAME:   profilingMetadata = #VPUIP.M2IProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 1 : i64>
    // CHECK-SAME:   profiling_data([[PROF_BUF_SLOT_2]] : memref<64xui8, [@CMX_NN, 0]>)

    // CHECK:        [[PROF_OUTPUT:%.+]] = VPUIP.SubView %arg2 [0] [128] : memref<128xui8> to memref<128xui8>
    // CHECK:        [[CONCAT_PROF_RES:%.+]] = VPUIP.ConcatView inputs([[OP_RESULT_PROF_1]], [[OP_RESULT_PROF_2]] : memref<64xui8, [@CMX_NN, 0]>, memref<64xui8, [@CMX_NN, 0]>) outputs([[PROF_BUF]] : memref<128xui8, [@CMX_NN, 0]>) -> memref<128xui8, [@CMX_NN, 0]>

    // CHECK:        [[PROF_BUF_COPY:%.+]] = VPUIP.NNDMA {profiling_buffer_mgmt} inputs([[CONCAT_PROF_RES]] : memref<128xui8, [@CMX_NN, 0]>) outputs([[PROF_OUTPUT]] : memref<128xui8>) -> memref<128xui8>
    // CHECK:        [[CONCAT_PROF_RES_FULL:%.+]] = VPUIP.ConcatView inputs([[PROF_BUF_COPY]] : memref<128xui8>) outputs(%arg2 : memref<128xui8>) -> memref<128xui8>

    // CHECK:        return [[R1:%.+]], [[CONCAT_PROF_RES_FULL]] : memref<2x512x512x3xui8, @DDR>, memref<128xui8>

}
