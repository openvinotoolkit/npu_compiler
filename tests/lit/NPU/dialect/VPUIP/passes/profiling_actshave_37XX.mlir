//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file  --init-compiler="platform=%platform%" --act-shave-profiling %s | FileCheck %s
// REQUIRES: platform-NPU3720

// CHECK-LABEL: @ActShaveProfiling
module @ActShaveProfiling {
    net.NetworkInfo entryPoint : @main inputsInfo :  {
        DataInfo "input" : tensor<1x3x224x224xf32>
    } outputsInfo :  {
        DataInfo "output" : tensor<1x150528xf32>
    } profilingOutputsInfo :  {
    }

    VPURT.SW.Runtime entryPoint: @VPU.SW::@runtime stack_configuration: [4096, 4096, 4096, 4096]

    module @VPU.SW {
        func.func nested @builtin_SoftMax(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64) attributes {VPU.kernel_code = "softmaxx.cpp", VPU.kernel_entry = "softmax"}
        func.func nested @builtin_ConvertF32F16(memref<*xf32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert"}
        func.func nested @builtin_ConvertF16F32(memref<*xf32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert"}
        func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    func.func @main(%arg0: memref<1x3x224x224xf32>, %arg1: memref<1x150528xf32>) -> memref<1x150528xf32> {
        %0 = memref.alloc() : memref<1x3x224x224xf16, @DDR>
        %1 = memref.alloc() : memref<1x3x224x224xf32, [@CMX_NN, 0]>
        %2 = VPUIP.NNDMA inputs(%arg0 : memref<1x3x224x224xf32>) outputs(%1 : memref<1x3x224x224xf32, [@CMX_NN, 0]>) -> memref<1x3x224x224xf32, [@CMX_NN, 0]>
        %3 = memref.alloc() : memref<1x3x224x224xf16, [@CMX_NN, 0]>
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ConvertF32F16 inputs(%2 as %arg2: memref<1x3x224x224xf32, [@CMX_NN, 0]>) outputs(%3 as %arg3: memref<1x3x224x224xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x3x224x224xf16, [@CMX_NN, 0]>  {
        VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x3x224x224xf32, [@CMX_NN, 0]>, memref<1x3x224x224xf16, [@CMX_NN, 0]>
        }
        %4 = VPUIP.NNDMA inputs(%results : memref<1x3x224x224xf16, [@CMX_NN, 0]>) outputs(%0 : memref<1x3x224x224xf16, @DDR>) -> memref<1x3x224x224xf16, @DDR>
        %5 = VPUIP.GenericReshape inputs(%4 : memref<1x3x224x224xf16, @DDR>) -> memref<1x150528xf16, @DDR>
        %6 = memref.alloc() : memref<1x150528xf16, [@CMX_NN, 0]>
        %7 = VPUIP.NNDMA inputs(%5 : memref<1x150528xf16, @DDR>) outputs(%6 : memref<1x150528xf16, [@CMX_NN, 0]>) -> memref<1x150528xf16, [@CMX_NN, 0]>
        %8 = memref.alloc() : memref<1x150528xf16, [@CMX_NN, 0]>
        %results_0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%7 as %arg2: memref<1x150528xf16, [@CMX_NN, 0]>) outputs(%8 as %arg3: memref<1x150528xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x150528xf16, [@CMX_NN, 0]>  {
        VPUIP.SW.Kernel.run {attrs = [0]}(%arg2, %arg3) : memref<1x150528xf16, [@CMX_NN, 0]>, memref<1x150528xf16, [@CMX_NN, 0]>
        }
        %9 = memref.alloc() : memref<1x150528xf32, [@CMX_NN, 0]>
        %results_1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ConvertF16F32 inputs(%results_0 as %arg2: memref<1x150528xf16, [@CMX_NN, 0]>) outputs(%9 as %arg3: memref<1x150528xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x150528xf32, [@CMX_NN, 0]>  {
        VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x150528xf16, [@CMX_NN, 0]>, memref<1x150528xf32, [@CMX_NN, 0]>
        }
        %10 = VPUIP.NNDMA inputs(%results_1 : memref<1x150528xf32, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x150528xf32>) -> memref<1x150528xf32>
        return %10 : memref<1x150528xf32>
    }

    //CHECK:        profilingOutputsInfo
    //CHECK-NEXT:   DataInfo "actshave" : tensor<24xui32>
    //CHECK:        func.func @main([[ARG_0:%[^:]+]]: memref<1x3x224x224xf32>, [[ARG_1:%[^:]+]]: memref<1x150528xf32>, [[ARG_2:%[^:]+]]: memref<24xui32>) -> (memref<1x150528xf32>, memref<24xui32>)
    //CHECK:        [[VAR0:%.+]] = memref.alloc() : memref<24xui32, [@CMX_NN, 0]>
    //CHECK:        [[VAR1:%.+]] = VPUIP.SubView [[VAR0]] [0] [8] : memref<24xui32, [@CMX_NN, 0]> to memref<8xui32, [@CMX_NN, 0]>
    //CHECK-NEXT:   VPUIP.SW.Kernel
    //CHECK-SAME:   @VPU.SW::@builtin_ConvertF32F16
    //CHECK-SAME:   profiling_data([[VAR1]] : memref<8xui32, [@CMX_NN, 0]>)
    //CHECK:        [[VAR2:%.+]] = VPUIP.SubView [[VAR0]] [8] [8] : memref<24xui32, [@CMX_NN, 0]> to memref<8xui32, [@CMX_NN, 0]>
    //CHECK-NEXT:   VPUIP.SW.Kernel
    //CHECK-SAME:   @VPU.SW::@builtin_SoftMax
    //CHECK-SAME:   profiling_data([[VAR2]] : memref<8xui32, [@CMX_NN, 0]>)
    //CHECK:        [[VAR3:%.+]] = VPUIP.SubView [[VAR0]] [16] [8] : memref<24xui32, [@CMX_NN, 0]> to memref<8xui32, [@CMX_NN, 0]>
    //CHECK-NEXT:   VPUIP.SW.Kernel
    //CHECK-SAME:   @VPU.SW::@builtin_ConvertF16F32
    //CHECK-SAME:   profiling_data([[VAR3]] : memref<8xui32, [@CMX_NN, 0]>)
    //CHECK:        return [[R1:%.+]], [[R2:%.+]] : memref<1x150528xf32>, memref<24xui32>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

!type_CMX = memref<1x128x64x32xf16, {order = #NWHC}, [@CMX_NN, 0]>
!type_CMX_subview = memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>

!type_DDR = memref<1x128x64x32xf16, {order = #NWHC}, @DDR>

// CHECK-LABEL: @ActShaveProfilingMultitile
module @ActShaveProfilingMultitile {
    net.NetworkInfo entryPoint : @main inputsInfo :  {
        DataInfo "input" : tensor<1x128x64x32xf16>
    } outputsInfo :  {
        DataInfo "output" : tensor<1x128x64x32xf16>
    } profilingOutputsInfo :  {
    }

    module @VPU.SW {
        func.func nested @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
        func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    func.func @main(%arg0: !type_DDR, %arg5: !type_DDR) -> !type_DDR {
        %0 = memref.alloc() : !type_CMX
        %1 = VPUIP.NNDMA inputs(%arg0 : !type_DDR) outputs(%0 : !type_CMX) -> !type_CMX
        %2 = memref.alloc() : !type_CMX
        %3 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 64, 64, 32] : !type_CMX to !type_CMX_subview
        %4 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 64, 64, 32] : !type_CMX to !type_CMX_subview
        %5 = VPUIP.SubView %1 [0, 64, 0, 0] [1, 64, 64, 32] : !type_CMX to !type_CMX_subview
        %6 = VPUIP.SubView %2 [0, 64, 0, 0] [1, 64, 64, 32] : !type_CMX to !type_CMX_subview
        %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN inputs(%3 as %arg1: !type_CMX_subview, %5 as %arg2: !type_CMX_subview) outputs(%4 as %arg3: !type_CMX_subview, %6 as %arg4: !type_CMX_subview) on tile 0 -> (!type_CMX_subview, !type_CMX_subview){
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg1, %arg3) : !type_CMX_subview, !type_CMX_subview
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg2, %arg4) : !type_CMX_subview, !type_CMX_subview
        }
        %7 = VPUIP.ConcatView inputs(%results#0, %results#1 : !type_CMX_subview, !type_CMX_subview) outputs(%2 : !type_CMX) -> !type_CMX
        %9 = VPUIP.NNDMA inputs(%7 : !type_CMX) outputs(%arg5 : !type_DDR) -> !type_DDR
        return %9 : !type_DDR
    }

    //CHECK:        profilingOutputsInfo
    //CHECK-NEXT:   DataInfo "actshave" : tensor<16xui32>
    //CHECK:        func.func @main([[ARG_0:%[^:]+]]: memref<1x128x64x32xf16, {order = #NWHC}, @DDR>, [[ARG_1:%[^:]+]]: memref<1x128x64x32xf16, {order = #NWHC}, @DDR>, [[ARG_2:%[^:]+]]: memref<16xui32>) -> (memref<1x128x64x32xf16, {order = #NWHC}, @DDR>, memref<16xui32>)
    //CHECK:        [[PROF_BUF:%.+]] = memref.alloc() : memref<16xui32, [@CMX_NN, 0]>
    //CHECK:        [[PROF_BUF_SLOT:%.+]] = VPUIP.SubView [[PROF_BUF]] [0] [16] : memref<16xui32, [@CMX_NN, 0]> to memref<16xui32, [@CMX_NN, 0]>

    //CHECK-NEXT:   [[OP_RESULT:%.+]], [[OP_RESULT_PROF:%.+]] = VPUIP.SW.Kernel
    //CHECK-SAME:   @VPU.SW::@builtin_MVN
    //CHECK-SAME:   profiling_data([[PROF_BUF_SLOT]] : memref<16xui32, [@CMX_NN, 0]>)
    //CHECK-NEXT:   VPUIP.SW.Kernel.run
    //CHECK-NEXT:   VPUIP.SW.Kernel.run

    //CHECK:        [[PROF_OUTPUT:%.+]] = VPUIP.SubView [[ARG_2]] [0] [16] : memref<16xui32> to memref<16xui32
    //CHECK:        [[CONCAT_PROF_RES:%.+]] = VPUIP.ConcatView inputs([[OP_RESULT_PROF]] : memref<16xui32, [@CMX_NN, 0]>) outputs([[PROF_BUF]] : memref<16xui32, [@CMX_NN, 0]>) -> memref<16xui32, [@CMX_NN, 0]>

    //CHECK:        [[PROF_BUF_COPY:%.+]] = VPUIP.NNDMA <{profiling_buffer_mgmt}> inputs([[CONCAT_PROF_RES]] : memref<16xui32, [@CMX_NN, 0]>) outputs([[PROF_OUTPUT]] : memref<16xui32>) -> memref<16xui32>
    //CHECK:        [[CONCAT_PROF_RES_FULL:%.+]] = VPUIP.ConcatView inputs([[PROF_BUF_COPY]] : memref<16xui32>) outputs([[ARG_2]] : memref<16xui32>) -> memref<16xui32>

    //CHECK:        return [[R1:%.+]], [[CONCAT_PROF_RES_FULL]] : memref<1x128x64x32xf16, {order = #NWHC}, @DDR>, memref<16xui32>

}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

!typeCmxDistributed = !VPUIP.DistributedBuffer<
    1x4x512x1xf16, #NCWH, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!type_CMX_memref = memref<1x4x512x1xf16, {order = #NCWH}, @CMX_NN>

!type_DDR  = memref<1x4x512x1xf16, {order = #NCWH}, @DDR>

// CHECK-LABEL: @ActShaveProfilingMulticluster
module @ActShaveProfilingMulticluster {
    net.NetworkInfo entryPoint : @main inputsInfo :  {
        DataInfo "input" : tensor<1x4x512x1xf16>
    } outputsInfo :  {
        DataInfo "output" : tensor<1x4x512x1xf16>
    } profilingOutputsInfo :  {
    }

    module @VPU.SW {
        func.func nested @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
        func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    func.func @main(%arg0: !type_DDR, %arg5: !type_DDR) -> !type_DDR {

        %1 = VPURT.AllocDistributed -> !typeCmxDistributed

        %2 = VPUIP.NNDMA inputs(%arg0 : !type_DDR) outputs(%1 : !typeCmxDistributed) -> !typeCmxDistributed

        %4 = VPURT.AllocDistributed -> !typeCmxDistributed
        %5 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs(%2 as %arg3: !type_CMX_memref) outputs(%4 as %arg4: !type_CMX_memref) on tile 0 -> !typeCmxDistributed {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg3, %arg4) : !type_CMX_memref, !type_CMX_memref
        }

        %6 = VPUIP.NNDMA inputs(%5 : !typeCmxDistributed) outputs(%arg5 : !type_DDR) -> !type_DDR

        return  %6 : !type_DDR
    }
    //CHECK:        profilingOutputsInfo
    //CHECK-NEXT:   DataInfo "actshave" : tensor<16xui32>
    //CHECK:         @main([[ARG_0:%[^:]+]]: memref<1x4x512x1xf16, {order = #NCWH}, @DDR>, [[ARG_1:%[^:]+]]: memref<1x4x512x1xf16, {order = #NCWH}, @DDR>, [[ARG_2:%[^:]+]]: memref<16xui32>) -> (memref<1x4x512x1xf16, {order = #NCWH}, @DDR>, memref<16xui32>)
    //CHECK:        [[PROF_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16xui32, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}>
    //CHECK:        [[PROF_BUF_SLOT:%.+]] = VPUIP.SubView [[PROF_BUF]] [0] [16] : !VPUIP.DistributedBuffer<16xui32, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}> to !VPUIP.DistributedBuffer<16xui32, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}>

    //CHECK:       [[OP_RESULT:%.+]], [[OP_RESULT_PROF:%.+]] = VPUIP.SW.Kernel
    //CHECK-SAME:       @VPU.SW::@builtin_MVN
    //CHECK-SAME:       profiling_data([[PROF_BUF_SLOT]] : !VPUIP.DistributedBuffer<16xui32, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}>) on tile 0 -> (!VPUIP.DistributedBuffer<1x4x512x1xf16, #NCWH, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<16xui32, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}>)
    //CHECK-NEXT:           VPUIP.SW.Kernel.run

    //CHECK:        [[PROF_OUTPUT:%.+]] = VPUIP.SubView [[ARG_2]] [0] [16] : memref<16xui32> to memref<16xui32
    //CHECK:        [[CONCAT_PROF_RES:%.+]] = VPUIP.ConcatView
    //CHECK-SAME:       inputs([[OP_RESULT_PROF]] : !VPUIP.DistributedBuffer<16xui32, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}>)
    //CHECK-SAME:       outputs([[PROF_BUF]] : !VPUIP.DistributedBuffer<16xui32, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}>)

    //CHECK:       [[NCE_RES_COPY:%.+]] = VPUIP.NNDMA <{profiling_buffer_mgmt}> inputs([[CONCAT_PROF_RES]] : !VPUIP.DistributedBuffer<16xui32, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}>) outputs([[PROF_OUTPUT]] : memref<16xui32>) -> memref<16xui32>

    //CHECK:        [[CONCAT_PROF_RES_FULL:%.+]] = VPUIP.ConcatView inputs([[NCE_RES_COPY]] : memref<16xui32>) outputs([[ARG_2]] : memref<16xui32>) -> memref<16xui32>

    //CHECK:        return [[R1:%.+]], [[CONCAT_PROF_RES_FULL]] : memref<1x4x512x1xf16, {order = #NCWH}, @DDR>, memref<16xui32>

}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

!type_CMX_Distributed = !VPUIP.DistributedBuffer<
    1x128x64x32xf16, #NWHC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!type_CMX_Distributed_subview = !VPUIP.DistributedBuffer<
    1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!type_CMX = memref<1x128x64x32xf16, {order = #NWHC}, @CMX_NN>

!type_CMX_subview = memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>

!type_DDR = memref<1x128x64x32xf16, {order = #NWHC}, @DDR>

// CHECK-LABEL: @ActShaveProfilingMulticlusterMultitile
module @ActShaveProfilingMulticlusterMultitile {
    net.NetworkInfo entryPoint : @main inputsInfo :  {
        DataInfo "input" : tensor<1x128x64x32xf16>
    } outputsInfo :  {
        DataInfo "output" : tensor<1x128x64x32xf16>
    } profilingOutputsInfo :  {
    }

    module @VPU.SW {
        func.func nested @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
        func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    func.func @main(%arg0: !type_DDR, %arg9: !type_DDR) -> !type_DDR {
        %0 = VPURT.AllocDistributed -> !type_CMX_Distributed
        %1 = VPUIP.NNDMA inputs(%arg0 : !type_DDR) outputs(%0 : !type_CMX_Distributed) -> !type_CMX_Distributed
        %2 = VPUIP.SubView %1 [0, 64, 0, 0] [1, 64, 64, 32] : !type_CMX_Distributed to !type_CMX_Distributed_subview
        %3 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 64, 64, 32] : !type_CMX_Distributed to !type_CMX_Distributed_subview
        %4 = VPURT.AllocDistributed -> !type_CMX_Distributed
        %5 = VPUIP.SubView %4 [0, 64, 0, 0] [1, 64, 64, 32] : !type_CMX_Distributed to !type_CMX_Distributed_subview
        %6 = VPUIP.SubView %4 [0, 0, 0, 0] [1, 64, 64, 32] : !type_CMX_Distributed to !type_CMX_Distributed_subview
        %7:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN inputs(%3 as %arg5: !type_CMX_Distributed_subview, %2 as %arg6: !type_CMX_Distributed_subview) outputs(%6 as %arg7: !type_CMX_Distributed_subview, %5 as %arg8: !type_CMX_Distributed_subview) on tile 0 -> (!type_CMX_Distributed_subview, !type_CMX_Distributed_subview){
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg5, %arg7) : !type_CMX_Distributed_subview, !type_CMX_Distributed_subview
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg6, %arg8) : !type_CMX_Distributed_subview, !type_CMX_Distributed_subview
        }
        %8 = VPUIP.ConcatView inputs(%7#0, %7#1 : !type_CMX_Distributed_subview, !type_CMX_Distributed_subview) outputs(%4 : !type_CMX_Distributed) -> !type_CMX_Distributed

        %9 = VPUIP.NNDMA inputs(%8 : !type_CMX_Distributed) outputs(%arg9 : !type_DDR) -> !type_DDR
        return %9 : !type_DDR
    }

    //CHECK:        profilingOutputsInfo
    //CHECK-NEXT:   DataInfo "actshave" : tensor<32xui32>
    //CHECK:         @main([[ARG_0:%[^:]+]]: memref<1x128x64x32xf16, {order = #NWHC}, @DDR>, [[ARG_1:%[^:]+]]: memref<1x128x64x32xf16, {order = #NWHC}, @DDR>, [[ARG_2:%[^:]+]]: memref<32xui32>) -> (memref<1x128x64x32xf16, {order = #NWHC}, @DDR>, memref<32xui32>)
    //CHECK:        [[PROF_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<32xui32, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}>
    //CHECK:        [[PROF_BUF_SLOT:%.+]] = VPUIP.SubView [[PROF_BUF]] [0] [32] : !VPUIP.DistributedBuffer<32xui32, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}> to !VPUIP.DistributedBuffer<32xui32, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}>

    //CHECK:       [[OP_RESULT:%.+]], [[OP_RESULT_PROF:%.+]] = VPUIP.SW.Kernel
    //CHECK-SAME:       @VPU.SW::@builtin_MVN
    //CHECK-SAME:       profiling_data([[PROF_BUF_SLOT]] : !VPUIP.DistributedBuffer<32xui32, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}>
    //CHECK-NEXT:           VPUIP.SW.Kernel.run
    //CHECK-NEXT:           VPUIP.SW.Kernel.run

    //CHECK:        [[PROF_OUTPUT:%.+]] = VPUIP.SubView [[ARG_2]] [0] [32] : memref<32xui32> to memref<32xui32
    //CHECK:        [[CONCAT_PROF_RES:%.+]] = VPUIP.ConcatView
    //CHECK-SAME:       inputs([[OP_RESULT_PROF]] : !VPUIP.DistributedBuffer<32xui32, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}>)
    //CHECK-SAME:       outputs([[PROF_BUF]] : !VPUIP.DistributedBuffer<32xui32, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}>)

    //CHECK:       [[NCE_RES_COPY:%.+]] = VPUIP.NNDMA <{profiling_buffer_mgmt}> inputs([[CONCAT_PROF_RES]] : !VPUIP.DistributedBuffer<32xui32, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64, uniform_distributed_segments}>) outputs([[PROF_OUTPUT]] : memref<32xui32>) -> memref<32xui32>

    //CHECK:        [[CONCAT_PROF_RES_FULL:%.+]] = VPUIP.ConcatView inputs([[NCE_RES_COPY]] : memref<32xui32>) outputs([[ARG_2]] : memref<32xui32>) -> memref<32xui32>

    //CHECK:        return [[R1:%.+]], [[CONCAT_PROF_RES_FULL]] : memref<1x128x64x32xf16, {order = #NWHC}, @DDR>, memref<32xui32>

}

// -----

// CHECK-LABEL: @ActShaveProfilingDynamicShapes
module @ActShaveProfilingDynamicShapes {
    net.NetworkInfo entryPoint : @main inputsInfo :  {
        DataInfo "input" : tensor<1x88xsi32>
    } outputsInfo :  {
        DataInfo "output" : tensor<2x88xsi32>
    } profilingOutputsInfo :  {
    }

    module @VPU.SW {
        func.func nested @builtin_NonZero(memref<*xsi32, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "non_zero.cpp", VPU.kernel_entry = "non_zero", VPU.task_type = @COMPUTE}
        func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    func.func @main(%arg0: memref<1x88xsi32, @DDR>, %arg5: memref<2x88xsi32, @DDR>) -> memref<2x88xsi32, @DDR> {
        %1 = memref.alloc() : memref<1x88xsi32, [@CMX_NN, 0]>

        %2 = VPUIP.NNDMA inputs(%arg0 : memref<1x88xsi32, @DDR>) outputs(%1 : memref<1x88xsi32, [@CMX_NN, 0]>) -> memref<1x88xsi32, [@CMX_NN, 0]>

        %3 = memref.alloc() : memref<2x88xsi32, [@CMX_NN, 0]>
        %4 = memref.alloc() : memref<2xsi32, [@CMX_NN, 0]>

        %5, %dynamicOutputShapes = VPUIP.SW.Kernel {dynamicInputShapesMap = array<i32: -1>, dynamicOutputShapesMap = array<i32: 0>, resultSegmentSizes = array<i32: 1, 1, 0>} @VPU.SW::@builtin_NonZero inputs(%2 as %arg3: memref<1x88xsi32, [@CMX_NN, 0]>) outputs(%3 as %arg4: memref<2x88xsi32, [@CMX_NN, 0]>) dynamicOutputShapes(%4 : memref<2xsi32, [@CMX_NN, 0]>) on tile 0 -> (memref<2x88xsi32, [@CMX_NN, 0]>, memref<2xsi32, [@CMX_NN, 0]>){
            VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x88xsi32, [@CMX_NN, 0]>, memref<2x88xsi32, [@CMX_NN, 0]>
        }

        %6 = VPUIP.NNDMA inputs(%5 : memref<2x88xsi32, [@CMX_NN, 0]>) outputs(%arg5 : memref<2x88xsi32, @DDR>) -> memref<2x88xsi32, @DDR>

        return  %6 : memref<2x88xsi32, @DDR>
    }
    //CHECK:        profilingOutputsInfo
    //CHECK-NEXT:   DataInfo "actshave" : tensor<8xui32>
    //CHECK:         @main([[ARG_0:%[^:]+]]: memref<1x88xsi32, @DDR>, [[ARG_1:%[^:]+]]: memref<2x88xsi32, @DDR>, [[ARG_2:%[^:]+]]: memref<8xui32>) -> (memref<2x88xsi32, @DDR>, memref<8xui32>)
    //CHECK:        [[PROF_BUF:%.+]] = memref.alloc() : memref<8xui32, [@CMX_NN, 0]>
    //CHECK:        [[PROF_BUF_SLOT:%.+]] = VPUIP.SubView [[PROF_BUF]] [0] [8] : memref<8xui32, [@CMX_NN, 0]> to memref<8xui32, [@CMX_NN, 0]>

    //CHECK:        [[OP_RESULT:%.+]], [[OP_DYN_OUTPUT_SHAPES:%.+]], [[OP_RESULT_PROF:%.+]] = VPUIP.SW.Kernel
    //CHECK-SAME:       @VPU.SW::@builtin_NonZero
    //CHECK-SAME:       profiling_data([[PROF_BUF_SLOT]] : memref<8xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<2x88xsi32, [@CMX_NN, 0]>, memref<2xsi32, [@CMX_NN, 0]>, memref<8xui32, [@CMX_NN, 0]>)
    //CHECK-NEXT:           VPUIP.SW.Kernel.run

    //CHECK:        [[PROF_OUTPUT:%.+]] = VPUIP.SubView [[ARG_2]] [0] [8] : memref<8xui32> to memref<8xui32>
    //CHECK:        [[CONCAT_PROF_RES:%.+]] = VPUIP.ConcatView
    //CHECK-SAME:       inputs([[OP_RESULT_PROF]] : memref<8xui32, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[PROF_BUF]] : memref<8xui32, [@CMX_NN, 0]>)

    //CHECK:        [[NCE_RES_COPY:%.+]] = VPUIP.NNDMA <{profiling_buffer_mgmt}> inputs([[CONCAT_PROF_RES]] : memref<8xui32, [@CMX_NN, 0]>) outputs([[PROF_OUTPUT]] : memref<8xui32>) -> memref<8xui32>

    //CHECK:        [[CONCAT_PROF_RES_FULL:%.+]] = VPUIP.ConcatView inputs([[NCE_RES_COPY]] : memref<8xui32>) outputs([[ARG_2]] : memref<8xui32>) -> memref<8xui32>

    //CHECK:        return [[R1:%.+]], [[CONCAT_PROF_RES_FULL]] : memref<2x88xsi32, @DDR>, memref<8xui32>

}
