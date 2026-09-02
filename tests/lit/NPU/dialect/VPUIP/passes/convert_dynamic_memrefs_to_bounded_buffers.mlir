//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-dynamic-memrefs-to-bounded-buffers %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @EmptyFunction
module @EmptyFunction {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "Input" : tensor<1x18x3x3xf32>
    } outputsInfo : {
        DataInfo "Output" : tensor<1x18x3x3xf32>
    }

    // CHECK: @main([[ARG0:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, {order = #NHWC}>, dynamic_shape=memref<4xsi32>>)
    // CHECK-SAME: -> !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, {order = #NHWC}>, dynamic_shape=memref<4xsi32>>
    func.func @main(%arg0: memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}>)
        -> memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}> {
        return %arg0 : memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}>
        // CHECK-NEXT: return [[ARG0]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @DynamicMemrefCopy
module @DynamicMemrefCopy {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "Input" : tensor<1x18x3x3xf32>
    } outputsInfo : {
        DataInfo "Output" : tensor<1x18x3x3xf32>
    }

    // CHECK: @main([[ARG0:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, {order = #NHWC}>, dynamic_shape=memref<4xsi32>>)
    // CHECK-SAME: -> !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, {order = #NHWC}, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>
    func.func @main(%arg0: memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}>)
        -> memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN> {
        %dim = arith.constant 3 : index
        %alloc = memref.alloc(%dim) : memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>
        // CHECK: [[DATA_ALLOC:%.+]] = memref.alloc() : memref<1x18x3x3xf32, {order = #NHWC}, @CMX_NN>
        // CHECK: [[SHAPE_ALLOC:%.+]] = memref.alloc() : memref<4xsi32, @CMX_NN>
        // CHECK: [[GROUP:%.+]] = VPUIP.GroupBoundedBuffer([[DATA_ALLOC]], [[SHAPE_ALLOC]])

        %copy = VPUIP.Copy
            inputs(%arg0: memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}>)
            outputs(%alloc: memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>)
            -> memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>
        // CHECK: [[COPY:%.+]] = VPUIP.Copy inputs([[ARG0]] {{.*}} outputs([[GROUP]]

        return %copy : memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>
        // CHECK-NEXT: return [[COPY]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @DynamicMemrefSwKernel
module @DynamicMemrefSwKernel {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "Input" : tensor<1x18x3x3xf32>
    } outputsInfo : {
        DataInfo "Output0" : tensor<1x18x3x3xf32>
        DataInfo "Output1" : tensor<1x18x3x3xf32>
    }

    module @VPU.SW {
        func.func nested @builtin_Convert(memref<*xf32>, memref<*xf16>) attributes {VPU.kernel_name = "convert", VPU.task_type = @COMPUTE}
        // CHECK: @builtin_Convert(memref<*xf32>, memref<*xsi32>, memref<*xf16>, memref<*xsi32>)
        // CHECK-SAME: attributes {VPU.kernel_name = "convert", VPU.task_type = @COMPUTE}

        func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    // CHECK: @main([[ARG0:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, {order = #NHWC}>, dynamic_shape=memref<4xsi32>>)
    // CHECK-SAME: -> (!VPUIP.BoundedBuffer<data=memref<1x18x3x3xf16, {order = #NHWC}>, dynamic_shape=memref<4xsi32>>,
    // CHECK-SAME:     !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf16, {order = #NHWC}>, dynamic_shape=memref<4xsi32>>)
    func.func @main(%arg0: memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}>)
        -> (memref<1x18x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}>,
            memref<1x18x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}>) {
        %dim = arith.constant 3 : index
        %cmx_alloc = memref.alloc(%dim) : memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>
        %cmx_input = VPUIP.Copy
            inputs(%arg0: memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}>)
            outputs(%cmx_alloc: memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>)
            -> memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>
        // CHECK: [[CMX_DATA_ALLOC:%.+]] = memref.alloc() : memref<1x18x3x3xf32, {order = #NHWC}, @CMX_NN>
        // CHECK: [[CMX_SHAPE_ALLOC:%.+]] = memref.alloc() : memref<4xsi32, @CMX_NN>
        // CHECK: [[CMX_GROUP:%.+]] = VPUIP.GroupBoundedBuffer([[CMX_DATA_ALLOC]], [[CMX_SHAPE_ALLOC]])
        // CHECK: [[CMX_INPUT:%.+]] = VPUIP.Copy inputs([[ARG0]] {{.*}} outputs([[CMX_GROUP]]

        %sw_out0 = memref.alloc(%dim) : memref<1x18x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>
        %sw_res0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert
            inputs(%cmx_input as %in: memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>)
            outputs(%sw_out0 as %out: memref<1x18x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>)
            on tile 0
            -> memref<1x18x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN> {
            VPUIP.SW.Kernel.run(%in, %out) :
                memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>,
                memref<1x18x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>
        }
        // CHECK: [[SW_DATA_ALLOC0:%.+]] = memref.alloc() : memref<1x18x3x3xf16, {order = #NHWC}, @CMX_NN>
        // CHECK: [[SW_SHAPE_ALLOC0:%.+]] = memref.alloc() : memref<4xsi32, @CMX_NN>
        // CHECK: [[SW_GROUP0:%.+]] = VPUIP.GroupBoundedBuffer([[SW_DATA_ALLOC0]], [[SW_SHAPE_ALLOC0]])
        // CHECK: [[SW0:%.+]] = VPUIP.SW.Kernel {{.*}} @VPU.SW::@builtin_Convert inputs([[CMX_INPUT]] {{.*}} outputs([[SW_GROUP0]]
        // CHECK-NEXT: VPUIP.SW.Kernel.run
        // CHECK-SAME:  !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, {order = #NHWC}, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>,
        // CHECK-SAME:  !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf16, {order = #NHWC}, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>

        %ddr_alloc0 = memref.alloc(%dim) : memref<1x18x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}>
        %ddr_output0 = VPUIP.Copy
            inputs(%sw_res0: memref<1x18x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>)
            outputs(%ddr_alloc0: memref<1x18x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}>)
            -> memref<1x18x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}>
        // CHECK: [[DDR_DATA_ALLOC0:%.+]] = memref.alloc() : memref<1x18x3x3xf16, {order = #NHWC}>
        // CHECK: [[DDR_SHAPE_ALLOC0:%.+]] = memref.alloc() : memref<4xsi32>
        // CHECK: [[DDR_GROUP0:%.+]] = VPUIP.GroupBoundedBuffer([[DDR_DATA_ALLOC0]], [[DDR_SHAPE_ALLOC0]])
        // CHECK: [[DDR_OUTPUT0:%.+]] = VPUIP.Copy inputs([[SW0]] {{.*}} outputs([[DDR_GROUP0]]

        %sw_out1 = memref.alloc(%dim) : memref<1x18x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>
        %sw_res1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert
            inputs(%cmx_input as %in: memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>)
            outputs(%sw_out1 as %out: memref<1x18x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>)
            on tile 0
            -> memref<1x18x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN> {
            VPUIP.SW.Kernel.run(%in, %out) :
                memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>,
                memref<1x18x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>
        }
        // CHECK: [[SW_DATA_ALLOC1:%.+]] = memref.alloc() : memref<1x18x3x3xf16, {order = #NHWC}, @CMX_NN>
        // CHECK: [[SW_SHAPE_ALLOC1:%.+]] = memref.alloc() : memref<4xsi32, @CMX_NN>
        // CHECK: [[SW_GROUP1:%.+]] = VPUIP.GroupBoundedBuffer([[SW_DATA_ALLOC1]], [[SW_SHAPE_ALLOC1]])
        // CHECK: [[SW1:%.+]] = VPUIP.SW.Kernel {{.*}} @VPU.SW::@builtin_Convert inputs([[CMX_INPUT]] {{.*}} outputs([[SW_GROUP1]]
        // CHECK-NEXT: VPUIP.SW.Kernel.run
        // CHECK-SAME:  !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, {order = #NHWC}, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>,
        // CHECK-SAME:  !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf16, {order = #NHWC}, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>

        %ddr_alloc1 = memref.alloc(%dim) : memref<1x18x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}>
        %ddr_output1 = VPUIP.Copy
            inputs(%sw_res1: memref<1x18x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>)
            outputs(%ddr_alloc1: memref<1x18x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}>)
            -> memref<1x18x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}>
        // CHECK: [[DDR_DATA_ALLOC1:%.+]] = memref.alloc() : memref<1x18x3x3xf16, {order = #NHWC}>
        // CHECK: [[DDR_SHAPE_ALLOC1:%.+]] = memref.alloc() : memref<4xsi32>
        // CHECK: [[DDR_GROUP1:%.+]] = VPUIP.GroupBoundedBuffer([[DDR_DATA_ALLOC1]], [[DDR_SHAPE_ALLOC1]])
        // CHECK: [[DDR_OUTPUT1:%.+]] = VPUIP.Copy inputs([[SW1]] {{.*}} outputs([[DDR_GROUP1]]

        return %ddr_output0, %ddr_output1 :
            memref<1x18x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}>,
            memref<1x18x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}>
        // CHECK: return [[DDR_OUTPUT0]], [[DDR_OUTPUT1]]
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DynamicReshape
module @DynamicReshape {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "Input" : tensor<1x3x10x16xf16>
    } outputsInfo : {
        DataInfo "Output" : tensor<1x1x2x240xf16>
    }

    module @VPU.SW {
        func.func nested @builtin_DynamicReshape(memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64)
            attributes {
                VPU.kernel_code = "dynamic_reshape.cpp",
                VPU.kernel_entry = "dynamic_reshape",
                VPU.kernel_name = "dynamic_reshape",
                VPU.task_type = @COMPUTE
            }
        // CHECK: func.func nested @builtin_DynamicReshape
        // CHECK-SAME: (memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, i64)
        func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    // CHECK: @main([[ARG0:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x3x10x16xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>)
    // CHECK-SAME: -> !VPUIP.BoundedBuffer<data=memref<1x1x2x240xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>
    func.func @main(%arg0: memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]>)
            -> memref<1x1x2x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 2, 240]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]> {
        %shape = const.Declare memref<4xsi32> = dense<[1, 1, 2, -1]> : tensor<4xsi64>, [#const.CastElemType<si32>]
        %shape_alloc = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
        %cmx_shape = VPUIP.Copy inputs(%shape : memref<4xsi32>) outputs(%shape_alloc : memref<4xsi32, [@CMX_NN, 0]>) -> memref<4xsi32, [@CMX_NN, 0]>
        // CHECK: [[SHAPE:%.+]] = const.Declare memref<4xsi32> = dense<[1, 1, 2, -1]>
        // CHECK: [[SHAPE_ALLOC:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
        // CHECK: [[CMX_SHAPE:%.+]] = VPUIP.Copy inputs([[SHAPE]] {{.*}} outputs([[SHAPE_ALLOC]]

        %c240 = arith.constant 240 : index
        %out = memref.alloc(%c240) : memref<1x1x2x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 2, 240]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]>
        // CHECK: [[DATA_ALLOC:%.+]] = memref.alloc() : memref<1x1x2x240xf16, [@CMX_NN, 0]>
        // CHECK: [[SHAPE_ALLOC:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
        // CHECK: [[OUT:%.+]] = VPUIP.GroupBoundedBuffer([[DATA_ALLOC]], [[SHAPE_ALLOC]])

        %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DynamicReshape
            inputs(%arg0 as %arg1: memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]>,
                %cmx_shape as %arg2: memref<4xsi32, [@CMX_NN, 0]>)
            outputs(%out as %arg3: memref<1x1x2x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 2, 240]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]>) on tile 0
            -> memref<1x1x2x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 2, 240]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]> {
            VPUIP.SW.Kernel.run {attrs = [0]} (%arg1, %arg2, %arg3)
                : memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]>,
                memref<4xsi32, [@CMX_NN, 0]>, memref<1x1x2x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 2, 240]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]>
        }
        // CHECK: [[RES:%.+]] = VPUIP.SW.Kernel {{.*}} @VPU.SW::@builtin_DynamicReshape inputs([[ARG0]] {{.*}} [[CMX_SHAPE]] {{.*}} outputs([[OUT]]
        // CHECK-SAME: -> !VPUIP.BoundedBuffer<data=memref<1x1x2x240xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>
        // CHECK-NEXT: VPUIP.SW.Kernel.run
        // CHECK-SAME: !VPUIP.BoundedBuffer<data=memref<1x3x10x16xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>,
        // CHECK-SAME: memref<4xsi32, [@CMX_NN, 0]>,
        // CHECK-SAME: !VPUIP.BoundedBuffer<data=memref<1x1x2x240xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>

        return %res : memref<1x1x2x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 2, 240]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]>
        // CHECK: return [[RES]]
    }
}

// -----

// Note: this is a DynamicLSTMSequence test case from bufferization to make sure
// the IR after the pass is similar to what bufferization would produce before

#C = affine_map<(d0) -> (d0)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!CmxInputDataDist = !VPUIP.DistributedBuffer<1x2x35x512xf16, #NCHW, @CMX_NN,
    {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64,
     uniform_distributed_segments, compute_shapes = [[1, 1, 35, 512], [1, 1, 35, 512]],
     compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]], memory_shapes = [[1, 1, 35, 512], [1, 1, 35, 512]],
     memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>
!CmxInputShapeDist = !VPUIP.DistributedBuffer<4xsi32, {order = #C}, @CMX_NN,
    {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments}>

!CmxInputStateDist = !VPUIP.DistributedBuffer<1x2x1x128xf16, #NCHW, @CMX_NN,
    {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
    memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>
!CmxInputRecWeightsDist = !VPUIP.DistributedBuffer<2x4x128x128xf16, #NCHW, @CMX_NN,
    {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 4, 128, 128], [1, 4, 128, 128]], compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]],
    memory_shapes = [[1, 4, 128, 128], [1, 4, 128, 128]], memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]}>
!CmxInputSyncBufferDist = !VPUIP.DistributedBuffer<1x1x1x2496xsi32, #NCHW, @CMX_NN,
    {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 2496], [1, 1, 1, 2496]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1, 1, 2496], [1, 1, 1, 2496]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

!CmxOutputDataDist = !VPUIP.DistributedBuffer<1x2x35x128xf16, #NCHW, @CMX_NN,
    {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 1, 35, 128], [1, 1, 35, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
    memory_shapes = [[1, 1, 35, 128], [1, 1, 35, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>
!CmxOutputShapeDist = !VPUIP.DistributedBuffer<4xsi32, {order = #C}, @CMX_NN,
    {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments}>

!CmxOutputStateDist = !VPUIP.DistributedBuffer<1x2x1x128xf16, #NCHW, @CMX_NN,
    {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
    memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>

// CHECK-LABEL: @DynamicLSTMSequence
module @DynamicLSTMSequence {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "Input1" : tensor<1x2x35x512xf16>
        DataInfo "Input2" : tensor<1x2x1x128xf16>
        DataInfo "Input3" : tensor<1x2x1x128xf16>
        DataInfo "Input4" : tensor<2x4x128x128xf16>
        DataInfo "Input5" : tensor<1x1x1x2496xsi32>
    } outputsInfo : {
        DataInfo "Output1" : tensor<1x2x35x128xf16>
        DataInfo "Output2" : tensor<1x2x1x128xf16>
        DataInfo "Output3" : tensor<1x2x1x128xf16>
    }

    module @VPU.SW {
        func.func nested @builtin_LSTMSequence(
                memref<*xf16, @CMX_NN>, memref<*xsi32, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>,
                memref<*xf16, @CMX_NN>, memref<*xsi32, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xsi32, @CMX_NN>,
                memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, none, i64, none)
            attributes {VPU.kernel_code = "lstm_dpu.cpp", VPU.kernel_entry = "lstm_dpu",
                        VPU.kernel_name = "lstm_dpu", VPU.task_type = @COMPUTE}
        func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    // Note: no change expected in the kernel function due to distributed types
    // CHECK: module @VPU.SW
    // CHECK: func.func nested @builtin_LSTMSequence
    // CHECK-SAME: (memref<*xf16, @CMX_NN>, memref<*xsi32, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>,
    // CHECK-SAME: memref<*xf16, @CMX_NN>, memref<*xsi32, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xsi32, @CMX_NN>,
    // CHECK-SAME: memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, none, i64, none)

    // CHECK: func.func @main([[IN0:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x2x35x512xf16>, dynamic_shape=memref<4xsi32>>,
    // CHECK-SAME: {{%.+}}: memref<1x2x1x128xf16>, {{%.+}}: memref<1x2x1x128xf16>, {{%.+}}: memref<2x4x128x128xf16>, {{%.+}}: memref<1x1x1x2496xsi32>)
    // CHECK-SAME: -> (!VPUIP.BoundedBuffer<data=memref<1x2x35x128xf16>, dynamic_shape=memref<4xsi32>>,
    // CHECK-SAME:    memref<1x2x1x128xf16>, memref<1x2x1x128xf16>
    func.func @main(%in0: memref<1x2x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 2, 35, 512]> : tensor<4xsi64>, order = #NCHW}>,
                    %in1: memref<1x2x1x128xf16>, %in2: memref<1x2x1x128xf16>, %in3: memref<2x4x128x128xf16>, %in4: memref<1x1x1x2496xsi32>)
            -> (memref<1x2x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 2, 35, 128]> : tensor<4xsi64>, order = #NCHW}>,
                memref<1x2x1x128xf16>, memref<1x2x1x128xf16>) {
        %0 = VPURT.AllocDistributed -> !CmxInputDataDist
        %1 = VPURT.AllocDistributed -> !CmxInputShapeDist
        %cmx_input_group = VPUIP.GroupBoundedBuffer(%0, %1) : !CmxInputDataDist, !CmxInputShapeDist
            -> !VPUIP.BoundedBuffer<data=!CmxInputDataDist, dynamic_shape=!CmxInputShapeDist>
        %cmx_input = VPUIP.Copy inputs(%in0 : memref<1x2x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 2, 35, 512]> : tensor<4xsi64>, order = #NCHW}>)
            outputs(%cmx_input_group : !VPUIP.BoundedBuffer<data=!CmxInputDataDist, dynamic_shape=!CmxInputShapeDist>)
            -> !VPUIP.BoundedBuffer<data=!CmxInputDataDist, dynamic_shape=!CmxInputShapeDist>
        // CHECK: [[CMX_IN0:%.+]] = VPUIP.Copy inputs([[IN0]]

        %cmx_input_hidden_state_alloc = VPURT.AllocDistributed -> !CmxInputStateDist
        %cmx_input_hidden_state = VPUIP.Copy inputs(%in1 : memref<1x2x1x128xf16>)
            outputs(%cmx_input_hidden_state_alloc : !CmxInputStateDist) -> !CmxInputStateDist

        %cmx_input_cell_state_alloc = VPURT.AllocDistributed -> !CmxInputStateDist
        %cmx_input_cell_state = VPUIP.Copy inputs(%in2 : memref<1x2x1x128xf16>)
            outputs(%cmx_input_cell_state_alloc : !CmxInputStateDist) -> !CmxInputStateDist

        %cmx_rec_weights_alloc = VPURT.AllocDistributed -> !CmxInputRecWeightsDist
        %cmx_rec_weights = VPUIP.Copy inputs(%in3 : memref<2x4x128x128xf16>)
            outputs(%cmx_rec_weights_alloc : !CmxInputRecWeightsDist) -> !CmxInputRecWeightsDist

        %cmx_sync_buf_alloc = VPURT.AllocDistributed -> !CmxInputSyncBufferDist
        %cmx_sync_buf = VPUIP.Copy inputs(%in4 : memref<1x1x1x2496xsi32>)
            outputs(%cmx_sync_buf_alloc : !CmxInputSyncBufferDist) -> !CmxInputSyncBufferDist

        %2 = VPURT.AllocDistributed -> !CmxOutputDataDist
        %3 = VPURT.AllocDistributed -> !CmxOutputShapeDist
        %cmx_output = VPUIP.GroupBoundedBuffer(%2, %3) : !CmxOutputDataDist, !CmxOutputShapeDist
            -> !VPUIP.BoundedBuffer<data=!CmxOutputDataDist, dynamic_shape=!CmxOutputShapeDist>

        %cmx_output_hidden_state_alloc = VPURT.AllocDistributed -> !CmxOutputStateDist
        %cmx_output_cell_state_alloc = VPURT.AllocDistributed -> !CmxOutputStateDist

        %results:3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 3, 0, 0>} @VPU.SW::@builtin_LSTMSequence
            inputs(%cmx_input as %arg0: !VPUIP.BoundedBuffer<data=!CmxInputDataDist, dynamic_shape=!CmxInputShapeDist>,
                   %cmx_input_hidden_state as %arg1: !CmxInputStateDist,
                   %cmx_input_cell_state as %arg2: !CmxInputStateDist,
                   %cmx_rec_weights as %arg3: !CmxInputRecWeightsDist,
                   %cmx_sync_buf as %arg4: !CmxInputSyncBufferDist)
            outputs(%cmx_output as %arg5: !VPUIP.BoundedBuffer<data=!CmxOutputDataDist, dynamic_shape=!CmxOutputShapeDist>,
                    %cmx_output_hidden_state_alloc as %arg6: !CmxOutputStateDist,
                    %cmx_output_cell_state_alloc as %arg7: !CmxOutputStateDist)
            -> (!VPUIP.BoundedBuffer<data=!CmxOutputDataDist, dynamic_shape=!CmxOutputShapeDist>, !CmxOutputStateDist, !CmxOutputStateDist) {
            VPUIP.SW.Kernel.run {attrs = [[-9223372036854775808, 1015], 2, [-1, -1]]}
                (%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7)
                : !VPUIP.BoundedBuffer<data=!CmxInputDataDist, dynamic_shape=!CmxInputShapeDist>,
                  !CmxInputStateDist, !CmxInputStateDist, !CmxInputRecWeightsDist, !CmxInputSyncBufferDist,
                  !VPUIP.BoundedBuffer<data=!CmxOutputDataDist, dynamic_shape=!CmxOutputShapeDist>,
                  !CmxOutputStateDist, !CmxOutputStateDist
        }
        // CHECK: [[SW_RES:%.+]]:3 = VPUIP.SW.Kernel {{.*}} @VPU.SW::@builtin_LSTMSequence inputs([[CMX_IN0]]

        %c35 = arith.constant 35 : index
        %alloc0 = memref.alloc(%c35) : memref<1x2x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 2, 35, 128]> : tensor<4xsi64>, order = #NCHW}>
        %output = VPUIP.Copy inputs(%results#0 : !VPUIP.BoundedBuffer<data=!CmxOutputDataDist, dynamic_shape=!CmxOutputShapeDist>)
            outputs(%alloc0 : memref<1x2x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 2, 35, 128]> : tensor<4xsi64>, order = #NCHW}>)
            -> memref<1x2x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 2, 35, 128]> : tensor<4xsi64>, order = #NCHW}>
        // CHECK: [[OUT_DATA:%.+]] = memref.alloc() : memref<1x2x35x128xf16>
        // CHECK: [[OUT_SHAPE:%.+]] = memref.alloc() : memref<4xsi32>
        // CHECK: [[OUT_GROUP:%.+]] = VPUIP.GroupBoundedBuffer([[OUT_DATA]], [[OUT_SHAPE]])
        // CHECK: [[OUT0:%.+]] = VPUIP.Copy inputs([[SW_RES]]#0 {{.*}} outputs([[OUT_GROUP]]

        %alloc1 = memref.alloc() : memref<1x2x1x128xf16>
        %output_hidden_state = VPUIP.Copy inputs(%results#1 : !CmxOutputStateDist) outputs(%alloc1 : memref<1x2x1x128xf16>) -> memref<1x2x1x128xf16>

        %alloc2 = memref.alloc() : memref<1x2x1x128xf16>
        %output_cell_state = VPUIP.Copy inputs(%results#2 : !CmxOutputStateDist) outputs(%alloc2 : memref<1x2x1x128xf16>) -> memref<1x2x1x128xf16>

        return %output, %output_hidden_state, %output_cell_state
            : memref<1x2x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 2, 35, 128]> : tensor<4xsi64>, order = #NCHW}>,
              memref<1x2x1x128xf16>, memref<1x2x1x128xf16>
        // CHECK: return [[OUT0]], {{%.+}}, {{%.+}}
    }
}

// -----

// Note: this is a ScaleParameterInterpolateLayer test case from
// bufferization to make sure the IR after the pass is similar to what
// bufferization would produce before

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ScaleParameterInterpolateLayer
module @ScaleParameterInterpolateLayer {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "Input" : tensor<1x3x4x6xf16>
        DataInfo "Scale" : tensor<2xf16>
        DataInfo "AuxBuffer" : tensor<1x1x1x1024xui8>
    } outputsInfo : {
        DataInfo "Output" : tensor<1x3x32x48xf16>
    }

    module @VPU.SW {
        func.func nested @builtin_InterpolateDMA(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>,
            memref<*xui8, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xui8, [@CMX_NN, 0]>,
            i64, i64, i64, i64, none, f64, f64, i64)
            attributes {VPU.kernel_code = "interpolate_dma.cpp", VPU.kernel_entry = "interpolate_dma",
                        VPU.kernel_name = "interpolate_dma", VPU.task_type = @COMPUTE}
        // Note: a new shape argument is added to the kernel
        // CHECK: func.func nested @builtin_InterpolateDMA(
        // CHECK-SAME: memref<*xf16, [@CMX_NN, 0]>
        // CHECK-SAME: memref<*xf16, [@CMX_NN, 0]>
        // CHECK-SAME: memref<*xui8, [@CMX_NN, 0]>
        // CHECK-SAME: memref<*xf16, [@CMX_NN, 0]>
        // CHECK-SAME: memref<*xsi32, [@CMX_NN, 0]>
        // CHECK-SAME: memref<*xui8, [@CMX_NN, 0]>
        // CHECK-SAME: ) attributes {VPU.kernel_code = "interpolate_dma.cpp", VPU.kernel_entry = "interpolate_dma"

        func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    // CHECK: func.func @main([[IN:%.+]]: memref<1x3x4x6xf16, [@CMX_NN, 0]>, [[SCALE:%.+]]: memref<2xf16, [@CMX_NN, 0]>,
    // CHECK-SAME:      [[AUX:%.+]]: memref<1x1x1x1024xui8, [@CMX_NN, 0]>)
    // CHECK-SAME: -> !VPUIP.BoundedBuffer<data=memref<1x3x32x48xf16>, dynamic_shape=memref<4xsi32>>
    func.func @main(%input: memref<1x3x4x6xf16, [@CMX_NN, 0]>, %scales: memref<2xf16, [@CMX_NN, 0]>,
                    %aux_buffer: memref<1x1x1x1024xui8, [@CMX_NN, 0]>)
            -> memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}> {
        %c32 = arith.constant 32 : index
        %c48 = arith.constant 48 : index
        %sw_alloc = memref.alloc(%c32, %c48)
            : memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]>
        // CHECK: [[SW_DATA:%.+]] = memref.alloc() : memref<1x3x32x48xf16, [@CMX_NN, 0]>
        // CHECK: [[SW_SHAPE:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
        // CHECK: [[SW_GROUP:%.+]] = VPUIP.GroupBoundedBuffer([[SW_DATA]], [[SW_SHAPE]])

        %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_InterpolateDMA
            inputs(%input as %arg0: memref<1x3x4x6xf16, [@CMX_NN, 0]>,
                   %scales as %arg1: memref<2xf16, [@CMX_NN, 0]>,
                   %aux_buffer as %arg2: memref<1x1x1x1024xui8, [@CMX_NN, 0]>)
            outputs(%sw_alloc as %arg3: memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]>,
                    %aux_buffer as %arg4: memref<1x1x1x1024xui8, [@CMX_NN, 0]>)
            on tile 0
            -> (memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]>,
                memref<1x1x1x1024xui8, [@CMX_NN, 0]>) {
            VPUIP.SW.Kernel.run {attrs = [1, 0, 2, 0, [2, 3], -7.500000e-01, 0.000000e+00, 8589934593]}
                (%arg0, %arg1, %arg2, %arg3, %arg4)
                : memref<1x3x4x6xf16, [@CMX_NN, 0]>, memref<2xf16, [@CMX_NN, 0]>, memref<1x1x1x1024xui8, [@CMX_NN, 0]>,
                memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]>,
                memref<1x1x1x1024xui8, [@CMX_NN, 0]>
        }
        // CHECK: [[SW_RES:%.+]]:2 = VPUIP.SW.Kernel {{.*}} @VPU.SW::@builtin_InterpolateDMA
        // CHECK-SAME:  inputs([[IN]] {{.*}} [[SCALE]] {{.*}} [[AUX]] {{.*}} outputs([[SW_GROUP]]
        // CHECK-SAME: -> (!VPUIP.BoundedBuffer<data=memref<1x3x32x48xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>
        // CHECK-NEXT: VPUIP.SW.Kernel.run
        // CHECK-SAME:  !VPUIP.BoundedBuffer<data=memref<1x3x32x48xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>

        %out_alloc = memref.alloc(%c32, %c48)
            : memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
        %out = VPUIP.Copy
            inputs(%results#0 : memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]>)
            outputs(%out_alloc : memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>)
            -> memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
        // CHECK: [[OUT_GROUP:%.+]] = VPUIP.GroupBoundedBuffer({{%.+}}, {{%.+}})
        // CHECK: [[OUT:%.+]] = VPUIP.Copy inputs([[SW_RES]]#0 {{.*}} outputs([[OUT_GROUP]]

        return %out : memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
        // CHECK: return [[OUT]]
    }
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: module @ReinterpretCastToBoundedBuffer
module @ReinterpretCastToBoundedBuffer {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "Input" : tensor<?x1x16xf16>
    } outputsInfo : {
        DataInfo "Output" : tensor<8x1x16xf16>
    }

    // CHECK: func.func @main([[IN:%.+]]: memref<?x1x16xf16>)
    // CHECK-SAME:  -> !VPUIP.BoundedBuffer<data=memref<8x1x16xf16>, dynamic_shape=memref<3xsi32>>
    func.func @main(%input: memref<?x1x16xf16>)
            -> memref<?x1x16xf16, {bounds = #const.OpaqueI64Elements<[8, 1, 16]> : tensor<3xsi64>, order = #CHW}> {
        %output = Core.ReinterpretCast(%input) : memref<?x1x16xf16>
            -> memref<?x1x16xf16, {bounds = #const.OpaqueI64Elements<[8, 1, 16]> : tensor<3xsi64>, order = #CHW}>
        return %output : memref<?x1x16xf16, {bounds = #const.OpaqueI64Elements<[8, 1, 16]> : tensor<3xsi64>, order = #CHW}>

        // CHECK: [[DATA_CAST:%.+]] = Core.ReinterpretCast([[IN]]) {{.*}}  -> memref<8x1x16xf16>
        // CHECK: [[SHAPE_ALLOC:%.+]] = memref.alloc() : memref<3xsi32>
        // CHECK: [[GROUP:%.+]] = VPUIP.GroupBoundedBuffer([[DATA_CAST]], [[SHAPE_ALLOC]])
        // CHECK: return [[GROUP]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: module @ReinterpretCastToBoundedBufferNHWC
module @ReinterpretCastToBoundedBufferNHWC {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "Input" : tensor<1x3x?x?xf16>
    } outputsInfo : {
        DataInfo "Output" : tensor<1x3x16x24xf16>
    }

    // CHECK: func.func @main([[IN:%.+]]: memref<1x3x?x?xf16>)
    // CHECK-SAME:  -> !VPUIP.BoundedBuffer<data=memref<1x3x16x24xf16, {order = #NHWC}>, dynamic_shape=memref<4xsi32>>
    func.func @main(%input: memref<1x3x?x?xf16>)
            -> memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 24]> : tensor<4xsi64>, order = #NHWC}> {
        %output = Core.ReinterpretCast(%input) : memref<1x3x?x?xf16>
            -> memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 24]> : tensor<4xsi64>, order = #NHWC}>
        return %output : memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 24]> : tensor<4xsi64>, order = #NHWC}>

        // CHECK: [[DATA_CAST:%.+]] = Core.ReinterpretCast([[IN]]) {{.*}}  -> memref<1x3x16x24xf16, {order = #NHWC}>
        // CHECK: [[SHAPE_ALLOC:%.+]] = memref.alloc() : memref<4xsi32>
        // CHECK: [[GROUP:%.+]] = VPUIP.GroupBoundedBuffer([[DATA_CAST]], [[SHAPE_ALLOC]])
        // CHECK: return [[GROUP]]
    }
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: module @ReinterpretCastFromBoundedBuffer
module @ReinterpretCastFromBoundedBuffer {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "Input" : tensor<?x1x16xf16>
    } outputsInfo : {
        DataInfo "Output" : tensor<8x1x16xf16>
    }

    // CHECK: func.func @main
    // CHECK-SAME: ([[IN:%.+]]: !VPUIP.BoundedBuffer<data=memref<8x1x16xf16>, dynamic_shape=memref<3xsi32>>)
    // CHECK-SAME:  -> memref<?x1x16xf16>
    func.func @main(
            %input: memref<?x1x16xf16, {bounds = #const.OpaqueI64Elements<[8, 1, 16]> : tensor<3xsi64>, order = #CHW}>)
            -> memref<?x1x16xf16> {
        %output = Core.ReinterpretCast(%input)
            : memref<?x1x16xf16, {bounds = #const.OpaqueI64Elements<[8, 1, 16]> : tensor<3xsi64>, order = #CHW}>
            -> memref<?x1x16xf16>
        return %output : memref<?x1x16xf16>

        // CHECK: [[IN_DATA:%.+]], {{%.+}} = VPUIP.UngroupBoundedBuffer([[IN]])
        // CHECK: [[DATA_CAST:%.+]] = Core.ReinterpretCast([[IN_DATA]]) {{.*}} -> memref<?x1x16xf16>
        // CHECK: return [[DATA_CAST]]
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Note: the case appears in the host compilation pipeline executing a SHAVE kernel that manages DMA

// CHECK-LABEL: module @ReinterpretCastFromSwKernelBoundedBuffer
module @ReinterpretCastFromSwKernelBoundedBuffer {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "Input" : tensor<1x1x1x8xf16>
    } outputsInfo : {
        DataInfo "Output" : tensor<1x1x1x8xf16>
    }

    module @VPU.SW {
        func.func nested @builtin_Atan(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_name = "atan", VPU.task_type = @COMPUTE}
        // CHECK: func.func nested @builtin_Atan(memref<*xf16>, memref<*xsi32>, memref<*xf16>, memref<*xsi32>)

        func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    // CHECK: func.func @main([[IN:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x1x1x8xf16>, dynamic_shape=memref<4xsi32>>)
    // CHECK-SAME:  -> memref<1x1x1x?xf16>
    func.func @main(%input: memref<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 8]> : tensor<4xsi64>, order = #NCHW}>)
            -> memref<1x1x1x?xf16> {
        %dim = arith.constant 8 : index
        %out_alloc = memref.alloc(%dim) : memref<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 8]> : tensor<4xsi64>, order = #NCHW}>
        // CHECK: [[DATA_ALLOC:%.+]] = memref.alloc() : memref<1x1x1x8xf16>
        // CHECK: [[SHAPE_ALLOC:%.+]] = memref.alloc() : memref<4xsi32>
        // CHECK: [[GROUP:%.+]] = VPUIP.GroupBoundedBuffer([[DATA_ALLOC]], [[SHAPE_ALLOC]])

        %sw_res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Atan
            inputs(%input as %in: memref<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 8]> : tensor<4xsi64>, order = #NCHW}>)
            outputs(%out_alloc as %out: memref<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 8]> : tensor<4xsi64>, order = #NCHW}>)
            on tile 0
            -> memref<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 8]> : tensor<4xsi64>, order = #NCHW}> {
            VPUIP.SW.Kernel.run(%in, %out) :
                memref<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 8]> : tensor<4xsi64>, order = #NCHW}>,
                memref<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 8]> : tensor<4xsi64>, order = #NCHW}>
        }
        // CHECK: [[SW:%.+]] = VPUIP.SW.Kernel {{.*}} @VPU.SW::@builtin_Atan inputs([[IN]] {{.*}} outputs([[GROUP]]
        // CHECK-NEXT: VPUIP.SW.Kernel.run
        // CHECK-SAME:  !VPUIP.BoundedBuffer<data=memref<1x1x1x8xf16>, dynamic_shape=memref<4xsi32>>,
        // CHECK-SAME:  !VPUIP.BoundedBuffer<data=memref<1x1x1x8xf16>, dynamic_shape=memref<4xsi32>>

        %result = Core.ReinterpretCast(%sw_res)
            : memref<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 8]> : tensor<4xsi64>, order = #NCHW}>
            -> memref<1x1x1x?xf16>
        return %result : memref<1x1x1x?xf16>

        // CHECK: [[SW_DATA:%.+]], {{%.+}} = VPUIP.UngroupBoundedBuffer([[SW]])
        // CHECK: [[DATA_CAST:%.+]] = Core.ReinterpretCast([[SW_DATA]]) {{.*}} -> memref<1x1x1x?xf16>
        // CHECK: return [[DATA_CAST]]
    }
}

// -----


#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: module @ShapeOf
module @ShapeOf {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "Input" : tensor<1x8x48x48xf16>
    } outputsInfo : {
        DataInfo "Output" : tensor<4xsi32>
    }

    module @VPU.SW {
        func.func nested @builtin_ShapeOf(memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>)
            attributes {VPU.kernel_code = "shape_of.cpp", VPU.kernel_entry = "shape_of",
                        VPU.kernel_name = "shape_of", VPU.task_type = @COMPUTE}
        // Note: ShapeOf is special as it drops dynamic memref and adopts shape
        // CHECK: func.func nested @builtin_ShapeOf(memref<*xsi32, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>)

        func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    // CHECK: func.func @main([[IN:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x8x48x48xf16>, dynamic_shape=memref<4xsi32>>)
    // CHECK-SAME:  -> memref<4xsi32>
    func.func @main(
            %in: memref<1x8x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 48, 48]> : tensor<4xsi64>, order = #NCHW}>)
            -> memref<4xsi32> {
        %c48 = arith.constant 48 : index
        %cmx_in_alloc = memref.alloc(%c48, %c48)
            : memref<1x8x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 48, 48]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]>
        %cmx_in = VPUIP.Copy
            inputs(%in : memref<1x8x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 48, 48]> : tensor<4xsi64>, order = #NCHW}>)
            outputs(%cmx_in_alloc : memref<1x8x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 48, 48]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]>)
            -> memref<1x8x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 48, 48]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]>
        // CHECK: [[CMX_DATA:%.+]] = memref.alloc() : memref<1x8x48x48xf16, [@CMX_NN, 0]>
        // CHECK: [[CMX_SHAPE:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
        // CHECK: [[CMX_GROUP:%.+]] = VPUIP.GroupBoundedBuffer([[CMX_DATA]], [[CMX_SHAPE]])
        // CHECK: [[CMX_IN:%.+]] = VPUIP.Copy inputs([[IN]] {{.*}} outputs([[CMX_GROUP]]

        %cmx_out_alloc = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
        // CHECK: [[CMX_OUT_ALLOC:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>

        %cmx_out = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ShapeOf
            inputs(%cmx_in as %arg1: memref<1x8x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 48, 48]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]>)
            outputs(%cmx_out_alloc as %arg2: memref<4xsi32, [@CMX_NN, 0]>) on tile 0
            -> memref<4xsi32, [@CMX_NN, 0]>
        {
            VPUIP.SW.Kernel.run(%arg1, %arg2) :
                memref<1x8x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 48, 48]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]>,
                memref<4xsi32, [@CMX_NN, 0]>
        }
        // CHECK: {{%.+}}, [[CMX_IN_SHAPE:%.+]] = VPUIP.UngroupBoundedBuffer([[CMX_IN]])
        // CHECK: [[CMX_OUT:%.+]] = VPUIP.SW.Kernel {{.*}} @VPU.SW::@builtin_ShapeOf
        // CHECK-SAME:  inputs([[CMX_IN_SHAPE]] {{.*}} outputs([[CMX_OUT_ALLOC]]
        // CHECK-NEXT: VPUIP.SW.Kernel.run
        // CHECK-SAME:  memref<4xsi32, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>

        %out_alloc = memref.alloc() : memref<4xsi32>
        %out = VPUIP.Copy inputs(%cmx_out : memref<4xsi32, [@CMX_NN, 0]>) outputs(%out_alloc : memref<4xsi32>) -> memref<4xsi32>
        // CHECK: [[OUT_ALLOC:%.+]] = memref.alloc() : memref<4xsi32>
        // CHECK: [[OUT:%.+]] = VPUIP.Copy inputs([[CMX_OUT]] {{.*}} outputs([[OUT_ALLOC]]

        return %out : memref<4xsi32>
        // CHECK: return [[OUT]]
    }
}
