//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
!MemRef1 = memref<1x128x64x32xf16, #NWHC>
!Distributed0 = !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
!Distributed1 = !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
!Distributed2 = !VPUIP.DistributedBuffer<1x62x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
!MemRef0 = memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>
!MemRef2 = memref<1x62x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>

module @VPU.SW {
func.func private @builtin_MVN(!Distributed1, !Distributed2, !Distributed1, !Distributed2) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @ParsePrintDistributedBuffer(%arg0: !MemRef1) -> !MemRef1 {
    %0 = VPURT.AllocDistributed -> !Distributed0
    %1 = VPURT.AllocDistributed -> !Distributed0
    %2 = memref.alloc() : !MemRef1
    %token, %results = async.execute -> !async.value<!Distributed0> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64} {
        %4 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: !MemRef1) outputs(%0 as %arg2: memref<1x128x64x32xf16, #NWHC, @CMX_NN>) -> !Distributed0 {
        %5 = VPUIP.Copy inputs(%arg1 : !MemRef1) outputs(%arg2 : memref<1x128x64x32xf16, #NWHC, @CMX_NN>) -> memref<1x128x64x32xf16, #NWHC, @CMX_NN>
        }
        async.yield %4 : !Distributed0
    }
    %token_0, %results_1:2 = async.execute [%token] (%results as %arg1: !async.value<!Distributed0>) -> (!async.value<!Distributed1>, !async.value<!Distributed2>) attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
        %4 = VPUIP.SubView %arg1 [0, 62, 0, 0] [1, 62, 64, 32] : !Distributed0 to !Distributed2
        %5 = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 64, 64, 32] : !Distributed0 to !Distributed1
        %6 = VPUIP.SubView %1 [0, 62, 0, 0] [1, 62, 64, 32] : !Distributed0 to !Distributed2
        %7 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 64, 64, 32] : !Distributed0 to !Distributed1
        %8:2 = VPUIP.SW.Kernel {
                resultSegmentSizes = array<i32: 2, 0, 0>
            } @VPU.SW::@builtin_MVN
            inputs(%5 as %arg6: !Distributed1, %4 as %arg7: !Distributed2)
            outputs(%7 as %arg8: !Distributed1, %6 as %arg9: !Distributed2) on tile 0 -> (!Distributed1, !Distributed2){
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg6, %arg8) : !Distributed1, !Distributed1
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg7, %arg9) : !Distributed2, !Distributed2
        }
        async.yield %8#0, %8#1 : !Distributed1, !Distributed2
    }
    %token_2, %results_3 = async.execute [%token_0] (%results_1#0 as %arg1: !async.value<!Distributed1>, %results_1#1 as %arg2: !async.value<!Distributed2>) -> !async.value<!MemRef1> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 2 : i64} {
        %4 = VPUIP.ConcatView inputs(%arg1, %arg2 : !Distributed1, !Distributed2) outputs(%1 : !Distributed0) -> !Distributed0
        %5 = VPUIP.NCEClusterTiling inputs(%4 as %arg3: memref<1x128x64x32xf16, #NWHC, @CMX_NN>) outputs(%2 as %arg4: !MemRef1) -> !MemRef1 {
        %6 = VPUIP.Copy inputs(%arg3 : memref<1x128x64x32xf16, #NWHC, @CMX_NN>) outputs(%arg4 : !MemRef1) -> !MemRef1
        }
        async.yield %5 : !MemRef1
    }
    %3 = async.await %results_3 : !async.value<!MemRef1>

    return %3 : !MemRef1

    //CHECK:        %token_0, [[RESULTS:%.*]] = async.execute [%token]
    //CHECK-SAME:        attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
    //CHECK:                [[RESULTS_SW:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN
    //CHECK-SAME:               inputs({{[^:]+}} as %arg2: !VPUIP.DistributedBuffer
    //CHECK-SAME:                      {{[^:]+}} as %arg3: !VPUIP.DistributedBuffer
    //CHECK-SAME:               outputs({{[^:]+}} as %arg4: !VPUIP.DistributedBuffer
    //CHECK-SAME:                       {{[^:]+}} as %arg5: !VPUIP.DistributedBuffer
    //CHECK:                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg2, %arg4) : !VPUIP.DistributedBuffer
    //CHECK:                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg3, %arg5) : !VPUIP.DistributedBuffer
    //CHECK:              async.yield [[RESULTS_SW]]#0, [[RESULTS_SW]]#1

}

// -----

module {
  module @VPU.SW {
    func.func private @builtin_dummy(memref<*xf16>, memref<*xf16>, i64, i64) attributes {VPU.kernel_code = "dummy.cpp", VPU.kernel_entry = "dummy"}
  }
  // CHECK-LABEL: @UseBoundedBufferAsSWKernelInput
  func.func @UseBoundedBufferAsSWKernelInput() -> !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>> {
    %alloc_0 = memref.alloc() : memref<1x8x384x384xf16>
    %alloc_1 = memref.alloc() : memref<4xsi32>
    %input = VPUIP.GroupBoundedBuffer(%alloc_0, %alloc_1) : memref<1x8x384x384xf16>, memref<4xsi32>
        -> !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>>
    %alloc_2 = memref.alloc() : memref<1x8x384x384xf16>
    %alloc_3 = memref.alloc() : memref<4xsi32>
    %output = VPUIP.GroupBoundedBuffer(%alloc_2, %alloc_3) : memref<1x8x384x384xf16>, memref<4xsi32>
        -> !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>>
    %results = VPUIP.SW.Kernel {
            resultSegmentSizes = array<i32: 1, 0, 0>
        }
        @VPU.SW::@builtin_dummy
            inputs(%input as %arg0: !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>>)
            outputs(%output as %arg1: !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>>)
                -> !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>>{
            VPUIP.SW.Kernel.run(%arg0, %arg1) :
                !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>>,
                !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>>
    }
    return %results : !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>>
  }

    //CHECK:        [[ALOC_0:%.*]] = memref.alloc() : memref<1x8x384x384xf16>
    //CHECK:        [[ALOC_1:%.*]] = memref.alloc() : memref<4xsi32>
    //CHECK:        [[SW_OP_INPUT:%.*]] = VPUIP.GroupBoundedBuffer([[ALOC_0]], [[ALOC_1]])
    //CHECK:        [[ALOC_2:%.*]] = memref.alloc() : memref<1x8x384x384xf16>
    //CHECK:        [[ALOC_3:%.*]] = memref.alloc() : memref<4xsi32>
    //CHECK:        [[SW_OP_OUTPUT:%.*]] = VPUIP.GroupBoundedBuffer([[ALOC_2]], [[ALOC_3]])
    //CHECK:        [[RESULT:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
    //CHECK-SAME:   inputs([[SW_OP_INPUT]] as %arg0: !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>>)
    //CHECK-SAME:   outputs([[SW_OP_OUTPUT]] as %arg1: !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>>)
    //CHECK:        return [[RESULT]] : !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>>
}

// -----

module {
  module @VPU.SW {
    func.func private @builtin_dummy(memref<*xf16>, memref<*xf16>, i64, i64) attributes {VPU.kernel_code = "dummy.cpp", VPU.kernel_entry = "dummy"}
  }
  // CHECK-LABEL: @SWKernelDynamicInputs
  func.func @SWKernelDynamicInputs() -> (memref<1x8x384x384xf16>, memref<4xsi32>) {
    %alloc_0 = memref.alloc() : memref<1x8x384x384xf16>
    %alloc_1 = memref.alloc() : memref<4xsi32>

    %alloc_2 = memref.alloc() : memref<1x8x384x384xf16>
    %alloc_3 = memref.alloc() : memref<4xsi32>

    %results:2 = VPUIP.SW.Kernel {
            dynamicInputShapesMap = array<i32: 0>,
            dynamicOutputShapesMap = array<i32: 0>,
            resultSegmentSizes = array<i32: 2, 0, 0>
        }
        @VPU.SW::@builtin_dummy
            inputs(%alloc_0 as %arg0: memref<1x8x384x384xf16>)
            dynamicInputShapes(%alloc_1 : memref<4xsi32>)
            outputs(%alloc_2 as %arg1: memref<1x8x384x384xf16>)
            dynamicOutputShapes(%alloc_3 : memref<4xsi32>)
                -> (memref<1x8x384x384xf16>, memref<4xsi32>) {
            VPUIP.SW.Kernel.run(%arg0, %arg1) :
                memref<1x8x384x384xf16>,
                memref<1x8x384x384xf16>
    }
    return %results#0, %alloc_3 : memref<1x8x384x384xf16>, memref<4xsi32>
  }
    //CHECK:        [[ALOC:%.*]] = memref.alloc() : memref<1x8x384x384xf16>
    //CHECK:        [[ALOC_0:%.*]] = memref.alloc() : memref<4xsi32>
    //CHECK:        [[ALOC_1:%.*]] = memref.alloc() : memref<1x8x384x384xf16>
    //CHECK:        [[ALOC_2:%.*]] = memref.alloc() : memref<4xsi32>
    //CHECK:        [[RESULT:%.*]]:2 = VPUIP.SW.Kernel {dynamicInputShapesMap = array<i32: 0>, dynamicOutputShapesMap = array<i32: 0>, resultSegmentSizes = array<i32: 2, 0, 0>}
    //CHECK-SAME:   inputs([[ALOC]] as %arg0: memref<1x8x384x384xf16>)
    //CHECK-SAME:   outputs([[ALOC_1]] as %arg1: memref<1x8x384x384xf16>)
    //CHECK:        return [[RESULT]]#0, [[ALOC_2]] : memref<1x8x384x384xf16>, memref<4xsi32>
}
