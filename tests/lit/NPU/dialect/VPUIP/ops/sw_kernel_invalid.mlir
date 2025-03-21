//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --verify-diagnostics --init-compiler="vpu-arch=%arch%" %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
module {
  module @VPU.SW {
    func.func private @builtin_dummy(memref<*xf16>, memref<*xf16>, i64, i64) attributes {VPU.kernel_code = "dummy.cpp", VPU.kernel_entry = "dummy"}
  }
  func.func @UseBoundedBufferAsSWKernelInputInvalid_1() -> !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>> {
    %alloc_0 = memref.alloc() : memref<1x8x384x384xf16>
    %alloc_1 = memref.alloc() : memref<4xsi32>
    %input = VPUIP.GroupBoundedBuffer(%alloc_0, %alloc_1) : memref<1x8x384x384xf16>, memref<4xsi32>
        -> !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>>
    %alloc_2 = memref.alloc() : memref<1x8x384x384xf16>
    %alloc_3 = memref.alloc() : memref<4xsi32>
    %output = VPUIP.GroupBoundedBuffer(%alloc_2, %alloc_3) : memref<1x8x384x384xf16>, memref<4xsi32>
        -> !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>>
    // expected-error@+1 {{SW Kernel has inputs of BoundedBufferType but dynamicInputShapes or dynamicInputShapesMap provided. Ambiguous dynamicShape information}}
    %results = VPUIP.SW.Kernel {
            dynamicInputShapesMap = array<i32: -1>,
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
}

// -----

module {
  module @VPU.SW {
    func.func private @builtin_dummy(memref<*xf16>, memref<*xf16>, i64, i64) attributes {VPU.kernel_code = "dummy.cpp", VPU.kernel_entry = "dummy"}
  }
  func.func @UseBoundedBufferAsSWKernelInputInvalid_2() -> !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>> {
    %alloc_0 = memref.alloc() : memref<1x8x384x384xf16>
    %alloc_1 = memref.alloc() : memref<4xsi32>
    %input = VPUIP.GroupBoundedBuffer(%alloc_0, %alloc_1) : memref<1x8x384x384xf16>, memref<4xsi32>
        -> !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>>
    %alloc_2 = memref.alloc() : memref<1x8x384x384xf16>
    %alloc_3 = memref.alloc() : memref<4xsi32>
    %output = VPUIP.GroupBoundedBuffer(%alloc_2, %alloc_3) : memref<1x8x384x384xf16>, memref<4xsi32>
        -> !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>>
    // expected-error@+1 {{SW Kernel has outputs of BoundedBufferType but dynamicOutputShapes or dynamicOutputShapesMap provided. Ambiguous dynamicShape information}}
    %results = VPUIP.SW.Kernel {
            dynamicOutputShapesMap = array<i32: -1>,
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
}

// -----

module {
  module @VPU.SW {
    func.func private @builtin_dummy(memref<*xf16>, memref<*xf16>, i64, i64) attributes {VPU.kernel_code = "dummy.cpp", VPU.kernel_entry = "dummy"}
  }
  func.func @UseBoundedBufferAsSWKernelInputInvalid_3() -> (memref<1x8x384x384xf16>, memref<4xsi32>) {
    %alloc_0 = memref.alloc() : memref<1x8x384x384xf16>
    %alloc_1 = memref.alloc() : memref<4xsi32>

    %alloc_2 = memref.alloc() : memref<1x8x384x384xf16>
    %alloc_3 = memref.alloc() : memref<4xsi32>

    // expected-error@+1 {{SW Kernel dynamicInputShapesMap contains values which are out of dynamicInputShapes operand range}}
    %results = VPUIP.SW.Kernel {
            dynamicInputShapesMap = array<i32: 1>,
            resultSegmentSizes = array<i32: 1, 0, 0>
        }
        @VPU.SW::@builtin_dummy
            inputs(%alloc_0 as %arg0: memref<1x8x384x384xf16>)
            dynamicInputShapes(%alloc_1 : memref<4xsi32>)
            outputs(%alloc_2 as %arg1: memref<1x8x384x384xf16>)
            dynamicOutputShapes(%alloc_3 : memref<4xsi32>)
                -> memref<1x8x384x384xf16> {
            VPUIP.SW.Kernel.run(%arg0, %arg1) :
                memref<1x8x384x384xf16>,
                memref<1x8x384x384xf16>
    }
    return %results, %alloc_3 : memref<1x8x384x384xf16>, memref<4xsi32>
  }
}

// -----

module {
  module @VPU.SW {
    func.func private @builtin_dummy(memref<*xf16>, memref<*xf16>, i64, i64) attributes {VPU.kernel_code = "dummy.cpp", VPU.kernel_entry = "dummy"}
  }
  func.func @UseBoundedBufferAsSWKernelInputInvalid_4() -> (memref<1x8x384x384xf16>, memref<4xsi32>) {
    %alloc_0 = memref.alloc() : memref<1x8x384x384xf16>
    %alloc_1 = memref.alloc() : memref<4xsi32>

    %alloc_2 = memref.alloc() : memref<1x8x384x384xf16>
    %alloc_3 = memref.alloc() : memref<4xsi32>
    // expected-error@+1 {{SW Kernel dynamicOutputShapesMap contains values which are out of dynamicOutputShapes operand range}}
    %results = VPUIP.SW.Kernel {
            dynamicOutputShapesMap = array<i32: 1>,
            resultSegmentSizes = array<i32: 1, 0, 0>
        }
        @VPU.SW::@builtin_dummy
            inputs(%alloc_0 as %arg0: memref<1x8x384x384xf16>)
            dynamicInputShapes(%alloc_1 : memref<4xsi32>)
            outputs(%alloc_2 as %arg1: memref<1x8x384x384xf16>)
            dynamicOutputShapes(%alloc_3 : memref<4xsi32>)
                -> memref<1x8x384x384xf16> {
            VPUIP.SW.Kernel.run(%arg0, %arg1) :
                memref<1x8x384x384xf16>,
                memref<1x8x384x384xf16>
    }
    return %results, %alloc_3 : memref<1x8x384x384xf16>, memref<4xsi32>
  }
}

// -----

module {
  module @VPU.SW {
    func.func private @builtin_dummy(memref<*xf16>, memref<*xf16>, i64, i64) attributes {VPU.kernel_code = "dummy.cpp", VPU.kernel_entry = "dummy"}
  }
  func.func @UseBoundedBufferAsSWKernelInputInvalid_5() -> (memref<1x8x384x384xf16>, memref<4xsi32>) {
    %alloc_0 = memref.alloc() : memref<1x8x384x384xf16>
    %alloc_1 = memref.alloc() : memref<4xsi32>

    %alloc_2 = memref.alloc() : memref<1x8x384x384xf16>
    %alloc_3 = memref.alloc() : memref<4xsi32>

    // expected-error@+1 {{SW Kernel has inconsistent inputs and dynamicInputShapesMap. Inputs size [1] does not match dynamicInputShapesMap size [2]}}
    %results = VPUIP.SW.Kernel {
            dynamicInputShapesMap = array<i32: 1, 2>,
            resultSegmentSizes = array<i32: 1, 0, 0>
        }
        @VPU.SW::@builtin_dummy
            inputs(%alloc_0 as %arg0: memref<1x8x384x384xf16>)
            dynamicInputShapes(%alloc_1 : memref<4xsi32>)
            outputs(%alloc_2 as %arg1: memref<1x8x384x384xf16>)
            dynamicOutputShapes(%alloc_3 : memref<4xsi32>)
                -> memref<1x8x384x384xf16> {
            VPUIP.SW.Kernel.run(%arg0, %arg1) :
                memref<1x8x384x384xf16>,
                memref<1x8x384x384xf16>
    }
    return %results, %alloc_3 : memref<1x8x384x384xf16>, memref<4xsi32>
  }
}

// -----

module {
  module @VPU.SW {
    func.func private @builtin_dummy(memref<*xf16>, memref<*xf16>, i64, i64) attributes {VPU.kernel_code = "dummy.cpp", VPU.kernel_entry = "dummy"}
  }
  func.func @UseBoundedBufferAsSWKernelInputInvalid_6() -> (memref<1x8x384x384xf16>, memref<4xsi32>) {
    %alloc_0 = memref.alloc() : memref<1x8x384x384xf16>
    %alloc_1 = memref.alloc() : memref<4xsi32>

    %alloc_2 = memref.alloc() : memref<1x8x384x384xf16>
    %alloc_3 = memref.alloc() : memref<4xsi32>

    // expected-error@+1 {{SW Kernel has inconsistent outputs and dynamicOutputShapesMap. Outputs size [1] does not match dynamicOutputShapesMap size [2]}}
    %results = VPUIP.SW.Kernel {
            dynamicOutputShapesMap = array<i32: 1, 2>,
            resultSegmentSizes = array<i32: 1, 0, 0>
        }
        @VPU.SW::@builtin_dummy
            inputs(%alloc_0 as %arg0: memref<1x8x384x384xf16>)
            dynamicInputShapes(%alloc_1 : memref<4xsi32>)
            outputs(%alloc_2 as %arg1: memref<1x8x384x384xf16>)
            dynamicOutputShapes(%alloc_3 : memref<4xsi32>)
                -> memref<1x8x384x384xf16> {
            VPUIP.SW.Kernel.run(%arg0, %arg1) :
                memref<1x8x384x384xf16>,
                memref<1x8x384x384xf16>
    }
    return %results, %alloc_3 : memref<1x8x384x384xf16>, memref<4xsi32>
  }
}
