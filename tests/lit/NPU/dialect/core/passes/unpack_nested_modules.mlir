//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unpack-nested-modules --verify-diagnostics %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX


// CHECK-LABEL: @NoNesting
module @NoNesting {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input": tensor<2x2xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<2x2xf32>
    }

    func.func private @foo(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> {
        return %arg : tensor<2x2xf32>
    }

    // CHECK: func.func private @foo([[ARG:%.+]]: tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:   return [[ARG]]

    func.func @main(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> {
        %call1 = call @foo(%arg) : (tensor<2x2xf32>) -> tensor<2x2xf32>
        return %call1 : tensor<2x2xf32>
    }

    // CHECK: func.func @main([[ARG:%.+]]: tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:   [[CALL1:%.+]] = call @foo([[ARG]])
    // CHECK:   return [[CALL1]]
}

// -----

// Note: this tests that same-name functions inside modules produce an error
//       when "unpacked"

module @ConflictingNestedNames {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input": tensor<2x2xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<2x2xf32>
    }

    module @Nested1 {
        // expected-note@+1 {{see existing symbol definition here}}
        func.func @foo(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> {
            return %arg : tensor<2x2xf32>
        }
    }

    module @Nested2 {
        // expected-error@+1 {{redefinition of symbol named 'foo'}}
        func.func @foo(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> {
            return %arg : tensor<2x2xf32>
        }
    }

    func.func @main(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> {
        %call1 = Core.NestedCall @Nested1::@foo(%arg) : (tensor<2x2xf32>) -> tensor<2x2xf32>
        %call2 = Core.NestedCall @Nested2::@foo(%call1) : (tensor<2x2xf32>) -> tensor<2x2xf32>
        return %call2 : tensor<2x2xf32>
    }
}

// -----

// CHECK-LABEL: @SimpleNesting
module @SimpleNesting {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input": tensor<2x2xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<2x2xf32>
    }

    module @Nested {
        net.NetworkInfo entryPoint : @foo inputsInfo : {
            DataInfo "in0": tensor<2x2xf32>
        } outputsInfo : {
            DataInfo "out0" : tensor<2x2xf16>
        }

        // Note: non-private function is going to be converted to private
        func.func @foo(%arg: tensor<2x2xf32>) -> tensor<2x2xf16> {
            %out = VPU.Convert(%arg) {dstElemType = f16} : tensor<2x2xf32> -> tensor<2x2xf16>
            return %out : tensor<2x2xf16>
        }
    }
    // CHECK-NOT: module @Nested

    // CHECK: func.func private @foo([[ARG:%.+]]: tensor<2x2xf32>) -> tensor<2x2xf16>
    // CHECK:   [[OUT:%.+]] = VPU.Convert([[ARG]]) {dstElemType = f16}
    // CHECK:   return [[OUT]]

    func.func @main(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> {
        %call = Core.NestedCall @Nested::@foo(%arg) : (tensor<2x2xf32>) -> tensor<2x2xf16>
        %cvt = VPU.Convert(%call) {dstElemType = f32} : tensor<2x2xf16> -> tensor<2x2xf32>
        return %cvt : tensor<2x2xf32>
    }

    // CHECK: func.func @main([[ARG:%.+]]: tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:   [[CALL:%.+]] = call @foo([[ARG]])
    // CHECK:   [[CVT:%.+]] = VPU.Convert([[CALL]]) {dstElemType = f32}
    // CHECK:   return [[CVT]]
}

// -----

// CHECK-LABEL: @MultiNesting
module @MultiNesting {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input": tensor<2x2xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<2x2xf32>
    }

    module @Nested {
        module @Nested2 {
            func.func private @bar(%arg: tensor<2x2xf16>) -> tensor<2x2xf16> {
                return %arg : tensor<2x2xf16>
            }
        }

        func.func private @foo(%arg: tensor<2x2xf32>) -> tensor<2x2xf16> {
            %cvt = VPU.Convert(%arg) {dstElemType = f16} : tensor<2x2xf32> -> tensor<2x2xf16>
            %out = Core.NestedCall @Nested2::@bar(%cvt) : (tensor<2x2xf16>) -> tensor<2x2xf16>
            return %out : tensor<2x2xf16>
        }
    }
    // CHECK-NOT: module @Nested
    // CHECK-NOT: module @Nested2

    // CHECK: func.func private @bar([[ARG:%.+]]: tensor<2x2xf16>) -> tensor<2x2xf16>
    // CHECK:   return [[ARG]]

    // CHECK: func.func private @foo([[ARG:%.+]]: tensor<2x2xf32>) -> tensor<2x2xf16>
    // CHECK:   [[CVT:%.+]] = VPU.Convert([[ARG]]) {dstElemType = f16}
    // CHECK:   [[OUT:%.+]] = call @bar([[CVT]])
    // CHECK:   return [[OUT]]

    func.func @main(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> {
        %call = Core.NestedCall @Nested::@foo(%arg) : (tensor<2x2xf32>) -> tensor<2x2xf16>
        %call2 = Core.NestedCall @Nested::@Nested2::@bar(%call) : (tensor<2x2xf16>) -> tensor<2x2xf16>
        %cvt = VPU.Convert(%call2) {dstElemType = f32} : tensor<2x2xf16> -> tensor<2x2xf32>
        return %cvt : tensor<2x2xf32>
    }

    // CHECK: func.func @main([[ARG:%.+]]: tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:   [[CALL:%.+]] = call @foo([[ARG]])
    // CHECK:   [[CALL2:%.+]] = call @bar([[CALL]])
    // CHECK:   [[CVT:%.+]] = VPU.Convert([[CALL2]]) {dstElemType = f32}
    // CHECK:   return [[CVT]]
}

// -----

// CHECK-LABEL: @SiblingNesting
module @SiblingNesting {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input": tensor<2x2xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<2x2xf32>
    }

    module @Nested1 {
        module @special {
            func.func private @returnArg(%arg: tensor<2x2xf16>) -> tensor<2x2xf16> {
                return %arg : tensor<2x2xf16>
            }
        }

        func.func private @foo1(%arg: tensor<2x2xf16>) -> tensor<2x2xf16> {
            %out = Core.NestedCall @special::@returnArg(%arg) : (tensor<2x2xf16>) -> tensor<2x2xf16>
            return %out : tensor<2x2xf16>
        }
    }

    module @Nested2 {
        func.func private @foo2(%arg: tensor<2x2xf16>) -> tensor<2x2xf16> {
            return %arg : tensor<2x2xf16>
        }

        func.func private @somethingElse(%arg: tensor<2x2xf16>) -> tensor<2x2xf16> {
            %out = call @foo2(%arg) : (tensor<2x2xf16>) -> tensor<2x2xf16>
            return %out : tensor<2x2xf16>
        }
    }
    // CHECK-NOT: module @Nested1
    // CHECK-NOT: module @special
    // CHECK-NOT: module @Nested2

    // CHECK: func.func private @returnArg([[ARG:%.+]]: tensor<2x2xf16>) -> tensor<2x2xf16>
    // CHECK:   return [[ARG]]

    // CHECK: func.func private @foo1([[ARG:%.+]]: tensor<2x2xf16>) -> tensor<2x2xf16>
    // CHECK:   [[OUT:%.+]] = call @returnArg([[ARG]])
    // CHECK:   return [[OUT]]

    // CHECK: func.func private @foo2([[ARG:%.+]]: tensor<2x2xf16>) -> tensor<2x2xf16>
    // CHECK:   return [[ARG]]

    // CHECK: func.func private @somethingElse([[ARG:%.+]]: tensor<2x2xf16>) -> tensor<2x2xf16>
    // CHECK:   [[OUT:%.+]] = call @foo2([[ARG]])
    // CHECK:   return [[OUT]]

    func.func @main(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> {
        %cvt = VPU.Convert(%arg) {dstElemType = f16} : tensor<2x2xf32> -> tensor<2x2xf16>
        %call1 = Core.NestedCall @Nested1::@foo1(%cvt) : (tensor<2x2xf16>) -> tensor<2x2xf16>
        %call2 = Core.NestedCall @Nested2::@somethingElse(%call1) : (tensor<2x2xf16>) -> tensor<2x2xf16>
        %out = VPU.Convert(%call2) {dstElemType = f32} : tensor<2x2xf16> -> tensor<2x2xf32>
        return %out : tensor<2x2xf32>
    }

    // CHECK: func.func @main([[ARG:%.+]]: tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:   [[CVT:%.+]] = VPU.Convert([[ARG]]) {dstElemType = f16}
    // CHECK:   [[CALL1:%.+]] = call @foo1([[CVT]])
    // CHECK:   [[CALL2:%.+]] = call @somethingElse([[CALL1]])
    // CHECK:   [[OUT:%.+]] = VPU.Convert([[CALL2]]) {dstElemType = f32}
    // CHECK:   return [[OUT]]
}

// -----

// CHECK-LABEL: @VPUIP
module @VPUIP {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
      DataInfo "input" : tensor<f16>
    } outputsInfo : {
      DataInfo "output" : tensor<f16>
    }

    module @Module0 {
        func.func private @foo(%arg0: memref<f16, @DDR>, %arg1: memref<f16, @DDR>) -> memref<f16, @DDR>
    }

    // CHECK-NOT: module @Module0
    // CHECK:     func.func private @foo

    func.func @main(%arg0: memref<f16, @DDR>, %arg1: memref<f16, @DDR>) -> memref<f16, @DDR> {
    // CHECK: func.func @main
        %netIn = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<f16, @DDR>
        %netOut = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<f16, @DDR>

        %inAlloc = VPURT.DeclareBuffer <DDR> <0> -> memref<f16, @DDR>
        %outAlloc = VPURT.DeclareBuffer <DDR> <24576> -> memref<f16, @DDR>
        %b_fooCall1CopyIn = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %b_fooCall1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        VPURT.Task updates(%b_fooCall1CopyIn : !VPURT.Barrier) {
            %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%netIn : memref<f16, @DDR>)
                outputs(%inAlloc : memref<f16, @DDR>)
                -> memref<f16, @DDR>
        }
        VPURT.Task waits(%b_fooCall1CopyIn : !VPURT.Barrier) updates(%b_fooCall1 : !VPURT.Barrier) {
            %0 = Core.NestedCall @Module0::@foo(%inAlloc, %outAlloc)
                : (memref<f16, @DDR>, memref<f16, @DDR>) -> memref<f16, @DDR>
            // CHECK: call @foo
        }

        VPURT.Task waits(%b_fooCall1 : !VPURT.Barrier) {
            %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%outAlloc : memref<f16, @DDR>)
                outputs(%netOut : memref<f16, @DDR>)
                -> memref<f16, @DDR>
        }
        return %arg1 : memref<f16, @DDR>
    }
}
