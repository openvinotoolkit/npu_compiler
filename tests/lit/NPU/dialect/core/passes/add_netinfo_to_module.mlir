//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --vpu-arch=%arch% --split-input-file --mlir-elide-elementsattrs-if-larger 8 --pass-pipeline="builtin.module(builtin.module(add-netinfo-to-module))" --verify-diagnostics  %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX


//CHECK-LABEL:   module @CopyInputOutput {
module @CopyInputOutput {
// CHECK-COUNT-1:  net.NetworkInfo entryPoint : @main inputsInfo : {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "in_0" : tensor<1x3x60x60xf16>
  } outputsInfo : {
    DataInfo "out_0" : tensor<1x3x60x60xf16>
  }

// CHECK-LABEL:  module @Module0 {
  module @Module0 {
// CHECK:  net.NetworkInfo entryPoint : @main_part1 inputsInfo : {
// CHECK:    DataInfo "in_0" : tensor<1x3x60x60xf16>
// CHECK:  } outputsInfo : {
// CHECK:    DataInfo "out_0" : tensor<1x3x60x60xf16>

    func.func private @main_part1(%arg0: memref<1x3x60x60xf16>, %arg1: memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16> {
      %0 = VPUIP.Copy inputs(%arg0 : memref<1x3x60x60xf16>) outputs(%arg1 : memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16>
      return %0 : memref<1x3x60x60xf16>
    }
  }

  func.func @main(%arg0: memref<1x3x60x60xf16>, %arg1: memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16> {
    %alloc = memref.alloc() : memref<1x3x60x60xf16>
    %0 = Core.NestedCall @Module0::@main_part1(%arg0, %alloc) : (memref<1x3x60x60xf16>, memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16>
    %1 = VPUIP.Copy inputs(%0 : memref<1x3x60x60xf16>) outputs(%arg1 : memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16>
    return %arg1 : memref<1x3x60x60xf16>
  }
}

// -----

//CHECK-LABEL:   module @MultipleNestedModules {
module @MultipleNestedModules {
// CHECK-NOT:  net.NetworkInfo entryPoint : @main inputsInfo : {

// CHECK-LABEL:  module @Module0 {
  module @Module0 {
// CHECK:  net.NetworkInfo entryPoint : @main_part1 inputsInfo : {
// CHECK:    DataInfo "in_0" : tensor<1x3x60x60xf16>
// CHECK:  } outputsInfo : {
// CHECK:    DataInfo "out_0" : tensor<1x3x60x60xf16>

    func.func private @main_part1(%arg0: memref<1x3x60x60xf16>, %arg1: memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16> {
      %0 = VPUIP.Copy inputs(%arg0 : memref<1x3x60x60xf16>) outputs(%arg1 : memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16>
      return %0 : memref<1x3x60x60xf16>
    }
  }

// CHECK-LABEL:  module @Module1 {
  module @Module1 {
// CHECK:  net.NetworkInfo entryPoint : @main_part2 inputsInfo : {
// CHECK:    DataInfo "in_0" : tensor<1x3x60x60xf16>
// CHECK:  } outputsInfo : {
// CHECK:    DataInfo "out_0" : tensor<1x3x60x60xf16>

    func.func private @main_part2(%arg0: memref<1x3x60x60xf16>, %arg1: memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16> {
      %0 = VPUIP.Copy inputs(%arg0 : memref<1x3x60x60xf16>) outputs(%arg1 : memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16>
      return %0 : memref<1x3x60x60xf16>
    }
  }

// CHECK-LABEL:  module @Module2 {
  module @Module2 {
// CHECK:  net.NetworkInfo entryPoint : @main_part3 inputsInfo : {
// CHECK:    DataInfo "in_0" : tensor<1x3x60x60xf16>
// CHECK:  } outputsInfo : {
// CHECK:    DataInfo "out_0" : tensor<1x3x60x60xf16>

    func.func private @main_part3(%arg0: memref<1x3x60x60xf16>, %arg1: memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16> {
      %0 = VPUIP.Copy inputs(%arg0 : memref<1x3x60x60xf16>) outputs(%arg1 : memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16>
      return %0 : memref<1x3x60x60xf16>
    }
  }

  func.func @main(%arg0: memref<1x3x60x60xf16>, %arg1: memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16> {
    %alloc = memref.alloc() : memref<1x3x60x60xf16>
    %0 = Core.NestedCall @Module0::@main_part1(%arg0, %alloc) : (memref<1x3x60x60xf16>, memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16>
    %1 = VPUIP.Copy inputs(%0 : memref<1x3x60x60xf16>) outputs(%arg1 : memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16>
    return %arg1 : memref<1x3x60x60xf16>
  }
}

// -----

module @ExistingNetInfoFailure {
  // expected-error@+1 {{Module already contains a NetworkInfoOp, cannot add another one}}
  module @Module0 {
    net.NetworkInfo entryPoint : @main_part1 inputsInfo : {
      DataInfo "in_0" : tensor<1x3x60x60xf16>
    } outputsInfo : {
      DataInfo "out_0" : tensor<1x3x60x60xf16>
    }
    func.func private @main_part1(%arg0: memref<1x3x60x60xf16>, %arg1: memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16> {
      %0 = VPUIP.Copy inputs(%arg0 : memref<1x3x60x60xf16>) outputs(%arg1 : memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16>
      return %0 : memref<1x3x60x60xf16>
    }
  }

  func.func @main(%arg0: memref<1x3x60x60xf16>, %arg1: memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16> {
    %alloc = memref.alloc() : memref<1x3x60x60xf16>
    %0 = Core.NestedCall @Module0::@main_part1(%arg0, %alloc) : (memref<1x3x60x60xf16>, memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16>
    %1 = VPUIP.Copy inputs(%0 : memref<1x3x60x60xf16>) outputs(%arg1 : memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16>
    return %arg1 : memref<1x3x60x60xf16>
  }
}

// -----

module @MultipleFunctionFailure {
  // expected-error@+1 {{Module must contain exactly one function to add NetworkInfoOp}}
  module @Module0 {
    func.func private @main_part0(%arg0: memref<1x3x60x60xf16>, %arg1: memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16> {
      %0 = VPUIP.Copy inputs(%arg0 : memref<1x3x60x60xf16>) outputs(%arg1 : memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16>
      return %0 : memref<1x3x60x60xf16>
    }

    func.func private @main_part1(%arg0: memref<1x3x60x60xf16>, %arg1: memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16> {
      %0 = VPUIP.Copy inputs(%arg0 : memref<1x3x60x60xf16>) outputs(%arg1 : memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16>
      return %0 : memref<1x3x60x60xf16>
    }
  }
}
