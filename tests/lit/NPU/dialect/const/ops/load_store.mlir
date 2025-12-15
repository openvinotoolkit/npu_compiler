//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @Load
module @Load {

const.Data @ov_bin {
    const.Rodata @weights dense<1.000000e+00> : tensor<2x2xf32>
}

func.func @init() -> tensor<2x2xf32> {
    %cst = const.Load @ov_bin::@weights -> tensor<2x2xf32>
    return %cst : tensor<2x2xf32>
}

// CHECK:  const.Data @ov_bin {
// CHECK:      const.Rodata @weights dense<1.000000e+00> : tensor<2x2xf32>
// CHECK:  }
// CHECK:  func.func @init() -> tensor<2x2xf32> {
// CHECK:      [[CST:%.+]] = const.Load @ov_bin::@weights -> tensor<2x2xf32>
// CHECK:      return [[CST]] : tensor<2x2xf32>
// CHECK:  }

}

// -----

// CHECK-LABEL: @Store
module @Store {

const.Data @init_res {
    const.Rodata @value dense<1.000000e+00> : tensor<2x2xf32>
}

func.func @init(%arg0 : tensor<2x2xf32>) {
    const.Store %arg0, @init_res::@value : tensor<2x2xf32>
    return
}

// CHECK:  const.Data @init_res {
// CHECK:      const.Rodata @value dense<1.000000e+00> : tensor<2x2xf32>
// CHECK:  }
// CHECK:  func.func @init([[ARG:%.+]]: tensor<2x2xf32>) {
// CHECK:      const.Store [[ARG]], @init_res::@value : tensor<2x2xf32>
// CHECK:      return
// CHECK:  }

}

// -----

// CHECK-LABEL: @LoadStore
module @LoadStore {

const.Data @ov_bin {
    const.Rodata @weights dense<1.000000e+00> : tensor<2x2xf32>
}

const.Data @init_res {
    const.Rodata @value dense<1.000000e+00> : tensor<2x2xf32>
}

func.func @init() {
    %cst = const.Load @ov_bin::@weights -> tensor<2x2xf32>
    const.Store %cst, @init_res::@value : tensor<2x2xf32>
    return
}

// CHECK:  const.Data @ov_bin {
// CHECK:      const.Rodata @weights dense<1.000000e+00> : tensor<2x2xf32>
// CHECK:  }
// CHECK:  const.Data @init_res {
// CHECK:      const.Rodata @value dense<1.000000e+00> : tensor<2x2xf32>
// CHECK:  }
// CHECK:  func.func @init() {
// CHECK:      [[CST:%.+]] = const.Load @ov_bin::@weights -> tensor<2x2xf32>
// CHECK:      const.Store [[CST]], @init_res::@value : tensor<2x2xf32>
// CHECK:      return
// CHECK:  }

}
