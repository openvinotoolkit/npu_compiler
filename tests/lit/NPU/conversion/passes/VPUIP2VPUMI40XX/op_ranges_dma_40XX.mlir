//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-VPUIP-to-VPUMI40XX %s | FileCheck %s
// REQUIRES: platform-NPU4000

module @SingleDMATile0List0Range {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  func.func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
    VPURT.Task {
      %0 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR>
    }
    return %arg1 : memref<1x2x3x4xf16, @DDR>
  }

  // CHECK-NOT: return
  // CHECK: VPUMI40XX.OpRanges
  // CHECK-SAME: types([#VPURegMapped.task_type<DMA>])
  // CHECK-SAME: begins([[VAR:%[0-9]+]] : !VPURegMapped.Index<0:0:0>)
  // CHECK-SAME: ends([[VAR]] : !VPURegMapped.Index<0:0:0>)
}

// -----

module @ThreeDMATile0List0Range {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  func.func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
    VPURT.Task {
      %0 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR>
    }
    VPURT.Task {
      %0 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR>
    }
    VPURT.Task {
      %0 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR>
    }
    return %arg1 : memref<1x2x3x4xf16, @DDR>
  }

  // CHECK-NOT: return
  // CHECK: VPUMI40XX.OpRanges
  // CHECK-SAME: types([#VPURegMapped.task_type<DMA>])
  // CHECK-SAME: begins([[VAR:%[0-9]+]] : !VPURegMapped.Index<0:0:0>)
  // CHECK-SAME: ends([[VAR2:%[2-9]+]] : !VPURegMapped.Index<0:0:2>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @SingleDMATile0List1Range {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  func.func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
    %5 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    VPURT.Task {
      %0 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%5 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%5 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    return %arg1 : memref<1x2x3x4xf16, @DDR>
  }

  // CHECK-NOT: return
  // CHECK: VPUMI40XX.OpRanges
  // CHECK-SAME: types([#VPURegMapped.task_type<DMA>])
  // CHECK-SAME: begins([[VAR:%[0-9]+]] : !VPURegMapped.Index<0:1:0>)
  // CHECK-SAME: ends([[VAR]] : !VPURegMapped.Index<0:1:0>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @ThreeDMATile0List1Range {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  func.func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
    %5 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    VPURT.Task {
      %0 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%5 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%5 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    VPURT.Task {
      %0 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%5 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%5 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    VPURT.Task {
      %0 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%5 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%5 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    return %arg1 : memref<1x2x3x4xf16, @DDR>
  }

  // CHECK-NOT: return
  // CHECK: VPUMI40XX.OpRanges
  // CHECK-SAME: types([#VPURegMapped.task_type<DMA>])
  // CHECK-SAME: begins([[VAR:%[0-9]+]] : !VPURegMapped.Index<0:1:0>)
  // CHECK-SAME: ends([[VAR2:%[2-9]+]] : !VPURegMapped.Index<0:1:2>)
}

// -----

module @SingleDMATile1List0Range {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  func.func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
    VPURT.Task {
      %0 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR>
    }
    return %arg1 : memref<1x2x3x4xf16, @DDR>
  }

  // CHECK-NOT: return
  // CHECK: VPUMI40XX.OpRanges
  // CHECK-SAME: types([#VPURegMapped.task_type<DMA>])
  // CHECK-SAME: begins([[VAR:%[0-9]+]] : !VPURegMapped.Index<1:0:0>)
  // CHECK-SAME: ends([[VAR]] : !VPURegMapped.Index<1:0:0>)
}

// -----

module @SingleDMATile1List0Range {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  func.func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
    VPURT.Task {
      %0 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR>
    }
    VPURT.Task {
      %0 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR>
    }
    VPURT.Task {
      %0 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR>
    }
    return %arg1 : memref<1x2x3x4xf16, @DDR>
  }

  // CHECK-NOT: return
  // CHECK: VPUMI40XX.OpRanges
  // CHECK-SAME: types([#VPURegMapped.task_type<DMA>])
  // CHECK-SAME: begins([[VAR:%[0-9]+]] : !VPURegMapped.Index<1:0:0>)
  // CHECK-SAME: ends([[VAR2:%[2-9]+]] : !VPURegMapped.Index<1:0:2>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @SingleDMATile1List1Range {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  func.func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
    %5 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    VPURT.Task {
      %0 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%5 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%5 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    return %arg1 : memref<1x2x3x4xf16, @DDR>
  }

  // CHECK-NOT: return
  // CHECK: VPUMI40XX.OpRanges
  // CHECK-SAME: types([#VPURegMapped.task_type<DMA>])
  // CHECK-SAME: begins([[VAR:%[0-9]+]] : !VPURegMapped.Index<1:1:0>)
  // CHECK-SAME: ends([[VAR]] : !VPURegMapped.Index<1:1:0>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @ThreeDMATile1List1Range {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  func.func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
    %5 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    VPURT.Task {
      %0 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%5 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%5 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    VPURT.Task {
      %0 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%5 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%5 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    VPURT.Task {
      %0 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%5 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%5 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    return %arg1 : memref<1x2x3x4xf16, @DDR>
  }

  // CHECK-NOT: return
  // CHECK: VPUMI40XX.OpRanges
  // CHECK-SAME: types([#VPURegMapped.task_type<DMA>])
  // CHECK-SAME: begins([[VAR:%[0-9]+]] : !VPURegMapped.Index<1:1:0>)
  // CHECK-SAME: ends([[VAR2:%[2-9]+]] : !VPURegMapped.Index<1:1:2>)
}
