//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --lower-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

//
// The 'lower-VPU-to-VPUIP' pipeline:
//
//   * Fully replaces VPU Dialect with VPUIP Dialect
//   * Changes all Value types from `tensor` to `memref`
//   * Adds result arguments to Function signature
//   * Inserts `VPUIP.Copy` to store result in output buffer
//   * Uses activation SHAVE kernels `VPUIP.SW.Kernel` for software ops
//

module {
  net.NetworkInfo entryPoint : @SingleLayer inputsInfo : {
    DataInfo "Parameter" : tensor<1x1000xf16>
  } outputsInfo : {
    DataInfo "Result" : tensor<1x1000xf16>
  }
// CHECK:       module @VPU.SW  {
// CHECK:           func.func private @builtin_SoftMax(memref<*xf16>, memref<*xf16>, i64, i64)
// CHECK-SAME:           attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.kernel_name = "softmax", VPU.task_type = @COMPUTE}
// CHECK:           func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK:       }

// CHECK: func.func @SingleLayer([[ARG0:%.*]]: memref<1x1000xf16>, [[ARG1:%.*]]: memref<1x1000xf16>) -> memref<1x1000xf16> {
func.func @SingleLayer(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 1} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    return %0 : tensor<1x1000xf16>

    // CHECK:       [[ALLOC:%.+]] = memref.alloc() : memref<1x1000xf16>
    // CHECK:       [[SW:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
    // CHECK-SAME:      @VPU.SW::@builtin_SoftMax inputs([[ARG0]] as %arg2: memref<1x1000xf16>)
    // CHECK-SAME:                                outputs([[ALLOC]] as %arg3: memref<1x1000xf16>) on tile 0
    // CHECK-SAME:               -> memref<1x1000xf16>{
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg2, %arg3)
    // CHECK-SAME:               : memref<1x1000xf16>, memref<1x1000xf16>
    // CHECK:       }
    // CHECK:       [[OUT:%.+]] = VPUIP.Copy inputs([[SW]] : memref<1x1000xf16>) outputs([[ARG1]] : memref<1x1000xf16>) -> memref<1x1000xf16>
    // CHECK:       return [[OUT]] : memref<1x1000xf16>
}
}
// -----

module {
  net.NetworkInfo entryPoint : @ReshapeInGraph inputsInfo : {
    DataInfo "Parameter" : tensor<1x512x1x1xf16>
  } outputsInfo : {
    DataInfo "Result" : tensor<1x512x1x1xf16>
  }
// CHECK:       module @VPU.SW  {
// CHECK:           func.func private @builtin_SoftMax(memref<*xf16>, memref<*xf16>, i64, i64)
// CHECK-SAME:           attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.kernel_name = "softmax", VPU.task_type = @COMPUTE}
// CHECK:           func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK:       }

// CHECK: func.func @ReshapeInGraph([[ARG0:%.*]]: memref<1x512x1x1xf16>, [[ARG1:%.*]]: memref<1x512x1x1xf16>) -> memref<1x512x1x1xf16> {
func.func @ReshapeInGraph(%arg0 : tensor<1x512x1x1xf16>) -> tensor<1x512x1x1xf16> {
    %0 = VPU.Reshape(%arg0) {shape_value = [1, 512]} : tensor<1x512x1x1xf16> -> tensor<1x512xf16>
    %1 = VPU.SoftMax(%0) {axisInd = 1} : tensor<1x512xf16> -> tensor<1x512xf16>
    %2 = VPU.Reshape(%1) {shape_value = [1, 512, 1, 1]} : tensor<1x512xf16> -> tensor<1x512x1x1xf16>
    return %2 : tensor<1x512x1x1xf16>

    // CHECK:       [[RESHAPE_IN:%.+]] = VPUIP.GenericReshape inputs([[ARG0]] : memref<1x512x1x1xf16>) -> memref<1x512xf16>
    // CHECK:       [[ALLOC:%.+]] = memref.alloc() : memref<1x512xf16>
    // CHECK:       [[SW:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
    // CHECK-SAME:      @VPU.SW::@builtin_SoftMax inputs([[RESHAPE_IN]] as %arg2: memref<1x512xf16>)
    // CHECK-SAME:                                outputs([[ALLOC]] as %arg3: memref<1x512xf16>) on tile 0
    // CHECK-SAME:               -> memref<1x512xf16>{
    // CHECK:         VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg2, %arg3)
    // CHECK-SAME:               : memref<1x512xf16>, memref<1x512xf16>
    // CHECK:       }
    // CHECK:       [[RESHAPE_OUT:%.+]] = VPUIP.GenericReshape inputs([[SW]] : memref<1x512xf16>) -> memref<1x512x1x1xf16>
    // CHECK:       [[OUT:%.+]] = VPUIP.Copy inputs([[RESHAPE_OUT]] : memref<1x512x1x1xf16>) outputs([[ARG1]] : memref<1x512x1x1xf16>) -> memref<1x512x1x1xf16>
    // CHECK:       return [[OUT]] : memref<1x512x1x1xf16>
}
}

// -----
module {
  net.NetworkInfo entryPoint : @ConstantLayer inputsInfo : {
  } outputsInfo : {
    DataInfo "Result" : tensor<1x2x2x2xf16>
  }
// CHECK: func.func @ConstantLayer([[ARG0:%.*]]: memref<1x2x2x2xf16>) -> memref<1x2x2x2xf16> {
func.func @ConstantLayer() -> tensor<1x2x2x2xf16> {
    %0 = const.Declare tensor<1x2x2x2xf16> = dense<1.0> : tensor<1x2x2x2xf16>
    return %0 : tensor<1x2x2x2xf16>

    // CHECK-DAG:  [[VAR0:%.*]] = const.Declare memref<1x2x2x2xf16> = dense<1.000000e+00> : tensor<1x2x2x2xf16>
    // CHECK:  [[VAR1:%.*]] = VPUIP.Copy inputs([[VAR0]] : memref<1x2x2x2xf16>) outputs([[ARG0]] : memref<1x2x2x2xf16>) -> memref<1x2x2x2xf16>
    // CHECK:  return [[VAR1]] : memref<1x2x2x2xf16>
}
}
// -----
module {
  net.NetworkInfo entryPoint : @Reshape inputsInfo : {
    DataInfo "Parameter" : tensor<1x512x1x1xf32>
  } outputsInfo : {
    DataInfo "Result" : tensor<1x512xf32>
  }
// CHECK: func.func @Reshape([[ARG0:%.*]]: memref<1x512x1x1xf32>, [[ARG1:%.*]]: memref<1x512xf32>) -> memref<1x512xf32> {
func.func @Reshape(%arg0 : tensor<1x512x1x1xf32>) -> tensor<1x512xf32> {
    %0 = VPU.Reshape(%arg0) {shape_value = [1, 512]} : tensor<1x512x1x1xf32> -> tensor<1x512xf32>
    return %0 : tensor<1x512xf32>

    // CHECK: [[VAR0:%.*]] = VPUIP.GenericReshape inputs([[ARG0]] : memref<1x512x1x1xf32>) -> memref<1x512xf32>
    // CHECK: [[VAR1:%.*]] = VPUIP.Copy inputs([[VAR0]] : memref<1x512xf32>) outputs([[ARG1]] : memref<1x512xf32>) -> memref<1x512xf32>
    // CHECK: return [[VAR1]] : memref<1x512xf32>
}
}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @NCEConv inputsInfo : {
    DataInfo "Parameter" : tensor<1x32x16x16xf16>
  } outputsInfo : {
    DataInfo "Result" : tensor<1x64x14x14xf16>
  }
// CHECK: func.func @NCEConv([[ARG0:%.*]]: memref<1x32x16x16xf16, #NHWC, @CMX_NN>, [[ARG1:%.*]]: memref<1x64x14x14xf16, #NHWC, @CMX_NN>) -> memref<1x64x14x14xf16, #NHWC, @CMX_NN> {
func.func @NCEConv(%arg0 : tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    %weights = const.Declare tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x3x3xf16, {mem_space = @CMX_NN}>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}> = dense<1> : tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}>

    %out = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [64, 32, 3, 3],
            strides = [1, 1]
        } : tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}> -> tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
        }

    return %out : tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare memref<64x32x3x3xf16, #NHWC, @CMX_NN> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<64x32x3x3xf16, {mem_space = @CMX_NN}>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare memref<64x1x1x4xsi32, @CMX_NN> = dense<1>
    // CHECK-SAME:      : tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}>

    // CHECK:       [[OUT_BUF:%.+]] = memref.alloc() : memref<1x64x14x14xf16, #NHWC, @CMX_NN>
    // CHECK:       [[OUT:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:          {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME:          input([[ARG0]]
    // CHECK-SAME:          weights([[WEIGHTS]]
    // CHECK-SAME:          weight_table([[WEIGHTS_TABLE]]
    // CHECK-SAME:          parent_input([[ARG0]]
    // CHECK-SAME:          parent_output([[OUT_BUF]]
    // CHECK-SAME:          outputs([[OUT_BUF]]
    // CHECK-SAME:      -> memref<1x64x14x14xf16, #NHWC, @CMX_NN> variants :  {
    // CHECK:               DPUTask {mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 15, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:           } PPE :  {
    // CHECK:           }

    // CHECK:       [[RESULT:%.+]] = VPUIP.Copy inputs([[OUT]] : memref<1x64x14x14xf16, #NHWC, @CMX_NN>) outputs([[ARG1]] : memref<1x64x14x14xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      -> memref<1x64x14x14xf16, #NHWC, @CMX_NN>
    // CHECK:       return [[RESULT]] : memref<1x64x14x14xf16, #NHWC, @CMX_NN>
}
}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @SparseNCEConv inputsInfo : {
    DataInfo "Parameter1" : tensor<1x32x16x16xf16>
    DataInfo "Parameter2" : tensor<1x32x16x16xi1>
  } outputsInfo : {
    DataInfo "Result" : tensor<1x64x14x14xf16>
  }
// CHECK:       func.func @SparseNCEConv([[ARG0:%.+]]: memref<1x32x16x16xf16, #NHWC>,
// CHECK-SAME:                      [[ARG1:%.+]]: memref<1x32x16x16xi1, #NHWC>,
// CHECK-SAME:                      [[ARG2:%.+]]: !VPUIP.SparseBuffer<data=memref<1x64x14x14xf16, #NHWC>,
// CHECK-SAME:                                                        sparsity_map=memref<1x64x14x14xi1, #NHWC>>)
// CHECK-SAME:       -> !VPUIP.SparseBuffer<data=memref<1x64x14x14xf16, #NHWC>, sparsity_map=memref<1x64x14x14xi1, #NHWC>> {
func.func @SparseNCEConv(%arg0 : tensor<1x32x16x16xf16, {order = #NHWC}>, %arg1 : tensor<1x32x16x16xi1, {order = #NHWC}>)
        -> !VPU.SparseTensor<data=tensor<1x64x14x14xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x64x14x14xi1, {order = #NHWC}>> {

    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1)
        -> !VPU.SparseTensor<data=tensor<1x32x16x16xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x32x16x16xi1, {order = #NHWC}>>
    %input_sparse_cmx = VPU.Copy(%input_sparse) {out_mem_space = @CMX_NN} :
           !VPU.SparseTensor<data=tensor<1x32x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x32x16x16xi1, {order = #NHWC}>>
        -> !VPU.SparseTensor<data=tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                             sparsity_map=tensor<1x32x16x16xi1, {mem_space = @CMX_NN, order = #NHWC}>>

    %weights = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<64x1x1x384xi1> = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<64x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x384xi1>, is_weights>
    %weights_sparse_cmx = VPU.Copy(%weights_sparse) {out_mem_space = @CMX_NN} :
           !VPU.SparseTensor<data=tensor<64x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x384xi1>, is_weights>
        -> !VPU.SparseTensor<data=tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                             sparsity_map=tensor<64x1x1x384xi1, {mem_space = @CMX_NN}>, is_weights>

    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %weights_table_cmx = VPU.Copy(%weights_table) {out_mem_space = @CMX_NN} :
            tensor<64x1x1x4xsi32> -> tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}>

    %output_sparse_cmx = VPU.NCE.Convolution(%input_sparse_cmx, %weights_sparse_cmx, %weights_table_cmx) {
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [64, 32, 3, 3],
            strides = [1, 1]
        } : !VPU.SparseTensor<data=tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x32x16x16xi1, {mem_space = @CMX_NN, order = #NHWC}>>, !VPU.SparseTensor<data=tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<64x1x1x384xi1, {mem_space = @CMX_NN}>, is_weights>, tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}> -> !VPU.SparseTensor<data=tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                               sparsity_map=tensor<1x64x14x14xi1, {mem_space = @CMX_NN, order = #NHWC}>> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
        }
    %output_sparse = VPU.Copy(%output_sparse_cmx) :
           !VPU.SparseTensor<data=tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                             sparsity_map=tensor<1x64x14x14xi1, {mem_space = @CMX_NN, order = #NHWC}>>
        -> !VPU.SparseTensor<data=tensor<1x64x14x14xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x64x14x14xi1, {order = #NHWC}>>

    return %output_sparse : !VPU.SparseTensor<data=tensor<1x64x14x14xf16, {order = #NHWC}>,
                                              sparsity_map=tensor<1x64x14x14xi1, {order = #NHWC}>>

    // CHECK-DAG:       [[CST_WEIGHTS_TABLE:%.+]] = const.Declare memref<64x1x1x4xsi32> = dense<1>
    // CHECK-SAME:      : tensor<64x1x1x4xsi32>

    // CHECK-DAG:       [[CST_WEIGHTS_SM:%.+]] = const.Declare memref<64x1x1x384xi1> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    // CHECK-DAG:       [[CST_WEIGHTS:%.+]] = const.Declare memref<64x32x3x3xf16, #NHWC> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]

    // CHECK:       [[IN_SPARSE_DDR:%.+]] = VPUIP.GroupSparseBuffer([[ARG0]], [[ARG1]])
    // CHECK-SAME:      -> !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, #NHWC>,
    // CHECK-SAME:                             sparsity_map=memref<1x32x16x16xi1, #NHWC>>
    // CHECK:       [[IN_DATA_CMX:%.+]] = memref.alloc() : memref<1x32x16x16xf16, #NHWC, @CMX_NN>
    // CHECK:       [[IN_SM_CMX:%.+]] = memref.alloc() : memref<1x32x16x16xi1, #NHWC, @CMX_NN>
    // CHECK:       [[IN_SPARSE_CMX:%.+]] = VPUIP.GroupSparseBuffer([[IN_DATA_CMX]], [[IN_SM_CMX]])
    // CHECK-SAME:      -> !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:                             sparsity_map=memref<1x32x16x16xi1, #NHWC, @CMX_NN>>
    // CHECK:       [[IN_SPARSE:%.+]] = VPUIP.Copy inputs([[IN_SPARSE_DDR]]
    // CHECK-SAME:                                 outputs([[IN_SPARSE_CMX]]

    // CHECK:       [[WEIGHTS_SPARSE_DDR:%.+]] = VPUIP.GroupSparseBuffer([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]]) {is_weights}
    // CHECK-SAME:      -> !VPUIP.SparseBuffer<data=memref<64x32x3x3xf16, #NHWC>,
    // CHECK-SAME:                             sparsity_map=memref<64x1x1x384xi1>, is_weights>
    // CHECK:       [[WEIGHTS_DATA_CMX:%.+]] = memref.alloc() : memref<64x32x3x3xf16, #NHWC, @CMX_NN>
    // CHECK:       [[WEIGHTS_SM_CMX:%.+]] = memref.alloc() : memref<64x1x1x384xi1, @CMX_NN>
    // CHECK:       [[WEIGHTS_SPARSE_CMX:%.+]] = VPUIP.GroupSparseBuffer([[WEIGHTS_DATA_CMX]], [[WEIGHTS_SM_CMX]])
    // CHECK-SAME:      -> !VPUIP.SparseBuffer<data=memref<64x32x3x3xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:                             sparsity_map=memref<64x1x1x384xi1, @CMX_NN>, is_weights>
    // CHECK:       [[WEIGHTS_SPARSE:%.+]] = VPUIP.Copy inputs([[WEIGHTS_SPARSE_DDR]]
    // CHECK-SAME:                                      outputs([[WEIGHTS_SPARSE_CMX]]

    // CHECK:       [[WEIGHTS_TABLE_CMX:%.+]] = memref.alloc() : memref<64x1x1x4xsi32, @CMX_NN>
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = VPUIP.Copy inputs([[CST_WEIGHTS_TABLE]]
    // CHECK-SAME:                                     outputs([[WEIGHTS_TABLE_CMX]]

    // CHECK:       [[OUT_BUF:%.+]] = memref.alloc() : memref<1x64x14x14xf16, #NHWC, @CMX_NN>
    // CHECK:       [[OUT_SM_BUF:%.+]] = memref.alloc() : memref<1x64x14x14xi1, #NHWC, @CMX_NN>

    // CHECK:       [[INPUT:%.+]], [[INPUT_SM:%.+]] = VPUIP.UngroupSparseBuffer([[IN_SPARSE]])
    // CHECK-SAME:      -> memref<1x32x16x16xf16, #NHWC, @CMX_NN>, memref<1x32x16x16xi1, #NHWC, @CMX_NN>
    // CHECK:       [[WEIGHTS:%.+]], [[WEIGHTS_SM:%.+]] = VPUIP.UngroupSparseBuffer([[WEIGHTS_SPARSE]])
    // CHECK-SAME:      -> memref<64x32x3x3xf16, #NHWC, @CMX_NN>, memref<64x1x1x384xi1, @CMX_NN>

    // CHECK:       [[OUT:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:          {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME:          input([[INPUT]]
    // CHECK-SAME:          input_sparsity_map([[INPUT_SM]]
    // CHECK-SAME:          weights([[WEIGHTS]]
    // CHECK-SAME:          weights_sparsity_map([[WEIGHTS_SM]]
    // CHECK-SAME:          weight_table([[WEIGHTS_TABLE]]
    // CHECK-SAME:          parent_input([[INPUT]]
    // CHECK-SAME:          parent_input_sparsity_map([[INPUT_SM]]
    // CHECK-SAME:          parent_output([[OUT_BUF]]
    // CHECK-SAME:          parent_output_sparsity_map([[OUT_SM_BUF]]
    // CHECK-SAME:          outputs([[OUT_BUF]]
    // CHECK-SAME:          output_sparsity_map([[OUT_SM_BUF]]
    // CHECK-SAME:      -> memref<1x64x14x14xf16, #NHWC, @CMX_NN>, memref<1x64x14x14xi1, #NHWC, @CMX_NN> variants :  {
    // CHECK:               DPUTask {mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 15, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:           } PPE :  {
    // CHECK:           }

    // CHECK:       [[OUT_SPARSE_CMX:%.+]] = VPUIP.GroupSparseBuffer([[OUT]]#0, [[OUT]]#1)
    // CHECK-SAME:      -> !VPUIP.SparseBuffer<data=memref<1x64x14x14xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:                             sparsity_map=memref<1x64x14x14xi1, #NHWC, @CMX_NN>>

    // CHECK:       [[OUT_DATA_DDR:%.+]] = memref.alloc() : memref<1x64x14x14xf16, #NHWC>
    // CHECK:       [[OUT_SM_DDR:%.+]] = memref.alloc() : memref<1x64x14x14xi1, #NHWC>
    // CHECK:       [[OUT_SPARSE_DDR:%.+]] = VPUIP.GroupSparseBuffer([[OUT_DATA_DDR]], [[OUT_SM_DDR]])
    // CHECK-SAME:      -> !VPUIP.SparseBuffer<data=memref<1x64x14x14xf16, #NHWC>,
    // CHECK-SAME:                             sparsity_map=memref<1x64x14x14xi1, #NHWC>>
    // CHECK:       [[OUT_SPARSE:%.+]] = VPUIP.Copy inputs([[OUT_SPARSE_CMX]]
    // CHECK-SAME:                                  outputs([[OUT_SPARSE_DDR]]
    // CHECK:       [[RESULT:%.+]] = VPUIP.Copy inputs([[OUT_SPARSE]]
    // CHECK-SAME:                              outputs([[ARG2]]
    // CHECK:       return [[RESULT]] : !VPUIP.SparseBuffer<data=memref<1x64x14x14xf16, #NHWC>, sparsity_map=memref<1x64x14x14xi1, #NHWC>>
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPU.DistributedTensor<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!InputSMDistributed = !VPU.DistributedTensor<
    1x32x16x16xi1, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!WeightsDistributed = !VPU.DistributedTensor<
    64x32x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsSMDistributed = !VPU.DistributedTensor<
    64x1x1x384xi1, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = !VPU.DistributedTensor<
    64x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x64x14x14xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!OutputSMDistributed = !VPU.DistributedTensor<
    1x64x14x14xi1, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!Input_DDR = tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}>
!InputSM_DDR = tensor<1x32x16x16xi1, {mem_space = @DDR, order = #NHWC}>
!Weights_DDR = tensor<64x32x3x3xf16, {mem_space = @DDR, order = #NHWC}>
!WeightsSM_DDR = tensor<64x1x1x384xi1, {mem_space = @DDR}>
!WeightsTable_DDR = tensor<64x1x1x4xsi32, {mem_space = @DDR}>
!Output_DDR = tensor<1x64x14x14xf16, {mem_space = @DDR, order = #NHWC}>
!OutputSM_DDR = tensor<1x64x14x14xi1, {mem_space = @DDR, order = #NHWC}>

!Input_CMX = tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
!InputSM_CMX = tensor<1x32x16x16xi1, {mem_space = @CMX_NN, order = #NHWC}>
!Weights_CMX = tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsSM_CMX = tensor<64x1x1x384xi1, {mem_space = @CMX_NN}>
!WeightsTable_CMX = tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}>
!Output_CMX = tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
!OutputSM_CMX = tensor<1x64x14x14xi1, {mem_space = @CMX_NN, order = #NHWC}>

module {
  net.NetworkInfo entryPoint : @SparseNCEConvSOH inputsInfo : {
    DataInfo "Parameter1" : tensor<1x32x16x16xf16>
    DataInfo "Parameter2" : tensor<1x32x16x16xi1>
  } outputsInfo : {
    DataInfo "Result" : tensor<1x64x14x14xf16>
  }
// CHECK:       func.func @SparseNCEConvSOH([[ARG0:%.+]]: memref<1x32x16x16xf16, #NHWC, @DDR>,
// CHECK-SAME:                         [[ARG1:%.+]]: memref<1x32x16x16xi1, #NHWC, @DDR>,
// CHECK-SAME:                         [[ARG2:%.+]]: !VPUIP.SparseBuffer<data=memref<1x64x14x14xf16, #NHWC, @DDR>,
// CHECK-SAME:                                                           sparsity_map=memref<1x64x14x14xi1, #NHWC, @DDR>>)
// CHECK-SAME:       -> !VPUIP.SparseBuffer<data=memref<1x64x14x14xf16, #NHWC, @DDR>, sparsity_map=memref<1x64x14x14xi1, #NHWC, @DDR>> {
func.func @SparseNCEConvSOH(%arg0 : !Input_DDR, %arg1 : !InputSM_DDR) -> !VPU.SparseTensor<data=!Output_DDR, sparsity_map=!OutputSM_DDR> {
    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1) -> !VPU.SparseTensor<data=!Input_DDR, sparsity_map=!InputSM_DDR>

    %weights = const.Declare !Weights_DDR = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare !WeightsSM_DDR = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
        -> !VPU.SparseTensor<data=!Weights_DDR, sparsity_map=!WeightsSM_DDR, is_weights>

    %weights_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<64x1x1x4xsi32>

    %input_sparse_cmx = VPU.Copy(%input_sparse) {out_mem_space = @CMX_NN} : !VPU.SparseTensor<data=!Input_DDR, sparsity_map=!InputSM_DDR>
            -> !VPU.SparseTensor<data=!InputDistributed, sparsity_map=!InputSMDistributed>

    %weights_sparse_cmx = VPU.Copy(%weights_sparse) {out_mem_space = @CMX_NN} : !VPU.SparseTensor<data=!Weights_DDR, sparsity_map=!WeightsSM_DDR, is_weights>
            -> !VPU.SparseTensor<data=!WeightsDistributed, sparsity_map=!WeightsSMDistributed, is_weights>

    %weights_table_sparse_cmx = VPU.Copy(%weights_table) {out_mem_space = @CMX_NN} : !WeightsTable_DDR -> !WeightsTableDistributed

    %output_sparse_cmx = VPU.NCE.Convolution(%input_sparse_cmx, %weights_sparse_cmx, %weights_table_sparse_cmx) {
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                rawFilterShape = [64, 32, 3, 3],
                strides = [1, 1]
            } : !VPU.SparseTensor<data=!InputDistributed, sparsity_map=!InputSMDistributed>, !VPU.SparseTensor<data=!WeightsDistributed, sparsity_map=!WeightsSMDistributed, is_weights>, !WeightsTableDistributed -> !VPU.SparseTensor<data=!OutputDistributed, sparsity_map=!OutputSMDistributed> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
        }

    %output_sparse = VPU.Copy(%output_sparse_cmx) {out_mem_space = @DDR} : !VPU.SparseTensor<data=!OutputDistributed, sparsity_map=!OutputSMDistributed>
            -> !VPU.SparseTensor<data=!Output_DDR, sparsity_map=!OutputSM_DDR>

    return %output_sparse : !VPU.SparseTensor<data=!Output_DDR, sparsity_map=!OutputSM_DDR>

    // CHECK-DAG:       [[CST_WEIGHTS_TABLE:%.+]] = const.Declare memref<64x1x1x4xsi32, @DDR> = dense<1> : tensor<64x1x1x4xsi32>

    // CHECK-DAG:       [[CST_WEIGHTS_SM:%.+]] = const.Declare memref<64x1x1x384xi1, @DDR> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    // CHECK-DAG:       [[CST_WEIGHTS:%.+]] = const.Declare memref<64x32x3x3xf16, #NHWC, @DDR> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]

    // CHECK:       [[INPUT_SPARSE_DDR:%.+]] = VPUIP.GroupSparseBuffer([[ARG0]], [[ARG1]])
    // CHECK-SAME:      -> !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, #NHWC, @DDR>,
    // CHECK-SAME:                             sparsity_map=memref<1x32x16x16xi1, #NHWC, @DDR>>
    // CHECK:       [[WEIGHTS_SPARSE_DDR:%.+]] = VPUIP.GroupSparseBuffer([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]]) {is_weights}
    // CHECK-SAME:      -> !VPUIP.SparseBuffer<data=memref<64x32x3x3xf16, #NHWC, @DDR>,
    // CHECK-SAME:                             sparsity_map=memref<64x1x1x384xi1, @DDR>, is_weights>

    // CHECK:       [[INPUT_DIST:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[INPUT_SM_DIST:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x16x16xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[INPUT_SPARSE_DIST:%.+]] = VPUIP.GroupSparseBuffer([[INPUT_DIST]], [[INPUT_SM_DIST]])
    // CHECK:       [[INPUT_SPARSE_DIST_CMX:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[INPUT_SPARSE_DDR]] : !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, #NHWC, @DDR>, sparsity_map=memref<1x32x16x16xi1, #NHWC, @DDR>>)
    // CHECK-SAME:          outputs([[INPUT_SPARSE_DIST]] : !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x32x16x16xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>>)
    // CHECK-SAME:      -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                             sparsity_map=!VPUIP.DistributedBuffer<1x32x16x16xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>>

    // CHECK:       [[WEIGHTS_DIST:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<64x32x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[WEIGHTS_SM_DIST:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<64x1x1x384xi1, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[WEIGHTS_SPARSE_DIST:%.+]] = VPUIP.GroupSparseBuffer([[WEIGHTS_DIST]], [[WEIGHTS_SM_DIST]])
    // CHECK:       [[WEIGHTS_SPARSE_DIST_CMX:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[WEIGHTS_SPARSE_DDR]]
    // CHECK-SAME:          outputs([[WEIGHTS_SPARSE_DIST]]
    // CHECK-SAME:      -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<64x32x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
    // CHECK-SAME:                             sparsity_map=!VPUIP.DistributedBuffer<64x1x1x384xi1, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
    // CHECK-SAME:                             is_weights>

    // CHECK:       [[WEIGHTS_TABLE_DIST:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<64x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[CST_WEIGHTS_TABLE]]
    // CHECK-SAME:          outputs([[WEIGHTS_TABLE_DIST]]
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:       [[OUTPUT_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[OUTPUT_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x14x14xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[INPUT_DATA:%.+]], [[INPUT_SM:%.+]] = VPUIP.UngroupSparseBuffer([[INPUT_SPARSE_DIST_CMX]])
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:         !VPUIP.DistributedBuffer<1x32x16x16xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[WEIGHTS_DATA:%.+]], [[WEIGHTS_SM:%.+]] = VPUIP.UngroupSparseBuffer([[WEIGHTS_SPARSE_DIST_CMX]])
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<64x32x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK-SAME:         !VPUIP.DistributedBuffer<64x1x1x384xi1, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:       [[CONV_OUTPUT:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:              {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME:              input([[INPUT_DATA]]
    // CHECK-SAME:              input_sparsity_map([[INPUT_SM]]
    // CHECK-SAME:              weights([[WEIGHTS_DATA]]
    // CHECK-SAME:              weights_sparsity_map([[WEIGHTS_SM]]
    // CHECK-SAME:              weight_table([[WEIGHTS_TABLE]]
    // CHECK-SAME:              parent_input([[INPUT_DATA]]
    // CHECK-SAME:              parent_input_sparsity_map([[INPUT_SM]]
    // CHECK-SAME:              parent_output([[OUTPUT_DATA]]
    // CHECK-SAME:              parent_output_sparsity_map([[OUTPUT_SM]]
    // CHECK-SAME:              outputs([[OUTPUT_DATA]]
    // CHECK-SAME:              output_sparsity_map([[OUTPUT_SM]]
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x64x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:          !VPUIP.DistributedBuffer<1x64x14x14xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> variants : {
    // CHECK:                   DPUTask {mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 15, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:               } PPE :  {
    // CHECK:               }

    // CHECK:       [[CONV_OUTPUT_SPARSE:%.+]] = VPUIP.GroupSparseBuffer([[CONV_OUTPUT]]#0, [[CONV_OUTPUT]]#1)

    // CHECK:       [[OUTPUT_BUF_DDR:%.+]] = memref.alloc() : memref<1x64x14x14xf16, #NHWC, @DDR>
    // CHECK:       [[OUTPUT_SM_BUF_DDR:%.+]] = memref.alloc() : memref<1x64x14x14xi1, #NHWC, @DDR>
    // CHECK:       [[OUTPUT_BUF_SPARSE:%.+]] = VPUIP.GroupSparseBuffer([[OUTPUT_BUF_DDR]], [[OUTPUT_SM_BUF_DDR]])
    // CHECK:       [[OUTPUT_SPARSE_DDR:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[CONV_OUTPUT_SPARSE]]
    // CHECK-SAME:          outputs([[OUTPUT_BUF_SPARSE]]
    // CHECK-SAME:      -> !VPUIP.SparseBuffer<data=memref<1x64x14x14xf16, #NHWC, @DDR>, sparsity_map=memref<1x64x14x14xi1, #NHWC, @DDR>>

    // CHECK:       [[RESULT:%.+]] = VPUIP.Copy inputs([[OUTPUT_SPARSE_DDR]]
    // CHECK-SAME:                              outputs([[ARG2]]
    // CHECK:       return [[RESULT]] : !VPUIP.SparseBuffer<data=memref<1x64x14x14xf16, #NHWC, @DDR>, sparsity_map=memref<1x64x14x14xi1, #NHWC, @DDR>>
}
}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
  net.NetworkInfo entryPoint : @SparseNCEConvSETable inputsInfo : {
    DataInfo "Parameter1" : tensor<1x32x16x16xf16>
    DataInfo "Parameter2" : tensor<1x32x16x16xi1>
  } outputsInfo : {
    DataInfo "Result" : tensor<1x64x14x14xf16>
  }
func.func @SparseNCEConvSETable(%arg0 : tensor<1x32x16x16xf16, {order = #NHWC}>, %arg1 : tensor<1x32x16x16xi1, {order = #NHWC}>)
        -> tensor<1x64x14x14xf16, {order = #NHWC}> {
    %se_table = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 32, 16, 16], seDepth = 1 : i64, seSize = [32]} -> tensor<1x1x16x16xi32, {order = #NHWC}>
    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1, %se_table)
        -> !VPU.SparseTensor<data=tensor<1x32x16x16xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x32x16x16xi1, {order = #NHWC}>,
                             storage_element_table=tensor<1x1x16x16xi32, {order = #NHWC}>>
    %input_sparse_cmx = VPU.Copy(%input_sparse) {out_mem_space = @CMX_NN} :
           !VPU.SparseTensor<data=tensor<1x32x16x16xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x32x16x16xi1, {order = #NHWC}>,
                             storage_element_table=tensor<1x1x16x16xi32, {order = #NHWC}>>
        -> !VPU.SparseTensor<data=tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                             sparsity_map=tensor<1x32x16x16xi1, {mem_space = @CMX_NN, order = #NHWC}>,
                             storage_element_table=tensor<1x1x16x16xi32, {mem_space = @CMX_NN, order = #NHWC}>>

    %weights = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_cmx = VPU.Copy(%weights) {out_mem_space = @CMX_NN} :
           tensor<64x32x3x3xf16, {order = #NHWC}>
        -> tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %weights_table_cmx = VPU.Copy(%weights_table) {out_mem_space = @CMX_NN} :
            tensor<64x1x1x4xsi32> -> tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}>

    %output_cmx = VPU.NCE.Convolution(%input_sparse_cmx, %weights_cmx, %weights_table_cmx) {
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [64, 32, 3, 3],
            strides = [1, 1]
        } : !VPU.SparseTensor<data=tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x32x16x16xi1, {mem_space = @CMX_NN, order = #NHWC}>, storage_element_table=tensor<1x1x16x16xi32, {mem_space = @CMX_NN, order = #NHWC}>>, tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}> -> tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
        }
    %output = VPU.Copy(%output_cmx) :
           tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x64x14x14xf16, {order = #NHWC}>

    return %output : tensor<1x64x14x14xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare memref<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare memref<64x32x3x3xf16, #NHWC> = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]
    // CHECK:       [[SE_TABLE:%.+]] = VPUIP.StorageElementTable
    // CHECK-SAME:      {dataElemType = f16, dataShape = [1, 32, 16, 16], seDepth = 1 : i64, seSize = [32]} -> memref<1x1x16x16xi32, #NHWC>
    // CHECK:       [[GROUP_SPARSE_BUFFER:%.+]] = VPUIP.GroupSparseBuffer(%arg0, %arg1, [[SE_TABLE]])
    // CHECK-SAME:      -> !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, #NHWC>,
    // CHECK-SAME:                             sparsity_map=memref<1x32x16x16xi1, #NHWC>,
    // CHECK-SAME:                             storage_element_table=memref<1x1x16x16xi32, #NHWC>>

    // CHECK:       [[DATA_ALLOC:%.+]] = memref.alloc() : memref<1x32x16x16xf16, #NHWC, @CMX_NN>
    // CHECK:       [[SPARSITY_MAP_ALLOC:%.+]] = memref.alloc() : memref<1x32x16x16xi1, #NHWC, @CMX_NN>
    // CHECK:       [[SE_TABLE_ALLOC:%.+]] = memref.alloc() : memref<1x1x16x16xi32, #NHWC, @CMX_NN>
    // CHECK:       [[GROUP_SP_BUFF_ALLOC:%.+]] = VPUIP.GroupSparseBuffer([[DATA_ALLOC]], [[SPARSITY_MAP_ALLOC]], [[SE_TABLE_ALLOC]])
    // CHECK-SAME:      -> !VPUIP.SparseBuffer<
    // CHECK-SAME:      data=memref<1x32x16x16xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:      sparsity_map=memref<1x32x16x16xi1, #NHWC, @CMX_NN>,
    // CHECK-SAME:      storage_element_table=memref<1x1x16x16xi32, #NHWC, @CMX_NN>>

    // CHECK:       [[IN_SPARSE:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[GROUP_SPARSE_BUFFER]] : !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, #NHWC>,
    // CHECK-SAME:                                                           sparsity_map=memref<1x32x16x16xi1, #NHWC>,
    // CHECK-SAME:                                                           storage_element_table=memref<1x1x16x16xi32, #NHWC>>)
    // CHECK-SAME:      outputs([[GROUP_SP_BUFF_ALLOC]] : !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:                                                            sparsity_map=memref<1x32x16x16xi1, #NHWC, @CMX_NN>,
    // CHECK-SAME:                                                            storage_element_table=memref<1x1x16x16xi32, #NHWC, @CMX_NN>>)
    // CHECK-SAME:      -> !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:                             sparsity_map=memref<1x32x16x16xi1, #NHWC, @CMX_NN>,
    // CHECK-SAME:                             storage_element_table=memref<1x1x16x16xi32, #NHWC, @CMX_NN>>


    // CHECK:       [[WEIGHTS_ALLOC:%.+]] = memref.alloc() : memref<64x32x3x3xf16, #NHWC, @CMX_NN>
    // CHECK:       [[WEIGHTS_CMX:%.+]] = VPUIP.Copy inputs([[WEIGHTS]] : memref<64x32x3x3xf16, #NHWC>)
    // CHECK-SAME:      outputs([[WEIGHTS_ALLOC]] : memref<64x32x3x3xf16, #NHWC, @CMX_NN>) -> memref<64x32x3x3xf16, #NHWC, @CMX_NN>

    // CHECK:       [[WT_ALLOC:%.+]] = memref.alloc() : memref<64x1x1x4xsi32, @CMX_NN>
    // CHECK:       [[WT_CMX:%.+]] = VPUIP.Copy inputs([[WEIGHTS_TABLE]] : memref<64x1x1x4xsi32>)
    // CHECK-SAME:      outputs([[WT_ALLOC]] : memref<64x1x1x4xsi32, @CMX_NN>) -> memref<64x1x1x4xsi32, @CMX_NN>

    // CHECK:       [[OUT_NCE_CLUSTER:%.+]] = memref.alloc() : memref<1x64x14x14xf16, #NHWC, @CMX_NN>
    // CHECK:       [[INPUT:%.+]], [[INPUT_SM:%.+]],  [[INPUT_SE:%.+]] = VPUIP.UngroupSparseBuffer([[IN_SPARSE]])
    // CHECK-SAME:      -> memref<1x32x16x16xf16, #NHWC, @CMX_NN>, memref<1x32x16x16xi1, #NHWC, @CMX_NN>, memref<1x1x16x16xi32, #NHWC, @CMX_NN>
    // CHECK:       [[RESULT_CMX:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      kernel_size = [3, 3], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME:      input([[INPUT]] : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      input_sparsity_map([[INPUT_SM]] : memref<1x32x16x16xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:      input_storage_element_table([[INPUT_SE]] : memref<1x1x16x16xi32, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weights([[WEIGHTS_CMX]] : memref<64x32x3x3xf16, #NHWC, @CMX_NN>) weight_table([[WT_CMX]] : memref<64x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      parent_input([[INPUT]] : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_input_sparsity_map([[INPUT_SM]] : memref<1x32x16x16xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_input_storage_element_table([[INPUT_SE]] : memref<1x1x16x16xi32, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_output([[OUT_NCE_CLUSTER]] : memref<1x64x14x14xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[OUT_NCE_CLUSTER]] : memref<1x64x14x14xf16, #NHWC, @CMX_NN>) -> memref<1x64x14x14xf16, #NHWC, @CMX_NN> variants : {
    // CHECK:           DPUTask {mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 15, 31], outStart = [0, 0, 0],
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}

    // CHECK:       [[RESULT_ALLOC:%.+]] = memref.alloc() : memref<1x64x14x14xf16, #NHWC>
    // CHECK:       [[RESULT:%.+]] = VPUIP.Copy inputs([[RESULT_CMX]] : memref<1x64x14x14xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[RESULT_ALLOC]] : memref<1x64x14x14xf16, #NHWC>) -> memref<1x64x14x14xf16, #NHWC>
    // CHECK:       [[RESULT_OUT:%.+]] = VPUIP.Copy inputs([[RESULT]] : memref<1x64x14x14xf16, #NHWC>)
    // CHECK-SAME:      outputs(%arg2 : memref<1x64x14x14xf16, #NHWC>) -> memref<1x64x14x14xf16, #NHWC>

    // CHECK:       return [[RESULT_OUT]] : memref<1x64x14x14xf16, #NHWC>
}
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
module {
  net.NetworkInfo entryPoint : @NCEMatMul inputsInfo : {
    DataInfo "Parameter" : tensor<64x1x32x35x1xf16>
  } outputsInfo : {
    DataInfo "Result" : tensor<64x1x64x35x1xf16>
  }
// CHECK: func.func @NCEMatMul([[ARG0:%.+]]: memref<64x1x32x35x1xf16, #GNHWC, @CMX_NN>, [[ARG1:%.+]]: memref<64x1x64x35x1xf16, #GNHWC, @CMX_NN>) -> memref<64x1x64x35x1xf16, #GNHWC, @CMX_NN> {
func.func @NCEMatMul(%arg0 : tensor<64x1x32x35x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>) -> tensor<64x1x64x35x1xf16, {mem_space = @CMX_NN, order = #GNHWC}> {
    %weights = const.Declare tensor<64x64x32x1x1xf16, {mem_space = @CMX_NN, order = #GNHWC}> = dense<1.000000e+00> : tensor<64x64x32x1x1xf16, {mem_space = @CMX_NN}>, [#const.Reorder<#GNHWC>]
    %weights_table = const.Declare tensor<64x64x1x1x4xsi32, {mem_space = @CMX_NN}> = dense<1> : tensor<64x64x1x1x4xsi32, {mem_space = @CMX_NN}>

    %1 = VPU.NCE.MatMul(%arg0, %weights, %weights_table) {
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                rawFilterShape = [64, 64, 32, 1, 1], strides = [1, 1]
            } -> tensor<64x1x64x35x1xf16, {mem_space = @CMX_NN, order = #GNHWC}> {
        VPU.DPU.Workload inOffsets [0, 0, 0, 0, 0] inSizes [64, 1, 32, 35, 1] outOffsets [0, 0, 0, 0, 0]
         outSizes [64, 1, 64, 35, 1] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16>
    }
    return %1 : tensor<64x1x64x35x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare memref<64x64x32x1x1xf16, #GNHWC, @CMX_NN> = dense<1.000000e+00> : tensor<64x64x32x1x1xf16,
    // CHECK-SAME:  {mem_space = @CMX_NN}>, [#const.Reorder<#GNHWC>]
    // CHECK:       [[WEIGHT_TABLE:%.+]] = const.Declare memref<64x64x1x1x4xsi32, @CMX_NN> = dense<1> : tensor<64x64x1x1x4xsi32, {mem_space = @CMX_NN}>
    // CHECK:       [[OUTPUT_BUF:%.+]] = memref.alloc() : memref<64x1x64x35x1xf16, #GNHWC, @CMX_NN>

    // CHECK:       [[OUT:%.+]] = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:  kernel_size = [1, 1], kernel_strides = [1, 1],
    // CHECK-SAME:  task_type = #VPUIP.nce_task_type<CONV>} input([[ARG0]] : memref<64x1x32x35x1xf16, #GNHWC, @CMX_NN>)
    // CHECK-SAME:  weights([[WEIGHTS]] : memref<64x64x32x1x1xf16, #GNHWC, @CMX_NN>) weight_table([[WEIGHT_TABLE]] : memref<64x64x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:  parent_input([[ARG0]] : memref<64x1x32x35x1xf16, #GNHWC, @CMX_NN>) parent_output([[OUTPUT_BUF]] : memref<64x1x64x35x1xf16, #GNHWC, @CMX_NN>)
    // CHECK-SAME:  outputs([[OUTPUT_BUF]] : memref<64x1x64x35x1xf16, #GNHWC, @CMX_NN>) -> memref<64x1x64x35x1xf16, #GNHWC, @CMX_NN> variants : {

    // CHECK:       DPUTask {inEnd = [0, 34, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [0, 34, 63],
    // CHECK-SAME:  outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}

    // CHECK:       [[RESULT:%.+]] = VPUIP.Copy inputs([[OUT]] : memref<64x1x64x35x1xf16, #GNHWC, @CMX_NN>)
    // CHECK-SAME:  outputs([[ARG1]] : memref<64x1x64x35x1xf16, #GNHWC, @CMX_NN>) -> memref<64x1x64x35x1xf16, #GNHWC, @CMX_NN>
    // CHECK:       return [[RESULT]] : memref<64x1x64x35x1xf16, #GNHWC, @CMX_NN>

}
}
