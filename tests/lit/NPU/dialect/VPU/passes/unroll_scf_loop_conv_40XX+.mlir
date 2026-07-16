//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile allow-custom-values=true" --unroll-scf-loop="loop-unroll-factor=1,1,2,1 enable-cascaded-unrolling=false" --canonicalize  %s | FileCheck %s

// REQUIRES: platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<()[s0] -> (s0 - 48)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0, 48)>
#map2 = affine_map<(d0) -> (0, d0 - 1)>
#map3 = affine_map<(d0) -> (-d0 + 1, 0)>
#map4 = affine_map<()[s0] -> (1, s0)>
#map5 = affine_map<(d0, d1)[s0] -> (0, d0 + d1 - s0 + 2)>
#map6 = affine_map<(d0, d1, d2) -> (d0 - d1 - d2 + 2)>
module @NPUModule {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "Parameter_13" : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
  } outputsInfo : {
    DataInfo "Convolution_15" friendlyName = "Result_16" : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
  }
  func.func @main_func1_dims_H_cases_2_static(%arg0: tensor<1x16x49x1280xf16, {order = #NHWC}> {func.dynamicStrides = true}) -> (tensor<1x16x48x1280xf16> {func.dynamicStrides = true}) {
    %cst = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<16x16x3x3xf16, {order = #NHWC}>, [#const.Sparsify<false>]
    %cst_0 = const.Declare tensor<16x1x1x256xi1> = dense<0.000000e+00> : tensor<16x16x3x3xf16, {order = #NHWC}>, [#const.GetSparsityMap]
    %0 = VPU.GroupSparseTensor(%cst, %cst_0) {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[134, 127, 130, 129, 130, 129, 134, 135, 131, 136, 136, 134, 132, 135, 125, 124]> : tensor<16xi64>, alignment = 16 : i64>} -> !VPU.SparseTensor<data=tensor<16x16x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x256xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[134, 127, 130, 129, 130, 129, 134, 135, 131, 136, 136, 134, 132, 135, 125, 124]> : tensor<16xi64>, alignment = 16 : i64>>
    %1 = VPU.NCE.Convolution(%arg0, %0) rawFilterShape [16, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x16x49x1280xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<16x16x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x256xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[134, 127, 130, 129, 130, 129, 134, 135, 131, 136, 136, 134, 132, 135, 125, 124]> : tensor<16xi64>, alignment = 16 : i64>> -> tensor<1x16x48x1280xf16>
    return %1 : tensor<1x16x48x1280xf16>
  }
  func.func @main_func1_dims_H_cases_1_static(%arg0: tensor<1x16x49x1280xf16, {order = #NHWC}> {func.dynamicStrides = true}) -> (tensor<1x16x48x1280xf16> {func.dynamicStrides = true}) {
    %cst = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<16x16x3x3xf16, {order = #NHWC}>, [#const.Sparsify<false>]
    %cst_0 = const.Declare tensor<16x1x1x256xi1> = dense<0.000000e+00> : tensor<16x16x3x3xf16, {order = #NHWC}>, [#const.GetSparsityMap]
    %0 = VPU.GroupSparseTensor(%cst, %cst_0) {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[134, 127, 130, 129, 130, 129, 134, 135, 131, 136, 136, 134, 132, 135, 125, 124]> : tensor<16xi64>, alignment = 16 : i64>} -> !VPU.SparseTensor<data=tensor<16x16x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x256xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[134, 127, 130, 129, 130, 129, 134, 135, 131, 136, 136, 134, 132, 135, 125, 124]> : tensor<16xi64>, alignment = 16 : i64>>
    %1 = VPU.NCE.Convolution(%arg0, %0) rawFilterShape [16, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x16x49x1280xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<16x16x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x256xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[134, 127, 130, 129, 130, 129, 134, 135, 131, 136, 136, 134, 132, 135, 125, 124]> : tensor<16xi64>, alignment = 16 : i64>> -> tensor<1x16x48x1280xf16>
    return %1 : tensor<1x16x48x1280xf16>
  }
  func.func @main_func1_dims_H_cases_0_static(%arg0: tensor<1x16x50x1280xf16, {order = #NHWC}> {func.dynamicStrides = true}) -> (tensor<1x16x48x1280xf16> {func.dynamicStrides = true}) {
    %cst = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<16x16x3x3xf16, {order = #NHWC}>, [#const.Sparsify<false>]
    %cst_0 = const.Declare tensor<16x1x1x256xi1> = dense<0.000000e+00> : tensor<16x16x3x3xf16, {order = #NHWC}>, [#const.GetSparsityMap]
    %0 = VPU.GroupSparseTensor(%cst, %cst_0) {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[134, 127, 130, 129, 130, 129, 134, 135, 131, 136, 136, 134, 132, 135, 125, 124]> : tensor<16xi64>, alignment = 16 : i64>} -> !VPU.SparseTensor<data=tensor<16x16x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x256xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[134, 127, 130, 129, 130, 129, 134, 135, 131, 136, 136, 134, 132, 135, 125, 124]> : tensor<16xi64>, alignment = 16 : i64>>
    %1 = VPU.NCE.Convolution(%arg0, %0) rawFilterShape [16, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x16x50x1280xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<16x16x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x256xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[134, 127, 130, 129, 130, 129, 134, 135, 131, 136, 136, 134, 132, 135, 125, 124]> : tensor<16xi64>, alignment = 16 : i64>> -> tensor<1x16x48x1280xf16>
    return %1 : tensor<1x16x48x1280xf16>
  }

  // CHECK:   func.func @merged_vpu_func_0_1([[VAL_0:%.+]]: tensor<1x16x97x1280xf16
  // CHECK:           [[VAL_1:%.+]] = VPU.Slice [[VAL_0]] [0, 0, 0, 0] [1, 16, 50, 1280]
  // CHECK:           [[VAL_2:%.+]] = VPU.NCE.Convolution([[VAL_1]]
  // CHECK-SAME:        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK:           [[VAL_3:%.+]] = VPU.Slice [[VAL_0]] [0, 0, 48, 0] [1, 16, 49, 1280]
  // CHECK:           [[VAL_4:%.+]] = VPU.NCE.Convolution([[VAL_3]]
  // CHECK-SAME:        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>
  // CHECK:           [[VAL_5:%.+]] = VPU.Concat([[VAL_2]], [[VAL_4]])
  // CHECK:           return [[VAL_5]] : tensor<1x16x96x1280xf16>
  // CHECK:         }

  // CHECK-DAG:   func.func @merged_vpu_func_0_0
  // CHECK-DAG:   func.func @merged_vpu_func_2_1
  // CHECK-DAG:   func.func @merged_vpu_func_2_0
  // CHECK-DAG:   func.func @main_func1_dims_H_cases_2_static
  // CHECK-DAG:   func.func @main_func1_dims_H_cases_1_static
  // CHECK-DAG:   func.func @main_func1_dims_H_cases_0_static

  func.func @main(%arg0: tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c21 = arith.constant 21 : index
    %c48 = arith.constant 48 : index

    %dim = tensor.dim %arg0, %c2 : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    %0 = tensor.empty(%dim) : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>

    %3 = scf.for %arg1 = %c0 to %dim step %c48 iter_args(%arg2 = %0) -> (tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>) {
      %5 = arith.addi %arg1, %c48 : index
      %6 = arith.cmpi sgt, %5, %dim : index
      %7 = scf.if %6 -> (index) {
        %20 = affine.apply #map()[%dim]
        scf.yield %20 : index
      } else {
        scf.yield %arg1 : index
      }
      %8 = affine.min #map1(%7)[%dim]
      %9 = affine.max #map2(%7)
      %10 = affine.max #map3(%7)
      %11 = affine.min #map4()[%10]
      %12 = affine.max #map5(%8, %9)[%dim]
      %13 = affine.min #map4()[%12]
      %14 = affine.apply #map6(%8, %11, %13)
      %15 = arith.cmpi eq, %9, %c0 : index
      %16 = scf.if %15 -> (index) {
        %20 = arith.cmpi eq, %14, %dim : index
        %21 = arith.select %20, %c3, %c2 : index
        scf.yield %21 : index
      } else {
        %20 = arith.addi %9, %14 : index
        %21 = arith.cmpi slt, %20, %dim : index
        %22 = arith.select %21, %c0, %c1 : index
        scf.yield %22 : index
      }
      %extracted_slice = tensor.extract_slice %arg0[0, 0, %9, 0] [1, 16, 49, 1280] [1, 1, 1, 1] : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x49x1280xf16, {order = #NHWC}>
      %cast = tensor.cast %extracted_slice : tensor<1x16x49x1280xf16, {order = #NHWC}> to tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 1280]> : tensor<4xsi64>, order = #NHWC}>
      %17 = arith.shli %16, %c2 : index
      %18 = arith.ori %c0, %17 : index
      %19 = scf.index_switch %18 -> tensor<1x16x48x1280xf16>
      case 0 {
        %extracted_slice_15 = tensor.extract_slice %arg0[0, 0, %9, 0] [1, 16, 50, 1280] [1, 1, 1, 1] : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x50x1280xf16, {order = #NHWC}>
        %20 = func.call @main_func1_dims_H_cases_0_static(%extracted_slice_15) : (tensor<1x16x50x1280xf16, {order = #NHWC}>) -> tensor<1x16x48x1280xf16>
        scf.yield %20 : tensor<1x16x48x1280xf16>
      }
      case 4 {
        %extracted_slice_15 = tensor.extract_slice %arg0[0, 0, %9, 0] [1, 16, 49, 1280] [1, 1, 1, 1] : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x49x1280xf16, {order = #NHWC}>
        %20 = func.call @main_func1_dims_H_cases_1_static(%extracted_slice_15) : (tensor<1x16x49x1280xf16, {order = #NHWC}>) -> tensor<1x16x48x1280xf16>
        scf.yield %20 : tensor<1x16x48x1280xf16>
      }
      case 8 {
        %extracted_slice_15 = tensor.extract_slice %arg0[0, 0, %9, 0] [1, 16, 49, 1280] [1, 1, 1, 1] : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x49x1280xf16, {order = #NHWC}>
        %20 = func.call @main_func1_dims_H_cases_2_static(%extracted_slice_15) : (tensor<1x16x49x1280xf16, {order = #NHWC}>) -> tensor<1x16x48x1280xf16>
        scf.yield %20 : tensor<1x16x48x1280xf16>
      }
      default {
        %false = arith.constant false
        cf.assert %false, "Unsupported case"
        %extracted_slice_15 = tensor.extract_slice %arg0[0, 0, %9, 0] [1, 16, 50, 1280] [1, 1, 1, 1] : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x50x1280xf16, {order = #NHWC}>
        %cast_16 = tensor.cast %extracted_slice_15 : tensor<1x16x50x1280xf16, {order = #NHWC}> to tensor<1x16x50x1280xf16, {order = #NHWC}>
        %20 = func.call @main_func1_dims_H_cases_0_static(%extracted_slice_15) : (tensor<1x16x50x1280xf16, {order = #NHWC}>) -> tensor<1x16x48x1280xf16>
        scf.yield %20 : tensor<1x16x48x1280xf16>
      }
      %cast_14 = tensor.cast %19 : tensor<1x16x48x1280xf16> to tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 1280]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %cast_14 into %arg2[0, 0, %7, 0] [1, 16, %c48, 1280] [1, 1, 1, 1] : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 1280]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %3 : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
  }
// CHECK:             func.func @main([[ARG_0:%.+]]: tensor<1x16x?x1280xf16
// CHECK-DAG:           [[CST_0:%.+]] = arith.constant 0 : index
// CHECK-DAG:           [[CST_2:%.+]] = arith.constant 2 : index
// CHECK-DAG:           [[STEP_INITIAL_CST:%.+]] = arith.constant 47 : index
// CHECK-DAG:           [[STEP_INCREASED_CST:%.+]] = arith.constant 48 : index
// CHECK-DAG:           [[STEP_UNROLLED_CST:%.+]] = arith.constant 96 : index

// CHECK:               [[DIM:%.+]] = tensor.dim [[ARG_0]], [[CST_2:%.+]] : tensor<1x16x?x1280xf16
// CHECK:               [[OUTPUT_TENSOR:%.+]] = tensor.empty([[DIM]]) : tensor<1x16x?x1280xf16

// CHECK:               [[CEIL_DIV_PREP:%.+]] = arith.addi [[DIM]], [[STEP_INITIAL_CST]] : index
// CHECK:               [[TOTAL_ITERATIONS:%.+]] = arith.divui [[CEIL_DIV_PREP]], [[STEP_INCREASED_CST]] : index
// CHECK:               [[REMAINDER:%.+]] = arith.remsi [[TOTAL_ITERATIONS]], [[CST_2]] : index
// CHECK:               [[ALIGNED_ITERATIONS:%.+]] = arith.subi [[TOTAL_ITERATIONS]], [[REMAINDER]] : index
// CHECK:               [[RAW_UPPERBOUND:%.+]] = arith.muli [[ALIGNED_ITERATIONS]], [[STEP_INCREASED_CST]] : inde
// CHECK:               [[CAN_EXECUTE:%.+]] = arith.cmpi uge, [[DIM]], [[STEP_UNROLLED_CST]] : index
// CHECK:               [[BASE_SAFE_UPPERBOUND:%.+]] = arith.minui [[RAW_UPPERBOUND]], [[DIM]] : index
// CHECK:               [[SAFE_UPPERBOUND:%.+]] = arith.select [[CAN_EXECUTE]], [[BASE_SAFE_UPPERBOUND]], [[CST_0]] : index

// CHECK:               [[MAIN_LOOP_RESULT:%.+]] = scf.for {{.+}} = [[CST_0]] to [[SAFE_UPPERBOUND]] step [[STEP_UNROLLED_CST]] iter_args({{.+}} = [[OUTPUT_TENSOR]]
// CHECK:                 scf.index_switch
// CHECK-COUNT-5:           func.call @merged_vpu_func

// CHECK:               [[RESIDUAL_RESULT:%.+]] = scf.for {{.+}} = [[SAFE_UPPERBOUND]] to [[DIM]] step [[STEP_INCREASED_CST]] iter_args({{.+}} = [[MAIN_LOOP_RESULT]]
// CHECK:                 scf.index_switch
// CHECK-COUNT-4:           func.call @main_func

// CHECK:               return [[RESIDUAL_RESULT]] : tensor<1x16x?x1280xf16
}
