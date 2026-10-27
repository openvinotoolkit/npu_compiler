//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true enable-is-reduce-supported" --scf-vertical-fusion="vf-merge-configuration=GREEDY" --resolve-shaped-type-result-dims --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

//CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (-d0 + 512, 112)>

// CHECK-LABEL: @MergeNCEReduce
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x32x175x512xf16>
func.func @MergeNCEReduce(%arg0: tensor<1x32x175x512xf16>) -> tensor<1x1x175x512xf16, {order = #NHWC}> {
   %0 = VPU.NCE.Permute(%arg0) {
     dstElemType = f16, dstOrder = #NHWC, expandedChannels = 32 : i64,
     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
     ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
     clamp_high = 3.4028234663852886E+38 : f64, scale = 5.000000e-01 : f64,
     prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64,
     adder = 0.000000e+00 : f64>, tilingStrategy = [1, 1, 1, 3]
   } -> tensor<1x32x175x512xf16, {order = #NHWC}>
   %1 = VPU.NCE.Reduce(%0) {
     axes = [1], input_padding = [0, 12, 0, 0],
     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
     op_type = #VPU.reduce_type<SUM>, ppe = #VPU.PPEFp<mode = <NOOP>,
     clamp_low = -3.4028234663852886E+38 : f64,
     clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
     prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64,
     adder = 0.000000e+00 : f64>, tilingStrategy = [1, 1, 1, 5]
   } -> tensor<1x1x175x512xf16, {order = #NHWC}>
   return %1 : tensor<1x1x175x512xf16, {order = #NHWC}>

   // CHECK-DAG:    [[C112:%.+]] = arith.constant 112 : index
   // CHECK-DAG:    [[C512:%.+]] = arith.constant 512 : index
   // CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
   // CHECK-DAG:    [[EMPT:%.+]] = tensor.empty() : tensor<1x1x175x512xf16, {order = #NHWC}>
   // CHECK:        [[LOOP:%.+]] = scf.for [[ITER:%.+]] = [[C0]] to [[C512]] step [[C112]] iter_args([[ARG2:%.+]] = [[EMPT]]) -> (tensor<1x1x175x512xf16, {order = #NHWC}>) {
   // CHECK-NEXT:      [[MIN:%.+]] = affine.min #[[$MAP1]]([[ITER]])
   // CHECK-NEXT:      [[IN_SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[ITER]]] [1, 32, 175, [[MIN]]] [1, 1, 1, 1] : tensor<1x32x175x512xf16> to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NCHW}>
   // CHECK-NEXT:      [[PERMUTE:%.+]] = VPU.NCE.Permute([[IN_SLICE]])
   // CHECK-SAME:           -> tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
   // CHECK-NEXT:      [[REDUCE:%.+]] = VPU.NCE.Reduce([[PERMUTE]])
   // CHECK-SAME:           axes = [1]
   // CHECK-SAME:           input_padding = [0, 12, 0, 0]
   // CHECK-SAME:           -> tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
   // CHECK-NEXT:      [[INSERT:%.+]] = tensor.insert_slice [[REDUCE]] into [[ARG2]][0, 0, 0, [[ITER]]] [1, 1, 175, [[MIN]]] [1, 1, 1, 1] : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x1x175x512xf16, {order = #NHWC}>
   // CHECK-NEXT:      scf.yield [[INSERT]] : tensor<1x1x175x512xf16, {order = #NHWC}>
   // CHECK-NEXT:   }
   // CHECK-NEXT:   return [[LOOP]] : tensor<1x1x175x512xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!outputStaticType = tensor<1x32x540x960x!quant.uniform<u8:f16, 0.030033121856988646:110>, {order = #NHWC}>

// CHECK-LABEL: @MergeSliceStaticInput
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x16x1080x1920xf16, {order = #NHWC}>
func.func @MergeSliceStaticInput(
    %arg0: tensor<1x16x1080x1920xf16, {order = #NHWC}>
  ) -> !outputStaticType {

  %weights_dw = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    = dense<1.0> : tensor<16x16x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

  %weights_cc = const.Declare tensor<32x1x1x32x!quant.uniform<u8:f16, 0.034925088695451328:128>, {order = #NHWC}>
    = dense<1> : tensor<32x3x3x3xsi8>,
      [#const.CastElemType<f16>,
       #const.CastElemType<!quant.uniform<i8:f16, 0.034925088695451328>>,
       #const.ConvertElemType<!quant.uniform<u8:f16, 0.034925088695451328:128>>,
       #const.Reorder<#NHWC>,
       #const.Reshape<[32, 1, 1, 27]>,
       #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 5]>]
  %bias_cc = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

  %depth = VPU.NCE.DepthConvolution(%arg0, %weights_dw) rawFilterShape [16, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64,
                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 507.0 : f64>,

    strides = [1, 1],
    tilingStrategy = [1, 1, 2, 1]
  } -> tensor<1x16x1080x1920x!quant.uniform<u8:f16, 0.0019697112195632038>, {order = #NHWC}>

  %slice = VPU.Slice %depth [0, 0, 0, 0] [1, 4, 1080, 1920]
    : tensor<1x16x1080x1920x!quant.uniform<u8:f16, 0.0019697112195632038>, {order = #NHWC}>
    to tensor<1x4x1080x1920x!quant.uniform<u8:f16, 0.0019697112195632038>, {order = #NHWC}>

  %compress = VPU.NCE.CompressConvolution(%slice, %weights_cc, %bias_cc) rawFilterShape [32, 3, 3, 3] {
    cm_sp_pattern = 7 : i64,
    multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
    pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64,
                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.0 : f64>,

    strides = [2, 2],
    tilingStrategy = [1, 1, 7, 3]
  } -> !outputStaticType

  return %compress : !outputStaticType
}

// CHECK:       [[C0_I8:%.+]] = arith.constant 0 : i8
// CHECK:       [[CSTEP:%.+]] = arith.constant {{[0-9]+}} : index
// CHECK:       [[C960:%.+]] = arith.constant 960 : index
// CHECK:       [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:   [[CST_W_DW:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
// CHECK-DAG:   [[CST_W_CC:%.+]] = const.Declare tensor<32x1x1x32x!qElemType{{.*}}, {order = #NHWC}>
// CHECK-DAG:   [[CST_B_CC:%.+]] = const.Declare tensor<32x1x1x4xsi32>
// CHECK:       [[EMPTY:%.+]] = tensor.empty() : tensor<1x32x540x960x!qElemType, {order = #NHWC}>
// CHECK:       [[LOOP:%.+]] = scf.for [[IV:%.+]] = [[C0]] to [[C960]] step [[CSTEP]] iter_args([[ARG_ITER:%.+]] = [[EMPTY]])
// CHECK:         [[EXTRACT:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, {{%.+}}] [1, 16, 1080, {{%.+}}] [1, 1, 1, 1]
// CHECK-SAME:      tensor<1x16x1080x1920xf16, {order = #NHWC}> to tensor<1x16x1080x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:         [[DEPTH:%.+]] = VPU.NCE.DepthConvolution([[EXTRACT]], [[CST_W_DW]])
// CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
// CHECK-SAME:      -> tensor<1x16x1080x?x!qElemType{{.*}}, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:         [[SLICE:%.+]] = VPU.Slice [[DEPTH]] [0, 0, 0, 0] [1, 4, 1080, -9223372036854775808]
// CHECK-SAME:      tensor<1x16x1080x?x!qElemType{{.*}}, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x4x1080x?x!qElemType{{.*}}, {bounds = #const.OpaqueI64Elements<[1, 4, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:         [[CAST:%.+]] = builtin.unrealized_conversion_cast [[C0_I8]] : i8 to !qElemType{{.*}}
// CHECK:         [[PAD:%.+]] = tensor.pad [[SLICE]] low[0, 0, 1, {{%.+}}] high[0, 0, 0, 0]
// CHECK:         [[COMPRESS:%.+]] = VPU.NCE.CompressConvolution([[PAD]], [[CST_W_CC]], [[CST_B_CC]])
// CHECK-SAME:      cm_sp_pattern = 7
// CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
// CHECK-SAME:      -> tensor<1x32x540x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:         [[INSERT:%.+]] = tensor.insert_slice {{%.+}} into [[ARG_ITER]][0, 0, 0, [[OUT_OFF_W:%.+]]] [1, 32, 540, [[TILE_W:.+]]] [1, 1, 1, 1]
// CHECK:         scf.yield [[INSERT]] : tensor<1x32x540x960x!qElemType, {order = #NHWC}>
// CHECK:       return [[LOOP]] : tensor<1x32x540x960x!qElemType, {order = #NHWC}>

// -----

// check no assertion error for group sparse tensor creation for a quantized model
func.func @SparseWeight(%input: tensor<1x?x3840x32xui8, {bounds = #const.OpaqueI64Elements<[1, 2160, 3840, 32]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>) -> tensor<1x?x3840x32xui8, {bounds = #const.OpaqueI64Elements<[1, 2160, 3840, 32]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}> {
  %cst_2 = const.Declare tensor<32x32x3x3x!quant.uniform<u8:f16, 0.002:147>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> = dense<1> : tensor<32x32x3x3xui8, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, [#const.CastElemType<!quant.uniform<u8:f16, 0.002:147>>, #const.Sparsify<false>]
  %cst_19 = const.Declare tensor<32x1x1x384xi1> = dense<1> : tensor<32x32x3x3xui8, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, [#const.CastElemType<!quant.uniform<u8:f16, 0.002:147>>, #const.GetSparsityMap]
  %0 = VPU.GroupSparseTensor(%cst_2, %cst_19) {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>} -> !VPU.SparseTensor<data=tensor<32x32x3x3x!quant.uniform<u8:f16, 0.002:147>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, sparsity_map=tensor<32x1x1x384xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>>
  %transposed = VPU.PermuteCast(%input) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>} : tensor<1x?x3840x32xui8, {bounds = #const.OpaqueI64Elements<[1, 2160, 3840, 32]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}> -> tensor<1x32x?x3840xui8, {bounds = #const.OpaqueI64Elements<[1, 32, 2160, 3840]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  %converted = VPU.QuantizeCast(%transposed) {dstElemType = !quant.uniform<u8:f16, 0.003921>} : tensor<1x32x?x3840xui8, {bounds = #const.OpaqueI64Elements<[1, 32, 2160, 3840]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x32x?x3840x!quant.uniform<u8:f16, 0.003921>, {bounds = #const.OpaqueI64Elements<[1, 32, 2160, 3840]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  %conv_weight = VPU.Slice %0 [0, 0, 0, 0] [32, 32, 3, 3] : !VPU.SparseTensor<data=tensor<32x32x3x3x!quant.uniform<u8:f16, 0.002:147>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, sparsity_map=tensor<32x1x1x384xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>> to !VPU.SparseTensor<data=tensor<32x32x3x3x!quant.uniform<u8:f16, 0.002:147>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, sparsity_map=tensor<32x1x1x384xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>>
  %conv_output = VPU.NCE.Convolution(%converted, %conv_weight) rawFilterShape [32, 32, 3, 3] {input_padding = [0, 0, 0, 0], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.190000e+02 : f64, clamp_high = 1.360000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 1.190000e+02 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], tilingStrategy = [1, 1, 8, 3]} : tensor<1x32x?x3840x!quant.uniform<u8:f16, 0.003921>, {bounds = #const.OpaqueI64Elements<[1, 32, 2160, 3840]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, !VPU.SparseTensor<data=tensor<32x32x3x3x!quant.uniform<u8:f16, 0.002:147>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, sparsity_map=tensor<32x1x1x384xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<1> : tensor<32xi64>, alignment = 16 : i64>> -> tensor<1x32x?x3840x!quant.uniform<u8:f16, 0.003921>, {bounds = #const.OpaqueI64Elements<[1, 32, 2160, 3840]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  %transposed_output = VPU.PermuteCast(%conv_output) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>} : tensor<1x32x?x3840x!quant.uniform<u8:f16, 0.003921>, {bounds = #const.OpaqueI64Elements<[1, 32, 2160, 3840]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x?x3840x32x!quant.uniform<u8:f16, 0.003921>, {bounds = #const.OpaqueI64Elements<[1, 2160, 3840, 32]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
  %output = VPU.QuantizeCast(%transposed_output) {dstElemType = ui8} : tensor<1x?x3840x32x!quant.uniform<u8:f16, 0.003921>, {bounds = #const.OpaqueI64Elements<[1, 2160, 3840, 32]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}> -> tensor<1x?x3840x32xui8, {bounds = #const.OpaqueI64Elements<[1, 2160, 3840, 32]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
  return %output : tensor<1x?x3840x32xui8, {bounds = #const.OpaqueI64Elements<[1, 2160, 3840, 32]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
}

// CHECK:   func.func @SparseWeight
// CHECK:       [[ARG_0:%.+]]: tensor<1x?x3840x32xui8, {bounds = #const.OpaqueI64Elements<[1, 2160, 3840, 32]> : tensor<4xsi64>
// CHECK:       [[C_1:%.+]] = arith.constant 1 : index
// CHECK:       [[C_0:%.+]] = arith.constant 0 : index
// CHECK:       [[CONST_0:%.+]] = const.Declare tensor<32x32x3x3x!qElemType
// CHECK:       [[CONST_1:%.+]] = const.Declare tensor<32x1x1x384xi1>
// CHECK:       [[SPARSE_W:%.+]] = VPU.GroupSparseTensor([[CONST_0]], [[CONST_1]])
// CHECK:       [[DIM_3:%.+]] = tensor.dim [[ARG_0]], [[C_1]]
// CHECK:       [[CONV_OUTPUT:%.+]] = scf.for [[ARG_1:%.+]] = [[C_0]] to [[DIM_3]]
// CHECK:           [[SLICE_0:%.+]] = tensor.extract_slice
// CHECK:           [[PADDED:%.+]] = tensor.pad
// CHECK:           [[OUTPUT:%.+]] = VPU.NCE.Convolution([[PADDED]], [[SPARSE_W]])
// CHECK:       [[TRANSPOSED_1:%.+]] = VPU.PermuteCast([[CONV_OUTPUT]])
// CHECK:       [[QUANTIZED_OUTPUT:%.+]] = VPU.QuantizeCast([[TRANSPOSED_1]])


// -----

!qAct117 = !quant.uniform<u8:f16, 0.0034980668741113998:117>
!qAct138 = !quant.uniform<u8:f16, 0.0033925405904358507:138>
!qAct151 = !quant.uniform<u8:f16, 0.004351120485978968:151>
!qW20 = !quant.uniform<u8:f16, 5.792334675788879E-4:128>
!qW21 = !quant.uniform<u8:f16, 5.42596274731206E-4:128>
!qW22 = !quant.uniform<u8:f16, 6.1786350081948676E-4:128>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @ConvEltwiseD2SSequence {
config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
    config.Option @config.AutoPaddingIDU : true
}

config.Resources 3 of @NCE at 2.100000e+03 MHz {
    builtin.module @ReservedMemory {
      module @CMXMetadataReservedMemory {
        config.MemoryResource 82944 bytes of @CMX_NN offset 15360
      }
      module @CMXStackFramesReservedMemory {
        config.MemoryResource 15360 bytes of @CMX_NN offset 0
      }
    }
    config.MemoryResource 1571840 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-DAG: #[[$MAP_A0:.*]] = affine_map<(d0) -> (d0 floordiv 2 - 1, 0)>
// CHECK-DAG: #[[$MAP_A1:.*]] = affine_map<(d0) -> (0, d0 - 1)>
// CHECK-DAG: #[[$MAP_PAD0_W_RAW:.*]] = affine_map<(d0) -> (d0 * -2 + 1, 0)>
// CHECK-DAG: #[[$MAP_MIN_CLAMP:.*]] = affine_map<()[s0] -> (1, s0)>
// CHECK-DAG: #[[$MAP_PAD2_W_RAW:.*]] = affine_map<(d0) -> (0, d0 - 1268)>
// CHECK-DAG: #[[$MAP_PAD1_W_RAW:.*]] = affine_map<(d0) -> (-(d0 floordiv 2) + 1, 0)>
// CHECK-DAG: #[[$MAP_PAD2_H_RAW:.*]] = affine_map<(d0, d1, d2) -> (0, d0 - d1 - d2 - 1266)>
// CHECK-DAG: #[[$MAP_PAD1_H_RAW:.*]] = affine_map<(d0) -> (-d0 + 1, 0)>
// CHECK-DAG: #[[$MAP_PAD0_H_RAW:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (0, d0 - d1 - d2 - d3 - d4 - 1264)>
// CHECK-DAG: #[[$MAP_IN_SZ_H:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (-d0 - d1 * 2 - d2 * 2 - d3 * 2 - d4 * 2 - d5 * 2 - d6 * 2 + 33)>
// CHECK-DAG: #[[$MAP_IN_OFF_H:.*]] = affine_map<(d0) -> (0, d0 * 2 - 1)>
// CHECK-DAG: #[[$MAP_SKIP_OFF_W:.*]] = affine_map<(d0, d1) -> (d0 - d1)>
// CHECK-DAG: #[[$MAP_SKIP0_SZ_W:.*]] = affine_map<(d0, d1, d2, d3) -> (-d0 - d1 - d2 - d3 + 14)>
// CHECK-DAG: #[[$MAP_SKIP1_SZ_W:.*]] = affine_map<(d0, d1) -> (-d0 - d1 + 12)>

// CHECK-LABEL:  @main
// CHECK-SAME: [[ARG0:%.+]]: tensor<1x1600x2560x4xf32>

func.func @main(%arg0: tensor<1x1600x2560x4xf32>) -> tensor<1x1600x2560x8xf32> {
  %cst = const.Declare tensor<32x1x1x144xf16, {order = #NHWC}> = dense<1.0> : tensor<32x4x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.Reshape<[32, 1, 1, 36]>, #const.LayoutCast<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 108]>]
  %cst_20 = const.Declare tensor<32x32x3x3x!qW20, {order = #NHWC}> = dense<1> : tensor<32x32x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!qW20>, #const.Reorder<#NHWC>]
  %cst_21 = const.Declare tensor<32x32x3x3x!qW21, {order = #NHWC}> = dense<1> : tensor<32x32x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!qW21>, #const.Reorder<#NHWC>]
  %cst_22 = const.Declare tensor<32x32x3x3x!qW22, {order = #NHWC}> = dense<1> : tensor<32x32x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!qW22>, #const.Reorder<#NHWC>]
  %0 = VPU.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1600x2560x4xf32> -> tensor<1x4x1600x2560xf32, {order = #NHWC}>
  %1 = VPU.Convert(%0) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tilingStrategy = [1, 1, 1, 23]} : tensor<1x4x1600x2560xf32, {order = #NHWC}> -> tensor<1x4x1600x2560xf16, {order = #NHWC}>
  %conv0 = VPU.NCE.Convolution(%1, %cst) rawFilterShape [32, 4, 3, 3] {
    mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>,
    strides = [2, 2], tilingStrategy = [1, 1, 1, 28]
  } : tensor<1x4x1600x2560xf16, {order = #NHWC}>, tensor<32x1x1x144xf16, {order = #NHWC}>
    -> tensor<1x32x800x1280x!qAct117, {order = #NHWC}>
  %conv1 = VPU.NCE.Convolution(%conv0, %cst_22) rawFilterShape [32, 32, 3, 3] {
    mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>, weight_zp = 128 : i64, activation_zp = 117 : i64>,
    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    pinnedStrategy, ppe = #VPU.PPEStub<>,
    resultSegmentSizes = array<i32: 1, 0, 0, 0>,
    strides = [1, 1], tilingStrategy = [1, 1, 3, 4]
  } : tensor<1x32x800x1280x!qAct117, {order = #NHWC}>, tensor<32x32x3x3x!qW22, {order = #NHWC}>
    -> tensor<1x32x800x1280x!qAct117, {order = #NHWC}>
  %eltwise0 = VPU.NCE.Eltwise(%conv0, %conv1) {
    is_inplace = true, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>, weight_zp = 117 : i64, activation_zp = 117 : i64>,
    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>,
    ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>,
    tilingStrategy = [1, 1, 1, 16]
  } -> tensor<1x32x800x1280x!qAct138, {order = #NHWC}>
  %conv2 = VPU.NCE.Convolution(%eltwise0, %cst_21) rawFilterShape [32, 32, 3, 3] {
    mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>, weight_zp = 128 : i64, activation_zp = 138 : i64>,
    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    pinnedStrategy, ppe = #VPU.PPEStub<>,
    resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1],
    tilingStrategy = [1, 1, 17, 4]
  } : tensor<1x32x800x1280x!qAct138, {order = #NHWC}>, tensor<32x32x3x3x!qW21, {order = #NHWC}>
    -> tensor<1x32x800x1280x!qAct138, {order = #NHWC}>
  %eltwise1 = VPU.NCE.Eltwise(%eltwise0, %conv2) {
    is_inplace = true, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>, weight_zp = 138 : i64, activation_zp = 138 : i64>,
    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>,
    ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>,
    tilingStrategy = [1, 1, 1, 16]
  } -> tensor<1x32x800x1280x!qAct151, {order = #NHWC}>
  %conv3 = VPU.NCE.Convolution(%eltwise1, %cst_20) rawFilterShape [32, 32, 3, 3] {
    mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>, weight_zp = 128 : i64, activation_zp = 151 : i64>,
    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    pinnedStrategy, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>,
    strides = [1, 1], tilingStrategy = [1, 1, 17, 4]
  } : tensor<1x32x800x1280x!qAct151, {order = #NHWC}>, tensor<32x32x3x3x!qW20, {order = #NHWC}>
    -> tensor<1x32x800x1280x!qAct117, {order = #NHWC}>
  %d2s = VPU.DepthToSpace(%conv3) {
    block_size = 2 : i64, dstElemType = f32, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>,
    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    tilingStrategy = [1, 1, 27, 1]
  } : tensor<1x32x800x1280x!qAct117, {order = #NHWC}>
    -> tensor<1x8x1600x2560xf32, {order = #NHWC}>
  %3 = VPU.PermuteCast(%d2s) {dst_order = #NCHW, mem_perm = #NCHW}
    : tensor<1x8x1600x2560xf32, {order = #NHWC}> -> tensor<1x1600x2560x8xf32>
  return %3 : tensor<1x1600x2560x8xf32>
}

// CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x8x1600x2560xf32, {order = #NHWC}>
// CHECK:         [[SCF_FOR:%.+]] = scf.for [[ARG1:%.+]] = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[ARG2:%.+]] = [[EMPTY]]) -> (tensor<1x8x1600x2560xf32, {order = #NHWC}>) {
// CHECK:           [[A0:%.+]] = affine.max #[[$MAP_A0]]([[ARG1]])
// CHECK:           [[A1:%.+]] = affine.max #[[$MAP_A1]]([[A0]])
// CHECK:           [[A2:%.+]] = affine.max #[[$MAP_A1]]([[A1]])
// CHECK:           [[PAD0_W_RAW:%.+]] = affine.max #[[$MAP_PAD0_W_RAW]]([[A2]])
// CHECK:           [[PAD0_W_LOW:%.+]] = affine.min #[[$MAP_MIN_CLAMP]]()[[[PAD0_W_RAW]]]
// CHECK:           [[PAD2_W_RAW:%.+]] = affine.max #[[$MAP_PAD2_W_RAW]]([[A0]])
// CHECK:           [[PAD2_W_HIGH:%.+]] = affine.min #[[$MAP_MIN_CLAMP]]()[[[PAD2_W_RAW]]]
// CHECK:           [[PAD1_W_RAW:%.+]] = affine.max #[[$MAP_PAD1_W_RAW]]([[ARG1]])
// CHECK:           [[PAD1_W_HIGH:%.+]] = affine.min #[[$MAP_MIN_CLAMP]]()[[[PAD1_W_RAW]]]
// CHECK:           [[PAD2_H_RAW:%.+]] = affine.max #[[$MAP_PAD2_H_RAW]]([[A1]], [[PAD1_W_HIGH]], [[PAD2_W_HIGH]])
// CHECK:           [[PAD2_H_HIGH:%.+]] = affine.min #[[$MAP_MIN_CLAMP]]()[[[PAD2_H_RAW]]]
// CHECK:           [[PAD1_H_RAW:%.+]] = affine.max #[[$MAP_PAD1_H_RAW]]([[A0]])
// CHECK:           [[PAD1_H_HIGH:%.+]] = affine.min #[[$MAP_MIN_CLAMP]]()[[[PAD1_H_RAW]]]
// CHECK:           [[PAD0_H_RAW:%.+]] = affine.max #[[$MAP_PAD0_H_RAW]]([[A2]], [[PAD1_H_HIGH]], [[PAD2_H_HIGH]], [[PAD1_W_HIGH]], [[PAD2_W_HIGH]])
// CHECK:           [[PAD0_H_LOW:%.+]] = affine.min #[[$MAP_MIN_CLAMP]]()[[[PAD0_H_RAW]]]
// CHECK:           [[PAD1_H_RAW2:%.+]] = affine.max #[[$MAP_PAD1_H_RAW]]([[A1]])
// CHECK:           [[PAD1_H_LOW:%.+]] = affine.min #[[$MAP_MIN_CLAMP]]()[[[PAD1_H_RAW2]]]
// CHECK:           [[IN_SZ_H:%.+]] = affine.apply #[[$MAP_IN_SZ_H]]([[PAD0_W_LOW]], [[PAD1_H_LOW]], [[PAD0_H_LOW]], [[PAD1_H_HIGH]], [[PAD2_H_HIGH]], [[PAD1_W_HIGH]], [[PAD2_W_HIGH]])
// CHECK:           [[IN_OFF_H:%.+]] = affine.max #[[$MAP_IN_OFF_H]]([[A2]])
// CHECK:           [[EXTRACT_SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IN_OFF_H]], 0] [1, 1600, [[IN_SZ_H]], 4] [1, 1, 1, 1] : tensor<1x1600x2560x4xf32> to tensor<1x1600x?x4xf32
// CHECK-SAME:        {bounds = #const.OpaqueI64Elements<[1, 1600, 2560, 4]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:           [[PERMUTE_CAST:%.+]] = VPU.PermuteCast([[EXTRACT_SLICE]])
// CHECK-SAME:        dst_order = #NHWC, mem_perm = #NCHW
// CHECK-SAME:        -> tensor<1x4x1600x?xf32, {bounds = #const.OpaqueI64Elements<[1, 4, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:           [[CONVERT:%.+]] = VPU.Convert([[PERMUTE_CAST]])
// CHECK-SAME:        -> tensor<1x4x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:           [[CONV0_PADDED:%.+]] = tensor.pad [[CONVERT]] low[0, 0, 1, {{%.+}}] high[0, 0, 0, 0]
// CHECK:             : tensor<1x4x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-SAME:        to tensor<1x4x1601x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1601, 2561]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:           [[CONV0_TILED:%.+]] = VPU.NCE.Convolution([[CONV0_PADDED]], {{%.+}})
// CHECK-SAME:        -> tensor<1x32x800x?x[[QTYPE0:!qElemType[0-9]*]], {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:           [[PADDED:%.+]] = tensor.pad {{%.+}} low[0, 0, 1, {{%.+}}] high[0, 0, 1, {{%.+}}]
// CHECK:             to tensor<1x32x802x?x[[QTYPE0]], {bounds = #const.OpaqueI64Elements<[1, 32, 802, 1282]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:           [[CONV1_TILED:%.+]] = VPU.NCE.Convolution([[PADDED]], {{%.+}})
// CHECK-SAME:        -> tensor<1x32x800x?x[[QTYPE0]], {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:           [[SKIP0_OFF_W:%.+]] = affine.apply #[[$MAP_SKIP_OFF_W]]([[A1]], [[A2]])
// CHECK:           [[SKIP0_SZ_W:%.+]] = affine.apply #[[$MAP_SKIP0_SZ_W]]([[PAD1_H_HIGH]], [[PAD2_H_HIGH]], [[PAD1_W_HIGH]], [[PAD2_W_HIGH]])
// CHECK:           [[EXTRACT_SKIP:%.+]] = tensor.extract_slice [[CONV0_TILED]][0, 0, 0, [[SKIP0_OFF_W]]] [1, 32, 800, [[SKIP0_SZ_W]]] [1, 1, 1, 1]
// CHECK-SAME:        : tensor<1x32x800x?x[[QTYPE0]], {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x800x?x[[QTYPE0]]
// CHECK:           [[ELT0_TILED:%.+]] = VPU.NCE.Eltwise([[EXTRACT_SKIP]], [[CONV1_TILED]])
// CHECK-SAME:        -> tensor<1x32x800x?x[[QTYPE1:!qElemType[0-9]*]], {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:           [[PADDED2:%.+]] = tensor.pad {{%.+}} low[0, 0, 1, {{%.+}}] high[0, 0, 1, {{%.+}}]
// CHECK:             to tensor<1x32x802x?x[[QTYPE1]], {bounds = #const.OpaqueI64Elements<[1, 32, 802, 1282]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:           [[CONV2_TILED:%.+]] = VPU.NCE.Convolution([[PADDED2]], {{%.+}})
// CHECK-SAME:        -> tensor<1x32x800x?x[[QTYPE1]], {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:           [[SKIP1_OFF_W:%.+]] = affine.apply #[[$MAP_SKIP_OFF_W]]([[A0]], [[A1]])
// CHECK:           [[SKIP1_SZ_W:%.+]] = affine.apply #[[$MAP_SKIP1_SZ_W]]([[PAD1_W_HIGH]], [[PAD2_W_HIGH]])
// CHECK:           [[EXTRACT_SKIP2:%.+]] = tensor.extract_slice [[ELT0_TILED]][0, 0, 0, [[SKIP1_OFF_W]]] [1, 32, 800, [[SKIP1_SZ_W]]] [1, 1, 1, 1]
// CHECK-SAME:        : tensor<1x32x800x?x[[QTYPE1]], {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-SAME:        to tensor<1x32x800x?x[[QTYPE1]]
// CHECK:           [[ELT1_TILED:%.+]] = VPU.NCE.Eltwise([[EXTRACT_SKIP2]], [[CONV2_TILED]])
// CHECK-SAME:        -> tensor<1x32x800x?x[[QTYPE2:!qElemType[0-9]*]], {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:           [[PADDED3:%.+]] = tensor.pad {{%.+}} low[0, 0, 1, {{%.+}}] high[0, 0, 1, {{%.+}}]
// CHECK:             to tensor<1x32x802x?x[[QTYPE2]], {bounds = #const.OpaqueI64Elements<[1, 32, 802, 1282]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:           [[CONV3_TILED:%.+]] = VPU.NCE.Convolution([[PADDED3]], {{%.+}})
// CHECK-SAME:        -> tensor<1x32x800x?x[[QTYPE0]], {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:           [[CASTED:%.+]] = tensor.cast [[CONV3_TILED]] : tensor<1x32x800x?x[[QTYPE0]]

// CHECK:           [[D2S:%.+]] = VPU.DepthToSpace([[CASTED]])
// CHECK-SAME:        -> tensor<1x8x1600x20xf32, {order = #NHWC}>

// CHECK:           [[INSERT_SLICE:%.+]] = tensor.insert_slice [[D2S]] into [[ARG2]][0, 0, 0, [[ARG1]]] [1, 8, 1600, 20] [1, 1, 1, 1]
// CHECK-SAME:        : tensor<1x8x1600x20xf32, {order = #NHWC}> into tensor<1x8x1600x2560xf32, {order = #NHWC}>
// CHECK:           scf.yield [[INSERT_SLICE]] : tensor<1x8x1600x2560xf32, {order = #NHWC}>
// CHECK:         }
// CHECK:         [[PERMUTE_FINAL:%.+]] = VPU.PermuteCast([[SCF_FOR]]) {dst_order = #NCHW, mem_perm = #NCHW}
// CHECK:         return [[PERMUTE_FINAL]]

}
