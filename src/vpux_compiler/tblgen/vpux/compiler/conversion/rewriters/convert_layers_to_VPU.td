//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_CONVERSION_REWRITERS_CONVERT_LAYERS_TO_VPU
#define VPUX_COMPILER_CONVERSION_REWRITERS_CONVERT_LAYERS_TO_VPU

include "vpux/compiler/dialect/IE/ops.td"
include "vpux/compiler/dialect/VPU/ops.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"

//
// IE.Convert -> VPU.Convert
//

def createConvertOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::ConvertOp>($_loc, $0, $1)
        }]>;

def RewriteConvert :
        Pat<
            (IE_ConvertOp $input, $dstElemType),
            (createConvertOp $input, $dstElemType)
        >;

//
// IE.SoftMax -> VPU.SoftMax
//

def createSoftMaxOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::SoftMaxOp>($_loc, $0, $1, $2)
        }]>;

def RewriteSoftMax :
        Pat<
            (IE_SoftMaxOp $input, $axisInd, $padSize),
            (createSoftMaxOp $input, $axisInd, $padSize)
        >;

//
// IE.LogSoftmax -> VPU.LogSoftmax
//

def createLogSoftmaxOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::LogSoftmaxOp>($_loc, $0, $1)
        }]>;

def RewriteLogSoftmax :
        Pat<
            (IE_LogSoftmaxOp $input, $axisInd),
            (createLogSoftmaxOp $input, $axisInd)
        >;

//
// IE.LoopSelect -> VPU.LoopSelect
//

def createLoopSelectOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::LoopSelectOp>($_loc, $0, $1, $2, $3, $4, $5)
        }]>;

def RewriteLoopSelect :
        Pat<
            (IE_LoopSelectOp $initExecCond, $execConds, $input, $do_concat, $axis, $stride),
            (createLoopSelectOp $initExecCond, $execConds, $input, $do_concat, $axis, $stride)
        >;

//
// IE.Tile -> VPU.Tile
//

def createTileOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::TileOp>($_loc, $0, $1)
        }]>;

def RewriteTile :
        Pat<
            (IE_TileOp $input, $repeats, $repeats_values),
            (createTileOp $input, $repeats_values)
        >;

//
// IE.Concat -> VPU.Concat
//

def createConcatOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::ConcatOp>($_loc, $0, $1, $2)
        }]>;

def RewriteConcat :
        Pat<
            (IE_ConcatOp $inputs, $per_axis, $static_offsets),
            (createConcatOp $inputs, $per_axis, $static_offsets)
        >;

//
// IE.AvgPool -> VPU.AvgPool
//

def createAvgPoolOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::AvgPoolOp>(
                $_loc, $0, $1, $2, $3, $4, $5, $6)
        }]>;

class ShouldBeNull<string desc> : Constraint<CPred<"!$_self">, desc>;
def RewriteAvgPool :
        Pat<
            (IE_AvgPoolOp $input, $kernel_size, $strides, $pads_begin, $pads_end, $rounding_type, $exclude_pads, $post_op, $clamp, $static_scale, $output_channels, $input_channels),
            (createAvgPoolOp $input, $kernel_size, $strides, $pads_begin, $pads_end, $rounding_type, $exclude_pads, $static_scale),
            [
                (ShouldBeNull<"static_scale must be null">:$static_scale)
            ]
        >;

//
// IE.LRN -> VPU.LRN
//

def createLRNOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::LRNOp> (
                $_loc, $0, $1, $2, $3, $4, $5)
        }]>;

def RewriteLRN :
        Pat<
            (IE_LRNOp $input, $axes, $axes_value, $alpha, $beta, $bias, $size),
            (createLRNOp $input, $axes_value, $alpha, $beta, $bias, $size)
        >;

//
// IE.LRN_IE -> VPU.LRN_IE
//

def createLRN_IEOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::LRN_IEOp> (
                $_loc, $0, $1, $2, $3, $4, $5)
        }]>;

def RewriteLRN_IE :
        Pat<
            (IE_LRN_IEOp $input, $alpha, $beta, $bias, $size, $region),
            (createLRN_IEOp $input, $alpha, $beta, $bias, $size, $region)
        >;

//
// IE.Broadcast -> VPU.Broadcast
//

def createBroadcastOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::BroadcastOp> (
                $_loc, $0, $1,
                $2.size() == 1 ? $2[0] : nullptr,
                $3)
        }]>;

def RewriteBroadcast :
        Pat<
            (IE_BroadcastOp $input, $target_shape, $axes_mapping, $mode),
            (createBroadcastOp $input, $target_shape, $axes_mapping, $mode)
        >;

//
// IE.Bucketize -> VPU.Bucketize
//

def createBucketizeOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::BucketizeOp> (
                $_loc, $0, $1, $2, $3)
        }]>;

def RewriteBucketize :
        Pat<
            (IE_BucketizeOp $data, $buckets, $output_type, $with_right_bound),
            (createBucketizeOp $data, $buckets, $output_type, $with_right_bound)
        >;

//
// IE.ExtractImagePatches -> VPU.ExtractImagePatches
//

def createExtractImagePatchesOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::ExtractImagePatchesOp>(
                $_loc, $0, $1, $2, $3, $4)
        }]>;

def RewriteExtractImagePatches :
        Pat<
            (IE_ExtractImagePatchesOp $data, $sizes, $strides, $rates, $autoPad),
            (createExtractImagePatchesOp $data, $sizes, $strides, $rates, $autoPad)
        >;

//
// IE.Reduce[Max|Mean|Prod|Sum|Min|L1|L2|LogicalOr|LogicalAnd]Op -> VPU.Reduce[Max|Mean|Prod|Sum|Min|L1|L2|LogicalOr|LogicalAnd]Op
//

class createReduceOp<string type> :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::Reduce}] # type # [{Op>(
                $_loc, $0, $1, $2)
        }]>;

def RewriteReduceMax :
        Pat<
            (IE_ReduceMaxOp $input, $axes, $axes_value, $keep_dims),
            (createReduceOp<"Max"> $input, $axes_value, $keep_dims)
        >;

def RewriteReduceMean :
        Pat<
            (IE_ReduceMeanOp $input, $axes, $axes_value, $keep_dims),
            (createReduceOp<"Mean"> $input, $axes_value, $keep_dims)
        >;

def RewriteReduceProd :
        Pat<
            (IE_ReduceProdOp $input, $axes, $axes_value, $keep_dims),
            (createReduceOp<"Prod"> $input, $axes_value, $keep_dims)
        >;

def RewriteReduceSum :
        Pat<
            (IE_ReduceSumOp $input, $axes, $axes_value, $keep_dims),
            (createReduceOp<"Sum"> $input, $axes_value, $keep_dims)
        >;

def RewriteReduceMin :
        Pat<
            (IE_ReduceMinOp $input, $axes, $axes_value, $keep_dims),
            (createReduceOp<"Min"> $input, $axes_value, $keep_dims)
        >;

def RewriteReduceL1 :
        Pat<
            (IE_ReduceL1Op $input, $axes, $axes_value, $keep_dims),
            (createReduceOp<"L1"> $input, $axes_value, $keep_dims)
        >;

def RewriteReduceLogicalOr :
        Pat<
            (IE_ReduceLogicalOrOp $input, $axes, $axes_value, $keep_dims),
            (createReduceOp<"LogicalOr"> $input, $axes_value, $keep_dims)
        >;

def RewriteReduceLogicalAnd :
        Pat<
            (IE_ReduceLogicalAndOp $input, $axes, $axes_value, $keep_dims),
            (createReduceOp<"LogicalAnd"> $input, $axes_value, $keep_dims)
        >;

def RewriteReduceL2 :
        Pat<
            (IE_ReduceL2Op $input, $axes, $axes_value, $keep_dims),
            (createReduceOp<"L2"> $input, $axes_value, $keep_dims)
        >;

//
// IE.FakeQuantize -> VPU.FakeQuantize
//

def createFakeQuantizeOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::FakeQuantizeOp>(
                $_loc, $0, $1, $2, $3, $4, $5, $6, $7)
        }]>;

def RewriteFakeQuantize :
        Pat<
            (IE_FakeQuantizeOp $input, $input_low, $input_high, $output_low, $output_high, $levels, $low_fp_type, $auto_broadcast),
            (createFakeQuantizeOp $input, $input_low, $input_high, $output_low, $output_high, $levels, $low_fp_type, $auto_broadcast)
        >;

//
// IE.MaxPool -> VPU.MaxPool
//

def createMaxPoolOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::MaxPoolOp>(
                $_loc, $0, $1, $2, $3, $4, $5, $6)
        }]>;

def RewriteMaxPool :
        Pat<
            (IE_MaxPoolOp $input, $kernel_size, $strides, $pads_begin, $pads_end, $rounding_type, $post_op, $clamp, $output_channels, $input_channels),
            (createMaxPoolOp $input, $kernel_size, $strides, $pads_begin, $pads_end, $rounding_type, $post_op, $clamp)
        >;

//
// IE.MaxPool8 -> VPU.MaxPool8
//

def createMaxPool8Op :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::MaxPool8Op>(
                $_loc, $0, $1, $2, $3, $4, $5, $6, $7, $8)
        }]>;

def RewriteMaxPool8 :
        Pat<
            (IE_MaxPool8Op $input, $kernel_size, $strides, $dilations, $pads_begin, $pads_end, $rounding_type, $index_element_type, $axis),
            (VPU_MaxPool8Op $input, $kernel_size, $strides, $dilations, $pads_begin, $pads_end, $rounding_type, $index_element_type, $axis)
        >;

//
// IE.Convolution -> VPU.Convolution
//

def createConvolutionOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::ConvolutionOp>(
                $_loc, $0, $1,
                $2.size() == 1 ? $2[0] : nullptr,
                $3, $4, $5, $6, $7)
        }]>;

// Note: AttrConstraint is not allowed so cannot use IsNullAttr from OpBase.td
class IsNull<string desc> : Constraint<CPred<"!$_self">, desc>;
def RewriteConvolution :
        Pat<
            (IE_ConvolutionOp $input, $filter, $bias, $strides, $pads_begin, $pads_end, $dilations, $post_op, $clamp, $static_scale, $output_channels, $input_channels),
            (createConvolutionOp $input, $filter, $bias, $strides, $pads_begin, $pads_end, $dilations, $post_op, $clamp, $static_scale),
            [
                (IsNull<"static_scale must be null">:$static_scale)
            ]
        >;


//
// IE.ReLU -> VPU.ReLU
//

def createReLUOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::ReLUOp>($_loc, $0)
        }]>;

def RewriteReLU :
        Pat<
            (IE_ReLUOp $input),
            (createReLUOp $input)
        >;

//
// IE.Sigmoid -> VPU.Sigmoid
//

def createSigmoidOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::SigmoidOp>($_loc, $0)
        }]>;

def RewriteSigmoid :
        Pat<
            (IE_SigmoidOp $input),
            (createSigmoidOp $input)
        >;

//
// IE.HSigmoid -> VPU.HSigmoid
//

def createHSigmoidOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::HSigmoidOp>($_loc, $0)
        }]>;

def RewriteHSigmoid :
        Pat<
            (IE_HSigmoidOp $input),
            (createHSigmoidOp $input)
        >;

//
// IE.HardSigmoid -> VPU.HardSigmoid
//

def createHardSigmoidOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::HardSigmoidOp>($_loc, $0, $1, $2)
        }]>;

def RewriteHardSigmoid :
        Pat<
            (IE_HardSigmoidOp $input, $alpha, $beta, $alpha_value, $beta_value),
            (createHardSigmoidOp $input, $alpha_value, $beta_value)
        >;

//
// IE.GridSample -> VPU.GridSample
//

def RewriteGridSample :
        Pat<
            (IE_GridSampleOp $input, $grip, $align_corners, $mode, $padding_mode),
            (VPU_GridSampleOp $input, $grip, $align_corners, $mode, $padding_mode)
        >;

//
// IE.Clamp -> VPU.Clamp
//

def createClampOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::ClampOp>($_loc, $0, $1, $2)
        }]>;

def RewriteClamp :
        Pat<
            (IE_ClampOp $input, $min, $max),
            (createClampOp $input, $min, $max)
        >;

//
// IE.Elu -> VPU.Elu
//

def createEluOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::EluOp>($_loc, $0, $1)
        }]>;

def RewriteElu :
        Pat<
            (IE_EluOp $input, $x),
            (createEluOp $input, $x)
        >;

//
// IE.Selu -> VPU.Selu
//

def createSeluOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::SeluOp>($_loc, $0, $1, $2)
        }]>;

def RewriteSelu :
        Pat<
            (IE_SeluOp $data, $alpha, $lambda, $alphaValue, $lambdaValue),
            (createSeluOp $data, $alphaValue, $lambdaValue)
        >;

//
// IE.Reshape -> VPU.Reshape
//

def createReshapeOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::ReshapeOp>(
                $_loc, $0,
                $1.size() == 1 ? $1[0] : nullptr,
                $2, $3)
        }]>;

def RewriteReshape :
        Pat<
            (IE_ReshapeOp $input, $shape, $special_zero, $shape_value),
            (createReshapeOp $input, $shape, $special_zero, $shape_value)
        >;

//
// IE.AffineReshape -> VPU.AffineReshape
//

def createAffineReshapeOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::AffineReshapeOp>($_loc, $0, $1, $2)
        }]>;

def RewriteAffineReshape :
        Pat<
            (IE_AffineReshapeOp $input, $dim_mapping, $shape_value),
            (createAffineReshapeOp $input, $dim_mapping, $shape_value)
        >;

//
// IE.Squeeze -> VPU.Squeeze
//

def createSqueezeOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::SqueezeOp>(
                $_loc, $0,
                $1.size() == 1 ? $1[0] : nullptr,
                $2)
        }]>;

def RewriteSqueeze :
        Pat<
            (IE_SqueezeOp $input, $axes, $axes_value),
            (createSqueezeOp $input, $axes, $axes_value)
        >;

//
// IE.Unsqueeze -> VPU.Unsqueeze
//

def createUnsqueezeOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::UnsqueezeOp>(
                $_loc, $0,
                $1.size() == 1 ? $1[0] : nullptr,
                $2)
        }]>;

def RewriteUnsqueeze :
        Pat<
            (IE_UnsqueezeOp $input, $axes, $axes_value),
            (createUnsqueezeOp $input, $axes, $axes_value)
        >;

//
// IE.HSwish -> VPU.HSwish
//

def createHSwishOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::HSwishOp>($_loc, $0)
        }]>;

def RewriteHSwish :
        Pat<
            (IE_HSwishOp $input),
            (createHSwishOp $input)
        >;

//
// IE.Floor -> VPU.Floor
//

def createFloorOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::FloorOp>($_loc, $0)
        }]>;

def RewriteFloor :
        Pat<
            (IE_FloorOp $input),
            (createFloorOp $input)
        >;

//
// IE.Round -> VPU.Round
//

def createRoundOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::RoundOp>($_loc, $0, $1)
        }]>;

def RewriteRound :
        Pat<
            (IE_RoundOp $input, $mode),
            (createRoundOp $input, $mode)
        >;

//
// IE.Mish -> VPU.Mish
//

def createMishOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::MishOp>($_loc, $0)
        }]>;

def RewriteMish :
        Pat<
            (IE_MishOp $input),
            (createMishOp $input)
        >;

//
// IE.Erf -> VPU.Erf
//

def createErfOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::ErfOp>($_loc, $0)
        }]>;

def RewriteErf :
        Pat<
            (IE_ErfOp $input),
            (createErfOp $input)
        >;

//
// IE.Tan -> VPU.Tan
//

def createTanOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::TanOp>($_loc, $0)
        }]>;

def RewriteTan :
        Pat<
            (IE_TanOp $input),
            (createTanOp $input)
        >;

//
// IE.Tanh -> VPU.Tanh
//

def createTanhOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::TanhOp>($_loc, $0)
        }]>;

def RewriteTanh :
        Pat<
            (IE_TanhOp $input),
            (createTanhOp $input)
        >;

//
// IE.Sin -> VPU.Sin
//

def createSinOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::SinOp>($_loc, $0)
        }]>;

def RewriteSin :
        Pat<
            (IE_SinOp $input),
            (createSinOp $input)
        >;

//
// IE.Cos -> VPU.Cos
//

def createCosOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::CosOp>($_loc, $0)
        }]>;

def RewriteCos :
        Pat<
            (IE_CosOp $input),
            (createCosOp $input)
        >;

//
// IE.Sqrt -> VPU.Sqrt
//

def createSqrtOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::SqrtOp>($_loc, $0)
        }]>;

def RewriteSqrt :
        Pat<
            (IE_SqrtOp $input),
            (createSqrtOp $input)
        >;

//
// IE.Sinh -> VPU.Sinh
//

def createSinhOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::SinhOp>($_loc, $0)
        }]>;

def RewriteSinh :
        Pat<
            (IE_SinhOp $input),
            (createSinhOp $input)
        >;

//
// IE.Cosh -> VPU.Cosh
//

def createCoshOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::CoshOp>($_loc, $0)
        }]>;

def RewriteCosh :
        Pat<
            (IE_CoshOp $input),
            (createCoshOp $input)
        >;

//
// IE.Asinh -> VPU.Asinh
//

def createAsinhOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::AsinhOp>($_loc, $0)
        }]>;

def RewriteAsinh :
        Pat<
            (IE_AsinhOp $input),
            (createAsinhOp $input)
        >;


//
// IE.Acosh -> VPU.Acosh
//

def createAcoshOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::AcoshOp>($_loc, $0)
        }]>;

def RewriteAcosh :
        Pat<
            (IE_AcoshOp $input),
            (createAcoshOp $input)
        >;

//
// IE.Abs -> VPU.Abs
//

def createAbsOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::AbsOp>($_loc, $0)
        }]>;

def RewriteAbs :
        Pat<
            (IE_AbsOp $input),
            (createAbsOp $input)
        >;

//
// IE.Atan -> VPU.Atan
//

def createAtanOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::AtanOp>($_loc, $0)
        }]>;

def RewriteAtan :
        Pat<
            (IE_AtanOp $input),
            (createAtanOp $input)
        >;

//
// IE.Atanh -> VPU.Atanh
//

def createAtanhOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::AtanhOp>($_loc, $0)
        }]>;

def RewriteAtanh :
        Pat<
            (IE_AtanhOp $input),
            (createAtanhOp $input)
        >;

//
// IE.Asin -> VPU.Asin
//

def createAsinOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::AsinOp>($_loc, $0)
        }]>;

def RewriteAsin :
        Pat<
            (IE_AsinOp $input),
            (createAsinOp $input)
        >;

//
// IE.Acos -> VPU.Acos
//

def createAcosOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::AcosOp>($_loc, $0)
        }]>;

def RewriteAcos :
        Pat<
            (IE_AcosOp $input),
            (createAcosOp $input)
        >;

//
// IE.Log -> VPU.Log
//

def createLogOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::LogOp>($_loc, $0)
        }]>;

def RewriteLog :
        Pat<
            (IE_LogOp $input),
            (createLogOp $input)
        >;

//
// IE.Gelu -> VPU.Gelu
//

def createGeluOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::GeluOp>($_loc, $0)
        }]>;

def RewriteGelu :
        Pat<
            (IE_GeluOp $input),
            (createGeluOp $input)
        >;

//
// IE.Exp -> VPU.Exp
//

def createExpOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::ExpOp>($_loc, $0)
        }]>;

def RewriteExp :
        Pat<
            (IE_ExpOp $input),
            (createExpOp $input)
        >;

//
// IE.PRelu -> VPU.PRelu
//

def createPReluOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::PReluOp>(
                $_loc, $0, $1)
        }]>;

def RewritePRelu :
        Pat<
            (IE_PReluOp $input, $negative_slope),
            (createPReluOp $input, $negative_slope)
        >;

//
// IE.LeakyRelu -> VPU.LeakyRelu
//

def createLeakyReluOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::LeakyReluOp>(
                $_loc, $0, $1)
        }]>;

def RewriteLeakyRelu :
        Pat<
            (IE_LeakyReluOp $input, $negative_slope),
            (createLeakyReluOp $input, $negative_slope)
        >;

//
// IE.Quantize -> VPU.Quantize
//

def createQuantizeOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::QuantizeOp>($_loc, $0, $1)
        }]>;

def RewriteQuantize :
        Pat<
            (IE_QuantizeOp $input, $dstElemType),
            (createQuantizeOp $input, $dstElemType)
        >;

//
// IE.Dequantize -> VPU.Dequantize
//

def createDequantizeOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::DequantizeOp>($_loc, $0, $1)
        }]>;

def RewriteDequantize :
        Pat<
            (IE_DequantizeOp $input, $dstElemType),
            (createDequantizeOp $input, $dstElemType)
        >;

//
// IE.DynamicQuantize -> VPU.DynamicQuantize
//

def RewriteDynamicQuantize :
        Pat<
            (IE_DynamicQuantizeOp $input),
            (VPU_DynamicQuantizeOp $input)
        >;

//
// IE.QuantizeCast -> VPU.QuantizeCast
//

def createQuantizeCastOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::QuantizeCastOp>($_loc, $0, $1)
        }]>;

def RewriteQuantizeCast :
        Pat<
            (IE_QuantizeCastOp $input, $dstElemType),
            (createQuantizeCastOp $input, $dstElemType)
        >;

//
// IE.PerAxisTile -> VPU.PerAxisTile
//

def createPerAxisTileOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::PerAxisTileOp>(
                $_loc, $0, $1, $2)
        }]>;

def RewritePerAxisTile :
        Pat<
            (IE_PerAxisTileOp $input, $axis, $tiles),
            (createPerAxisTileOp $input, $axis, $tiles)
        >;

//
// IE.[Add|Subtract|Multiply|Divide|SquaredDiff|Power|FloorMod|Mod|Min|Max|And|BitwiseAnd|BitwiseOr|BitwiseXor|Equal|Less|LessEqual|NotEqual|Greater|GreaterEqual|LogicalOr|LogicalXor]Pool
//     -> VPU.[Add|Subtract|Multiply|Divide|SquaredDiff|Power|FloorMod|Mod|Min|Max|And|BitwiseAnd|BitwiseOr|BitwiseXor|Equal|Less|LessEqual|NotEqual|Greater|GreaterEqual|LogicalOr|LogicalXor]Pool
//

class createEltwiseOp<string type> :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::}] # type # [{Op>(
                $_loc, $0, $1, $2)
        }]>;

class createEltwiseOpWithPostOp<string type> :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::}] # type # [{Op>(
                $_loc, $0, $1, $2, $3)
        }]>;

def RewritePower :
        Pat<
            (IE_PowerOp $input1, $input2, $auto_broadcast),
            (createEltwiseOp<"Power"> $input1, $input2, $auto_broadcast)
        >;

def RewriteAdd :
        Pat<
            (IE_AddOp $input1, $input2, $auto_broadcast, $post_op, $clamp, $output_channels, $input_channels),
            (createEltwiseOpWithPostOp<"Add"> $input1, $input2, $auto_broadcast, $post_op, $clamp)
        >;

def RewriteSubtract :
        Pat<
            (IE_SubtractOp $input1, $input2, $auto_broadcast, $post_op, $clamp, $output_channels, $input_channels),
            (createEltwiseOpWithPostOp<"Subtract"> $input1, $input2, $auto_broadcast, $post_op, $clamp)
        >;

def RewriteDivide :
        Pat<
            (IE_DivideOp $input1, $input2, $auto_broadcast),
            (createEltwiseOp<"Divide"> $input1, $input2, $auto_broadcast)
        >;

def RewriteSquaredDiff :
        Pat<
            (IE_SquaredDifferenceOp $input1, $input2, $auto_broadcast),
            (createEltwiseOp<"SquaredDifference"> $input1, $input2, $auto_broadcast)
        >;

def RewriteFloorMod :
        Pat<
            (IE_FloorModOp $input1, $input2, $auto_broadcast),
            (createEltwiseOp<"FloorMod"> $input1, $input2, $auto_broadcast)
        >;

def RewriteMod :
        Pat<
            (IE_ModOp $input1, $input2, $auto_broadcast),
            (createEltwiseOp<"Mod"> $input1, $input2, $auto_broadcast)
        >;

def RewriteLess :
        Pat<
            (IE_LessOp $input1, $input2, $auto_broadcast),
            (createEltwiseOp<"Less"> $input1, $input2, $auto_broadcast)
        >;

def RewriteLessEqual :
        Pat<
            (IE_LessEqualOp $input1, $input2, $auto_broadcast),
            (createEltwiseOp<"LessEqual"> $input1, $input2, $auto_broadcast)
        >;

def RewriteGreater :
        Pat<
            (IE_GreaterOp $input1, $input2, $auto_broadcast),
            (createEltwiseOp<"Greater"> $input1, $input2, $auto_broadcast)
        >;

def RewriteGreaterEqual :
        Pat<
            (IE_GreaterEqualOp $input1, $input2, $auto_broadcast),
            (createEltwiseOp<"GreaterEqual"> $input1, $input2, $auto_broadcast)
        >;

def RewriteLogicalOr :
        Pat<
            (IE_LogicalOrOp $input1, $input2, $auto_broadcast),
            (createEltwiseOp<"LogicalOr"> $input1, $input2, $auto_broadcast)
        >;

def RewriteLogicalXor :
        Pat<
            (IE_LogicalXorOp $input1, $input2, $auto_broadcast),
            (createEltwiseOp<"LogicalXor"> $input1, $input2, $auto_broadcast)
        >;

def RewriteMultiply :
        Pat<
            (IE_MultiplyOp $input1, $input2, $auto_broadcast, $post_op, $clamp, $output_channels, $input_channels),
            (createEltwiseOpWithPostOp<"Multiply"> $input1, $input2, $auto_broadcast, $post_op, $clamp)
        >;

def RewriteAnd :
        Pat<
            (IE_AndOp $input1, $input2, $auto_broadcast),
            (createEltwiseOp<"And"> $input1, $input2, $auto_broadcast)
        >;

def RewriteBitwiseAnd :
        Pat<
            (IE_BitwiseAndOp $input1, $input2, $auto_broadcast),
            (createEltwiseOp<"BitwiseAnd"> $input1, $input2, $auto_broadcast)
        >;

def RewriteBitwiseOr :
        Pat<
            (IE_BitwiseOrOp $input1, $input2, $auto_broadcast),
            (createEltwiseOp<"BitwiseOr"> $input1, $input2, $auto_broadcast)
        >;

def RewriteBitwiseXor :
        Pat<
            (IE_BitwiseXorOp $input1, $input2, $auto_broadcast),
            (createEltwiseOp<"BitwiseXor"> $input1, $input2, $auto_broadcast)
        >;

def RewriteMinimum :
        Pat<
            (IE_MinimumOp $input1, $input2, $auto_broadcast),
            (createEltwiseOp<"Minimum"> $input1, $input2, $auto_broadcast)
        >;

def RewriteMaximum :
        Pat<
            (IE_MaximumOp $input1, $input2, $auto_broadcast),
            (createEltwiseOp<"Maximum"> $input1, $input2, $auto_broadcast)
        >;

def RewriteEqual :
        Pat<
            (IE_EqualOp $input1, $input2, $auto_broadcast),
            (createEltwiseOp<"Equal"> $input1, $input2, $auto_broadcast)
        >;

def RewriteNotEqual :
        Pat<
            (IE_NotEqualOp $input1, $input2, $auto_broadcast),
            (createEltwiseOp<"NotEqual"> $input1, $input2, $auto_broadcast)
        >;

//
// IE.LogicalNot -> VPU.LogicalNot
//

def createLogicalNotOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::LogicalNotOp>($_loc, $0)
        }]>;

def RewriteLogicalNot :
        Pat<
            (IE_LogicalNotOp $input1),
            (createLogicalNotOp $input1)
        >;

//
// IE.BitwiseNot -> VPU.BitwiseNot
//

def createBitwiseNotOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::BitwiseNotOp>($_loc, $0)
        }]>;

def RewriteBitwiseNot :
        Pat<
            (IE_BitwiseNotOp $input1),
            (createBitwiseNotOp $input1)
        >;

//
// IE.EmbeddingBagPackedSum -> VPU.EmbeddingBagPackedSum
//

def createEmbeddingBagPackedSumOp :
        NativeCodeCall<[{
                $_builder.create<vpux::VPU::EmbeddingBagPackedSumOp>(
                      $_loc, $0, $1,
                      $2.size() == 1 ? $2[0] : nullptr)
        }]>;

def RewriteEmbeddingBagPackedSum :
        Pat<
            (IE_EmbeddingBagPackedSumOp $emb_table, $indices, $per_sample_weights),
            (createEmbeddingBagPackedSumOp $emb_table, $indices, $per_sample_weights)
        >;


//
// IE.FullyConnected -> VPU.FullyConnected
//

def createFullyConnectedOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::FullyConnectedOp>(
                $_loc, $0, $1,
                $2.size() == 1 ? $2[0] : nullptr)
        }]>;

def RewriteFullyConnected :
        Pat<
            (IE_FullyConnectedOp $input, $weights, $bias),
            (createFullyConnectedOp $input, $weights, $bias)
        >;

//
// IE.MatMul -> VPU.MatMul
//

def createMatMulOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::MatMulOp>(
                $_loc, $0, $1, $2, $3, $4)
        }]>;

def RewriteMatMul :
        Pat<
            (IE_MatMulOp $input1, $input2, $transpose_a, $transpose_b, $post_op),
            (createMatMulOp $input1, $input2, $transpose_a, $transpose_b, $post_op)
        >;

//
// IE.Select -> VPU.Select
//

def createSelectOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::SelectOp>(
                $_loc, $0, $1, $2, $3)
        }]>;

def RewriteSelect :
        Pat<
            (IE_SelectOp $input1, $input2, $input3, $auto_broadcast),
            (createSelectOp $input1, $input2, $input3, $auto_broadcast)
        >;

//
// IE.GRN -> VPU.GRN
//

def createGRNOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::GRNOp>(
                $_loc, $0, $1)
        }]>;

def RewriteGRN :
        Pat<
            (IE_GRNOp $input, $bias),
            (createGRNOp $input, $bias)
        >;

//
// IE.Swish -> VPU.Swish
//

def createSwishOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::SwishOp>(
                $_loc, $0,
                $1.size() == 1 ? $1[0] : nullptr,
                $2)
        }]>;

def RewriteSwish :
        Pat<
            (IE_SwishOp $input, $beta, $beta_value),
            (createSwishOp $input, $beta, $beta_value)
        >;

//
// IE.DetectionOutput -> VPU.DetectionOutput
//

def createDetectionOutputOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::DetectionOutputOp>(
                $_loc, $0, $1, $2,
                $3.size() == 1 ? $3[0] : nullptr,
                $4.size() == 1 ? $4[0] : nullptr,
                $5)
        }]>;

def RewriteDetectionOutput :
        Pat<
            (IE_DetectionOutputOp $in_box_logits, $in_class_preds, $in_proposals, $in_additional_preds, $in_additional_proposals, $attr),
            (createDetectionOutputOp $in_box_logits, $in_class_preds, $in_proposals, $in_additional_preds, $in_additional_proposals, $attr)
        >;

//
// IE.Reverse -> VPU.Reverse
//

def createReverseOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::ReverseOp>(
                $_loc, $0, $1, $2)
        }]>;

def RewriteReverse :
        Pat<
            (IE_ReverseOp $data, $axis, $axis_value, $mode),
            (createReverseOp $data, $axis_value, $mode)
        >;

//
// IE.ScaleShift -> VPU.ScaleShift
//

def createScaleShiftOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::ScaleShiftOp>(
                $_loc, $0,
                $1.size() == 1 ? $1[0] : nullptr,
                $2.size() == 1 ? $2[0] : nullptr)
        }]>;

def RewriteScaleShift :
        Pat<
            (IE_ScaleShiftOp $input, $weights, $biases),
            (createScaleShiftOp $input, $weights, $biases)
        >;

//
// IE.Upsampling -> VPU.Upsampling
//

def createUpsamplingOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::UpsamplingOp>(
                $_loc, $0, $1, $2)
        }]>;

def RewriteUpsampling :
        Pat<
            (IE_UpsamplingOp $input, $upsampling_factor, $pad, $output_channels),
            (createUpsamplingOp $input, $upsampling_factor, $pad)
        >;

//
// IE.Negative -> VPU.Negative
//

def createNegativeOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::NegativeOp>(
                $_loc, $0)
        }]>;

def RewriteNegative :
        Pat<
            (IE_NegativeOp $input),
            (createNegativeOp $input)
        >;

//
// IE.Sign -> VPU.Sign
//

def createSignOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::SignOp>(
                $_loc, $0)
        }]>;

def RewriteSign :
        Pat<
            (IE_SignOp $input),
            (createSignOp $input)
        >;

//
// IE.ROIPooling -> VPU.ROIPooling
//
def createROIPoolingOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::ROIPoolingOp>(
                $_loc, $0, $1, $2, $3, $4)
        }]>;
def RewriteROIPooling :
        Pat<
            (IE_ROIPoolingOp $input, $coords, $output_size, $spatial_scale, $method),
            (createROIPoolingOp $input, $coords, $output_size, $spatial_scale, $method)
        >;

//
// IE.PSROIPooling -> VPU.PSROIPooling
//
def createPSROIPoolingOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::PSROIPoolingOp>(
                $_loc, $0, $1, $2, $3, $4, $5, $6, $7)
        }]>;
def RewritePSROIPooling :
        Pat<
            (IE_PSROIPoolingOp $input, $coords, $output_dim, $spatial_scale, $group_size, $spatial_bins_x, $spatial_bins_y, $mode),
            (createPSROIPoolingOp $input, $coords, $output_dim, $spatial_scale, $group_size, $spatial_bins_x, $spatial_bins_y, $mode)
        >;

//
// IE.ROIAlign -> VPU.ROIAlign
//

def createROIAlignOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::ROIAlignOp>(
                $_loc, $0, $1, $2, $3, $4, $5, $6, $7, $8)
        }]>;
def RewriteROIAlign :
        Pat<
            (IE_ROIAlignOp $input, $coords, $roisIdx, $pooled_h, $pooled_w, $sampling_ratio, $spatial_scale, $poolingMode, $alignedMode),
            (createROIAlignOp $input, $coords, $roisIdx, $pooled_h, $pooled_w, $sampling_ratio, $spatial_scale, $poolingMode, $alignedMode)
        >;

//
// IE.CTCGreedyDecoder -> VPU.CTCGreedyDecoder
//

def createCTCGreedyDecoderOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::CTCGreedyDecoderOp>(
                $_loc, $0, $1, $2)
        }]>;

def RewriteCTCGreedyDecoder :
        Pat<
            (IE_CTCGreedyDecoderOp $input, $sequenceLengths, $mergeRepeated),
            (createCTCGreedyDecoderOp $input, $sequenceLengths, $mergeRepeated)
        >;

//
// IE.Pad -> VPU.Pad
//

def createPadOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::PadOp>(
                $_loc, $0,
                $1.size() == 1 ? $1[0] : nullptr,
                $2.size() == 1 ? $2[0] : nullptr,
                $3.size() == 1 ? $3[0] : nullptr,
                $4, $5, $6, $7, $8)
        }]>;

def RewritePad :
        Pat<
            (IE_PadOp $input, $pads_begin, $pads_end, $pad_value, $pads_begin_attr, $pads_end_attr, $pad_value_attr, $mode, $output_channels),
            (createPadOp $input, $pads_begin, $pads_end, $pad_value, $pads_begin_attr, $pads_end_attr, $pad_value_attr, $mode, $output_channels)
        >;

//
// IE.Copy -> VPU.Copy
//

def createCopyOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::CopyOp>(
                $_loc, $0, $1)
        }]>;

def RewriteCopy :
        Pat<
            (IE_CopyOp $input, $out_mem_space),
            (createCopyOp $input, $out_mem_space)
        >;

//
// IE.Slice -> VPU.Slice
//

def createSliceOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::SliceOp>(
                $_loc, $0, $1, $2)
        }]>;

def RewriteSlice :
        Pat<
            (IE_SliceOp $source, $static_offsets, $static_sizes),
            (createSliceOp $source, $static_offsets, $static_sizes)
        >;

//
// IE.StridedSlice -> VPU.StridedSlice
//

def createStridedSliceOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::StridedSliceOp>(
                $_loc, $0,
                $1.size() == 1 ? $1[0] : nullptr,
                $2.size() == 1 ? $2[0] : nullptr,
                $3.size() == 1 ? $3[0] : nullptr,
                $4, $5, $6, $7, $8, $9, $10, $11)
        }]>;

def RewriteStridedSlice :
        Pat<
            (IE_StridedSliceOp $input, $begins, $ends, $strides, $begins_attr,
                $ends_attr, $strides_attr, $begin_mask, $end_mask,
                $new_axis_mask, $shrink_mask, $ellipsis_mask),
            (createStridedSliceOp $input, $begins, $ends, $strides, $begins_attr,
                $ends_attr, $strides_attr, $begin_mask, $end_mask,
                $new_axis_mask, $shrink_mask, $ellipsis_mask)
        >;

//
// IE.Expand -> VPU.Expand
//

def createExpandOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::ExpandOp>(
                $_loc, $0, $1, $2)
        }]>;

def RewriteExpand :
        Pat<
            (IE_ExpandOp $input, $pads_begin, $pads_end),
            (createExpandOp $input, $pads_begin, $pads_end)
        >;

//
// IE.GroupNormalization -> VPU.GroupNormalization
//

def createGroupNormalizationOp :
    NativeCodeCall<[{
        $_builder.create<vpux::VPU::GroupNormalizationOp>($_loc, $0, $1, $2, $3, $4)
    }]>;

def RewriteGroupNormalization :
    Pat<
        (IE_GroupNormalizationOp $input, $scale, $bias, $num_groups, $epsilon),
        (createGroupNormalizationOp $input, $scale, $bias, $num_groups, $epsilon)
    >;

//
// IE.ExpandDilated -> VPU.ExpandDilated
//

def createExpandDilatedOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::ExpandDilatedOp>(
                $_loc, $0, $1)
        }]>;

def RewriteExpandDilated :
        Pat<
            (IE_ExpandDilatedOp $input, $dilations),
            (createExpandDilatedOp $input, $dilations)
        >;

//
// IE.RegionYolo -> VPU.RegionYolo
//

def createRegionYoloOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::RegionYoloOp>(
                $_loc, $0, $1, $2, $3, $4, $5, $6, $7, $8)
        }]>;

def RewriteRegionYolo :
        Pat<
            (IE_RegionYoloOp $input, $coords, $classes, $regions, $do_softmax, $mask, $axis, $end_axis, $anchors),
            (createRegionYoloOp $input, $coords, $classes, $regions, $do_softmax, $mask, $axis, $end_axis, $anchors)
        >;

//
// IE.ReorgYolo -> VPU.ReorgYolo
//

def createReorgYoloOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::ReorgYoloOp>(
                $_loc, $0, $1)
        }]>;

def RewriteReorgYolo :
        Pat<
            (IE_ReorgYoloOp $input, $stride),
            (createReorgYoloOp $input, $stride)
        >;

//
// IE.MVN -> VPU.MVN
//

def createMVNOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::MVNOp>(
                $_loc, $0, $1, $2, $3, $4)
        }]>;

def RewriteMVN :
        Pat<
            (IE_MVNOp $input, $across_channels, $normalize_variance, $eps, $internal_reshape),
            (createMVNOp $input, $across_channels, $normalize_variance, $eps, $internal_reshape)
        >;

//
// IE.MVN6 -> VPU.MVN6
//

def createMVN6Op :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::MVN6Op>(
                $_loc, $0,
                $1.size() == 1 ? $1[0] : nullptr,
                $2.size() == 1 ? $2[0] : nullptr,
                $3, $4, $5, $6)
        }]>;

def RewriteMVN6 :
        Pat<
            (IE_MVN6Op $input, $scale, $bias, $axes, $axes_value, $normalize_variance, $eps, $eps_mode),
            (createMVN6Op $input, $scale, $bias, $axes_value, $normalize_variance, $eps, $eps_mode)
        >;

//
// IE.Gather -> VPU.Gather
//

def createGatherOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::GatherOp>(
                $_loc, $0, $1,
                $2.size() == 1 ? $2[0] : nullptr,
                $3, $4, $5)
        }]>;

def RewriteGather :
        Pat<
            (IE_GatherOp $input, $indices, $axis, $axis_value, $batch_dims, $indices_rank),
            (createGatherOp $input, $indices, $axis, $axis_value, $batch_dims, $indices_rank)
        >;

//
// IE.GatherND -> VPU.GatherND
//

def createGatherNDOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::GatherNDOp>(
                $_loc, $0, $1, $2)
        }]>;

def RewriteGatherND :
        Pat<
            (IE_GatherNDOp $input, $indices, $batch_dims),
            (createGatherNDOp $input, $indices, $batch_dims)
        >;

//
// IE.GatherElements -> VPU.GatherElements
//

def createGatherElementsOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::GatherElementsOp>(
                    $_loc, $0, $1, $2)
        }]>;

def RewriteGatherElements :
        Pat<
            (IE_GatherElementsOp $input, $indices, $axis),
            (createGatherElementsOp $input, $indices, $axis)
        >;

//
// IE.GatherTree -> VPU.GatherTree
//

def createGatherTreeOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::GatherTreeOp>(
                    $_loc, $0, $1, $2, $3)
        }]>;

def RewriteGatherTree :
        Pat<
            (IE_GatherTreeOp $stepIds, $parentIds, $maxSeqLen, $endToken),
            (createGatherTreeOp $stepIds, $parentIds, $maxSeqLen, $endToken)
        >;

//
// IE.YuvToRgb -> VPU.YuvToRgb
//

def createYuvToRgbOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::YuvToRgbOp>(
                $_loc, $0,
                $1.size() == 1 ? $1[0] : nullptr,
                $2.size() == 1 ? $2[0] : nullptr,
                $3, $4)
        }]>;

def RewriteYuvToRgb :
        Pat<
            (IE_YuvToRgbOp $input1, $input2, $input3, $inFmt, $outFmt),
            (createYuvToRgbOp $input1, $input2, $input3, $inFmt, $outFmt)
        >;

//
// IE.RandomUniform -> VPU.RandomUniform
//

def RewriteRandomUniform :
        Pat<
            (IE_RandomUniformOp $min, $max, $shape, $outputType, $global_seed, $op_seed),
            (VPU_RandomUniformOp $min, $max, $shape, $outputType, $global_seed, $op_seed)
        >;

//
// IE.OneHot -> VPU.OneHot
//

def createOneHotOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::OneHotOp>(
            $_loc, $0, $1, $2, $3, $4, $5)
        }]>;

def RewriteOneHot :
        Pat<
            (IE_OneHotOp $input, $depth, $on_value, $off_value, $depth_attr,
             $on_value_attr, $off_value_attr, $axis_attr, $outputType),
            (createOneHotOp $input, $depth_attr, $on_value_attr, $off_value_attr, $axis_attr, $outputType)
        >;

//
// IE.SpaceToDepth -> VPU.SpaceToDepth
//

def createSpaceToDepthOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::SpaceToDepthOp>(
                $_loc, $0, $1, $2)
        }]>;

def RewriteSpaceToDepth :
        Pat<
            (IE_SpaceToDepthOp $input, $block_size, $mode),
            (createSpaceToDepthOp $input, $block_size, $mode)
        >;

//
// IE.SpaceToBatch -> VPU.SpaceToBatch
//

def createSpaceToBatch :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::SpaceToBatch>(
                $_loc, $0, $1, $2, $3)
        }]>;

def RewriteSpaceToBatch :
        Pat<
            (IE_SpaceToBatch $input, $block_shape, $pads_begin, $pads_end, $block_shape_value, $pads_begin_value, $pads_end_value),
            (createSpaceToBatch $input, $block_shape_value, $pads_begin_value, $pads_end_value)
        >;

//
// IE.BatchToSpace -> VPU.BatchToSpace
//

def createBatchToSpace :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::BatchToSpace>(
                $_loc, $0, $1, $2, $3)
        }]>;

def RewriteBatchToSpace :
        Pat<
            (IE_BatchToSpace $input, $block_shape, $crops_begin, $crops_end, $block_shape_value, $crops_begin_value, $crops_end_value),
            (createBatchToSpace $input, $block_shape_value, $crops_begin_value, $crops_end_value)
        >;

//
// IE.ScatterNDUpdate -> VPU.ScatterNDUpdate
//

def createScatterNDUpdateOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::ScatterNDUpdateOp>(
                $_loc, $0, $1, $2)
        }]>;

def RewriteScatterNDUpdate :
        Pat<
            (IE_ScatterNDUpdateOp $input, $indices, $updates),
            (createScatterNDUpdateOp $input, $indices, $updates)
        >;

//
// IE.ScatterUpdate -> VPU.ScatterUpdate
//

def createScatterUpdateOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::ScatterUpdateOp>(
                $_loc, $0, $1, $2, $3)
        }]>;

def RewriteScatterUpdate :
        Pat<
            (IE_ScatterUpdateOp $input, $indices, $updates, $axis, $axis_value),
            (createScatterUpdateOp $input, $indices, $updates, $axis_value)
        >;

//
// IE.ScatterElementsUpdate -> VPU.ScatterElementsUpdate
//

def createScatterElementsUpdateOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::ScatterElementsUpdateOp>(
                $_loc, $0, $1, $2, $3, $4, $5)
        }]>;

def RewriteScatterElementsUpdate :
        Pat<
            (IE_ScatterElementsUpdateOp $input, $indices, $updates, $axis, $axis_value, $reduction, $use_init_val),
            (createScatterElementsUpdateOp $input, $indices, $updates, $axis_value, $reduction, $use_init_val)
        >;

//
// IE.MemPermute -> VPU.MemPermute
//

def createMemPermuteOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::MemPermuteOp>(
                $_loc, $0, $1, $2)
        }]>;

def RewriteMemPermute :
        Pat<
            (IE_MemPermuteOp $input, $dst_order, $mem_perm),
            (createMemPermuteOp $input, $dst_order, $mem_perm)
            >;

//
// IE.PermuteCast -> VPU.PermuteCast
//

def createPermuteCastOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::PermuteCastOp>(
                $_loc, $0, $1, $2)
        }]>;

def RewritePermuteCast :
        Pat<
            (IE_PermuteCastOp $input, $dst_order, $mem_perm),
            (createPermuteCastOp $input, $dst_order, $mem_perm)
            >;

//
// IE.SoftPlus -> VPU.SoftPlus
//

def createSoftPlusOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::SoftPlusOp>($_loc, $0)
        }]>;

def RewriteSoftPlus :
        Pat<
            (IE_SoftPlusOp $input),
            (createSoftPlusOp $input)
        >;

//
// IE.Ceiling -> VPU.Ceiling
//

def createCeilingOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::CeilingOp>($_loc, $0)
        }]>;

def RewriteCeiling :
        Pat<
            (IE_CeilingOp $input),
            (createCeilingOp $input)
        >;


//
// IE.NormalizeIE -> VPU.NormalizeIE
//

def createNormalizeIEOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::NormalizeIEOp>(
                $_loc, $0, $1, $2, $3, $4)
        }]>;

def RewriteNormalizeIE :
        Pat<
            (IE_NormalizeIEOp $data, $weights, $eps, $across_spatial, $channel_shared),
            (createNormalizeIEOp $data, $weights, $eps, $across_spatial, $channel_shared)
        >;

//
// IE.CumSum -> VPU.CumSum
//

def createCumSumOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::CumSumOp>(
                $_loc, $0, $1, $2, $3)
        }]>;

def RewriteCumSum :
        Pat<
            (IE_CumSumOp $input, $axis, $axis_value, $exclusive, $reverse),
            (createCumSumOp $input, $axis_value, $exclusive, $reverse)
        >;

//
// IE.Eye -> VPU.Eye
//

def createEyeOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::EyeOp>(
                $_loc, $0, $1, $2, $3, $4)
        }]>;

def RewriteEye :
        Pat<
            (IE_EyeOp $num_rows, $num_columns, $diagonal_index, $batch_shape, $num_rows_value, $num_columns_value, $batch_shape_value, $outputType),
            (createEyeOp $diagonal_index, $num_rows_value, $num_columns_value, $batch_shape_value, $outputType)
        >;

//
// IE.Roll -> VPU.Roll
//

def createRollOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::RollOp>($_loc, $0, $1, $2)
        }]>;

def RewriteRoll :
        Pat<
            (IE_RollOp $data, $shift, $axes),
            (createRollOp $data, $shift, $axes)
        >;

//
// IE.ReverseSequence -> VPU.ReverseSequence
//

def createReverseSequenceOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::ReverseSequenceOp>(
                $_loc, $0, $1, $2, $3)
        }]>;

def RewriteReverseSequence :
        Pat<
            (IE_ReverseSequenceOp $data, $seq_length, $seq_axis, $batch_axis),
            (createReverseSequenceOp $data, $seq_length, $seq_axis, $batch_axis)
        >;

//
// IE.DepthToSpace -> VPU.DepthToSpace
//

def createDepthToSpaceOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::DepthToSpaceOp>(
                $_loc, $0, $1, $2, $3)
        }]>;

def RewriteDepthToSpace :
        Pat<
            (IE_DepthToSpaceOp $input, $block_size, $mode, $padded_channels),
            (createDepthToSpaceOp $input, $block_size, $mode, $padded_channels)
        >;

//
// IE.AdaptiveAvgPool -> VPU.AdaptiveAvgPool
//

def RewriteAdaptiveAvgPool :
        Pat<
            (IE_AdaptiveAvgPoolOp $input, $pooled_spatial_shape),
            (VPU_AdaptiveAvgPoolOp $input, $pooled_spatial_shape)
        >;

//
// IE.AdaptiveMaxPool -> VPU.AdaptiveMaxPool
//

def RewriteAdaptiveMaxPool :
        Pat<
            (IE_AdaptiveMaxPoolOp $input, $pooled_spatial_shape, $index_element_type),
            (VPU_AdaptiveMaxPoolOp $input, $pooled_spatial_shape, $index_element_type)
        >;

//
// IE.GRUSequence -> VPU.GRUSequence
//

def RewriteGRUSequence :
        Pat<
            (IE_GRUSequenceOp $input_data, $initial_hidden_state, $weights, $recurrence_weights, $biases, $hidden_size, $seq_length, $direction, $should_linear_before_reset, $clip),
            (VPU_GRUSequenceOp $input_data, $initial_hidden_state, $weights, $recurrence_weights, $biases, $hidden_size, $seq_length, $direction, $should_linear_before_reset, $clip)
        >;

//
// IE.GRUSequenceLastPart -> VPU.GRUSequenceLastPart
//

def RewriteGRUSequenceLastPart :
        Pat<
            (IE_GRUSequenceLastPartOp $first_part_output, $initial_hidden_state, $recurrence_weights, $biases, $hidden_size, $seq_length, $direction, $should_linear_before_reset, $clip),
            (VPU_GRUSequenceLastPartOp $first_part_output, $initial_hidden_state, $recurrence_weights, $biases, $hidden_size, $seq_length, $direction, $should_linear_before_reset, $clip)
        >;


//
// IE.DeformablePSROIPooling -> VPU.DeformablePSROIPooling
//
def createDeformablePSROIPoolingOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::DeformablePSROIPoolingOp>(
                $_loc, $0, $1, $2.size() == 1 ? $2[0] : nullptr, $3, $4, $5, $6, $7, $8, $9, $10)
        }]>;
def RewriteDeformablePSROIPooling :
        Pat<
            (IE_DeformablePSROIPoolingOp $input_score_maps, $input_rois, $input_transformations, $output_dim, $spatial_scale, $group_size, $spatial_bins_x, $spatial_bins_y, $trans_std, $part_size, $mode),
            (createDeformablePSROIPoolingOp $input_score_maps, $input_rois, $input_transformations, $output_dim, $spatial_scale, $group_size, $spatial_bins_x, $spatial_bins_y, $trans_std, $part_size, $mode)
        >;

//
// IE.PermuteQuantize -> VPU.PermuteQuantize
//

def createPermuteQuantizeOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::PermuteQuantizeOp>(
                $_loc, $0, $1, $2, $3, $4, $5)
        }]>;

def RewritePermuteQuantize :
        Pat<
            (IE_PermuteQuantizeOp $input, $dst_order, $mem_perm, $dstElemType, $pads_begin, $pads_end),
            (createPermuteQuantizeOp $input, $dst_order, $mem_perm, $dstElemType,  $pads_begin, $pads_end)
            >;

//
// IE.ShapeCast -> VPU.ShapeCast
//

def createShapeCastOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::ShapeCastOp>($_loc, $0, $1)
        }]>;

def RewriteShapeCast :
        Pat<
            (IE_ShapeCastOp $source, $shape),
            (createShapeCastOp $source, $shape)
        >;

//
// IE.LayoutCast -> VPU.LayoutCast
//

def createLayoutCastOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::LayoutCastOp>($_loc, $0, $1)
        }]>;

def RewriteLayoutCast :
        Pat<
            (IE_LayoutCastOp $input, $dst_order),
            (createLayoutCastOp $input, $dst_order)
        >;

//
// IE.DFT -> VPU.DFT
//

def createDFTOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::DFTOp>($_loc, $0, nullptr, $1, $2)
        }]>;

def RewriteDFT :
        Pat<
            (IE_DFTOp $input, $axes, $signal_size, $axes_attr, $signal_size_attr),
            (createDFTOp $input, $axes_attr, $signal_size_attr)
        >;

//
// IE.IDFT -> VPU.IDFT
//

def createIDFTOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::IDFTOp>($_loc, $0, nullptr, $1, $2)
        }]>;

def RewriteIDFT :
        Pat<
            (IE_IDFTOp $input, $axes, $signal_size, $axes_attr, $signal_size_attr),
            (createIDFTOp $input, $axes_attr, $signal_size_attr)
        >;

//
// IE.RDFT -> VPU.RDFT
//

def createRDFTOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::RDFTOp>($_loc, $0, nullptr, $1, $2)
        }]>;

def RewriteRDFT :
        Pat<
            (IE_RDFTOp $input, $axes, $signal_size, $axes_attr, $signal_size_attr),
            (createRDFTOp $input, $axes_attr, $signal_size_attr)
        >;

//
// IE.IRDFT -> VPU.IRDFT
//

def createIRDFTOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::IRDFTOp>($_loc, $0, nullptr, $1, $2)
        }]>;

def RewriteIRDFT :
        Pat<
            (IE_IRDFTOp $input, $axes, $signal_size, $axes_attr, $signal_size_attr),
            (createIRDFTOp $input, $axes_attr, $signal_size_attr)
        >;

//
// IE.ShapeOf -> VPU.ShapeOf
//

def createShapeOfOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::ShapeOfOp>($_loc, $0)
        }]>;

def RewriteShapeOf :
        Pat<
            (IE_ShapeOfOp $input, $dstElemType),
            (createShapeOfOp $input)
        >;

//
// IE.Range -> VPU.Range
//

def createRangeOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::RangeOp>($_loc, $0, $1, $2, $3)
        }]>;

def RewriteRange :
        Pat<
            (IE_RangeOp $start, $stop, $step, $dstElemType),
            (createRangeOp $start, $stop, $step, $dstElemType)
        >;

//
// IE.NonZero -> VPU.NonZero
//

def createNonZeroOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::NonZeroOp>($_loc, $0)
        }]>;

def RewriteNonZero :
        Pat<
            (IE_NonZeroOp $input, $dstElemType),
            (createNonZeroOp $input)
        >;

//
// IE.RMSOp -> VPU.RMSOp
//

def createRMSOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::RMSOp>($_loc, $0, $1, $2)
        }]>;

def RewriteRMS :
        Pat<
            (IE_RMSOp $input, $gamma, $epsilon),
            (createRMSOp $input, $gamma, $epsilon)
        >;

//
// IE.Inverse -> VPU.Inverse
//

def RewriteInverse :
        Pat<
            (IE_InverseOp $input, $adjoint),
            (VPU_InverseOp $input, $adjoint)
        >;

//
// IE.DeformableConvolution -> VPU.DeformableConvolution
//

def createDeformableConvolutionOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::DeformableConvolutionOp>(
                $_loc, $0, $1, $2,
                $3.size() == 1 ? $3[0] : nullptr,
                $4, $5, $6, $7, $8, $9, $10)
        }]>;

def RewriteDeformableConvolution :
        Pat<
            (IE_DeformableConvolutionOp $input, $offset, $kernel, $mask, $strides, $pads_begin, $pads_end, $dilations, $group, $deformable_group, $biliniar_interpolate_pad),
            (createDeformableConvolutionOp $input, $offset, $kernel, $mask, $strides, $pads_begin, $pads_end, $dilations, $group, $deformable_group, $biliniar_interpolate_pad)
        >;

//
// IE.DynamicExpandOp -> VPU.DynamicExpandOp
//

def RewriteDynamicExpandOp :
        Pat<
            (IE_DynamicExpandOp $input),
            (VPU_DynamicExpandOp $input)
        >;

//
// IE.DynamicDequantizeOp -> VPU.DynamicDequantizeOp
//

def createDynamicDequantizeOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::DynamicDequantizeOp>(
                $_loc, $0, $1,
                $2.size() == 1 ? $2[0] : nullptr,
                $3)
        }]>;

def RewriteDynamicDequantizeOp :
        Pat<
            (IE_DynamicDequantizeOp $input, $scale, $zp, $dstElemType),
            (createDynamicDequantizeOp $input, $scale, $zp, $dstElemType)
        >;

//
// IE.RoPEOp -> VPU.RoPEOp
//

def createRoPEOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::RoPEOp>($_loc, $0, $1, $2)
        }]>;

def RewriteRoPE :
        Pat<
            (IE_RoPEOp $input, $input_cos, $input_sin),
            (createRoPEOp $input, $input_cos, $input_sin)
        >;

#endif
