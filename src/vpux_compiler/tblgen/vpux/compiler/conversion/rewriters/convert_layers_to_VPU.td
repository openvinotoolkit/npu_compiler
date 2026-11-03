//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_CONVERSION_REWRITERS_CONVERT_LAYERS_TO_VPU
#define VPUX_COMPILER_CONVERSION_REWRITERS_CONVERT_LAYERS_TO_VPU

include "vpux/compiler/dialect/IE/ops/activation.td"
include "vpux/compiler/dialect/IE/ops/arithmetic.td"
include "vpux/compiler/dialect/IE/ops/bitwise.td"
include "vpux/compiler/dialect/IE/ops/comparison.td"
include "vpux/compiler/dialect/IE/ops/control_flow.td"
include "vpux/compiler/dialect/IE/ops/convolution.td"
include "vpux/compiler/dialect/IE/ops/data_movement.td"
include "vpux/compiler/dialect/IE/ops/data_type.td"
include "vpux/compiler/dialect/IE/ops/eltwise.td"
include "vpux/compiler/dialect/IE/ops/image.td"
include "vpux/compiler/dialect/IE/ops/logical.td"
include "vpux/compiler/dialect/IE/ops/normalization.td"
include "vpux/compiler/dialect/IE/ops/pooling.td"
include "vpux/compiler/dialect/IE/ops/recurrent.td"
include "vpux/compiler/dialect/IE/ops/reduce.td"
include "vpux/compiler/dialect/IE/ops/shape_manipulation.td"
include "vpux/compiler/dialect/IE/ops/specialized.td"

include "vpux/compiler/dialect/VPU/ops/activation.td"
include "vpux/compiler/dialect/VPU/ops/arithmetic.td"
include "vpux/compiler/dialect/VPU/ops/bitwise.td"
include "vpux/compiler/dialect/VPU/ops/comparison.td"
include "vpux/compiler/dialect/VPU/ops/control_flow.td"
include "vpux/compiler/dialect/VPU/ops/convolution.td"
include "vpux/compiler/dialect/VPU/ops/data_movement.td"
include "vpux/compiler/dialect/VPU/ops/data_type.td"
include "vpux/compiler/dialect/VPU/ops/eltwise.td"
include "vpux/compiler/dialect/VPU/ops/image.td"
include "vpux/compiler/dialect/VPU/ops/internal.td"
include "vpux/compiler/dialect/VPU/ops/logical.td"
include "vpux/compiler/dialect/VPU/ops/normalization.td"
include "vpux/compiler/dialect/VPU/ops/pooling.td"
include "vpux/compiler/dialect/VPU/ops/recurrent.td"
include "vpux/compiler/dialect/VPU/ops/reduce.td"
include "vpux/compiler/dialect/VPU/ops/shape_manipulation.td"
include "vpux/compiler/dialect/VPU/ops/specialized.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"

//
// IE.Convert -> VPU.Convert
//

def createConvertOp :
        NativeCodeCall<[{
            vpux::VPU::ConvertOp::create($_builder, $_loc, $0, $1)
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
            vpux::VPU::SoftMaxOp::create($_builder, $_loc, $0, ::mlir::Value{}, $1, $2, $3, $4)
        }]>;

def RewriteSoftMax :
        Pat<
            (IE_SoftMaxOp $input, $axisInd, $padSize, $dstElemType, $maskAware),
            (createSoftMaxOp $input, $axisInd, $padSize, $dstElemType, $maskAware)
        >;

//
// IE.LogSoftmax -> VPU.LogSoftmax
//

def createLogSoftmaxOp :
        NativeCodeCall<[{
            vpux::VPU::LogSoftmaxOp::create($_builder, $_loc, $0, $1, $2)
        }]>;

def RewriteLogSoftmax :
        Pat<
            (IE_LogSoftmaxOp $input, $axisInd, $padSize),
            (createLogSoftmaxOp $input, $axisInd, $padSize)
        >;

//
// IE.LoopSelect -> VPU.LoopSelect
//

def createLoopSelectOp :
        NativeCodeCall<[{
            vpux::VPU::LoopSelectOp::create($_builder, $_loc, $0, $1, $2, $3, $4, $5)
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
            vpux::VPU::TileOp::create($_builder, $_loc, $0, $1)
        }]>;

def RewriteTile :
        Pat<
            (IE_TileOp $input, $repeats_values),
            (createTileOp $input, $repeats_values)
        >;

//
// IE.Concat -> VPU.Concat
//

def createConcatOp :
        NativeCodeCall<[{
            vpux::VPU::ConcatOp::create($_builder, $_loc, $0, $1, $2)
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
            vpux::VPU::AvgPoolOp::create($_builder,
                $_loc, $0, $1, $2, $3, $4, $5, $6)
        }]>;

class ShouldBeNull<string desc> : Constraint<CPred<"!$_self">, desc>;
def RewriteAvgPool :
        Pat<
            (IE_AvgPoolOp $input, $scale, $kernel_size, $strides, $pads_begin, $pads_end, $rounding_type, $exclude_pads, $post_op, $clamp, $static_scale, $output_padding, $input_padding),
            (createAvgPoolOp $input, $kernel_size, $strides, $pads_begin, $pads_end, $rounding_type, $exclude_pads),
            [
                (ShouldBeNull<"scale must be null">:$scale),
                (ShouldBeNull<"post_op must be null">:$post_op),
                (ShouldBeNull<"clamp must be null">:$clamp),
                (ShouldBeNull<"static_scale must be null">:$static_scale)
            ]
        >;

//
// IE.AvgPool16 -> VPU.AvgPool16
//

def createAvgPool16Op :
        NativeCodeCall<[{
            vpux::VPU::AvgPool16Op::create($_builder,
                $_loc, $0, $1, $2, $3, $4, $5, $6, $7)
        }]>;

def RewriteAvgPool16 :
        Pat<
            (IE_AvgPool16Op $input, $kernel_size, $strides, $dilations, $pads_begin, $pads_end, $rounding_type, $exclude_pads, $post_op, $clamp, $static_scale, $output_padding, $input_padding),
            (createAvgPool16Op $input, $kernel_size, $strides, $dilations, $pads_begin, $pads_end, $rounding_type, $exclude_pads),
            [
                (ShouldBeNull<"post_op must be null">:$post_op),
                (ShouldBeNull<"clamp must be null">:$clamp),
                (ShouldBeNull<"static_scale must be null">:$static_scale)
            ]
        >;

//
// IE.LRN -> VPU.LRN
//

def createLRNOp :
        NativeCodeCall<[{
            vpux::VPU::LRNOp::create($_builder,
                $_loc, $0, $1, $2, $3, $4, $5, /*multiClusterStrategy=*/nullptr)
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
            vpux::VPU::LRN_IEOp::create($_builder,
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
            vpux::VPU::BroadcastOp::create($_builder,
                $_loc, $0, $1, $2, $3)
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
            vpux::VPU::BucketizeOp::create($_builder,
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
            vpux::VPU::ExtractImagePatchesOp::create($_builder,
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
            vpux::VPU::Reduce}] # type # [{Op::create($_builder,
                $_loc, $0, $1, $2)
        }]>;

def RewriteReduceMax :
        Pat<
            (IE_ReduceMaxOp $input, $axes_value, $keep_dims),
            (createReduceOp<"Max"> $input, $axes_value, $keep_dims)
        >;

def RewriteReduceProd :
        Pat<
            (IE_ReduceProdOp $input, $axes_value, $keep_dims),
            (createReduceOp<"Prod"> $input, $axes_value, $keep_dims)
        >;

def RewriteReduceSum :
        Pat<
            (IE_ReduceSumOp $input, $axes_value, $keep_dims, $output_padding, $input_padding),
            (createReduceOp<"Sum"> $input, $axes_value, $keep_dims)
        >;

def RewriteReduceMin :
        Pat<
            (IE_ReduceMinOp $input, $axes_value, $keep_dims),
            (createReduceOp<"Min"> $input, $axes_value, $keep_dims)
        >;

def RewriteReduceL1 :
        Pat<
            (IE_ReduceL1Op $input, $axes_value, $keep_dims),
            (createReduceOp<"L1"> $input, $axes_value, $keep_dims)
        >;

def RewriteReduceLogicalOr :
        Pat<
            (IE_ReduceLogicalOrOp $input, $axes_value, $keep_dims),
            (createReduceOp<"LogicalOr"> $input, $axes_value, $keep_dims)
        >;

def RewriteReduceLogicalAnd :
        Pat<
            (IE_ReduceLogicalAndOp $input, $axes_value, $keep_dims),
            (createReduceOp<"LogicalAnd"> $input, $axes_value, $keep_dims)
        >;

def RewriteReduceL2 :
        Pat<
            (IE_ReduceL2Op $input, $axes_value, $keep_dims),
            (createReduceOp<"L2"> $input, $axes_value, $keep_dims)
        >;

//
// IE.FakeQuantize -> VPU.FakeQuantize
//

def createFakeQuantizeOp :
        NativeCodeCall<[{
            vpux::VPU::FakeQuantizeOp::create($_builder,
                $_loc, $0, $1, $2, $3, $4, $5, $6)
        }]>;

def RewriteFakeQuantize :
        Pat<
            (IE_FakeQuantizeOp $input, $input_low, $input_high, $output_low, $output_high, $levels_or_low_fp_type, $auto_broadcast),
            (createFakeQuantizeOp $input, $input_low, $input_high, $output_low, $output_high, $levels_or_low_fp_type, $auto_broadcast)
        >;

//
// IE.FakeConvert -> VPU.FakeConvert
//

def createFakeConvertOp :
        NativeCodeCall<[{
            vpux::VPU::FakeConvertOp::create($_builder,
                $_loc, $0, $1, $2, $3)
        }]>;

def RewriteFakeConvert :
        Pat<
            (IE_FakeConvertOp $input, $scale, $shift, $dst_type),
            (createFakeConvertOp $input, $scale, $shift, $dst_type)
        >;

//
// IE.MaxPool -> VPU.MaxPool
//

def createMaxPoolOp :
        NativeCodeCall<[{
            vpux::VPU::MaxPoolOp::create($_builder,
                $_loc, $0, $1, $2, $3, $4, $5, $6)
        }]>;

def RewriteMaxPool :
        Pat<
            (IE_MaxPoolOp $input, $scale, $kernel_size, $strides, $pads_begin, $pads_end, $rounding_type, $post_op, $clamp, $static_scale, $output_padding, $input_padding),
            (createMaxPoolOp $input, $kernel_size, $strides, $pads_begin, $pads_end, $rounding_type, $post_op, $clamp),
            [
                (ShouldBeNull<"scale must be null">:$scale),
                (ShouldBeNull<"static_scale must be null">:$static_scale)
            ]
        >;

//
// IE.Convolution -> VPU.Convolution
//

def createConvolutionOp :
        NativeCodeCall<[{
            vpux::VPU::ConvolutionOp::create($_builder,
                $_loc, $0, $1, $2, $3, $4, $5, $6, $7)
        }]>;

// Note: AttrConstraint is not allowed so cannot use IsNullAttr from OpBase.td
class IsNullAttr<string desc> : Constraint<CPred<"!$_self">, desc>;
def RewriteConvolution :
        Pat<
            (IE_ConvolutionOp $input, $filter, $bias, $scale, $zero_points, $strides, $pads_begin, $pads_end, $dilations, $post_op, $clamp, $static_scale, $output_padding, $input_padding),
            (createConvolutionOp $input, $filter, $bias, $strides, $pads_begin, $pads_end, $dilations, $post_op, $clamp, $static_scale),
            [
                (IsNullAttr<"static_scale must be null">:$static_scale),
                (IsNullAttr<"scale must be null">:$scale),
                (IsNullAttr<"zero_points must be null">:$zero_points)
            ]
        >;


//
// IE.ReLU -> VPU.ReLU
//

def createReLUOp :
        NativeCodeCall<[{
            vpux::VPU::ReLUOp::create($_builder, $_loc, $0)
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
            vpux::VPU::SigmoidOp::create($_builder, $_loc, $0)
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
            vpux::VPU::HSigmoidOp::create($_builder, $_loc, $0)
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
            vpux::VPU::HardSigmoidOp::create($_builder, $_loc, $0, $1, $2)
        }]>;

def RewriteHardSigmoid :
        Pat<
            (IE_HardSigmoidOp $input, $alpha, $beta, $alpha_value, $beta_value),
            (createHardSigmoidOp $input, $alpha_value, $beta_value)
        >;

//
// IE.GridSample -> VPU.GridSample
//

def createGridSampleOp :
        NativeCodeCall<[{
            vpux::VPU::GridSampleOp::create($_builder, $_loc, $0, $1, $2, $3, $4)
        }]>;

def RewriteGridSample :
        Pat<
            (IE_GridSampleOp $input, $grip, $align_corners, $mode, $padding_mode),
            (createGridSampleOp $input, $grip, $align_corners, $mode, $padding_mode)
        >;

//
// IE.Clamp -> VPU.Clamp
//

def createClampOp :
        NativeCodeCall<[{
            vpux::VPU::ClampOp::create($_builder, $_loc, $0, $1, $2)
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
            vpux::VPU::EluOp::create($_builder, $_loc, $0, $1)
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
            vpux::VPU::SeluOp::create($_builder, $_loc, $0, $1, $2)
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
            vpux::VPU::ReshapeOp::create($_builder,
                $_loc, $0, $1)
        }]>;

def RewriteReshape :
        Pat<
            (IE_ReshapeOp $input, $shape_value),
            (createReshapeOp $input, $shape_value)
        >;

//
// IE.AffineReshape -> VPU.AffineReshape
//

def createAffineReshapeOp :
        NativeCodeCall<[{
            vpux::VPU::AffineReshapeOp::create($_builder, $_loc, $0, $1, $2)
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
            vpux::VPU::SqueezeOp::create($_builder,
                $_loc, $0, $1, $2)
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
            vpux::VPU::UnsqueezeOp::create($_builder,
                $_loc, $0, $1, $2)
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
            vpux::VPU::HSwishOp::create($_builder, $_loc, $0)
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
            vpux::VPU::FloorOp::create($_builder, $_loc, $0)
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
            vpux::VPU::RoundOp::create($_builder, $_loc, $0, $1)
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
            vpux::VPU::MishOp::create($_builder, $_loc, $0)
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
            vpux::VPU::ErfOp::create($_builder, $_loc, $0)
        }]>;

def RewriteErf :
        Pat<
            (IE_ErfOp $input),
            (createErfOp $input)
        >;

        //
// IE.ErfInv -> VPU.ErfInv
//

def createErfInvOp :
        NativeCodeCall<[{
            vpux::VPU::ErfInvOp::create($_builder, $_loc, $0)
        }]>;

def RewriteErfInv :
        Pat<
            (IE_ErfInvOp $input),
            (createErfInvOp $input)
        >;

//
// IE.Tan -> VPU.Tan
//

def createTanOp :
        NativeCodeCall<[{
            vpux::VPU::TanOp::create($_builder, $_loc, $0)
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
            vpux::VPU::TanhOp::create($_builder, $_loc, $0)
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
            vpux::VPU::SinOp::create($_builder, $_loc, $0)
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
            vpux::VPU::CosOp::create($_builder, $_loc, $0)
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
            vpux::VPU::SqrtOp::create($_builder, $_loc, $0)
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
            vpux::VPU::SinhOp::create($_builder, $_loc, $0)
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
            vpux::VPU::CoshOp::create($_builder, $_loc, $0)
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
            vpux::VPU::AsinhOp::create($_builder, $_loc, $0)
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
            vpux::VPU::AcoshOp::create($_builder, $_loc, $0)
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
            vpux::VPU::AbsOp::create($_builder, $_loc, $0)
        }]>;

def RewriteAbs :
        Pat<
            (IE_AbsOp $input),
            (createAbsOp $input)
        >;

//
// IE.Atanh -> VPU.Atanh
//

def createAtanhOp :
        NativeCodeCall<[{
            vpux::VPU::AtanhOp::create($_builder, $_loc, $0)
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
            vpux::VPU::AsinOp::create($_builder, $_loc, $0)
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
            vpux::VPU::AcosOp::create($_builder, $_loc, $0)
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
            vpux::VPU::LogOp::create($_builder, $_loc, $0)
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
            vpux::VPU::GeluOp::create($_builder, $_loc, $0)
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
            vpux::VPU::ExpOp::create($_builder, $_loc, $0)
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
            vpux::VPU::PReluOp::create($_builder,
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
            vpux::VPU::LeakyReluOp::create($_builder,
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
            vpux::VPU::QuantizeOp::create($_builder, $_loc, $0, $1)
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
            vpux::VPU::DequantizeOp::create($_builder, $_loc, $0, $1)
        }]>;

def RewriteDequantize :
        Pat<
            (IE_DequantizeOp $input, $dstElemType),
            (createDequantizeOp $input, $dstElemType)
        >;

//
// IE.QuantizeCast -> VPU.QuantizeCast
//

def createQuantizeCastOp :
        NativeCodeCall<[{
            vpux::VPU::QuantizeCastOp::create($_builder, $_loc, $0, $1)
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
            vpux::VPU::PerAxisTileOp::create($_builder,
                $_loc, $0, $1, $2)
        }]>;

def RewritePerAxisTile :
        Pat<
            (IE_PerAxisTileOp $input, $axis, $tiles),
            (createPerAxisTileOp $input, $axis, $tiles)
        >;

//
// IE.[Subtract|Divide|SquaredDiff|Power|FloorMod|Mod|Min|Max|And|BitwiseAnd|BitwiseOr|BitwiseXor|BitwiseRightShift|BitwiseLeftShift|Equal|Less|LessEqual|NotEqual|Greater|GreaterEqual|LogicalOr|LogicalXor]Pool
//     -> VPU.[Subtract|Divide|SquaredDiff|Power|FloorMod|Mod|Min|Max|And|BitwiseAnd|BitwiseOr|BitwiseXor|BitwiseRightShift|BitwiseLeftShift|Equal|Less|LessEqual|NotEqual|Greater|GreaterEqual|LogicalOr|LogicalXor]Pool
//

class createEltwiseOp<string type> :
        NativeCodeCall<[{
            vpux::VPU::}] # type # [{Op::create($_builder,
                $_loc, $0, $1, $2)
        }]>;

class createEltwiseOpWithPostOp<string type> :
        NativeCodeCall<[{
            vpux::VPU::}] # type # [{Op::create($_builder,
                $_loc, $0, $1, $2, $3)
        }]>;

def RewritePower :
        Pat<
            (IE_PowerOp $input1, $input2, $auto_broadcast),
            (createEltwiseOp<"Power"> $input1, $input2, $auto_broadcast)
        >;

def RewriteSubtract :
        Pat<
            (IE_SubtractOp $input1, $input2, $scale, $auto_broadcast, $post_op, $clamp, $output_padding, $input_padding),
            (createEltwiseOpWithPostOp<"Subtract"> $input1, $input2, $auto_broadcast, $post_op),
            [
                (ShouldBeNull<"scale must be null">:$scale)
            ]
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

def RewriteBitwiseRightShift :
        Pat<
            (IE_BitwiseRightShiftOp $input1, $input2, $auto_broadcast),
            (createEltwiseOp<"BitwiseRightShift"> $input1, $input2, $auto_broadcast)
        >;

def RewriteBitwiseLeftShift :
        Pat<
            (IE_BitwiseLeftShiftOp $input1, $input2, $auto_broadcast),
            (createEltwiseOp<"BitwiseLeftShift"> $input1, $input2, $auto_broadcast)
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
// IE.IsInf -> VPU.IsInf
//

def createIsInfOp :
        NativeCodeCall<[{
            vpux::VPU::IsInfOp::create($_builder, $_loc, $0)
        }]>;

def RewriteIsInf :
        Pat<
            (IE_IsInfOp $input1),
            (createIsInfOp $input1)
        >;

//
// IE.IsFinite -> VPU.IsFinite
//

def createIsFiniteOp :
        NativeCodeCall<[{
            vpux::VPU::IsFiniteOp::create($_builder, $_loc, $0)
        }]>;

def RewriteIsFinite :
        Pat<
            (IE_IsFiniteOp $input1),
            (createIsFiniteOp $input1)
        >;

//
// IE.LogicalNot -> VPU.LogicalNot
//

def createLogicalNotOp :
        NativeCodeCall<[{
            vpux::VPU::LogicalNotOp::create($_builder, $_loc, $0)
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
            vpux::VPU::BitwiseNotOp::create($_builder, $_loc, $0)
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
                vpux::VPU::EmbeddingBagPackedSumOp::create($_builder,
                      $_loc, $0, $1, $2)
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
            vpux::VPU::FullyConnectedOp::create($_builder,
                $_loc, $0, $1, $2)
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
            vpux::VPU::MatMulOp::create($_builder,
                $_loc, $0, $1, $2, $3, $4)
        }]>;

def RewriteMatMul :
        Pat<
            (IE_MatMulOp $input1, $input2, $bias, $scale, $zero_points, $transpose_a, $transpose_b, $post_op, $clamp),
            (createMatMulOp $input1, $input2, $transpose_a, $transpose_b, $post_op, $clamp),
            [
                (ShouldBeNull<"bias must be null">:$bias),
                (ShouldBeNull<"scale must be null">:$scale),
                (ShouldBeNull<"zero_points must be null">:$zero_points)
            ]
        >;

//
// IE.Select -> VPU.Select
//

def createSelectOp :
        NativeCodeCall<[{
            vpux::VPU::SelectOp::create($_builder,
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
            vpux::VPU::GRNOp::create($_builder,
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
            vpux::VPU::SwishOp::create($_builder,
                $_loc, $0, $1, $2)
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
            vpux::VPU::DetectionOutputOp::create($_builder,
                $_loc, $0, $1, $2, $3, $4, $5)
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
            vpux::VPU::ReverseOp::create($_builder,
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
            vpux::VPU::ScaleShiftOp::create($_builder,
                $_loc, $0, $1, $2)
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
            vpux::VPU::UpsamplingOp::create($_builder,
                $_loc, $0, $1, $2)
        }]>;

def RewriteUpsampling :
        Pat<
            (IE_UpsamplingOp $input, $upsampling_factor, $pad, $output_padding, $input_padding),
            (createUpsamplingOp $input, $upsampling_factor, $pad)
        >;

//
// IE.Negative -> VPU.Negative
//

def createNegativeOp :
        NativeCodeCall<[{
            vpux::VPU::NegativeOp::create($_builder,
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
            vpux::VPU::SignOp::create($_builder,
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
            vpux::VPU::ROIPoolingOp::create($_builder,
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
            vpux::VPU::PSROIPoolingOp::create($_builder,
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
            vpux::VPU::ROIAlignOp::create($_builder,
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
            vpux::VPU::CTCGreedyDecoderOp::create($_builder,
                $_loc, $0, $1, $2)
        }]>;

def RewriteCTCGreedyDecoder :
        Pat<
            (IE_CTCGreedyDecoderOp $input, $sequenceLengths, $mergeRepeated),
            (createCTCGreedyDecoderOp $input, $sequenceLengths, $mergeRepeated)
        >;

//
// IE.Slice -> VPU.Slice
//

def createSliceOp :
        NativeCodeCall<[{
            vpux::VPU::SliceOp::create($_builder,
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
            vpux::VPU::StridedSliceOp::create($_builder,
                $_loc, $0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
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
            vpux::VPU::ExpandOp::create($_builder,
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
        vpux::VPU::GroupNormalizationOp::create($_builder, $_loc, $0, $1, $2, $3, $4)
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
            vpux::VPU::ExpandDilatedOp::create($_builder,
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
            vpux::VPU::RegionYoloOp::create($_builder,
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
            vpux::VPU::ReorgYoloOp::create($_builder,
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
            vpux::VPU::MVNOp::create($_builder,
                $_loc, $0, $1, $2, $3, $4, $5)
        }]>;

def RewriteMVN :
        Pat<
            (IE_MVNOp $input, $across_channels, $normalize_variance, $eps, $internal_reshape, $norm_high_prc),
            (createMVNOp $input, $across_channels, $normalize_variance, $eps, $internal_reshape, $norm_high_prc)
        >;

//
// IE.MVN6 -> VPU.MVN6
//

def createMVN6Op :
        NativeCodeCall<[{
            vpux::VPU::MVN6Op::create($_builder,
                $_loc, $0, $1, $2, $3, $4, $5, $6)
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
            vpux::VPU::GatherOp::create($_builder,
                $_loc, $0, $1, $2, $3, $4)
        }]>;

def RewriteGather :
        Pat<
            (IE_GatherOp $input, $indices, $axis_value, $batch_dims, $indices_rank),
            (createGatherOp $input, $indices, $axis_value, $batch_dims, $indices_rank)
        >;

//
// IE.GatherND -> VPU.GatherND
//

def createGatherNDOp :
        NativeCodeCall<[{
            vpux::VPU::GatherNDOp::create($_builder,
                $_loc, $0, $1, $2, $3)
        }]>;

def RewriteGatherND :
        Pat<
            (IE_GatherNDOp $input, $indices, $batch_dims, $original_shape),
            (createGatherNDOp $input, $indices, $batch_dims, $original_shape)
        >;

//
// IE.GatherElements -> VPU.GatherElements
//

def createGatherElementsOp :
        NativeCodeCall<[{
            vpux::VPU::GatherElementsOp::create($_builder,
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
            vpux::VPU::GatherTreeOp::create($_builder,
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
            vpux::VPU::YuvToRgbOp::create($_builder,
                $_loc, $0, $1, $2, $3, $4, $5, nullptr)
        }]>;

def RewriteYuvToRgb :
        Pat<
            (IE_YuvToRgbOp $input1, $input2, $input3, $inFmt, $outFmt, $scale),
            (createYuvToRgbOp $input1, $input2, $input3, $inFmt, $outFmt, $scale)
        >;

//
// IE.OneHot -> VPU.OneHot
//

def createOneHotOp :
        NativeCodeCall<[{
            vpux::VPU::OneHotOp::create($_builder,
            $_loc, $0, $1, $2, $3, $4, $5, $6, $7)
        }]>;

def RewriteOneHot :
        Pat<
            (IE_OneHotOp $input, $depth, $on_value, $off_value, $depth_attr,
             $on_value_attr, $off_value_attr, $axis_attr, $mode, $rank_expanded, $outputType),
            (createOneHotOp $input, $depth_attr, $on_value_attr, $off_value_attr, $axis_attr, $mode, $rank_expanded, $outputType)
        >;

//
// IE.SpaceToDepth -> VPU.SpaceToDepth
//

def createSpaceToDepthOp :
        NativeCodeCall<[{
            vpux::VPU::SpaceToDepthOp::create($_builder,
                $_loc, $0.getType(), $1, $2, $3)
        }]>;

// The result type (including the layout chosen by layout assignment) is taken from the IE op,
// so the VPU op does not need any target-specific layout logic in its type inference.
def RewriteSpaceToDepth :
        Pat<
            (IE_SpaceToDepthOp:$origOp $input, $block_size, $mode),
            (createSpaceToDepthOp $origOp, $input, $block_size, $mode)
        >;

//
// IE.SpaceToBatch -> VPU.SpaceToBatch
//

def createSpaceToBatch :
        NativeCodeCall<[{
            vpux::VPU::SpaceToBatch::create($_builder,
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
            vpux::VPU::BatchToSpace::create($_builder,
                $_loc, $0, $1, $2, $3)
        }]>;

def RewriteBatchToSpace :
        Pat<
            (IE_BatchToSpace $input, $block_shape_value, $crops_begin_value, $crops_end_value),
            (createBatchToSpace $input, $block_shape_value, $crops_begin_value, $crops_end_value)
        >;

//
// IE.ScatterNDUpdate -> VPU.ScatterNDUpdate
//

def createScatterNDUpdateOp :
        NativeCodeCall<[{
            vpux::VPU::ScatterNDUpdateOp::create($_builder,
                $_loc, $0, $1, $2)
        }]>;

def RewriteScatterNDUpdate :
        Pat<
            (IE_ScatterNDUpdateOp $input, $indices, $updates),
            (createScatterNDUpdateOp $input, $indices, $updates)
        >;

//
// IE.ScatterElementsUpdate -> VPU.ScatterElementsUpdate
//

def createScatterElementsUpdateOp :
        NativeCodeCall<[{
            vpux::VPU::ScatterElementsUpdateOp::create($_builder,
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
            vpux::VPU::MemPermuteOp::create($_builder,
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
            vpux::VPU::PermuteCastOp::create($_builder,
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
            vpux::VPU::SoftPlusOp::create($_builder, $_loc, $0)
        }]>;

def RewriteSoftPlus :
        Pat<
            (IE_SoftPlusOp $input),
            (createSoftPlusOp $input)
        >;

//
// IE.SoftSign -> VPU.SoftSign
//

def createSoftSignOp :
        NativeCodeCall<[{
            vpux::VPU::SoftSignOp::create($_builder, $_loc, $0)
        }]>;

def RewriteSoftSign :
        Pat<
            (IE_SoftSignOp $input),
            (createSoftSignOp $input)
        >;


//
// IE.Ceiling -> VPU.Ceiling
//

def createCeilingOp :
        NativeCodeCall<[{
            vpux::VPU::CeilingOp::create($_builder, $_loc, $0)
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
            vpux::VPU::NormalizeIEOp::create($_builder,
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
            vpux::VPU::CumSumOp::create($_builder,
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
            vpux::VPU::EyeOp::create($_builder,
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
            vpux::VPU::RollOp::create($_builder, $_loc, $0, $1, $2)
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
            vpux::VPU::ReverseSequenceOp::create($_builder,
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
            vpux::VPU::DepthToSpaceOp::create($_builder,
                $_loc, $0, $1, $2, $3, nullptr)
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
// IE.GRUGates -> VPU.GRUGates
//

def createGRUGatesOp :
        NativeCodeCall<[{
            vpux::VPU::GRUGatesOp::create($_builder, $_loc, $0, $1, $2, $3)
        }]>;

def RewriteGRUGates :
        Pat<
            (IE_GRUGatesOp $input_data, $initial_hidden_state, $hidden_data, $biases),
            (createGRUGatesOp $input_data, $initial_hidden_state, $hidden_data, $biases)
        >;

//
// IE.DeformablePSROIPooling -> VPU.DeformablePSROIPooling
//
def createDeformablePSROIPoolingOp :
        NativeCodeCall<[{
            vpux::VPU::DeformablePSROIPoolingOp::create($_builder,
                $_loc, $0, $1, $2,  $3, $4, $5, $6, $7, $8, $9, $10)
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
            vpux::VPU::PermuteQuantizeOp::create($_builder,
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
            vpux::VPU::ShapeCastOp::create($_builder, $_loc, $0, $1)
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
            vpux::VPU::LayoutCastOp::create($_builder, $_loc, $0, $1)
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
            vpux::VPU::DFTOp::create($_builder, $_loc, $0, nullptr, $1, $2)
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
            vpux::VPU::IDFTOp::create($_builder, $_loc, $0, nullptr, $1, $2)
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
            vpux::VPU::RDFTOp::create($_builder, $_loc, $0, nullptr, $1, $2)
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
            vpux::VPU::IRDFTOp::create($_builder, $_loc, $0, nullptr, $1, $2)
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
            vpux::VPU::ShapeOfOp::create($_builder, $_loc, $0)
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
            vpux::VPU::RangeOp::create($_builder, $_loc, $0, $1, $2, $3, vpux::VPU::BoundsRepresentationAttr::get($_loc.getContext(), vpux::VPU::BoundsRepresentation::BOUNDS))
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
            vpux::VPU::NonZeroOp::create($_builder, $_loc, $0)
        }]>;

def RewriteNonZero :
        Pat<
            (IE_NonZeroOp $input, $dstElemType),
            (createNonZeroOp $input)
        >;
//
// IE.ReduceSquareOp -> VPU.ReduceSquareOp
//

def createReduceSquareOp :
        NativeCodeCall<[{
            vpux::VPU::ReduceSquareOp::create($_builder, $_loc, $0, $1, $2, $3)
        }]>;


def RewriteReduceSquare :
        Pat<
            (IE_ReduceSquareOp $input, $epsilon, $axes_value, $keep_dims),
            (createReduceSquareOp $input, $axes_value, $keep_dims, $epsilon)
        >;


//
// IE.RMSOp -> VPU.RMSOp
//

def createRMSOp :
        NativeCodeCall<[{
            vpux::VPU::RMSOp::create($_builder, $_loc, $0, $1, $2, $3)
        }]>;

def RewriteRMS :
        Pat<
            (IE_RMSOp $input, $gamma, $eps, $epsMode),
            (createRMSOp $input, $gamma, $eps, $epsMode)
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
            vpux::VPU::DeformableConvolutionOp::create($_builder,
                $_loc, $0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, nullptr)
        }]>;

def RewriteDeformableConvolution :
        Pat<
            (IE_DeformableConvolutionOp $input, $offset, $kernel, $mask, $strides, $pads_begin, $pads_end, $dilations, $group, $deformable_group, $bilinear_interpolate_pad),
            (createDeformableConvolutionOp $input, $offset, $kernel, $mask, $strides, $pads_begin, $pads_end, $dilations, $group, $deformable_group, $bilinear_interpolate_pad)
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
            vpux::VPU::DynamicDequantizeOp::create($_builder,
                $_loc, $0, $1, $2, $3)
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
            vpux::VPU::RoPEOp::create($_builder, $_loc, $0, $1, $2, $3)
        }]>;

def RewriteRoPE :
        Pat<
            (IE_RoPEOp $input, $input_cos, $input_sin, $mode),
            (createRoPEOp $input, $input_cos, $input_sin, $mode)
        >;

//
// IE.SDPAOp -> VPU.SDPAOp
//

def createSDPAOp :
        NativeCodeCall<[{
            vpux::VPU::SDPAOp::create($_builder, $_loc, $0, $1, $2, $3, $4, nullptr)
        }]>;

def RewriteSDPA :
        Pat<
            (IE_SDPAOp $inputQ, $inputK, $inputV, $inputMask, $inputScale, $inputSink, $causal),
            (createSDPAOp $inputQ, $inputK, $inputV, $inputMask, $inputScale)
        >;

//
// IE.AttentionOp -> VPU.AttentionOp
//

def createAttentionOp :
        NativeCodeCall<[{
            vpux::VPU::AttentionOp::create($_builder, $_loc, $0, $1, $2, $3, $4, $5, $6, $7)
        }]>;

def RewriteAttention :
        Pat<
            (IE_AttentionOp $inputQ, $inputK, $inputV, $inputMask, $inputScale, $inputBias, $inputSink,
                                                                $padSizeS),
            (createAttentionOp $inputQ, $inputK, $inputV, $inputMask, $inputScale, $inputBias, $inputSink, $padSizeS)
        >;

//
// IE.DynamicDataMaskOp -> VPU.DynamicDataMaskOp
//

def createDynamicDataMaskOp :
        NativeCodeCall<[{
            vpux::VPU::DynamicDataMaskOp::create($_builder,
                $_loc, $0, $1)
        }]>;

def RewriteDynamicDataMaskOp :
        Pat<
            (IE_DynamicDataMaskOp $realShape, $outputTensorType),
            (createDynamicDataMaskOp $realShape, $outputTensorType)
        >;

//
// IE.GroupQuantCorrectionOp -> VPU.GroupQuantCorrectionOp
//

def createGroupQuantCorrectionOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::GroupQuantCorrectionOp>(
                $_loc, $0, $1, $2, $3, /*multiClusterStrategy=*/nullptr)
        }]>;

def RewriteGroupQuantCorrectionOp :
        Pat<
            (IE_GroupQuantCorrectionOp $actOut, $actIn, $scales, $zp),
            (createGroupQuantCorrectionOp $actOut, $actIn, $scales, $zp)
        >;

#endif
