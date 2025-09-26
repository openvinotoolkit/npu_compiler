//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/utils/core/array_ref.hpp"

namespace vpux {

bool isBroadcastable(int64_t d0, int64_t d1);

struct ShapeInfo {
    SmallVector<int64_t> shape;
    SmallVector<int64_t> bounds;

    int64_t rank() const;
    bool isDynamic() const;

    static ShapeInfo fromNDType(NDTypeInterface type);
};

/**
 * @brief                        Infers the output shape for a StridedSlice operation
 *                               with the given parameters
 * @param inDataShapeInfo:       The shape information of the input data
 * @param begins:                1D tensor with begin indexes for input blob slicing. Use for constant begins
 * @param ends:                  1D tensor with end indexes for input blob slicing. Use for constant ends
 * @param strides:               1D tensor of the slicing strides. Use for constant strides
 * @param beginsSize:            Shape of begin indexes for input blob slicing. Use for non-constant begins
 * @param endsSize:              Shape of end indexes for input blob slicing. Use for non-constant ends
 * @param stridesSize:           Shape of the slicing strides. Use for non-constant strides
 * @param begin_mask:            Bitmask corresponding to the dimensions of the begin input
 * @param end_mask:              Bitmask corresponding to the dimensions of the end input
 * @param new_axis_mask:         Bitmask which specifies the insertion of 1 dimension
 * @param shrink_axis_mask:      Bitmask which specifies the deletion of 1 dimension
 * @param ellipsis_mask:         Bitmask which inserts missing dimensions on a position
 *                               of a non-zero bit
 * @return                       The output shape info as ShapeInfo
 */
ShapeInfo inferStridedSliceOutputShape(const ShapeInfo& inDataShapeInfo, ArrayRef<int64_t> begins,
                                       ArrayRef<int64_t> ends, ArrayRef<int64_t> strides, ArrayRef<int64_t> beginsShape,
                                       ArrayRef<int64_t> endsShape, ArrayRef<int64_t> stridesShape,
                                       ArrayRef<int64_t> beginMask, ArrayRef<int64_t> endMask,
                                       ArrayRef<int64_t> newAxisMask, ArrayRef<int64_t> shrinkAxisMask,
                                       ArrayRef<int64_t> ellipsisMask);

/**
 * @brief                        Infers the output shape for a MaxPool operation
 *                               with the given parameters
 * @param inDataShapeInfo:       The shape information of the input data
 * @param windowStrides:         The strides
 * @param dataPaddingBelow:      Builds the beginning of padding shape
 * @param dataPaddingAbove:      Builds the end of padding shape
 * @param windowShape:           The kernel window
 * @param roundingType:          Whether to use ceiling or floor rounding type while
 *                               computing output shape
 * @return                       The output shape info as ShapeInfo
 */
ShapeInfo inferMaxPoolOutputShape(const ShapeInfo& inDataShape, ArrayRef<int64_t> windowStrides,
                                  ArrayRef<int64_t> dataPaddingBelow, ArrayRef<int64_t> dataPaddingAbove,
                                  ArrayRef<int64_t> windowShape,
                                  IE::RoundingType roundingType = IE::RoundingType::FLOOR);

/**
 * @brief                        Infers the output shape for a MaxPool8 operation
 *                               with the given parameters
 * @param inDataShape:           The shape of the input data
 * @param windowStrides:         The strides
 * @param windowDilations:       The dilations of the pooling filter
 * @param dataPaddingBelow:      Builds the beginning of padding shape
 * @param dataPaddingAbove:      Builds the end of padding shape
 * @param windowShape:           The kernel window
 * @param roundingType:          Whether to use ceiling or floor rounding type while
 *                               computing output shape
 * @return                       The output shape as SmallVector
 */
ShapeInfo inferMaxPool8OutputShape(const ShapeInfo& inDataShape, ArrayRef<int64_t> windowStrides,
                                   ArrayRef<int64_t> windowDilations, ArrayRef<int64_t> dataPaddingBelow,
                                   ArrayRef<int64_t> dataPaddingAbove, ArrayRef<int64_t> windowShape,
                                   IE::RoundingType roundingType = IE::RoundingType::FLOOR);

/**
 * @brief                        Infers the output shape for a AvgPool operation
 *                               with the given parameters
 * @param inDataShape:           The shape of the input data
 * @param windowStrides:         The strides
 * @param dataPaddingBelow:      Builds the beginning of padding shape
 * @param dataPaddingAbove:      Builds the end of padding shape
 * @param windowShape:           The kernel window
 * @param roundingType:          Whether to use ceiling or floor rounding type while
 *                               computing output shape
 * @return                       The output shape as SmallVector
 */
ShapeInfo inferAvgPoolOutputShape(const ShapeInfo& inDataShape, ArrayRef<int64_t> windowStrides,
                                  ArrayRef<int64_t> dataPaddingBelow, ArrayRef<int64_t> dataPaddingAbove,
                                  ArrayRef<int64_t> windowShape,
                                  IE::RoundingType roundingType = IE::RoundingType::FLOOR);

/**
 * @brief                        Infers the output shape for an AvgPool operation
 *                               with the given parameters
 * @param inDataShape:           The shape of the input data
 * @param windowStrides:         The strides
 * @param windowDilations:       The dilations
 * @param dataPaddingBelow:      Builds the beginning of padding shape
 * @param dataPaddingAbove:      Builds the end of padding shape
 * @param windowShape:           The kernel window
 * @param roundingType:          Whether to use ceiling or floor rounding type while
 *                               computing output shape
 * @return                       The output shape as SmallVector
 */
ShapeInfo inferAvgPool16OutputShape(const ShapeInfo& inDataShape, ArrayRef<int64_t> windowStrides,
                                    ArrayRef<int64_t> windowDilations, ArrayRef<int64_t> dataPaddingBelow,
                                    ArrayRef<int64_t> dataPaddingAbove, ArrayRef<int64_t> windowShape,
                                    IE::RoundingType roundingType = IE::RoundingType::FLOOR);

/**
 * @brief                        Infers the output shape for a ConvolutionBackpropData operation
 *                               with the given parameters
 * @param inputShape:            The shape of the input data
 * @param filterShape:           The shape of the filter
 * @param windowStrides:         The strides
 * @param dataPaddingBelow:      Builds the beginning of padding shape
 * @param dataPaddingAbove:      Builds the end of padding shape
 * @param windowDilations:       The dilations
 * @param outputPadding:         The output padding
 *
 * @return                       The output shape as SmallVector
 */
SmallVector<int64_t> inferConvBackpropOutputShape(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> filterShape,
                                                  ArrayRef<int64_t> windowStrides, ArrayRef<int64_t> dataPaddingBelow,
                                                  ArrayRef<int64_t> dataPaddingAbove, ArrayRef<int64_t> windowDilations,
                                                  ArrayRef<int64_t> outputPadding);

SmallVector<int64_t> inferGroupConvBackpropOutputShape(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> filterShape,
                                                       ArrayRef<int64_t> windowStrides,
                                                       ArrayRef<int64_t> dataPaddingBelow,
                                                       ArrayRef<int64_t> dataPaddingAbove,
                                                       ArrayRef<int64_t> windowDilations,
                                                       ArrayRef<int64_t> outputPadding);

SmallVector<int64_t> inferTransposedConvBackpropOutputShape(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> filterShape,
                                                            ArrayRef<int64_t> windowStrides,
                                                            ArrayRef<int64_t> dataPaddingBelow,
                                                            ArrayRef<int64_t> dataPaddingAbove,
                                                            ArrayRef<int64_t> windowDilations,
                                                            ArrayRef<int64_t> outputPadding);

SmallVector<int64_t> inferTransposedGroupConvBackpropOutputShape(
        ArrayRef<int64_t> inputShape, ArrayRef<int64_t> filterShape, ArrayRef<int64_t> windowStrides,
        ArrayRef<int64_t> dataPaddingBelow, ArrayRef<int64_t> dataPaddingAbove, ArrayRef<int64_t> windowDilations,
        ArrayRef<int64_t> outputPadding);

/**
 * @brief                        Infers the output shape for a MatMul operation
 *                               with the given parameters
 * @param in1ShapeInfo:          The shape info of the first input
 * @param in2ShapeInfo:          The shape info of the second input
 * @param transposeA:            Apply transpose for the first input
 * @param transposeB:            Apply transpose for the second input
 *
 * @return                       The output shape info as ShapeInfo
 */
ShapeInfo inferMatMulOutputShapeInfo(const ShapeInfo& in1ShapeInfo, const ShapeInfo& in2ShapeInfo, bool transposeA,
                                     bool transposeB);
/**
 * @brief                        Infers the output shape for a Convolution operation
 *                               with the given parameters
 * @param inShapeInfo:           The shape info of the input data
 * @param filterShapeInfo:       The shape info of the filter
 * @param windowStrides:         The strides
 * @param dataPaddingBelow:      Builds the beginning of padding shape
 * @param dataPaddingAbove:      Builds the end of padding shape
 * @param windowDilations:       The dilations
 *
 * @return                       The output shape info as ShapeInfo
 */
ShapeInfo inferConvolutionOutputShapeInfo(const ShapeInfo& inShapeInfo, const ShapeInfo& filterShapeInfo,
                                          ArrayRef<int64_t> windowStrides, ArrayRef<int64_t> dataPaddingBelow,
                                          ArrayRef<int64_t> dataPaddingAbove, ArrayRef<int64_t> windowDilations);

/**
 * @brief                        Infers the output shape for a GroupConvolution operation
 *                               with the given parameters
 * @param inShapeInfo:           The shape info of the input data
 * @param filterShapeInfo:       The shape info of the filter
 * @param windowStrides:         The strides
 * @param dataPaddingBelow:      Builds the beginning of padding shape
 * @param dataPaddingAbove:      Builds the end of padding shape
 * @param windowDilations:       The dilations
 * @param maybeGroups:           Number of groups, optionally specified by the original op
 * @param hasOutputPadding:      Specifies if the original op has output padding
 *
 * @return                       The output shape info as ShapeInfo
 */
ShapeInfo inferGroupConvolutionOutputShapeInfo(ShapeInfo& inShapeInfo, ShapeInfo& filterShapeInfo,
                                               ArrayRef<int64_t> windowStrides, ArrayRef<int64_t> dataPaddingBelow,
                                               ArrayRef<int64_t> dataPaddingAbove, ArrayRef<int64_t> windowDilations,
                                               std::optional<int64_t> maybeGroups, bool hasOutputPadding);

ShapeInfo inferTransposedConvBackpropOutputShapeInfo(const ShapeInfo& inShapeInfo, const ShapeInfo& filterShapeInfo,
                                                     ArrayRef<int64_t> windowStrides,
                                                     ArrayRef<int64_t> dataPaddingBelow,
                                                     ArrayRef<int64_t> dataPaddingAbove,
                                                     ArrayRef<int64_t> windowDilations,
                                                     ArrayRef<int64_t> outputPadding);

//
// Tensor Reifiers
//

mlir::OpFoldResult reifyDim(mlir::OpBuilder builder, mlir::Value value, mlir::RankedTensorType type, size_t idx,
                            std::optional<mlir::Location> loc = std::nullopt);
mlir::OpFoldResult reifyDim(mlir::OpBuilder builder, mlir::Value value, size_t idx,
                            std::optional<mlir::Location> loc = std::nullopt);

SmallVector<mlir::OpFoldResult> reifyTrivialTensor(mlir::OpBuilder builder, mlir::Value input,
                                                   std::optional<mlir::Location> loc = std::nullopt);

mlir::FailureOr<SmallVector<mlir::OpFoldResult>> reifyEltwiseTensors(mlir::OpBuilder& builder, mlir::Value input1,
                                                                     mlir::Value input2,
                                                                     IE::AutoBroadcastType broadcastType,
                                                                     mlir::Location loc);
mlir::FailureOr<SmallVector<mlir::OpFoldResult>> reifyMatMulTensors(mlir::OpBuilder& builder, mlir::Value input1,
                                                                    mlir::Value input2, bool transposeA,
                                                                    bool transposeB, mlir::Location loc);

/**
 * @brief Reify tensors for convolution or pooling operations. Currently, it supports only convolution with dilation
 * equal to 1 and pooling. kernel size is passed along with tensor for maxpool operations
 *
 * @param builder - builder to create new operations
 * @param input - input tensor
 * @param output - output tensor
 * @param kernel - kernel tensor
 * @param kernelSize - kernel size
 * @param strides - strides
 * @param padBegin - padding begin
 * @param padEnd - padding end
 *
 * @return reified shapes for output tensor
 */
mlir::FailureOr<SmallVector<mlir::OpFoldResult>> reifyConvPoolTensors(mlir::OpBuilder& builder, mlir::Value input,
                                                                      mlir::Value output, mlir::Value kernel,
                                                                      ArrayRef<int64_t> kernelSize,
                                                                      ArrayRef<int64_t> strides,
                                                                      ArrayRef<int64_t> padBegin,
                                                                      ArrayRef<int64_t> padEnd, mlir::Location loc);

/**
 * @brief                        Infers the output shape for a PermuteQuantize operation
 *
 * @param input:                 The input tensor
 * @param inOrder:               The order of the input tensor
 * @param newType:               The new type obtained after applying padding to the input tensor
 * @param outOrder:              The order of the output tensor
 * @param memPerm:               The memory permutation attribute

 * @return                       The output shape info as ShapeInfo
 */

ShapeInfo inferPermuteQuantizeOutputShapeInfo(mlir::Value input, DimsOrder inOrder, vpux::NDTypeInterface newType,
                                              DimsOrder outOrder, mlir::AffineMap memPerm);

/**
 * @brief                        Infers the output shape for an Elementwise operation
 *
 * @param in1ShapeInfo:          The shape info of the first input
 * @param in2ShapeInfo:          The shape info of the second input
 * @param broadcastType:         The broadcast type
 * @param inputPaddingAttr:      The padding attribute for the input
 * @param outputPaddingAttr:     The padding attribute for the output
 * @param loc:                   The location
 *
 * @return                       The output shape info as ShapeInfo
 */
ShapeInfo inferEltwiseOutputShapeInfo(const ShapeInfo& in1ShapeInfo, const ShapeInfo& in2ShapeInfo,
                                      IE::AutoBroadcastType broadcastType, mlir::ArrayAttr inputPaddingAttr,
                                      mlir::ArrayAttr outputPaddingAttr, mlir::Location loc);

/**
 * @brief                        Infers the output shape for an Elementwise operation
 *
 * @param in1ShapeInfo:          The shape info of the first input
 * @param in2ShapeInfo:          The shape info of the second input
 * @param broadcastType:         The broadcast type
 * @param loc:                   The location
 *
 * @return                       The output shape info as ShapeInfo
 */
ShapeInfo inferEltwiseOutputShapeInfo(const ShapeInfo& in1ShapeInfo, const ShapeInfo& in2ShapeInfo,
                                      IE::AutoBroadcastType broadcastType, mlir::Location loc);

}  // namespace vpux
