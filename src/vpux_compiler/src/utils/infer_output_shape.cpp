//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/transposed_convolution_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/type_padding.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/range.hpp"

#include <openvino/op/avg_pool.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convolution.hpp>
#include <openvino/op/group_conv.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/max_pool.hpp>
#include <openvino/op/strided_slice.hpp>
#include "openvino/op/parameter.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypes.h>

#include <cstddef>

using namespace vpux;

bool vpux::isBroadcastable(int64_t d0, int64_t d1) {
    return d0 == 1 || d1 == 1 || d0 == d1;
}

//
// ShapeInfo
//

int64_t ShapeInfo::rank() const {
    return shape.size();
}

bool ShapeInfo::isDynamic() const {
    return !bounds.empty();
}

ShapeInfo ShapeInfo::fromNDType(NDTypeInterface type) {
    // NB: empty bounds means that the shape is static
    auto boundVals = [&type]() -> SmallVector<int64_t> {
        if (const auto sparseType = mlir::dyn_cast<VPU::SparseTensorType>(type)) {
            auto boundedData = mlir::dyn_cast<Core::BoundedTensorType>(sparseType.getData());
            if (boundedData == nullptr) {
                return SmallVector<int64_t>{};
            }

            auto bounds = to_small_vector(boundedData.getBounds());
            if (sparseType.getStorageElementTable() == nullptr) {
                return bounds;
            }

            auto setShape = mlir::cast<NDTypeInterface>(sparseType.getStorageElementTable()).getShape();
            bounds[Dims4D::Act::H.ind()] = setShape[Dims4D::Act::H];
            bounds[Dims4D::Act::W.ind()] = setShape[Dims4D::Act::W];

            return bounds;
        }

        if (const auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(type)) {
            const auto bounds = boundedType.getBounds();
            return to_small_vector(bounds);
        }
        return SmallVector<int64_t>{};
    }();

    return ShapeInfo{to_small_vector(type.getShape()), std::move(boundVals)};
}

namespace {

ShapeInfo createShapeInfoFromPartialShape(const ov::PartialShape& partialShape) {
    auto resultShape = to_small_vector(partialShape | transformed([](const ov::Dimension& val) {
                                           if (val.is_static()) {
                                               return checked_cast<int64_t>(val.get_length());
                                           }
                                           return checked_cast<int64_t>(mlir::ShapedType::kDynamic);
                                       }));

    if (partialShape.is_dynamic()) {
        const auto getUpperBound = [](const ov::Dimension& val) -> int64_t {
            if (val.is_static()) {
                return checked_cast<int64_t>(val.get_length());
            }
            return checked_cast<int64_t>(val.get_max_length());
        };
        auto resultBounds = to_small_vector(partialShape | transformed(getUpperBound));
        return {std::move(resultShape), std::move(resultBounds)};
    }

    return {std::move(resultShape), {}};
}

ov::PartialShape createPartialShapeFromShapeInfo(const ShapeInfo& shapeInfo) {
    ov::PartialShape partialShape = {};
    const auto toDimension = [](const int64_t val) -> ov::Dimension {
        if (val != mlir::ShapedType::kDynamic) {
            return ov::Dimension(val);
        }
        return ov::Dimension::dynamic();
    };
    const auto shape = shapeInfo.shape;
    std::transform(shape.begin(), shape.end(), std::back_inserter(partialShape), toDimension);
    if (partialShape.is_static()) {
        return partialShape;
    }
    const auto bounds = shapeInfo.bounds;
    for (const auto& idx : irange(partialShape.size())) {
        if (partialShape[idx].is_dynamic() && partialShape.size() == bounds.size()) {
            partialShape[idx] = ov::Dimension(1, bounds[idx]);
        }
    }
    return partialShape;
}
}  // namespace

ShapeInfo vpux::inferStridedSliceOutputShape(const ShapeInfo& inDataShapeInfo, ArrayRef<int64_t> begins,
                                             ArrayRef<int64_t> ends, ArrayRef<int64_t> strides,
                                             ArrayRef<int64_t> beginsShape, ArrayRef<int64_t> endsShape,
                                             ArrayRef<int64_t> stridesShape, ArrayRef<int64_t> beginMask,
                                             ArrayRef<int64_t> endMask, ArrayRef<int64_t> newAxisMask,
                                             ArrayRef<int64_t> shrinkAxisMask, ArrayRef<int64_t> ellipsisMask) {
    auto extractPaddedMask = [](ArrayRef<int64_t> mask, std::size_t expandSize) -> std::vector<int64_t> {
        auto maskVector = to_std_vector(mask);
        if (maskVector.size() < expandSize) {
            maskVector.insert(maskVector.end(), expandSize - maskVector.size(), 0);
        }
        return maskVector;
    };

    SmallVector<ArrayRef<int64_t>> opMasks{beginMask, endMask, newAxisMask, shrinkAxisMask, ellipsisMask};
    const auto padSize = std::max_element(opMasks.begin(), opMasks.end(), [](auto const& lhs, auto const& rhs) {
                             return lhs.size() < rhs.size();
                         })->size();

    const auto paddedBeginMask = extractPaddedMask(opMasks[0], padSize);
    const auto paddedEndMask = extractPaddedMask(opMasks[1], padSize);
    const auto paddedNewAxisMask = extractPaddedMask(opMasks[2], padSize);
    const auto paddedShrinkAxisMask = extractPaddedMask(opMasks[3], padSize);
    const auto paddedEllipsisMask = extractPaddedMask(opMasks[4], padSize);

    ov::Output<ov::Node> ovBegins = {};
    if (!begins.empty()) {
        const auto beginsVec = to_std_vector(begins);
        ovBegins = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape({beginsVec.size()}), beginsVec);
    } else if (!beginsShape.empty()) {
        ovBegins = std::make_shared<ov::op::v0::Parameter>(ov::element::i64,
                                                           ov::Shape(beginsShape.begin(), beginsShape.end()));
    }

    ov::Output<ov::Node> ovEnds = {};
    if (!ends.empty()) {
        const auto endsVec = to_std_vector(ends);
        ovEnds = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape({endsVec.size()}), endsVec);
    } else if (!endsShape.empty()) {
        ovEnds = std::make_shared<ov::op::v0::Parameter>(ov::element::i64,
                                                         ov::Shape(endsShape.begin(), endsShape.end()));
    }

    ov::Output<ov::Node> ovStrides = {};
    if (!strides.empty()) {
        const auto stridesVec = to_std_vector(strides);
        ovStrides =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape({stridesVec.size()}), stridesVec);
    } else if (!stridesShape.empty()) {
        ovStrides = std::make_shared<ov::op::v0::Parameter>(ov::element::i64,
                                                            ov::Shape(stridesShape.begin(), stridesShape.end()));
    }

    const auto inDataShape = createPartialShapeFromShapeInfo(inDataShapeInfo);
    const auto ovOp = ov::op::v1::StridedSlice(std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inDataShape),
                                               ovBegins, ovEnds, ovStrides, paddedBeginMask, paddedEndMask,
                                               paddedNewAxisMask, paddedShrinkAxisMask, paddedEllipsisMask);

    return createShapeInfoFromPartialShape(ovOp.get_output_partial_shape(0));
}

ShapeInfo vpux::inferAvgPoolOutputShape(const ShapeInfo& inDataShapeInfo, ArrayRef<int64_t> windowStrides,
                                        ArrayRef<int64_t> dataPaddingBelow, ArrayRef<int64_t> dataPaddingAbove,
                                        ArrayRef<int64_t> windowShape, IE::RoundingType roundingType) {
    const auto padsBegin = ov::Shape(dataPaddingBelow.begin(), dataPaddingBelow.end());
    const auto padsEnd = ov::Shape(dataPaddingAbove.begin(), dataPaddingAbove.end());
    const auto inDataShape = createPartialShapeFromShapeInfo(inDataShapeInfo);
    const auto ovOp = ov::op::v1::AvgPool(std::make_shared<ov::op::v0::Parameter>(ov::element::i32, inDataShape),
                                          ov::Strides(windowStrides.begin(), windowStrides.end()), padsBegin, padsEnd,
                                          ov::Shape(windowShape.begin(), windowShape.end()), false,
                                          static_cast<ov::op::RoundingType>(roundingType), ov::op::PadType::EXPLICIT);
    return createShapeInfoFromPartialShape(ovOp.get_output_partial_shape(0));
}

ShapeInfo vpux::inferAvgPool16OutputShape(const ShapeInfo& inDataShapeInfo, ArrayRef<int64_t> windowStrides,
                                          ArrayRef<int64_t> windowDilations, ArrayRef<int64_t> dataPaddingBelow,
                                          ArrayRef<int64_t> dataPaddingAbove, ArrayRef<int64_t> windowShape,
                                          IE::RoundingType roundingType) {
    const auto padsBegin = ov::Shape(dataPaddingBelow.begin(), dataPaddingBelow.end());
    const auto padsEnd = ov::Shape(dataPaddingAbove.begin(), dataPaddingAbove.end());
    const auto inDataShape = createPartialShapeFromShapeInfo(inDataShapeInfo);
    const auto ovOp = ov::op::v16::AvgPool(std::make_shared<ov::op::v0::Parameter>(ov::element::i32, inDataShape),
                                           ov::Strides(windowStrides.begin(), windowStrides.end()),
                                           ov::Strides(windowDilations.begin(), windowDilations.end()), padsBegin,
                                           padsEnd, ov::Shape(windowShape.begin(), windowShape.end()), false,
                                           static_cast<ov::op::RoundingType>(roundingType), ov::op::PadType::EXPLICIT);
    return createShapeInfoFromPartialShape(ovOp.get_output_partial_shape(0));
}

ShapeInfo vpux::inferMaxPoolOutputShape(const ShapeInfo& inDataShapeInfo, ArrayRef<int64_t> windowStrides,
                                        ArrayRef<int64_t> dataPaddingBelow, ArrayRef<int64_t> dataPaddingAbove,
                                        ArrayRef<int64_t> windowShape, IE::RoundingType roundingType) {
    const auto padsBegin = ov::Shape(dataPaddingBelow.begin(), dataPaddingBelow.end());
    const auto padsEnd = ov::Shape(dataPaddingAbove.begin(), dataPaddingAbove.end());
    const auto inDataShape = createPartialShapeFromShapeInfo(inDataShapeInfo);
    const auto ovOp = ov::op::v1::MaxPool(std::make_shared<ov::op::v0::Parameter>(ov::element::i32, inDataShape),
                                          ov::Strides(windowStrides.begin(), windowStrides.end()), padsBegin, padsEnd,
                                          ov::Shape(windowShape.begin(), windowShape.end()),
                                          static_cast<ov::op::RoundingType>(roundingType), ov::op::PadType::EXPLICIT);

    return createShapeInfoFromPartialShape(ovOp.get_output_partial_shape(0));
}

ShapeInfo vpux::inferMaxPool8OutputShape(const ShapeInfo& inDataShapeInfo, ArrayRef<int64_t> windowStrides,
                                         ArrayRef<int64_t> windowDilations, ArrayRef<int64_t> dataPaddingBelow,
                                         ArrayRef<int64_t> dataPaddingAbove, ArrayRef<int64_t> windowShape,
                                         IE::RoundingType roundingType) {
    const auto padsBegin = ov::Shape(dataPaddingBelow.begin(), dataPaddingBelow.end());
    const auto padsEnd = ov::Shape(dataPaddingAbove.begin(), dataPaddingAbove.end());
    const auto inDataShape = createPartialShapeFromShapeInfo(inDataShapeInfo);
    const auto ovOp = ov::op::v8::MaxPool(std::make_shared<ov::op::v0::Parameter>(ov::element::i32, inDataShape),
                                          ov::Strides(windowStrides.begin(), windowStrides.end()),
                                          ov::Strides(windowDilations.begin(), windowDilations.end()), padsBegin,
                                          padsEnd, ov::Shape(windowShape.begin(), windowShape.end()),
                                          static_cast<ov::op::RoundingType>(roundingType), ov::op::PadType::EXPLICIT,
                                          ov::element::i64, 0);
    return createShapeInfoFromPartialShape(ovOp.get_output_partial_shape(0));
}

ov::PartialShape getConvBackpropOutputShape(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> filterShape,
                                            ArrayRef<int64_t> windowStrides, ArrayRef<int64_t> dataPaddingBelow,
                                            ArrayRef<int64_t> dataPaddingAbove, ArrayRef<int64_t> windowDilations,
                                            ArrayRef<int64_t> outputPadding) {
    return ov::op::v1::ConvolutionBackpropData(
                   std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                           ov::Shape(inputShape.begin(), inputShape.end())),
                   std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                           ov::Shape(filterShape.begin(), filterShape.end())),
                   ov::Strides(windowStrides.begin(), windowStrides.end()),
                   ov::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),
                   ov::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),
                   ov::Strides(windowDilations.begin(), windowDilations.end()), ov::op::PadType::EXPLICIT,
                   ov::CoordinateDiff(outputPadding.begin(), outputPadding.end()))
            .get_output_partial_shape(0);
}

ov::PartialShape getGroupConvBackpropOutputShape(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> filterShape,
                                                 ArrayRef<int64_t> windowStrides, ArrayRef<int64_t> dataPaddingBelow,
                                                 ArrayRef<int64_t> dataPaddingAbove, ArrayRef<int64_t> windowDilations,
                                                 ArrayRef<int64_t> outputPadding) {
    return ov::op::v1::GroupConvolutionBackpropData(
                   std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                           ov::Shape(inputShape.begin(), inputShape.end())),
                   std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                           ov::Shape(filterShape.begin(), filterShape.end())),
                   ov::Strides(windowStrides.begin(), windowStrides.end()),
                   ov::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),
                   ov::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),
                   ov::Strides(windowDilations.begin(), windowDilations.end()), ov::op::PadType::EXPLICIT,
                   ov::CoordinateDiff(outputPadding.begin(), outputPadding.end()))
            .get_output_partial_shape(0);
}

SmallVector<int64_t> vpux::inferConvBackpropOutputShape(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> filterShape,
                                                        ArrayRef<int64_t> windowStrides,
                                                        ArrayRef<int64_t> dataPaddingBelow,
                                                        ArrayRef<int64_t> dataPaddingAbove,
                                                        ArrayRef<int64_t> windowDilations,
                                                        ArrayRef<int64_t> outputPadding) {
    auto backpropFilter = to_std_vector(filterShape);
    backpropFilter[Dims4D::Filter::OC.ind()] = inputShape[Dims4D::Act::C.ind()];
    auto ovOpShape = getConvBackpropOutputShape(inputShape, backpropFilter, windowStrides, dataPaddingBelow,
                                                dataPaddingAbove, windowDilations, outputPadding)
                             .get_shape();

    ovOpShape[Dims4D::Act::N.ind()] = inputShape[Dims4D::Act::N.ind()];
    ovOpShape[Dims4D::Act::C.ind()] = filterShape[Dims4D::Filter::IC.ind()];

    return to_small_vector(ovOpShape | transformed([](size_t val) {
                               return checked_cast<int64_t>(val);
                           }));
}

SmallVector<int64_t> vpux::inferGroupConvBackpropOutputShape(
        ArrayRef<int64_t> inputShape, ArrayRef<int64_t> filterShape, ArrayRef<int64_t> windowStrides,
        ArrayRef<int64_t> dataPaddingBelow, ArrayRef<int64_t> dataPaddingAbove, ArrayRef<int64_t> windowDilations,
        ArrayRef<int64_t> outputPadding) {
    auto groups = filterShape[0];
    auto IC = filterShape[1];
    auto OC = filterShape[2];

    auto backpropIn = to_std_vector(inputShape);
    backpropIn[Dims4D::Act::C.ind()] = groups * IC;
    auto ovOpShape = getGroupConvBackpropOutputShape(backpropIn, filterShape, windowStrides, dataPaddingBelow,
                                                     dataPaddingAbove, windowDilations, outputPadding)
                             .get_shape();

    ovOpShape[Dims4D::Act::N.ind()] = inputShape[Dims4D::Act::N.ind()];
    ovOpShape[Dims4D::Act::C.ind()] = groups * OC;

    return to_small_vector(ovOpShape | transformed([](size_t val) {
                               return checked_cast<int64_t>(val);
                           }));
}

SmallVector<int64_t> vpux::inferTransposedConvBackpropOutputShape(
        ArrayRef<int64_t> inputShape, ArrayRef<int64_t> filterShape, ArrayRef<int64_t> windowStrides,
        ArrayRef<int64_t> dataPaddingBelow, ArrayRef<int64_t> dataPaddingAbove, ArrayRef<int64_t> windowDilations,
        ArrayRef<int64_t> outputPadding) {
    auto backpropFilter = to_std_vector(filterShape);
    backpropFilter[Dims4D::Filter::OC.ind()] = inputShape[Dims4D::Act::C.ind()];
    auto ovOpShape = getConvBackpropOutputShape(inputShape, backpropFilter, windowStrides, dataPaddingBelow,
                                                dataPaddingAbove, windowDilations, outputPadding)
                             .get_shape();

    ovOpShape[Dims4D::Act::N.ind()] = inputShape[Dims4D::Act::N.ind()];
    ovOpShape[Dims4D::Act::C.ind()] = filterShape[Dims4D::Filter::OC.ind()];

    return to_small_vector(ovOpShape | transformed([](size_t val) {
                               return checked_cast<int64_t>(val);
                           }));
}

SmallVector<int64_t> vpux::inferTransposedGroupConvBackpropOutputShape(
        ArrayRef<int64_t> inputShape, ArrayRef<int64_t> filterShape, ArrayRef<int64_t> windowStrides,
        ArrayRef<int64_t> dataPaddingBelow, ArrayRef<int64_t> dataPaddingAbove, ArrayRef<int64_t> windowDilations,
        ArrayRef<int64_t> outputPadding) {
    // For 2D GroupTransposedConvolution:
    // input tensor layout is [N, C_IN * GROUPS, H, W]
    // kernel tensor layout is [GROUPS, C_OUT, C_IN, kH, kW]
    auto groups = filterShape[IE::GROUP_TRANSPOSED_CONV_GROUPS_DIM_INDEX];
    auto OC = filterShape[IE::GROUP_TRANSPOSED_CONV_C_OUT_DIM_INDEX];
    auto groupedChannels = groups * OC;

    auto transposedBackpropIn = to_std_vector(inputShape);
    transposedBackpropIn[Dims4D::Act::C.ind()] = groupedChannels;
    auto ovOpShape = getGroupConvBackpropOutputShape(transposedBackpropIn, filterShape, windowStrides, dataPaddingBelow,
                                                     dataPaddingAbove, windowDilations, outputPadding)
                             .get_shape();

    ovOpShape[Dims4D::Act::N.ind()] = inputShape[Dims4D::Act::N.ind()];
    ovOpShape[Dims4D::Act::C.ind()] = groupedChannels;

    return to_small_vector(ovOpShape | transformed([](size_t val) {
                               return checked_cast<int64_t>(val);
                           }));
}

ShapeInfo vpux::inferMatMulOutputShapeInfo(const ShapeInfo& in1ShapeInfo, const ShapeInfo& in2ShapeInfo,
                                           bool transposeA, bool transposeB) {
    const auto inPartialShape1 = createPartialShapeFromShapeInfo(in1ShapeInfo);
    const auto inPartialShape2 = createPartialShapeFromShapeInfo(in2ShapeInfo);

    auto op = ov::op::v0::MatMul(std::make_shared<ov::op::v0::Parameter>(ov::element::i32, inPartialShape1),
                                 std::make_shared<ov::op::v0::Parameter>(ov::element::i32, inPartialShape2), transposeA,
                                 transposeB);
    return createShapeInfoFromPartialShape(op.get_output_partial_shape(0));
}

ShapeInfo vpux::inferPermuteQuantizeOutputShapeInfo(mlir::Value input, DimsOrder inOrder, vpux::NDTypeInterface newType,
                                                    DimsOrder outOrder, mlir::AffineMap memPerm) {
    const auto inMemShape = inOrder.toMemoryOrder(newType.getShape());
    const auto outMemShape = applyPerm(inMemShape, memPerm);
    const auto outShape = outOrder.toLogicalOrder(outMemShape);

    SmallVector<int64_t> bounds{};
    if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(input.getType())) {
        auto newBoundsRaw = mlir::dyn_cast<Core::BoundedTensorType>(newType).getBounds().raw();
        auto newShape = newType.getShape();

        SmallVector<int64_t> transformedBounds(newBoundsRaw.size());
        std::transform(newBoundsRaw.begin(), newBoundsRaw.end(), newShape.raw().begin(), transformedBounds.begin(),
                       [](int64_t bound, int64_t shape) {
                           return shape == mlir::ShapedType::kDynamic ? bound : shape;
                       });
        bounds = std::move(transformedBounds);
    }

    return {std::move(outShape.raw()), std::move(bounds)};
}

ShapeInfo vpux::inferConvolutionOutputShapeInfo(const ShapeInfo& inShapeInfo, const ShapeInfo& filterShapeInfo,
                                                ArrayRef<int64_t> windowStrides, ArrayRef<int64_t> dataPaddingBelow,
                                                ArrayRef<int64_t> dataPaddingAbove, ArrayRef<int64_t> windowDilations) {
    const auto inPartialShape = createPartialShapeFromShapeInfo(inShapeInfo);
    const auto filterPartialShape = createPartialShapeFromShapeInfo(filterShapeInfo);

    const auto op =
            ov::op::v1::Convolution(std::make_shared<ov::op::v0::Parameter>(ov::element::i32, inPartialShape),
                                    std::make_shared<ov::op::v0::Parameter>(ov::element::i32, filterPartialShape),
                                    ov::Strides(windowStrides.begin(), windowStrides.end()),
                                    ov::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),
                                    ov::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),
                                    ov::Strides(windowDilations.begin(), windowDilations.end()));
    return createShapeInfoFromPartialShape(op.get_output_partial_shape(0));
}

ShapeInfo vpux::inferGroupConvolutionOutputShapeInfo(ShapeInfo& inShapeInfo, ShapeInfo& filterShapeInfo,
                                                     ArrayRef<int64_t> windowStrides,
                                                     ArrayRef<int64_t> dataPaddingBelow,
                                                     ArrayRef<int64_t> dataPaddingAbove,
                                                     ArrayRef<int64_t> windowDilations,
                                                     std::optional<int64_t> maybeGroups, bool hasOutputPadding) {
    VPUX_THROW_WHEN(filterShapeInfo.isDynamic(), "Filters with dynamic shapes aren't currently supported");

    // We need to adjust the input and filter shapes to reuse OV helpers for normal convolution
    auto groups = maybeGroups.value_or(0);
    if (groups != 0) {
        VPUX_THROW_UNLESS(filterShapeInfo.rank() == inShapeInfo.rank(),
                          "Input rank '{0}' does not match filter rank '{1}'. (groups != 0)", inShapeInfo.rank(),
                          filterShapeInfo.rank());
    } else {
        VPUX_THROW_UNLESS(filterShapeInfo.rank() == inShapeInfo.rank() + 1,
                          "Input rank '{0}' does not match filter rank '{1}'. (groups == 0)", inShapeInfo.rank(),
                          filterShapeInfo.rank());

        groups = filterShapeInfo.shape[Dims4D::Act::N.ind()];
        filterShapeInfo.shape[Dims4D::Act::C.ind()] *= groups;
        filterShapeInfo.shape.erase(filterShapeInfo.shape.begin());
    }

    const auto adjustShapeChannels = [&](int64_t& channels) {
        // The number of groups is influenced by the output channels so the division computes a wrong value because the
        // input is still expanded in case of ODU autopad
        if (hasOutputPadding) {
            channels = filterShapeInfo.shape[Dims4D::Act::C.ind()];
        } else {
            channels /= groups;
        }
    };

    adjustShapeChannels(inShapeInfo.shape[Dims4D::Act::C.ind()]);
    if (inShapeInfo.isDynamic()) {
        adjustShapeChannels(inShapeInfo.bounds[Dims4D::Act::C.ind()]);
    }

    const auto inPartialShape = createPartialShapeFromShapeInfo(inShapeInfo);
    const auto filterPartialShape = createPartialShapeFromShapeInfo(filterShapeInfo);

    auto op = ov::op::v1::Convolution(std::make_shared<ov::op::v0::Parameter>(ov::element::i32, inPartialShape),
                                      std::make_shared<ov::op::v0::Parameter>(ov::element::i32, filterPartialShape),
                                      ov::Strides(windowStrides.begin(), windowStrides.end()),
                                      ov::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),
                                      ov::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),
                                      ov::Strides(windowDilations.begin(), windowDilations.end()));

    return createShapeInfoFromPartialShape(op.get_output_partial_shape(0));
}

ShapeInfo vpux::inferTransposedConvBackpropOutputShapeInfo(
        const ShapeInfo& inShapeInfo, const ShapeInfo& filterShapeInfo, ArrayRef<int64_t> windowStrides,
        ArrayRef<int64_t> dataPaddingBelow, ArrayRef<int64_t> dataPaddingAbove, ArrayRef<int64_t> windowDilations,
        ArrayRef<int64_t> outputPadding) {
    const auto inPartialShape = createPartialShapeFromShapeInfo(inShapeInfo);
    const auto filterPartialShape = createPartialShapeFromShapeInfo(filterShapeInfo);

    auto backpropFilter = to_std_vector(filterPartialShape);
    backpropFilter[Dims4D::Filter::OC.ind()] = inPartialShape[Dims4D::Act::C.ind()];

    auto ovOpShape = ov::op::v1::ConvolutionBackpropData(
                             std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inPartialShape),
                             std::make_shared<ov::op::v0::Parameter>(ov::element::f32, backpropFilter),
                             ov::Strides(windowStrides.begin(), windowStrides.end()),
                             ov::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),
                             ov::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),
                             ov::Strides(windowDilations.begin(), windowDilations.end()), ov::op::PadType::EXPLICIT,
                             ov::CoordinateDiff(outputPadding.begin(), outputPadding.end()))
                             .get_output_partial_shape(0);

    ovOpShape[Dims4D::Act::N.ind()] = inPartialShape[Dims4D::Act::N.ind()];
    ovOpShape[Dims4D::Act::C.ind()] = filterPartialShape[Dims4D::Filter::OC.ind()];

    return createShapeInfoFromPartialShape(ovOpShape);
}

//
// Tensor Reifiers
//

mlir::OpFoldResult vpux::reifyDim(mlir::OpBuilder builder, mlir::Value value, mlir::RankedTensorType type, size_t idx,
                                  std::optional<mlir::Location> loc) {
    if (type.isDynamicDim(idx)) {
        const auto actualLoc = loc.value_or(value.getLoc());
        auto dimLoc = appendLoc(actualLoc, llvm::StringLiteral("dim_{0}"), idx);
        auto index = builder.create<mlir::arith::ConstantIndexOp>(appendLoc(dimLoc, "const_index"), idx);
        auto dimOp = builder.createOrFold<mlir::tensor::DimOp>(dimLoc, value, index);
        return mlir::getValueOrCreateConstantIndexOp(builder, appendLoc(actualLoc, "const_index"), dimOp);
    }
    return builder.getIndexAttr(type.getDimSize(idx));
}

mlir::OpFoldResult vpux::reifyDim(mlir::OpBuilder builder, mlir::Value value, size_t idx,
                                  std::optional<mlir::Location> loc) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
    return reifyDim(builder, value, type, idx, loc);
}

SmallVector<mlir::OpFoldResult> vpux::reifyTrivialTensor(mlir::OpBuilder builder, mlir::Value input,
                                                         std::optional<mlir::Location> loc) {
    // For operations which do not modify input shape in any way.
    // If output dimension is static, corresponding input dimension is also static.
    // If output dimension is dynamic, corresponding input dimension is also dynamic.
    // [N, C, H, ?] reifies to [N, C, H, ?], [N, ?, H, ?] reifies to [N, ?, H, ?], etc.
    const auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
    const auto rank = inputType.getRank();

    SmallVector<mlir::OpFoldResult> dims;
    dims.reserve(rank);
    for (auto i : irange(rank)) {
        dims.push_back(reifyDim(builder, input, inputType, i, loc));
    }

    return dims;
}

mlir::FailureOr<SmallVector<mlir::OpFoldResult>> vpux::reifyMatMulTensors(mlir::OpBuilder& builder, mlir::Value input1,
                                                                          mlir::Value input2, bool transposeA,
                                                                          bool transposeB, mlir::Location loc) {
    const auto type1 = mlir::cast<mlir::RankedTensorType>(input1.getType());
    const auto type2 = mlir::cast<mlir::RankedTensorType>(input2.getType());

    const auto shape1 = type1.getShape();
    const auto shape2 = type2.getShape();

    // Step 1: Apply transpositions if needed
    auto getTransposedShape = [&](ArrayRef<int64_t> shape, bool transpose) {
        if (transpose && shape.size() >= 2) {
            // Assume the default dimensions order is used. It means the H and W dimensions are the last two elements in
            // the array, so we can swap them to transpose the tensor
            SmallVector<int64_t> transposedShape(shape.begin(), shape.end());
            std::swap(transposedShape[shape.size() - 2], transposedShape[shape.size() - 1]);
            return transposedShape;
        }
        return SmallVector<int64_t>(shape.begin(), shape.end());
    };

    auto shape1Transposed = getTransposedShape(shape1, transposeA);
    auto shape2Transposed = getTransposedShape(shape2, transposeB);

    // Step 2: Unsqueeze 1D tensors
    // If rank of the first input is equal to 1, it is always unsqueezed to 2D tensor row vector
    // If rank of the second input is equal to 1, it is always unsqueezed to 2D tensor column vector
    auto unsqueeze1D = [](ArrayRef<int64_t> shape, bool isRowVector) {
        if (shape.size() == 1) {
            if (isRowVector) {
                return SmallVector<int64_t>{1, shape[0]};
            } else {
                return SmallVector<int64_t>{shape[0], 1};
            }
        }
        return SmallVector<int64_t>(shape.begin(), shape.end());
    };

    shape1Transposed = unsqueeze1D(shape1Transposed, true);
    shape2Transposed = unsqueeze1D(shape2Transposed, false);

    // Step 3: Align ranks by unsqueezing from the left
    std::pair<uint32_t, uint32_t> alignmentAxesCnt{0, 0};
    auto alignRanks = [&alignmentAxesCnt](SmallVector<int64_t>& shape1, SmallVector<int64_t>& shape2) {
        while (shape1.size() < shape2.size()) {
            ++alignmentAxesCnt.first;
            shape1.insert(shape1.begin(), 1);
        }
        while (shape2.size() < shape1.size()) {
            ++alignmentAxesCnt.second;
            shape2.insert(shape2.begin(), 1);
        }
    };

    alignRanks(shape1Transposed, shape2Transposed);

    // Step 4: Broadcast batch dimensions
    SmallVector<mlir::OpFoldResult> outDims;
    for (size_t i = 0; i < shape1Transposed.size() - 2; ++i) {
        if (shape1Transposed[i] == 1) {
            outDims.push_back(reifyDim(builder, input2, type2, i, loc));
        } else if (shape2Transposed[i] == 1 || shape1Transposed[i] == shape2Transposed[i]) {
            outDims.push_back(reifyDim(builder, input1, type1, i, loc));
        } else {
            return errorAt(loc, "Incompatible batch dimensions: '{0}' and '{1}'", shape1Transposed[i],
                           shape2Transposed[i]);
        }
    }

    // Step 5: Determine the output matrix dimensions, taking into account transposition and rank alignment if applied
    // result H dim is equal to input1.H
    outDims.push_back(reifyDim(builder, input1, type1,
                               shape1Transposed.size() - (transposeA ? 1 : 2) - alignmentAxesCnt.first, loc));
    // result W dim is equal to input2.W
    outDims.push_back(reifyDim(builder, input2, type2,
                               shape2Transposed.size() - (transposeB ? 2 : 1) - alignmentAxesCnt.second, loc));

    return outDims;
}

mlir::FailureOr<SmallVector<mlir::OpFoldResult>> vpux::reifyEltwiseTensors(mlir::OpBuilder& builder, mlir::Value input1,
                                                                           mlir::Value input2,
                                                                           IE::AutoBroadcastType broadcastType,
                                                                           mlir::Location loc) {
    const auto type1 = mlir::cast<mlir::RankedTensorType>(input1.getType());
    const auto type2 = mlir::cast<mlir::RankedTensorType>(input2.getType());

    const auto shape1 = type1.getShape();
    const auto shape2 = type2.getShape();

    if (broadcastType == IE::AutoBroadcastType::NONE_OR_EXPLICIT) {
        if (shape1 != shape2) {
            return errorAt(loc, "Input shapes must be equal in case BroadcastType is NONE");
        }

        const auto outRank = shape1.size();
        SmallVector<mlir::OpFoldResult> outDims;
        outDims.reserve(outRank);
        auto rankRange = irange(outRank);
        std::transform(rankRange.begin(), rankRange.end(), std::back_inserter(outDims), [&](auto i) {
            return reifyDim(builder, input1, type1, i, loc);
        });
        return outDims;
    } else if (broadcastType == IE::AutoBroadcastType::NUMPY) {
        const auto in1Rank = shape1.size();
        const auto in2Rank = shape2.size();

        auto in1ShapeIter = shape1.rbegin();
        auto in2ShapeIter = shape2.rbegin();

        const auto outRank = std::max(shape1.size(), shape2.size());
        SmallVector<mlir::OpFoldResult> outDims(outRank);

        for (auto i : irange(outRank)) {
            if (in1ShapeIter != shape1.rend() && in2ShapeIter != shape2.rend()) {
                if (!isBroadcastable(*in1ShapeIter, *in2ShapeIter)) {
                    return errorAt(loc, "Got non broadcastable dimensions pair : '{0}' and {1}'", *in1ShapeIter,
                                   *in2ShapeIter);
                }
            }

            if (in1ShapeIter == shape1.rend() || (in2ShapeIter != shape2.rend() && (*in1ShapeIter == 1))) {
                outDims[outRank - i - 1] = reifyDim(builder, input2, type2, in2Rank - i - 1, loc);
            } else {
                VPUX_THROW_UNLESS(in1ShapeIter != shape1.rend(), "Failed to broadcast shapes: {0}, {1} at {2}", shape1,
                                  shape2, loc);
                outDims[outRank - i - 1] = reifyDim(builder, input1, type1, in1Rank - i - 1, loc);
            }

            if (in1ShapeIter != shape1.rend()) {
                ++in1ShapeIter;
            }
            if (in2ShapeIter != shape2.rend()) {
                ++in2ShapeIter;
            }
        }

        return outDims;
    }

    return errorAt(loc, "Unsupported BroadcastType '{0}'", broadcastType);
}

mlir::FailureOr<SmallVector<mlir::OpFoldResult>> vpux::reifyConvPoolTensors(
        mlir::OpBuilder& builder, mlir::Value input, mlir::Value output, mlir::Value kernel,
        ArrayRef<int64_t> kernelSize, ArrayRef<int64_t> strides, ArrayRef<int64_t> padBegin, ArrayRef<int64_t> padEnd,
        mlir::Location loc) {
    const auto inputShapedType = mlir::cast<mlir::ShapedType>(input.getType());
    const auto outputShapedType = mlir::cast<mlir::ShapedType>(output.getType());

    VPUX_THROW_WHEN(inputShapedType.getRank() != 4 || outputShapedType.getRank() != 4,
                    "reifyConvPoolTensors: Unsupported input or output rank: {0} , {1}", inputShapedType.getRank(),
                    outputShapedType.getRank());

    if (kernel != nullptr) {
        const auto kernelShapedType = mlir::cast<mlir::ShapedType>(kernel.getType());
        VPUX_THROW_WHEN(kernelShapedType.getRank() != 4, "reifyConvPoolTensors: Unsupported kernel rank: {0}",
                        kernelShapedType.getRank());
    }

    auto makeIndex = [&](int64_t value, mlir::Location indexLoc) {
        return builder.createOrFold<mlir::arith::ConstantIndexOp>(appendLoc(indexLoc, "const_index"), value);
    };

    auto calculateDimSize = [&](mlir::Value inputDim, int64_t kernelDim, int64_t padBegin, int64_t padEnd,
                                int64_t stride, mlir::Location dimLoc) {
        // output = (input + padBegin + padEnd - kernelDim + stride) / stride
        auto padConst = padBegin + padEnd - kernelDim + stride;
        auto addLoc = appendLoc(dimLoc, "add");
        auto sum = builder.createOrFold<mlir::arith::AddIOp>(addLoc, inputDim, makeIndex(padConst, addLoc));
        auto divLoc = appendLoc(dimLoc, "div");
        return builder.createOrFold<mlir::arith::DivSIOp>(divLoc, sum, makeIndex(stride, divLoc));
    };

    // Use generator functions based on index for each output dimension
    auto computeShapeForDim = [&](int64_t idx) -> mlir::OpFoldResult {
        if (idx == Dims4D::Act::N.ind()) {
            return reifyDim(builder, input, idx, loc);
        } else if (idx == Dims4D::Act::C.ind()) {
            return kernel == nullptr ? reifyDim(builder, input, Dims4D::Act::C.ind(), loc)
                                     : reifyDim(builder, kernel, Dims4D::Filter::OC.ind(), loc);
        } else if (idx == Dims4D::Act::H.ind() || idx == Dims4D::Act::W.ind()) {
            auto inputDim = reifyDim(builder, input, idx, loc);
            auto inputDimVal = mlir::dyn_cast<mlir::Value>(inputDim);
            VPUX_THROW_WHEN(inputDimVal == nullptr, "Failed to reify input dimension {0} for input {1} at location {2}",
                            idx, input, loc);
            auto adjustedIdx = idx - 2;
            return calculateDimSize(inputDimVal, kernelSize[adjustedIdx], padBegin[adjustedIdx], padEnd[adjustedIdx],
                                    strides[adjustedIdx], appendLoc(loc, llvm::StringLiteral("dim_{0}"), adjustedIdx));
        } else {
            VPUX_THROW("Unexpected dimension index {0}", idx);
        }
    };

    SmallVector<mlir::OpFoldResult> shapes;
    for (const auto dim : llvm::seq<int64_t>(0, outputShapedType.getRank())) {
        if (outputShapedType.isDynamicDim(dim)) {
            shapes.push_back(mlir::getValueOrCreateConstantIndexOp(builder, appendLoc(loc, "const_index"),
                                                                   computeShapeForDim(dim)));
        } else {
            shapes.push_back(builder.getIndexAttr(outputShapedType.getDimSize(dim)));
        }
    }

    return shapes;
}

ShapeInfo vpux::inferEltwiseOutputShapeInfo(const ShapeInfo& in1ShapeInfo, const ShapeInfo& in2ShapeInfo,
                                            IE::AutoBroadcastType broadcastType, mlir::ArrayAttr inputPaddingAttr,
                                            mlir::ArrayAttr outputPaddingAttr, mlir::Location loc) {
    auto in1Shape = in1ShapeInfo.shape;
    auto in2Shape = in2ShapeInfo.shape;
    if (inputPaddingAttr) {
        if (mlir::failed(IE::unpadInputShape(in1Shape, inputPaddingAttr, loc))) {
            return {};
        }
        if (mlir::failed(IE::unpadInputShape(in2Shape, inputPaddingAttr, loc))) {
            return {};
        }
    }
    auto outShapeRes = IE::broadcastEltwiseShape(in1Shape, in2Shape, broadcastType, loc);
    if (mlir::failed(outShapeRes)) {
        return {};
    }
    auto outShape = outShapeRes.value();
    if (outputPaddingAttr) {
        if (mlir::failed(IE::padOutputShape(outShape, outputPaddingAttr, loc))) {
            return {};
        }
    }

    if (in1ShapeInfo.bounds.empty() && in2ShapeInfo.bounds.empty()) {
        return {std::move(outShape), {}};
    }

    // In case if we don't have bounds - use shape as bounds
    const auto in1Bounds = in1ShapeInfo.bounds.empty() ? in1ShapeInfo.shape : in1ShapeInfo.bounds;
    const auto in2Bounds = in2ShapeInfo.bounds.empty() ? in2ShapeInfo.shape : in2ShapeInfo.bounds;

    auto outBoundRes = IE::broadcastEltwiseShape(in1Bounds, in2Bounds, broadcastType, loc);
    if (mlir::failed(outBoundRes)) {
        return {};
    }

    return {std::move(outShape), outBoundRes.value()};
}

ShapeInfo vpux::inferEltwiseOutputShapeInfo(const ShapeInfo& in1ShapeInfo, const ShapeInfo& in2ShapeInfo,
                                            IE::AutoBroadcastType broadcastType, mlir::Location loc) {
    return inferEltwiseOutputShapeInfo(in1ShapeInfo, in2ShapeInfo, broadcastType, {}, {}, loc);
}
