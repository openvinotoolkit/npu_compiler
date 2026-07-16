//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <algorithm>
#include <cassert>
#include <climits>
#include <functional>
#include <numeric>

using namespace vpux;

namespace {

template <typename WidenedT, typename InElemT>
void addConstToOutput(ArrayRef<InElemT> inBuf, Const::Content& output, int64_t preDims, int64_t singleCopyElements,
                      int64_t outPlaneSize, int64_t planeOffset) {
    auto outBuf = output.getTempBuf<WidenedT>();
    for (int64_t n = 0; n < preDims; ++n) {
        const auto* src = inBuf.data() + n * singleCopyElements;
        auto* dst = outBuf.data() + n * outPlaneSize + planeOffset;
        std::transform(src, src + singleCopyElements, dst, [](const InElemT& v) {
            return static_cast<WidenedT>(v);
        });
    }
}

}  // namespace

//
// ConcatAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::ConcatAttr::inferOutputType(vpux::NDTypeInterface /*input*/) const {
    auto constants = getConstants();
    VPUX_THROW_WHEN(constants.empty(), "ConcatAttr: constants must not be empty");

    auto firstType = constants.front().getType();
    auto shape = to_small_vector(firstType.getShape());
    const auto axisInd = getAxis().getInt();

    for (size_t i = 1; i < constants.size(); ++i) {
        shape[axisInd] += constants[i].getType().getShape()[Dim(axisInd)];
    }

    // For per-axis quantized types where the quantized dimension coincides with the concat axis,
    // the scales and zero-points from all inputs must be merged to produce a valid output type.
    const auto elemType = firstType.getElementType();
    if (auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType)) {
        if (perAxisType.getQuantizedDimension() == static_cast<int32_t>(axisInd)) {
            SmallVector<double> scales(perAxisType.getScales().begin(), perAxisType.getScales().end());
            SmallVector<int64_t> zeroPoints(perAxisType.getZeroPoints().begin(), perAxisType.getZeroPoints().end());
            for (size_t i = 1; i < constants.size(); ++i) {
                auto inputPerAxisType =
                        mlir::cast<mlir::quant::UniformQuantizedPerAxisType>(constants[i].getType().getElementType());
                scales.append(inputPerAxisType.getScales().begin(), inputPerAxisType.getScales().end());
                zeroPoints.append(inputPerAxisType.getZeroPoints().begin(), inputPerAxisType.getZeroPoints().end());
            }
            auto mergedElemType = mlir::quant::UniformQuantizedPerAxisType::get(
                    perAxisType.getFlags(), perAxisType.getStorageType(), perAxisType.getExpressedType(), scales,
                    zeroPoints, perAxisType.getQuantizedDimension(), perAxisType.getStorageTypeMin(),
                    perAxisType.getStorageTypeMax());
            return firstType.changeShapeElemType(ShapeRef(shape), mergedElemType);
        }
    }

    return firstType.changeShape(ShapeRef(shape));
}

//
// ConcatAttr::inferOutputSplat
//

bool vpux::Const::ConcatAttr::inferOutputSplat(bool /*inputIsSplat*/, vpux::NDTypeInterface /*input*/) const {
    return false;
}

//
// ConcatAttr::transform
//

Const::Content vpux::Const::ConcatAttr::transform(vpux::Const::Content& input) const {
    // The input comes from a dummy base content (zero-sized tensor) and is not used for data.
    // All actual data comes from the constants stored in this attribute.
    auto dummy = Const::getDummyBaseContent(input.getType().getContext());
    VPUX_THROW_UNLESS(input.getType() == dummy.getType(), "ConcatAttr expects a dummy base content, got {0}",
                      input.getType());

    auto outNdInterface = inferOutputType(input.getType());
    auto constants = getConstants();

    // Fast path: single constant can be returned directly without allocation
    if (constants.size() == 1) {
        auto foldedContent = constants.front().fold();
        return Const::Content::moveBuffer(outNdInterface, std::move(foldedContent));
    }

    const auto outputIsSplat = inferOutputSplat(input.isSplat(), outNdInterface);
    assert(!outputIsSplat && "ConcatAttr: splat output is not supported");

    // For quantized types, operate on the raw storage type (e.g. ui8) rather than the
    // expressed type (e.g. f16).
    const auto outElemType = outNdInterface.getElementType();
    const auto storageType = mlir::isa<mlir::quant::QuantizedType>(outElemType)
                                     ? mlir::cast<mlir::quant::QuantizedType>(outElemType).getStorageType()
                                     : outElemType;
    const bool isFloat = mlir::isa<mlir::FloatType>(storageType);

    const auto axisInd = getAxis().getInt();
    const auto axisValue = Dim(axisInd);

    auto offsetAttr = parseIntArrayOfArrayAttr<int64_t>(getStaticOffsets());

    auto outPhyShape = outNdInterface.getMemShape().raw();
    auto memDimIndex = outNdInterface.getDimsOrder().dimPos(axisValue);
    const auto preDims = std::accumulate(outPhyShape.begin(), outPhyShape.begin() + memDimIndex, int64_t{1},
                                         std::multiplies<int64_t>());
    const auto afterDims = std::accumulate(outPhyShape.begin() + memDimIndex + 1, outPhyShape.end(), int64_t{1},
                                           std::multiplies<int64_t>());

    // Use a widened storage type (f32 for floats, i64/u64 for integers) so that elements from
    // inputs with different storage types can be safely cast into a common representation.
    auto ctx = outNdInterface.getContext();
    const auto signSemantics = storageType.isSignedInteger() ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned;
    auto widenedStorageType = isFloat ? mlir::Type(mlir::Float32Type::get(ctx))
                                      : mlir::Type(mlir::IntegerType::get(ctx, 64, signSemantics));
    auto output = Const::Content::allocTempBuffer(outNdInterface, widenedStorageType, outputIsSplat);

    const auto outPlaneSize = afterDims * outPhyShape[memDimIndex];

    for (int64_t inIndex = 0; inIndex < static_cast<int64_t>(constants.size()); ++inIndex) {
        auto content = constants[inIndex].fold();
        auto cstShape = content.getType().getShape();
        const auto singleCopyElements = afterDims * cstShape[axisValue];
        const auto planeOffset = offsetAttr[inIndex][axisInd] * afterDims;
        const auto numElements = content.getType().getNumElements();

        content.read(widenedStorageType, [&](auto inBuf, auto dummy) {
            using InElemT = typename decltype(inBuf)::value_type;
            using WidenedT = std::decay_t<decltype(dummy)>;
            std::vector<InElemT> nonSplatStorage;
            assert(!inBuf.empty());
            if (inBuf.size() == 1) {
                nonSplatStorage.resize(numElements, inBuf[0]);
                inBuf = nonSplatStorage;
            }
            addConstToOutput<WidenedT>(inBuf, output, preDims, singleCopyElements, outPlaneSize, planeOffset);
        });
    }

    return output;
}

//
// ConcatAttr::getStableHashValue
//

llvm::hash_code vpux::Const::ConcatAttr::getStableHashValue() const {
    VPUX_THROW("Not implemented. It requires folding of the content, which is expensive");
}
