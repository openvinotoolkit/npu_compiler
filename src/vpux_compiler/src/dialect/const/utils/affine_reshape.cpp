//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/utils/affine_reshape.hpp"

std::optional<mlir::Type> vpux::Const::inferElemTypeAffineReshape(ShapeRef inputShape, mlir::Type inputElementType,
                                                                  const SmallVector<SmallVector<int64_t>>& dimMapping,
                                                                  ArrayRef<int64_t> shapeValue) {
    const auto perAxisQType = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(inputElementType);
    if (perAxisQType == nullptr) {
        return inputElementType;
    }

    const auto inputQAxis = perAxisQType.getQuantizedDimension();
    const auto outputShape = shapeValue;

    // get output dims for input Q axis
    const auto& outputDims = dimMapping[inputQAxis];
    int64_t outQAxis = -1;
    int64_t inputQAxisSize = inputShape.raw()[inputQAxis];

    if (inputQAxisSize == 1) {
        // Per tensor, but must be per channel, do not handle it here
        return std::nullopt;
    }

    for (const auto& dim : outputDims) {
        if (inputQAxisSize == outputShape[dim]) {
            // firstly check that element is unique and others == 1
            if (std::find_if(outputDims.begin(), outputDims.end(), [&](int64_t elem) {
                    return (outputShape[elem] != 1 && outputShape[elem] != inputQAxisSize);
                }) != outputDims.end()) {
                return std::nullopt;
            }
            outQAxis = dim;
            break;
        }
    }

    if (outQAxis == -1) {
        return std::nullopt;
    }

    if (const auto perAxisQuantileQType =
                inputElementType.dyn_cast_or_null<mlir::quant::QuantileQuantizedPerAxisType>()) {
        return mlir::quant::QuantileQuantizedPerAxisType::get(
                perAxisQuantileQType.getFlags(), perAxisQuantileQType.getStorageType(),
                perAxisQuantileQType.getQuantileType(), perAxisQuantileQType.getExpressedType(),
                perAxisQuantileQType.getQuantiles(), perAxisQuantileQType.getScales(),
                perAxisQuantileQType.getZeroPoints(), static_cast<int32_t>(outQAxis),
                perAxisQuantileQType.getStorageTypeMin(), perAxisQuantileQType.getStorageTypeMax());
    }

    return mlir::quant::UniformQuantizedPerAxisType::get(
            perAxisQType.getFlags(), perAxisQType.getStorageType(), perAxisQType.getExpressedType(),
            perAxisQType.getScales(), perAxisQType.getZeroPoints(), static_cast<int32_t>(outQAxis),
            perAxisQType.getStorageTypeMin(), perAxisQType.getStorageTypeMax());
}

std::optional<vpux::DimsOrder> vpux::Const::inferAffineReshapeOutputLayout(const DimArr& inPerm,
                                                                           mlir::ArrayAttr dimMapAttr) {
    VPUX_THROW_UNLESS(dimMapAttr != nullptr, "dimMapAttr is nullptr");
    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(dimMapAttr);
    SmallVector<vpux::Dim> perm;

    // Iterate over input dims in the given order and push back corresponding output dims as indicated by the op's
    // dim_mapping. The result is the permutation of output dims.
    bool layoutInferFail = false;
    for (auto pIt = inPerm.begin(); pIt != inPerm.end(); ++pIt) {
        const auto& outputDims = dimMapping[pIt->ind()];
        for (const auto& dim : outputDims) {
            const auto outDim = vpux::Dim(dim);

            // Ensure input dim order is not switched.
            // E.g. nchw -> c'h'w', with n = c', c = h', h * w = w'
            // Layouts 0123 and 0132 would both produce 012 output layout, but
            // the content of w' would not be the same.
            if (!perm.empty() && perm.back() == outDim) {
                layoutInferFail = std::prev(pIt)->ind() > pIt->ind();
                if (layoutInferFail) {
                    return std::nullopt;
                }

                continue;
            }
            perm.push_back(outDim);
        }
    }

    // Check that the resulting output permutation does not have duplicate dims
    SmallVector<vpux::Dim> temp(perm);
    llvm::sort(temp.begin(), temp.end(), [](const vpux::Dim& dim0, const vpux::Dim& dim1) {
        return dim0.ind() < dim1.ind();
    });

    if (std::adjacent_find(temp.begin(), temp.end()) != temp.end()) {
        return std::nullopt;
    }

    return DimsOrder::fromPermutation(ArrayRef(perm));
}
