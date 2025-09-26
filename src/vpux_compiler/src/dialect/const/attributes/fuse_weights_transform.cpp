//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/constant_fusion.hpp"

using namespace vpux;

NDTypeInterface Const::FuseWeightsAttr::inferOutputType(NDTypeInterface input) const {
    (void)input;
    return getFusedType();
}

bool vpux::Const::FuseWeightsAttr::inferOutputSplat(bool, vpux::NDTypeInterface) const {
    return false;
}

Const::Content Const::FuseWeightsAttr::transform(Const::Content& input) const {
    auto outputType = inferOutputType(input.getType());
    auto output = Const::Content::allocTempBuffer(outputType, outputType.getElementType(),
                                                  inferOutputSplat(input.isSplat(), input.getType()));

    auto fusedBuffer = output.getRawTempBuf();

    Const::ContentAttr contentVector[] = {getWeightsTable(), getWeights(), getSparsity(), getActivations()};

    size_t index = 0;
    for (auto content : contentVector) {
        if (content == nullptr) {
            continue;
        }
        auto foldedContent = content.fold();
        auto contentType = mlir::cast<vpux::NDTypeInterface>(foldedContent.getType());
        auto elemType = contentType.getElementType();

        if (elemType.isInteger(1)) {
            const auto packedNumElems = contentType.getNumElements() / CHAR_BIT;
            const auto packedElemType = getUInt8Type(contentType.getContext());
            const auto packedContentType =
                    contentType.changeShapeElemType(ShapeRef({1, 1, 1, packedNumElems}), packedElemType);
            auto packedContent = Const::Content::fromRawBuffer(packedContentType, foldedContent.getRawStorageBuf(),
                                                               packedElemType, foldedContent.isSplat());
            appendContentToVector(packedContent, fusedBuffer, index);
        } else {
            appendContentToVector(foldedContent, fusedBuffer, index);
        }
    }

    return output;
}

//
// FuseWeightsAttr::getStableHashValue
//

llvm::hash_code vpux::Const::FuseWeightsAttr::getStableHashValue() const {
    VPUX_THROW("Not implemented. It requires folding of the content, which is expensive");
}
