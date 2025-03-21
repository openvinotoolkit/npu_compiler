//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/constant_fusion.hpp"

using namespace vpux;

NDTypeInterface Const::FuseAttr::inferOutputType(NDTypeInterface input) const {
    (void)input;
    return getFusedType();
}

bool vpux::Const::FuseAttr::inferOutputSplat(bool, vpux::NDTypeInterface) {
    return false;
}

Const::Content Const::FuseAttr::transform(Const::Content& input) const {
    auto outputType = inferOutputType(input.getType());
    auto output = Const::Content::allocTempBuffer(outputType, outputType.getElementType(),
                                                  inferOutputSplat(input.isSplat(), input.getType()));

    auto fusedBuffer = output.getRawTempBuf();

    size_t index = 0;
    for (auto content : getConstants()) {
        if (content == nullptr) {
            continue;
        }
        auto foldedContent = content.fold();
        appendContentToVector(foldedContent, fusedBuffer, index);
    }

    return output;
}

//
// FuseAttr::getStableHashValue
//

llvm::hash_code vpux::Const::FuseAttr::getStableHashValue() const {
    VPUX_THROW("Not implemented. It requires folding of the content, which is expensive");
}
