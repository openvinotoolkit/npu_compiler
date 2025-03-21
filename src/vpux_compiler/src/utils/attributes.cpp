//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/IR/DialectResourceBlobManager.h>

namespace vpux {

mlir::StringRef getResourceName(mlir::DenseResourceElementsAttr attr) {
    auto key = attr.getRawHandle().getKey();
    static_assert(std::is_same_v<mlir::StringRef, decltype(key)>,
                  "Returning StringRef is possible because dense_resource<> returns it directly");
    return key;
}

mlir::StringRef getResourceName(mlir::ElementsAttr attr) {
    if (auto denseResource = mlir::dyn_cast<mlir::DenseResourceElementsAttr>(attr); denseResource != nullptr) {
        auto key = getResourceName(denseResource);
        static_assert(std::is_same_v<mlir::StringRef, decltype(key)>,
                      "Cannot return StringRef if the underlying getResourceName() doesn't return it - potential "
                      "dangling reference otherwise");
        return key;
    }
    return {};
}

}  // namespace vpux
