//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/memref_attr_utils.hpp"
#include "vpux/compiler/core/attributes/dims_order.hpp"

#include <mlir/IR/BuiltinAttributes.h>

using namespace vpux;

DimsOrder vpux::inferNewDimsOrder(const DimsOrder& origOrder, size_t numShapeDims) {
    if (origOrder.isIdentity()) {
        return DimsOrder::fromNumDims(numShapeDims);
    }
    if (numShapeDims == origOrder.numDims()) {
        return origOrder;
    }
    // We can infer new dims order for DMA fusion. In this case we just add new dim and increase mappings by one
    VPUX_THROW_WHEN(numShapeDims != origOrder.numDims() + 1, "Can't infer new dims order");
    VPUX_THROW_WHEN(numShapeDims >= vpux::MAX_NUM_DIMS, "Can't expand dims");
    // Create codeDelta, which is just 0x11111(based on num Dims). To get new DimOrder add this code to original
    // For example NHWC - 0x1342, new order will be 0x12453(GNHWC)
    DimsOrder::StorageType codeDelta = 0;
    for (size_t dim = 0; dim < numShapeDims; ++dim) {
        codeDelta = (codeDelta << DimsOrder::BITS_PER_DIM) | 1ul;
    }
    DimsOrder::StorageType newCode = origOrder.code() + codeDelta;
    return DimsOrder::fromCode(newCode);
}

bool vpux::hasStridedLayout(mlir::Type type) {
    if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type)) {
        return mlir::isa<mlir::StridedLayoutAttr>(memrefType.getLayout());
    }

    return false;
}
