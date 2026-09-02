//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/scheduling/utils.hpp"

#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <algorithm>

vpux::AddressType vpux::getRawBufferSize(mlir::Value buf) {
    return static_cast<vpux::AddressType>(vpux::getTotalSize(buf).count());
}

vpux::AddressType vpux::getPackedPersistentSize(llvm::ArrayRef<size_t> order,
                                                llvm::ArrayRef<PersistentFitBufferInfo> buffers) {
    vpux::AddressType offset = 0;
    for (size_t idx : order) {
        const auto& bufferInfo = buffers[idx];
        offset = vpux::alignValUp(offset, bufferInfo.alignment);
        offset += bufferInfo.size;
    }
    return offset;
}

bool vpux::haveSameAlignment(llvm::ArrayRef<size_t> order, llvm::ArrayRef<PersistentFitBufferInfo> buffers) {
    if (order.size() <= 1) {
        return true;
    }

    const auto alignment = buffers[order.front()].alignment;
    for (size_t idx : order) {
        if (buffers[idx].alignment != alignment) {
            return false;
        }
    }
    return true;
}

bool vpux::shouldUsePersistentFitSearch(llvm::ArrayRef<size_t> order, llvm::ArrayRef<PersistentFitBufferInfo> buffers) {
    return order.size() <= MAX_PERSISTENT_FIT_ORDER_SEARCH && !haveSameAlignment(order, buffers);
}

llvm::SmallVector<size_t> vpux::getWorstCasePersistentFitOrder(llvm::ArrayRef<size_t> order,
                                                               llvm::ArrayRef<PersistentFitBufferInfo> buffers) {
    llvm::SmallVector<size_t> candidateOrder(order.begin(), order.end());
    std::sort(candidateOrder.begin(), candidateOrder.end());

    llvm::SmallVector<size_t> worstCaseOrder = candidateOrder;
    auto worstCaseSize = getPackedPersistentSize(candidateOrder, buffers);
    while (std::next_permutation(candidateOrder.begin(), candidateOrder.end())) {
        const auto packedSize = getPackedPersistentSize(candidateOrder, buffers);
        if (packedSize > worstCaseSize) {
            worstCaseSize = packedSize;
            worstCaseOrder = candidateOrder;
        }
    }
    return worstCaseOrder;
}

vpux::AddressType vpux::saturatingSub(vpux::AddressType lhs, vpux::AddressType rhs) {
    return lhs > rhs ? (lhs - rhs) : 0;
}
