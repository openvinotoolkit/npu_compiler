//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/utils/dma_transaction_utils.hpp"

#include <mlir/IR/Operation.h>

namespace vpux::VPUIP {

// Assigned during unrolling passes to keep track of which DMAs belong to the same group.
// Used by DMA port balancing passes later on.
constexpr auto UNROLL_IDX = "unrollIdx";

int64_t getDMAPortValue(mlir::Operation* wrappedTaskOp);

std::string getDMAChannelTypeAsString(VPUIP::DmaChannelType channelType, config::ArchKind arch);
std::string getDMAChannelTypeAsString(int64_t dmaQueueIdEncoding, config::ArchKind arch);

DMATransaction getDMATransactionFromExpand(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType,
                                           mlir::ArrayAttr padsBegin, mlir::ArrayAttr padsEnd, bool stridedInput,
                                           bool stridedOutput);
DMATransaction getDMATransactionFromPermutation(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType,
                                                mlir::AffineMap mappingOrder, mlir::AffineMap loopOrder,
                                                bool stridedInput, bool stridedOutput);
DMATransaction getDMATransactionFromPermutation(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType,
                                                mlir::AffineMap mappingOrder, mlir::SmallVector<int64_t> loopOrder,
                                                bool stridedInput, bool stridedOutput);

}  // namespace vpux::VPUIP
