//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/schedule_builder_utils.hpp"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

namespace vpux {

constexpr size_t MAX_PERSISTENT_FIT_ORDER_SEARCH = 3;

// Buffer groups used by VF scheduling.
//
// PERSISTENT_CANDIDATE:
// Buffer reused across loop iterations (same value every time).
// Keeping it in CMX can avoid reloading it each iteration.
// This classifier only marks it as a candidate. Final keep/evict is decided later.
//
// TEMPORARY:
// Buffer used only inside one iteration.
// It is treated as short-lived and can be reused after that iteration.
enum class BufferCategory : uint8_t {
    PERSISTENT_CANDIDATE,
    TEMPORARY,
};

enum class VFPrefetchHint : uint8_t {
    FULL_PREFETCHING,
    LASTOP_PREFETCHING,
    WEIGHTS_PREFETCHING,
    MINIMAL,
};

struct PersistentFitBufferInfo {
    AddressType size = 0;
    AddressType alignment = 1;
};

AddressType getRawBufferSize(mlir::Value buf);

AddressType getPackedPersistentSize(ArrayRef<size_t> order, ArrayRef<PersistentFitBufferInfo> buffers);

bool haveSameAlignment(ArrayRef<size_t> order, ArrayRef<PersistentFitBufferInfo> buffers);

bool shouldUsePersistentFitSearch(ArrayRef<size_t> order, ArrayRef<PersistentFitBufferInfo> buffers);

SmallVector<size_t> getWorstCasePersistentFitOrder(ArrayRef<size_t> order, ArrayRef<PersistentFitBufferInfo> buffers);

AddressType saturatingSub(AddressType lhs, AddressType rhs);

}  // namespace vpux
