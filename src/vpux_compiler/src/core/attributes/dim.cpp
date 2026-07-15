//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/dim.hpp"

using namespace vpux;

//
// DimBase
//

void vpux::details::throwNegativeDim(StringRef className, int64_t ind) {
    VPUX_THROW("Got negative index {0} for {1}", ind, className);
}

void vpux::details::throwDimOutOfRange(StringRef className, uint64_t ind) {
    VPUX_THROW("{0} index {1} exceeds maximal supported value {2}", className, ind, MAX_NUM_DIMS);
}

//
// Dim
//

StringRef vpux::Dim::getClassName() {
    return "Dim";
}

//
// MemDim
//

StringRef vpux::MemDim::getClassName() {
    return "MemDim";
}
