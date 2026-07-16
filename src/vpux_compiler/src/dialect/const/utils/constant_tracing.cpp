//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/utils/constant_tracing.hpp"

namespace vpux::Const {

bool TraceId::operator==(TraceId other) const {
    return _ptr == other._ptr;
}

bool TraceId::operator!=(TraceId other) const {
    return !(*this == other);
}

bool TraceId::operator<(TraceId other) const {
    return _ptr < other._ptr;
}

TraceId::operator bool() const {
    return _ptr != nullptr;
}

bool TraceId::operator!() const {
    return _ptr == nullptr;
}

}  // namespace vpux::Const
