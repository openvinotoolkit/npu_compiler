//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/small_vector.hpp"

namespace vpux {

template <class T>
void broadcast(SmallVectorImpl<T>& values, size_t size) {
    VPUX_THROW_UNLESS(values.size() <= size, "Cannot broadcast to size {0}", size);

    if (values.size() == size) {
        return;
    }

    VPUX_THROW_UNLESS(!values.empty(), "Cannot broadcast an empty vector to size {0}", size);
    VPUX_THROW_UNLESS(size % values.size() == 0,
                      "Cannot broadcast {0} values to size {1}: {1} is not a multiple of {0}", values.size(), size);

    const size_t replicationFactor = size / values.size();
    const SmallVector<T> originalValues(values.begin(), values.end());
    values.reserve(size);
    for (size_t i = 1; i < replicationFactor; ++i) {
        values.append(originalValues.begin(), originalValues.end());
    }
}

}  // namespace vpux
