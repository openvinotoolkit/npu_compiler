//
// Copyright © 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/Block.h>

namespace vpux {

struct Factors final {
    int64_t first = 0;
    int64_t second = 0;

    Factors() {
    }
    Factors(int64_t first, int64_t second): first(first), second(second) {
    }
};

SmallVector<Factors> getFactorsList(int64_t n);
SmallVector<Factors> getFactorsListWithMaxLimit(int64_t n, int64_t limit);
SmallVector<Factors> getFactorsListWithMinLimit(int64_t n, int64_t limit);
SmallVector<int64_t> getPrimeFactors(int64_t n);
int64_t smallestDivisor(int64_t n);
}  // namespace vpux
