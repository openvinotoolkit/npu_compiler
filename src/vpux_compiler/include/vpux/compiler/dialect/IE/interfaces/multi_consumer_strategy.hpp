//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/Operation.h>

namespace vpux::IE {

// Strategy describing how the consumers of a shared op result are handled by the target's MPE engines.
// Some architectures require an operation (e.g. a FakeQuantize) to be preserved for the subset of its consumers that a
// specialized engine can map, while the remaining consumers follow the regular lowering path
class IMultiConsumerStrategy {
public:
    virtual ~IMultiConsumerStrategy() = default;

    // Collects the users of the given op that the target's specialized MPE engine can map. Returns an empty list when
    // the architecture has no such engine or none of the users qualify.
    virtual SmallVector<mlir::Operation*> getMPEEngineMappableConsumers(mlir::Operation* op) const = 0;
};

}  // namespace vpux::IE
