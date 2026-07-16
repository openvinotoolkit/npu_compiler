//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include <mlir/IR/Operation.h>

#include <cstdint>
#include <optional>

namespace vpux::IE {

class IConvertToConvAlignmentStrategy {
public:
    virtual ~IConvertToConvAlignmentStrategy() = default;

    virtual std::optional<int64_t> getRequiredInputChannelAlignment(
            mlir::Operation* originalOp, vpux::NDTypeInterface prospectiveInputType,
            vpux::NDTypeInterface prospectiveFilterType, vpux::NDTypeInterface prospectiveOutputType) const = 0;
};

}  // namespace vpux::IE
