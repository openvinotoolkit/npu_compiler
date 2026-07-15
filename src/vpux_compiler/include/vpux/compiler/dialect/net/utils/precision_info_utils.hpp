//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/utils/logger/logger.hpp"

namespace vpux::net {

/** @brief Utility class for identifying precision sensitive operations.

    This utility class is able to find those operations marked to be kept in FP32.
    Marking is done by extractPrecisionInfo() function.
 */
class PrecisionSensitiveOps {
public:
    PrecisionSensitiveOps(mlir::ModuleOp module, vpux::Logger log);
    bool isPrecisionSensitiveOp(mlir::Operation* op) const;

private:
    vpux::Logger _logger;
    std::multimap<std::string, net::PrecisionInfoOp, std::less<>> _lookup;
};

}  // namespace vpux::net
