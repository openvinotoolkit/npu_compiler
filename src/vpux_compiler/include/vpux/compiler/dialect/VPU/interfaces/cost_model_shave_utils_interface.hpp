//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>

#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace VPU {
/// alias for the map type used for translating operation names to SHAVE implementation names
using MapShaveNamesToVPUNN = std::map<std::string, std::string>;

/**
 * @brief IShaveCostModelUtils is an interface that defines the contract for SHAVE cost model utilities.
 * It provides methods to:
 * Retrieve the mapping of software kernel names
 * Check if a specific software kernel is supported
 *
 * Used to expose only the required methods to the CostModelConfig class
 */
class IShaveCostModelUtils {
public:
    virtual ~IShaveCostModelUtils() = default;

    // retrieve the mapping of the transformation from VPUx to VPUNN of SW kernel names
    virtual const MapShaveNamesToVPUNN& getSwKernelContainer() const = 0;

    // retrieve if a kernel is supported in the current mapping
    virtual bool isSwKernelOpSupported(const std::string& swKernelName) const = 0;
};
}  // namespace VPU
}  // namespace vpux
