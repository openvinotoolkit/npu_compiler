//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <mutex>
#include <string>
#include <vector>

#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace VPU {

/** @brief Class that manages SHAVE cost model utilities
 *  Stores the list of supported SHAVE operations retrieved from VPUCostModel.
 */
class CostModelShaveUtil {
public:
    // Check if a kernel is supported
    bool isSwKernelOpSupported(const std::string& swKernelName) const {
        const auto& supportedOps = getCachedSupportedOperations();
        return llvm::is_contained(supportedOps, swKernelName);
    }

    // Check if Shave2 API is used
    bool isShave2ApiUsed() const {
        return _isShave2ApiUsedInVPUNN;
    }

    CostModelShaveUtil(bool isShave2ApiUsed, const std::vector<std::string>& supportedOperations)
            : _isShave2ApiUsedInVPUNN(isShave2ApiUsed), _supportedOperations(supportedOperations) {
    }

    CostModelShaveUtil(bool isShave2ApiUsed, std::function<std::vector<std::string>()> lazySupportedOperations)
            : _isShave2ApiUsedInVPUNN(isShave2ApiUsed), _lazySupportedOperations(std::move(lazySupportedOperations)) {
    }

private:
    const std::vector<std::string>& getCachedSupportedOperations() const {
        std::call_once(_initFlag, [this]() {
            if (!_supportedOperations.empty()) {
                return;
            }
            VPUX_THROW_WHEN(_lazySupportedOperations == nullptr,
                            "Supported operations loader not initialized. Cannot retrieve supported operations.");
            _supportedOperations = _lazySupportedOperations();
        });
        return _supportedOperations;
    }

private:
    bool _isShave2ApiUsedInVPUNN;
    std::function<std::vector<std::string>()> _lazySupportedOperations;
    mutable std::once_flag _initFlag;
    mutable std::vector<std::string> _supportedOperations;
};
}  // namespace VPU
}  // namespace vpux
