//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"

namespace vpux::VPU {
/* @brief
 * Static class for generating MPEEngine attributes.
 */
class MPEEngineConfig {
public:
    static MPEEngineAttr retrieveMPEEngineAttribute(mlir::Operation* operation) {
        const auto arch = config::getArch(operation);

        VPUX_THROW_WHEN(arch == config::ArchKind::UNKNOWN,
                        "An unknown architecture is associated to the provided operation");

        return MPEEngine37XXAttr::get(operation->getContext(),
                                      MPEEngine37XXModeAttr::get(operation->getContext(), MPEEngine37XXMode::SCL));
    }

    template <typename ConcreteOp>
    static MPEEngineAttr retrieveMPEEngineAttribute(ConcreteOp operation, bool) {
        static_assert(std::is_same_v<ConcreteOp, IE::ConvolutionOp>, "Invalid operation, expected IE::ConvolutionOp");

        return retrieveMPEEngineAttribute(operation);
    }

    static bool useNewWeightTableFormat(mlir::Operation*, bool) {
        return false;
    }

    static bool isNewWeightTableFormatSupportedWithDwOps([[maybe_unused]] config::ArchKind arch) {
        return false;
    }
};

}  // namespace vpux::VPU
