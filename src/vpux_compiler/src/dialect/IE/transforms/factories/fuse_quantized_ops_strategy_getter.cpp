//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/factories/fuse_quantized_ops_strategy_getter.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/impl/fuse_quantized_ops_strategy.hpp"
#include "vpux/compiler/NPU50XX/dialect/IE/impl/fuse_quantized_ops_strategy.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

namespace vpux::IE {

std::unique_ptr<IGreedilyPassStrategy> createFuseQuantizedOpsStrategy(mlir::func::FuncOp funcOp,
                                                                      const bool seOpsEnabled,
                                                                      const bool seExperimentalOpsEnabled) {
    const auto arch = config::getArch(funcOp);
    switch (arch) {
    case config::ArchKind::NPU37XX:
    case config::ArchKind::NPU40XX: {
        return std::make_unique<arch37xx::FuseQuantizedOpsStrategy>(seOpsEnabled, seExperimentalOpsEnabled);
    }
    default: {
        return std::make_unique<arch50xx::FuseQuantizedOpsStrategy>(seOpsEnabled, seExperimentalOpsEnabled);
    }
    }
    VPUX_THROW("Unable to get FuseQuantizedOpsStrategy for arch {0}", arch);
}

}  // namespace vpux::IE
