//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/factories/fuse_outstanding_quant_strategy_getter.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/impl/fuse_outstanding_quant_strategy.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

namespace vpux::IE {

std::unique_ptr<IGreedilyPassStrategy> createFuseOutstandingQuantStrategy(mlir::func::FuncOp funcOp) {
    const auto arch = config::getArch(funcOp);
    switch (arch) {
    case config::ArchKind::NPU37XX:
    case config::ArchKind::NPU40XX:
        return std::make_unique<arch37xx::FuseOutstandingQuantStrategy>();
    default: {
    }
    }
    VPUX_THROW("Unable to get FuseOutstandingQuantStrategy for arch {0}", arch);
}

}  // namespace vpux::IE
