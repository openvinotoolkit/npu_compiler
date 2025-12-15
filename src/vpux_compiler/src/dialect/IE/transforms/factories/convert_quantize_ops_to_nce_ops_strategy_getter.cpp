//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/factories/convert_quantize_ops_to_nce_ops_strategy_getter.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/impl/convert_quantize_ops_to_nce_ops_strategy.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

namespace vpux::IE {

std::unique_ptr<IConvertQuantizeOpsToNceOpsStrategy> createConvertQuantizeOpsToNceOpsStrategy(
        mlir::func::FuncOp funcOp) {
    const auto arch = config::getArch(funcOp);
    switch (arch) {
    case config::ArchKind::NPU37XX:
    case config::ArchKind::NPU40XX:
    case config::ArchKind::NPU50XX:
        return std::make_unique<IE::arch37xx::ConvertQuantizeOpsToNceOpsStrategy>();

    default:
        VPUX_THROW("Unsupported architecture in createConvertQuantizeOpsToNceOpsStrategy: {0}", arch);
    }
}
}  // namespace vpux::IE
