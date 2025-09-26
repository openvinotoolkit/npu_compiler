//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/factories/convert_IE_to_VPU_NCE_strategy_getter.hpp"
#include "vpux/compiler/NPU37XX/conversion/passes/convert_IE_to_VPU_NCE_strategy.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

namespace vpux {
std::unique_ptr<IConvertIEToVPUNCEStrategy> createConvertIEToVPUNCEStrategy(mlir::func::FuncOp funcOp, Logger log) {
    auto module = funcOp->getParentOfType<mlir::ModuleOp>();
    const auto arch = config::getArch(module);
    switch (arch) {
    case config::ArchKind::NPU37XX:
    case config::ArchKind::NPU40XX:
        return std::make_unique<arch37xx::ConvertIEToVPUNCEStrategy>(log, arch);
    default: {
    }
    }
    VPUX_THROW("Unable to get ConvertIEToVPUNCEStrategy for arch {0}", arch);
}
}  // namespace vpux
