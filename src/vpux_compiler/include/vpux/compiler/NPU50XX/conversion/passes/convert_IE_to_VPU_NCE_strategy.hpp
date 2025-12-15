//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/conversion/factories/convert_IE_to_VPU_NCE_strategy_getter.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

namespace vpux::arch50xx {
class ConvertIEToVPUNCEStrategy final : public IConvertIEToVPUNCEStrategy {
public:
    ConvertIEToVPUNCEStrategy(const Logger& log, const config::ArchKind arch): IConvertIEToVPUNCEStrategy(log, arch) {
    }

    void addTargets(mlir::ConversionTarget& target, LogCb logCb) const override;
    void addPatterns(mlir::RewritePatternSet& patterns) const override;
};
}  // namespace vpux::arch50xx
