//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux {
class IConvertIEToVPUNCEStrategy {
public:
    IConvertIEToVPUNCEStrategy(const Logger& log, const config::ArchKind arch): _log(log), _arch(arch) {
    }

    virtual void addTargets(mlir::ConversionTarget& target, LogCb logCb) const = 0;
    virtual void addPatterns(mlir::RewritePatternSet& patterns) const = 0;

    virtual ~IConvertIEToVPUNCEStrategy() = default;

protected:
    Logger _log;
    config::ArchKind _arch;
};

std::unique_ptr<IConvertIEToVPUNCEStrategy> createConvertIEToVPUNCEStrategy(mlir::func::FuncOp funcOp, Logger log);
}  // namespace vpux
