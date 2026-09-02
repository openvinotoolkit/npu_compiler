//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>

#include <memory>

namespace vpux::IE {
#define GEN_PASS_DECL_FORBIDFOURBITOUTPUTS
#define GEN_PASS_DEF_FORBIDFOURBITOUTPUTS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

class ForbidFourBitOutputsPass final : public IE::impl::ForbidFourBitOutputsBase<ForbidFourBitOutputsPass> {
public:
    explicit ForbidFourBitOutputsPass(const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final {
        auto moduleOp = getOperation();
        const auto archKind = config::getArch(moduleOp);
        // 4-bit outputs are supported on NPU40XX+
        if (archKind != config::ArchKind::NPU37XX) {
            return;
        }
        auto netInfo = net::getNetworkInfo(moduleOp);
        auto outputsInfo = netInfo.getOutputsDataInfo();
        for (auto outputInfo : outputsInfo) {
            auto elemType = mlir::cast<NDTypeInterface>(outputInfo.getUserType()).getElementType();
            if (auto quantType = mlir::dyn_cast<mlir::quant::QuantizedType>(elemType)) {
                elemType = quantType.getStorageType();
            }
            if (elemType.isInteger(4)) {
                outputInfo.emitError("Network has 4-bit output, which is not yet supported");
                signalPassFailure();
                return;
            }
        }
    }
};

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createForbidFourBitOutputsPass(Logger log) {
    return std::make_unique<ForbidFourBitOutputsPass>(log);
}
