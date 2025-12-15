//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/VPU/IR/ops_interfaces.hpp"

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"

#include "vpux/compiler/dialect/VPU/interfaces/cost_model_utils.hpp"

namespace vpux {
namespace VPU {
namespace arch50xx {
class ICostModelUtilsInterface final : public vpux::VPU::ICostModelUtilsInterface {
public:
    ICostModelUtilsInterface(mlir::Dialect* dialect): vpux::VPU::ICostModelUtilsInterface(dialect) {
    }

    bool isNCEWithInt4WeightsSupported() const override {
        return true;
    }

    bool isNNCacheStatisticsSupported() const override {
        return true;
    }

    bool isMultiDimPipelineTilingSupported() const override {
        return true;
    }
};
}  // namespace arch50xx
}  // namespace VPU
}  // namespace vpux

//
// setupExtraInterfaces
//

void vpux::VPU::arch50xx::registerICostModelUtilsInterface(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext*, vpux::VPU::VPUDialect* dialect) {
        dialect->addInterfaces<vpux::VPU::arch50xx::ICostModelUtilsInterface>();
    });
}
