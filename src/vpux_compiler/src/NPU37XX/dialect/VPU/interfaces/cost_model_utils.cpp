//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/IR/ops_interfaces.hpp"

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/cost_model_utils.hpp"

namespace vpux {
namespace VPU {
namespace arch37xx {
class ICostModelUtilsInterface final : public vpux::VPU::ICostModelUtilsInterface {
public:
    ICostModelUtilsInterface(mlir::Dialect* dialect): vpux::VPU::ICostModelUtilsInterface(dialect) {
    }

    bool isNCEWithInt4WeightsSupported() const override {
        return false;
    }

    bool isNNCacheStatisticsSupported() const override {
        return false;
    }

    bool isMultiDimPipelineTilingSupported() const override {
        return false;
    }
};
}  // namespace arch37xx
}  // namespace VPU
}  // namespace vpux

//
// setupExtraInterfaces
//

void vpux::VPU::arch37xx::registerICostModelUtilsInterface(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext*, vpux::VPU::VPUDialect* dialect) {
        dialect->addInterfaces<vpux::VPU::arch37xx::ICostModelUtilsInterface>();
    });
}
