//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/sw_tiling_interface_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"

using namespace vpux;

namespace {

//
// SwLayerTilingInfoOpModel50XX
//

template <class MainOpType>
class SwLayerPipeliningTilingSupportedInfoOpModel final :
        public SwLayerTilingInfoOpModelBase<SwLayerPipeliningTilingSupportedInfoOpModel<MainOpType>, MainOpType> {
public:
    bool isPipeliningTilingSupported(mlir::Operation* /*origOp*/) const {
        return true;
    }
};

}  // namespace

//
// setupExtraInterfaces
//

void vpux::VPU::arch50xx::registerSWTilingInfoOpInterface(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*) {
        // Register pipelining-tiling-supported SW ops for NPU50XX
        VPU::MVN1NormalizeOp::attachInterface<SwLayerPipeliningTilingSupportedInfoOpModel<VPU::MVN1NormalizeOp>>(*ctx);
        VPU::RoPEOp::attachInterface<SwLayerPipeliningTilingSupportedInfoOpModel<VPU::RoPEOp>>(*ctx);
        VPU::CumSumOp::attachInterface<SwLayerPipeliningTilingSupportedInfoOpModel<VPU::CumSumOp>>(*ctx);
        VPU::MultiplyOp::attachInterface<SwLayerPipeliningTilingSupportedInfoOpModel<VPU::MultiplyOp>>(*ctx);
    });
    // Register common interface if the op is tiling-supported but doesn't have TilingInfoOpInterface yet
    vpux::VPU::registerSWTilingInfoOpInterfaceCommon(registry);
}
