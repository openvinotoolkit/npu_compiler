//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/native_attributes/distribution_info.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/utils/clustered_op_interface_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/ValueRange.h>

#include <initializer_list>
#include <numeric>

namespace vpux {
namespace VPU {

void registerAlignedChannelsOpInterfacesVPU(mlir::DialectRegistry& registry);

//
// SparseOpInterface
//

bool supportsSparseInputs(mlir::Operation* op);
bool supportsSparseOutputs(mlir::Operation* op);
bool supportsSparseData(mlir::Operation* op);
bool supportsSparseWeights(mlir::Operation* op);

//
// NCEOpInterface
//

template <typename ConcreteOp>
void setLayerMultiClusterStrategy(ConcreteOp mainOp, VPU::MultiClusterStrategy strategy) {
    const auto multiClusterStrategyAttr = VPU::MultiClusterStrategyAttr::get(mainOp->getContext(), strategy);
    mainOp.setMultiClusterStrategyAttr(multiClusterStrategyAttr);
}

int64_t getOptimalNumClusters(mlir::Operation* operation, ShapeRef outputShape, VPU::MultiClusterStrategy strategy);

namespace details {

mlir::LogicalResult validatePrecisionForNCE(mlir::Operation* op);
mlir::LogicalResult validateWorkloadsRegion(mlir::Location loc, mlir::Region& workloads);

mlir::Operation* addWorkload(mlir::Region& workloads, mlir::OpBuilder& builder, mlir::Location loc, ShapeRef offsets,
                             ShapeRef sizes, PaddingAttr pad, MPEMode mpeMode, mlir::IntegerAttr clusterId);

mlir::LogicalResult verifyInputTypeOp(mlir::Operation* op, vpux::NDTypeInterface inputType);
mlir::LogicalResult verifyInputQuantization(mlir::Operation* op);

}  // namespace details

//
// LayerOpInterface
//

mlir::LogicalResult verifyLayer(mlir::Operation* op);

//
// TilingBuilderOpInterface
//

mlir::Value makeTile(mlir::OpBuilder& builder, mlir::Location baseLoc, mlir::Value origVal, const TileInfo& tile,
                     StringRef valName);

//
// TilingInfoOpInterface
//

mlir::LogicalResult verifyTilingInfo(mlir::Operation* op);

//
// EltwiseOp
//

mlir::LogicalResult verifyEltwiseOp(mlir::Operation* op);

template <typename ConcreteOp>
class EltwiseOp : public mlir::OpTrait::TraitBase<ConcreteOp, EltwiseOp> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return VPU::verifyEltwiseOp(op);
    }

    InputTiling backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
        return backInferEltwiseTile(this->getOperation(), outputTile);
    }

    void adjustAttrs(const TilingInfo&, const TileInfo&) {
        // Do nothing
    }

    mlir::FailureOr<OutputTiling> getTilingStrategy(TilingMode tilingMode, Logger log) {
        return getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
    }
};

//
// NCEOpInterface
//

mlir::LogicalResult verifyNCEOp(mlir::Operation* op);

//
// ClusteredOpInterface
//

vpux::NDTypeInterface getDistributedTypeForOpOperand(mlir::Operation* op, mlir::OpOperand& operand,
                                                     bool hasExplicitDistributedAttr,
                                                     SiblingOpsAnalysis& siblingsAnalysis);

vpux::NDTypeInterface getDistributedTypeForOpResult(mlir::Operation* op, mlir::Value result,
                                                    VPU::MultiClusterStrategy strategy,
                                                    SiblingOpsAnalysis& siblingsAnalysis,
                                                    bool hasExplicitDistributedAttr);

//
// isPureViewOp
//

bool isPureViewOp(mlir::Operation* op);

//
// SwOpInterface
//

bool supportSwOpLoweringAsDMA(mlir::Operation* op);

//
// TilingInfoOpInterface for SW
//

void registerSWTilingInfoOpInterfaceCommon(mlir::DialectRegistry& registry);

}  // namespace VPU
}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/VPU/ops_interfaces.hpp.inc>
