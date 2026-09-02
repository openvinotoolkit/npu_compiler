//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/schedule_builder_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/init/dialects_registry.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>

#include <gtest/gtest.h>

namespace vpux {

// Common test fixture for scheduler loop creation tests (Tiling and VF)
class MLIR_SchedulerLoopCreationTestBase : public testing::TestWithParam<vpux::config::Platform> {
public:
    void SetUp() override;
    mlir::MLIRContext* getCtx();

private:
    mlir::DialectRegistry registry;
    std::unique_ptr<mlir::MLIRContext> ctx;
};

// Common test parameter name formatter
std::string schedulerTestParamName(const testing::TestParamInfo<vpux::config::Platform>& info);

// Debug print helpers
void printComputeRegions(const vpux::ComputeRegionVec& regions, vpux::Logger log);
void printDebugIR(mlir::ModuleOp module, vpux::Logger log);

// Common op-building utilities
VPU::MPEEngineAttr createMPEEngineAttr(mlir::MLIRContext* ctx, vpux::config::Platform platform);

VPUIP::NCEClusterTaskOp createNCEClusterTaskOp(mlir::OpBuilder& builder, mlir::MLIRContext* ctx, mlir::Location loc,
                                               int64_t kernel, int64_t padding, int64_t stride, mlir::Value inputTile,
                                               mlir::Value weightOp, mlir::Value weightTableOp, mlir::Value outputTile,
                                               VPU::MPEEngineAttr mpeEngineAttr);

mlir::Value createWeightsTable(mlir::OpBuilder& builder, mlir::Location loc, int64_t tileC,
                               const vpux::IndexedSymbolAttr& ddrSpace, const vpux::IndexedSymbolAttr& cmxSpace);

mlir::Value createWeights(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type elemType,
                          vpux::ShapeRef weightsShape, const vpux::IndexedSymbolAttr& ddrSpace,
                          const vpux::IndexedSymbolAttr& cmxSpace);

mlir::OwningOpRef<mlir::ModuleOp> createTiledConvolutionModule(mlir::MLIRContext* ctx, int numTilesH, int numTilesC,
                                                               config::Platform platform,
                                                               bool addDdr2DdrConsumers = false);

}  // namespace vpux
