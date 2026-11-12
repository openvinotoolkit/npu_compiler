//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant_odu.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/attributes.hpp"

namespace vpux::VPUIPDPU::arch60xx::ODU {

struct ODUConfig : public arch40xx::ODU::ODUConfig {
    struct ReduceLayer {
        bool reduceMax = false;
        bool reduceMin = false;
        bool reduceTotal = false;
        ODUReduceDataType reduceDataType;
        std::optional<int64_t> tileSelect;
    } reduceLayer;
    struct LocalRegion {
        int64_t beginX = 0;
        int64_t beginY = 0;
        int64_t endX = 0;
        int64_t endY = 0;
        bool suppressOutput = false;
        bool localRegionEnable = false;
    } localRegion;
};

int64_t getTileSelectMaskForReduceTotal(ELF::SymbolReferenceMap& symRefMap,
                                        std::optional<mlir::ArrayAttr> minMaxArrayAttr, const NDTypeInterface& outType);

mlir::LogicalResult configureReduceOutputLayer(const Logger& log,
                                               VPUIPDPU::arch60xx::ODU::ODUConfig::ReduceLayer& config,
                                               bool doReduceMaxPerXy, bool doReduceMinPerXy,
                                               bool doReduceMinMaxPerTensor, int64_t tileSelect,
                                               const NDTypeInterface& outActType);

mlir::LogicalResult configureLocalRegion(const Logger& log, VPUIPDPU::arch60xx::ODU::ODUConfig::LocalRegion& config,
                                         std::optional<vpux::VPUIP::LocalRegionAttr> localRegionAttr);

mlir::LogicalResult configureDataReuse(const Logger& log, ODUConfig::DataReuse& config, VPU::MPEMode& mpeFrequentMode,
                                       VPUIP::NCETaskType dpuTaskType, bool enableStencilSelection,
                                       bool allowExtension);

mlir::LogicalResult configureODU(const Logger& log, ODUConfig& config, const NDTypeInterface& outActType,
                                 VPU::MPEMode& mpeFrequentMode, VPUIP::NCETaskType dpuTaskType,
                                 std::optional<int64_t> outSwizzling, std::optional<bool> isSuperdense,
                                 bool outSparsityEnabled, int64_t minMaxTileSelect, bool doReduceMaxPerXy,
                                 bool doReduceMinPerXy, bool doReduceMinMaxPerTensor,
                                 const NDTypeInterface& minMaxDataType,
                                 std::optional<vpux::VPUIP::LocalRegionAttr> localRegionAttr,
                                 bool enableStencilSelection = true, bool allowExtension = false);

mlir::LogicalResult buildODUConfig(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                   const ODUConfig& config, mlir::Value outAct, mlir::Value outSparsityMap);

}  // namespace vpux::VPUIPDPU::arch60xx::ODU
