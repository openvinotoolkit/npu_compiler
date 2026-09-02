//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/native_attributes/distribution_info.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_fwd.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace VPUNN {
class VPUCostModel;
enum class VPUDevice;
enum class MemoryLocation;
}  // namespace VPUNN

namespace vpux {
bool isDMACostModelAccurate(config::ArchKind arch);
bool isDMAPortSplittingSupported(config::ArchKind arch);
double getStrideDMACorrectionThresholdByArch(config::ArchKind arch);
bool applyStrideDMACorrectionForTile(vpux::NDTypeInterface tileType, bool isStridedDMA, uint32_t& cost,
                                     config::ArchKind arch, bool isFullSearchVersion = false);
bool correctStrideDMACostOnAllTiles(
        ArrayRef<std::vector<std::pair<vpux::NDTypeInterface, llvm::DenseMap<mlir::Type, VPU::DistributionInfo>>>>
                tilesTypes,
        const std::function<vpux::NDTypeInterface(
                ArrayRef<std::pair<vpux::NDTypeInterface, llvm::DenseMap<mlir::Type, VPU::DistributionInfo>>>)>&
                tileTypeGetter,
        SmallVector<uint32_t>& dmaCost, bool isStridedDMA, config::ArchKind arch);

VPUNN::MemoryLocation getMemoryLocation(mlir::Type type);
// VPUNN DMA Costs utils
// VPU overloads
size_t getDMACost(vpux::NDTypeInterface inTensorType, vpux::NDTypeInterface outTensorType, config::ArchKind archKind,
                  VPUNN::VPUDevice vpuDevice, const std::shared_ptr<VPUNN::VPUCostModel>& costModel,
                  int64_t numDMAPorts);
size_t getDMACost(vpux::NDTypeInterface tensorType, config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                  const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts);
// VPUIP overload
size_t getDMACost(mlir::Value input, mlir::Value output, config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                  const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts);

// DMA Analytical cost utils (for archs for which VPUNN DMA cost model is not accurate)
double getAnalyticalDMACost(vpux::NDTypeInterface, const VPU::DistributionInfo& distribution, double ddrLatency,
                            double ddrBandwidth, int64_t numDMAPorts);

}  // namespace vpux
