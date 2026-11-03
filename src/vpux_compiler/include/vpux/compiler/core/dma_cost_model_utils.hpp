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
#include <optional>
#include <utility>
#include <vector>

namespace VPUNN {
class VPUCostModel;
enum class VPUDevice;
enum class MemoryLocation;
struct VPUDMATensor;
struct VPUDMADescriptor;
}  // namespace VPUNN

namespace vpux {
bool isDMACostModelAccurate(config::ArchKind arch);
// Default answer to "should this platform use the strided DMA cost interface?".
// True  -> costs are queried with a VPUNN DMA descriptor, which carries shape, byte strides and layout,
//          so a strided buffer is priced differently from a compact one.
// False -> costs are queried with the contiguous interface, which only knows the shape and therefore
//          prices every buffer as if it were packed.
// This is only the default: the getDMACost entry points below take a `useStrided` override.
bool usesStrideAwareDMACostModel(config::ArchKind arch);
double getStrideDMACorrectionThresholdByArch(config::ArchKind arch);
// Heuristic that inflates the cost of a strided DMA, needed only because the contiguous interface cannot see
// striding. It is therefore skipped when the strided interface is in use, since that one already prices it.
// NOTE: avoid applying this correction on platforms that use the new stride-aware VPUNN interface once that
// interface is adopted for them but is not yet supported/accurate there; the correction would double-count
// the striding cost already priced by that interface.
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

VPUNN::VPUDMATensor getVPUDMATensor(vpux::NDTypeInterface type, VPUNN::VPUDevice vpuDevice);
VPUNN::VPUDMATensor getVPUDMATensor(ShapeRef shape, StridesRef strides, const DimsOrder& dimsOrder, mlir::Type elemType,
                                    VPUNN::VPUDevice vpuDevice);
VPUNN::VPUDMADescriptor buildDMADescriptor(VPUNN::VPUDevice device, const VPUNN::VPUDMATensor& src,
                                           const VPUNN::VPUDMATensor& dst, VPUNN::MemoryLocation srcLoc,
                                           VPUNN::MemoryLocation dstLoc);
// VPUNN DMA Costs utils
// VPU overloads

// Direction of a DMA transfer, used to derive the VPUNN source/destination memory locations when
// they cannot be deduced from the operand types (e.g. a spill, where both sides are the same buffer)
enum class DmaDirectionType { CMX_TO_DDR, DDR_TO_CMX, CMX_TO_CMX, DDR_TO_DDR };

// `useStrided` picks which VPUNN interface prices the transfer:
//   - std::nullopt (the default): follow usesStrideAwareDMACostModel(archKind), i.e. whatever the platform
//     was validated with. Callers that do not care should keep this.
//   - true: always use the strided descriptor interface, even on a platform whose default is the flat one.
//     This is the opt-in knob for experimenting with strided costs on such a platform, e.g. from a pass
//     that wants a striding-sensitive comparison, from a study, or from a unit test. VPUNN may not have
//     been tuned for that platform, so treat the absolute numbers as indicative and only compare costs
//     obtained the same way.
//   - false: always use the contiguous interface, even on a platform whose default is the strided one. Useful
//     to reproduce previous numbers or to isolate a regression to the interface change.
// The choice must stay the same for all costs that are compared against each other: the two interfaces are
// separate models and their absolute values are not interchangeable.
size_t getDMACost(vpux::NDTypeInterface inTensorType, vpux::NDTypeInterface outTensorType, config::ArchKind archKind,
                  VPUNN::VPUDevice vpuDevice, const std::shared_ptr<VPUNN::VPUCostModel>& costModel,
                  int64_t numDMAPorts, std::optional<bool> useStrided = std::nullopt);
size_t getDMACost(vpux::NDTypeInterface tensorType, config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                  const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts,
                  vpux::DmaDirectionType dmaDirection, std::optional<bool> useStrided = std::nullopt);
// VPUIP overload
size_t getDMACost(mlir::Value input, mlir::Value output, config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                  const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts,
                  std::optional<bool> useStrided = std::nullopt);
// VPUIP overload with explicit DMA direction. Forces the VPUNN in/out memory locations from the
// direction instead of deriving both from the single buffer type
size_t getDMACost(mlir::Value input, config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                  const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts,
                  vpux::DmaDirectionType dmaDirection, std::optional<bool> useStrided = std::nullopt);

// DMA Analytical cost utils (for archs for which VPUNN DMA cost model is not accurate)
double getAnalyticalDMACost(vpux::NDTypeInterface, const VPU::DistributionInfo& distribution, double ddrLatency,
                            double ddrBandwidth, int64_t numDMAPorts);

}  // namespace vpux
