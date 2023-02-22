//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/attributes.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/attributes/structs.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>

#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/TypeSwitch.h>

#include "vpu/performance.h"

using namespace vpux;

//
// Generated
//

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/VPU/generated/attributes.cpp.inc>

//
// Dialect hooks
//

void VPU::VPUDialect::registerAttributes() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include <vpux/compiler/dialect/VPU/generated/attributes.cpp.inc>
            >();
}

//
// Run-time resources
//

namespace {

constexpr StringLiteral derateFactorAttrName = "VPU.derateFactor";
constexpr StringLiteral bandwidthAttrName = "VPU.bandwidth"; /*!< This attribute corresponds to a single JSON field
                      nested at header>resources>memory_bandwidth>number in the deserialized version of the blob.
                      */

}  // namespace

StringLiteral vpux::VPU::getMemoryDerateAttrName() {
    return derateFactorAttrName;
}

StringLiteral vpux::VPU::getMemoryBandwidthAttrName() {
    return bandwidthAttrName;
}

namespace {

constexpr int VPUX30XX_MAX_DPU_GROUPS = 4;
constexpr int VPUX37XX_MAX_DPU_GROUPS = 2;

constexpr int VPUX30XX_MAX_DMA_PORTS = 1;
constexpr int VPUX311X_MAX_DMA_PORTS = 2;
constexpr int VPUX37XX_MAX_DMA_PORTS = 2;

}  // namespace

uint32_t vpux::VPU::getMaxDPUClusterNum(ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::VPUX30XX:
        return VPUX30XX_MAX_DPU_GROUPS;
    case VPU::ArchKind::VPUX311X:
        return VPUX30XX_MAX_DPU_GROUPS;
    case VPU::ArchKind::VPUX37XX:
        return VPUX37XX_MAX_DPU_GROUPS;
    default:
        VPUX_THROW("Unsupported architecture '{0}'", arch);
    }
}

uint32_t vpux::VPU::getMaxDPUClusterNum(mlir::Operation* op) {
    return VPU::getMaxDPUClusterNum(VPU::getArch(op));
}

uint32_t vpux::VPU::getMaxDMAPorts(ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::VPUX30XX:
        return VPUX30XX_MAX_DMA_PORTS;
    case VPU::ArchKind::VPUX311X:
        return VPUX311X_MAX_DMA_PORTS;
    case VPU::ArchKind::VPUX37XX:
        return VPUX37XX_MAX_DMA_PORTS;
    default:
        VPUX_THROW("Unsupported architecture '{0}'", arch);
    }
}

double vpux::VPU::getDMABandwidth(ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::VPUX30XX:
        return 20000.0 / 700;
    case VPU::ArchKind::VPUX311X:
        return 20000.0 / 850;
    case VPU::ArchKind::VPUX37XX:
        return 31200.0f / 1300;
    default:
        VPUX_THROW("Unsupported architecture '{0}'", arch);
    }
}

double vpux::VPU::getNCEThroughput(ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::VPUX30XX:
        return 7168000.0;
    case VPU::ArchKind::VPUX311X:
        return 7168000.0;
    case VPU::ArchKind::VPUX37XX:
        return 8000000.0;
    default:
        VPUX_THROW("Unsupported architecture '{0}'", arch);
    }
}

unsigned int vpux::VPU::getDpuFrequency(vpux::VPU::ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::VPUX30XX:
        return VPUNN::get_dpu_fclk(VPUNN::VPUDevice::VPU_2_0);
    case VPU::ArchKind::VPUX311X:
        return VPUNN::get_dpu_fclk(VPUNN::VPUDevice::VPU_2_1);
    case VPU::ArchKind::VPUX37XX:
        return VPUNN::get_dpu_fclk(VPUNN::VPUDevice::VPU_2_7); /*!< The value 1300 corresponds to Highvcc of dpuclk.
                (See VPUX37XX HAS #voltage-and-frequency-targets section).
                 */
    default:
        VPUX_THROW("Unsupported architecture '{0}'", arch);
    }
}

double vpux::VPU::getDmaBandwidthGBps(mlir::ModuleOp module) {
    const ArchKind arch = getArch(module);
    return getDmaBandwidthGBps(arch);
}

double vpux::VPU::getDmaBandwidthGBps(vpux::VPU::ArchKind arch) {
    double BW = 0;
    switch (arch) {
    case VPU::ArchKind::VPUX30XX:
        BW = VPUNN::get_dram_bandwidth_MBps(VPUNN::VPUDevice::VPU_2_0);  // 20000 MB/s
        break;
    case VPU::ArchKind::VPUX311X:
        BW = VPUNN::get_dram_bandwidth_MBps(VPUNN::VPUDevice::VPU_2_1);  // 20000 MB/s
        break;
    case VPU::ArchKind::VPUX37XX:
        BW = VPUNN::get_dram_bandwidth_MBps(VPUNN::VPUDevice::VPU_2_7);  // 27000 MB/s
        break;
    default:
        VPUX_THROW("Unsupported architecture '{0}'", arch);
    };

    BW /= 1000;  // convert to GB/s
    return BW;
}

Byte vpux::VPU::getTotalCMXSize(mlir::ModuleOp module) {
    auto cmxRes = IE::getAvailableMemory(module, VPU::MemoryKind::CMX_NN);
    VPUX_THROW_UNLESS(cmxRes != nullptr, "Can't get information about {0} memory", VPU::MemoryKind::CMX_NN);

    const ArchKind arch = getArch(module);

    // This function is used to determine the best tile size. It tries to put maximum data in CMX.
    // Available CMX memory is decreased by two profilingBufferSize even if profiling is disabled
    // because we want to get exactly same compiled networks with profiling enabled and disabled.
    // Two buffer sizes are required in case when profiling allocates new buffer and old buffer
    // is still not disposed. Second buffer can be treated as an optimisation that prevents spilling.
    const int64_t profilingBufferSize =
            vpux::VPUIP::HW_DMA_PROFILING_MAX_BUFFER_SIZE + vpux::VPUIP::HW_DPU_PROFILING_MAX_BUFFER_SIZE +
            ((arch == ArchKind::VPUX37XX) ? vpux::VPUIP::HW_ACT_SHAVE_PROFILING_MAX_BUFFER_SIZE : 0);

    return cmxRes.size() - Byte(2 * profilingBufferSize);
}

Byte vpux::VPU::getTotalCMXSize(mlir::Operation* op) {
    return getTotalCMXSize(getModuleOp(op));
}

//
// ArchKind
//

namespace {

constexpr StringLiteral archAttrName = "VPU.arch";

constexpr Byte DDR_HEAP_SIZE_DEFAULT = 500_MB;
const int64_t DDR_HEAP_SIZE_LOWER_LIMIT = 500;
const int64_t DDR_HEAP_SIZE_UPPER_LIMIT = 1500;
constexpr Byte CSRAM_SIZE = 24_MB;

constexpr Byte KMB_CMX_WORKSPACE_SIZE = Byte(896_KB);

constexpr Byte VPUX37XX_CMX_WORKSPACE_SIZE = Byte(1936_KB);
}  // namespace

void vpux::VPU::setArch(mlir::ModuleOp module, ArchKind kind, Optional<int> numOfDPUGroups, Optional<int> numOfDMAPorts,
                        Optional<int> ddrHeapSize) {
    VPUX_THROW_WHEN(module->hasAttr(archAttrName),
                    "Architecture is already defined, probably you run '--init-compiler' twice");

    module->setAttr(archAttrName, ArchKindAttr::get(module.getContext(), kind));

    const auto addMem = [&](MemoryKind kind, Byte size, double derateFactor, uint32_t bandwidth) {
        auto mem = IE::addAvailableMemory(module, kind, size);
        mem->setAttr(derateFactorAttrName, getFPAttr(module.getContext(), derateFactor));
        mem->setAttr(bandwidthAttrName, getIntAttr(module.getContext(), bandwidth));
    };

    const auto addExecutor = [&](ExecutorKind kind, uint32_t count) {
        return IE::addAvailableExecutor(module, kind, count);
    };

    const auto getNumOfDPUGroupsVal = [&](int maxDpuGroups) {
        int numOfDPUGroupsVal = numOfDPUGroups.hasValue() ? numOfDPUGroups.getValue() : maxDpuGroups;
        VPUX_THROW_UNLESS(1 <= numOfDPUGroupsVal && numOfDPUGroupsVal <= maxDpuGroups,
                          "Invalid number of DPU groups: '{0}'", numOfDPUGroupsVal);
        return numOfDPUGroupsVal;
    };

    const auto getNumOfDMAPortsVal = [&](int maxDmaPorts) {
        int numOfDMAPortsVal = numOfDMAPorts.hasValue() ? numOfDMAPorts.getValue() : maxDmaPorts;
        VPUX_THROW_UNLESS(1 <= numOfDMAPortsVal && numOfDMAPortsVal <= maxDmaPorts,
                          "Invalid number of DMA ports: '{0}'", numOfDMAPortsVal);
        return numOfDMAPortsVal;
    };

    const auto getDDRHeapSize = [&]() {
        if (!ddrHeapSize.hasValue()) {
            return DDR_HEAP_SIZE_DEFAULT;
        } else {
            int64_t ddrHeapSizeVal = ddrHeapSize.getValue();
            VPUX_THROW_UNLESS(
                    DDR_HEAP_SIZE_LOWER_LIMIT <= ddrHeapSizeVal && ddrHeapSizeVal <= DDR_HEAP_SIZE_UPPER_LIMIT,
                    "Invalid number of DDR heap size: '{0}'",
                    ddrHeapSizeVal);  // FixMe: 1500 enlarges the limit for big networks such as unet. Not
                                      // sure if it is too large.
            return static_cast<Byte>(MB(ddrHeapSizeVal));
        }
    };

    IE::ExecutorResourceOp nceCluster;

    switch (kind) {
    case ArchKind::VPUX30XX: {
        addMem(MemoryKind::DDR, getDDRHeapSize(), 0.6, 8);
        addMem(MemoryKind::CMX_NN, KMB_CMX_WORKSPACE_SIZE, 1.0, 32);

        addExecutor(ExecutorKind::DMA_NN, getNumOfDMAPortsVal(VPUX30XX_MAX_DMA_PORTS));
        addExecutor(ExecutorKind::SHAVE_UPA, 16);
        nceCluster = IE::addAvailableExecutor(module, ExecutorKind::NCE, getNumOfDPUGroupsVal(VPUX30XX_MAX_DPU_GROUPS));
        nceCluster.addSubExecutor(ExecutorKind::DPU, 5);

        break;
    }
    case ArchKind::VPUX311X: {
        addMem(MemoryKind::DDR, getDDRHeapSize(), 0.6, 8);
        addMem(MemoryKind::CSRAM, CSRAM_SIZE, 0.85, 64);
        addMem(MemoryKind::CMX_NN, KMB_CMX_WORKSPACE_SIZE, 1.0, 32);

        addExecutor(ExecutorKind::DMA_NN, getNumOfDMAPortsVal(VPUX311X_MAX_DMA_PORTS));
        addExecutor(ExecutorKind::SHAVE_UPA, 16);
        nceCluster = IE::addAvailableExecutor(module, ExecutorKind::NCE, getNumOfDPUGroupsVal(VPUX30XX_MAX_DPU_GROUPS));
        nceCluster.addSubExecutor(ExecutorKind::DPU, 5);

        break;
    }
    case ArchKind::VPUX37XX: {
        addMem(MemoryKind::DDR, getDDRHeapSize(), 0.6, 8);
        addMem(MemoryKind::CMX_NN, VPUX37XX_CMX_WORKSPACE_SIZE, 1.0, 32);

        addExecutor(ExecutorKind::DMA_NN, getNumOfDMAPortsVal(VPUX37XX_MAX_DMA_PORTS));
        // TODO: SHAVE_NN shouldn't be used here
        addExecutor(ExecutorKind::SHAVE_NN, 1);
        // TODO: move SHAVE_ACT as a sub-executor for NCE
        addExecutor(ExecutorKind::SHAVE_ACT, 2);
        nceCluster = IE::addAvailableExecutor(module, ExecutorKind::NCE, getNumOfDPUGroupsVal(VPUX37XX_MAX_DPU_GROUPS));
        nceCluster.addSubExecutor(ExecutorKind::DPU, 1);

        break;
    }
    default:
        VPUX_THROW("Unsupported architecture '{0}'", kind);
    }

    nceCluster.setProcessorFrequency(getFPAttr(module.getContext(), getDpuFrequency(kind)));
}

VPU::ArchKind vpux::VPU::getArch(mlir::Operation* op) {
    auto module = getModuleOp(op);

    if (auto attr = module->getAttr(archAttrName)) {
        VPUX_THROW_UNLESS(attr.isa<VPU::ArchKindAttr>(), "Module attribute '{0}' has unsupported value '{1}'",
                          archAttrName, attr);
        return attr.cast<VPU::ArchKindAttr>().getValue();
    }

    return VPU::ArchKind::UNKNOWN;
}

//
// CompilationMode
//

namespace {

constexpr StringLiteral compilationModeAttrName = "VPU.compilationMode";

}  // namespace

void vpux::VPU::setCompilationMode(mlir::ModuleOp module, CompilationMode compilationMode) {
    VPUX_THROW_WHEN(module->hasAttr(compilationModeAttrName),
                    "CompilationMode is already defined, probably you run '--init-compiler' twice");

    module->setAttr(compilationModeAttrName, VPU::CompilationModeAttr::get(module.getContext(), compilationMode));
}

VPU::CompilationMode vpux::VPU::getCompilationMode(mlir::Operation* op) {
    auto module = getModuleOp(op);

    if (auto attr = module->getAttr(compilationModeAttrName)) {
        VPUX_THROW_UNLESS(attr.isa<VPU::CompilationModeAttr>(), "Module attribute '{0}' has unsupported value '{1}'",
                          compilationModeAttrName, attr);

        return attr.cast<VPU::CompilationModeAttr>().getValue();
    }

    // Use DefaultHW as a default mode
    return VPU::CompilationMode::DefaultHW;
}

//
// PaddingAttr
//

VPU::PaddingAttr vpux::VPU::getPaddingAttr(mlir::MLIRContext* ctx, int64_t left, int64_t right, int64_t top,
                                           int64_t bottom) {
    return PaddingAttr::get(getIntAttr(ctx, left), getIntAttr(ctx, right), getIntAttr(ctx, top),
                            getIntAttr(ctx, bottom), ctx);
}

VPU::PaddingAttr vpux::VPU::getPaddingAttr(mlir::MLIRContext* ctx, ArrayRef<int64_t> padsBegin,
                                           ArrayRef<int64_t> padsEnd) {
    VPUX_THROW_UNLESS(padsBegin.size() == 2, "Paddings array has unsuppoted size '{0}'", padsBegin.size());
    VPUX_THROW_UNLESS(padsEnd.size() == 2, "Paddings array has unsuppoted size '{0}'", padsEnd.size());
    return getPaddingAttr(ctx, padsBegin[1], padsEnd[1], padsBegin[0], padsEnd[0]);
}

VPU::PaddingAttr vpux::VPU::getPaddingAttr(mlir::MLIRContext* ctx, const PadInfo& pad) {
    return getPaddingAttr(ctx, pad.left, pad.right, pad.top, pad.bottom);
}

PadInfo vpux::VPU::toPadInfo(PaddingAttr attr) {
    const auto left = attr.left().getValue().getSExtValue();
    const auto right = attr.right().getValue().getSExtValue();
    const auto top = attr.top().getValue().getSExtValue();
    const auto bottom = attr.bottom().getValue().getSExtValue();
    return PadInfo(left, right, top, bottom);
}

//
// PPETaskAttr
//

VPU::PPETaskAttr vpux::VPU::getPPETaskAttr(mlir::MLIRContext* ctx, VPU::PPEMode mode) {
    return VPU::PPETaskAttr::get(VPU::PPEModeAttr::get(ctx, mode), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                 nullptr, nullptr, nullptr, nullptr, nullptr, ctx);
}

VPU::PPETaskAttr vpux::VPU::getPPETaskAttr(mlir::MLIRContext* ctx, VPU::PPEMode mode, int64_t clampLow,
                                           int64_t clampHigh, int64_t lreluMult, int64_t lreluShift) {
    return VPU::PPETaskAttr::get(VPU::PPEModeAttr::get(ctx, mode), getIntAttr(ctx, clampLow),
                                 getIntAttr(ctx, clampHigh), getIntAttr(ctx, lreluMult), getIntAttr(ctx, lreluShift),
                                 nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, ctx);
}

VPU::PPETaskAttr vpux::VPU::getPPETaskAttr(mlir::MLIRContext* ctx, VPU::PPEMode mode, int64_t clampLow,
                                           int64_t clampHigh, int64_t lreluMult, int64_t lreluShift,
                                           ArrayRef<int64_t> quantMult, ArrayRef<int64_t> quantShift,
                                           int64_t quantPostShift) {
    return VPU::PPETaskAttr::get(VPU::PPEModeAttr::get(ctx, mode), getIntAttr(ctx, clampLow),
                                 getIntAttr(ctx, clampHigh), getIntAttr(ctx, lreluMult), getIntAttr(ctx, lreluShift),
                                 nullptr, getIntArrayAttr(ctx, quantMult), getIntArrayAttr(ctx, quantShift),
                                 getIntAttr(ctx, quantPostShift), nullptr, nullptr, nullptr, ctx);
}

VPU::PPETaskAttr vpux::VPU::getPPETaskAttr(mlir::MLIRContext* ctx, VPU::PPEMode mode, int64_t clampLow,
                                           int64_t clampHigh, int64_t lreluMult, int64_t lreluShift,
                                           ArrayRef<double> quantScale) {
    return VPU::PPETaskAttr::get(VPU::PPEModeAttr::get(ctx, mode), getIntAttr(ctx, clampLow),
                                 getIntAttr(ctx, clampHigh), getIntAttr(ctx, lreluMult), getIntAttr(ctx, lreluShift),
                                 getFPArrayAttr(ctx, quantScale), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                 ctx);
}

VPU::PPETaskAttr vpux::VPU::getPPETaskAttr(mlir::MLIRContext* ctx, VPU::PPEMode mode, int64_t clampLow,
                                           int64_t clampHigh, int64_t lreluMult, int64_t lreluShift,
                                           ArrayRef<int64_t> quantMult, ArrayRef<int64_t> quantShift,
                                           int64_t quantPostShift, ArrayRef<int64_t> in1QuantMult,
                                           ArrayRef<int64_t> in2QuantMult, double fpPReluAlpha) {
    return VPU::PPETaskAttr::get(VPU::PPEModeAttr::get(ctx, mode), getIntAttr(ctx, clampLow),
                                 getIntAttr(ctx, clampHigh), getIntAttr(ctx, lreluMult), getIntAttr(ctx, lreluShift),
                                 nullptr, getIntArrayAttr(ctx, quantMult), getIntArrayAttr(ctx, quantShift),
                                 getIntAttr(ctx, quantPostShift), getIntArrayAttr(ctx, in1QuantMult),
                                 getIntArrayAttr(ctx, in2QuantMult), getFPAttr(ctx, fpPReluAlpha), ctx);
}

VPU::PPEMode vpux::VPU::getPPEMode(VPU::EltwiseType type) {
    switch (type) {
    case VPU::EltwiseType::ADD:
        return vpux::VPU::PPEMode::ADD;
    case VPU::EltwiseType::AND:
        return vpux::VPU::PPEMode::AND;
    case VPU::EltwiseType::MULTIPLY:
        return vpux::VPU::PPEMode::MULT;
    case VPU::EltwiseType::SUBTRACT:
        return vpux::VPU::PPEMode::SUB;
    case VPU::EltwiseType::MIN:
        return vpux::VPU::PPEMode::MINIMUM;
    case VPU::EltwiseType::MAX:
        return vpux::VPU::PPEMode::MAXIMUM;
    default:
        VPUX_THROW("Unsupported EltwiseType '{0}' for PPEMode", type);
    }
}

//
// DistributedTensorAttr
//

mlir::LogicalResult vpux::VPU::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                      DistributedTensorAttr distributedAttr, ArrayRef<int64_t> shape) {
    const auto distributionMode = distributedAttr.mode().getValue();

    if (distributionMode != VPU::DistributionMode::NONE) {
        if (distributedAttr.num_clusters() == nullptr) {
            return printTo(emitError(), "Missing number of clusters.");
        }

        auto numClusters = distributedAttr.num_clusters().getInt();
        if (numClusters <= 0) {
            return printTo(emitError(), "The number of clusters must be greater than 0. Got: {0}", numClusters);
        }
    }

    const auto isTiledMode = [](VPU::DistributionMode mode) {
        return VPU::bitEnumContains(mode, VPU::DistributionMode::SEGMENTED) ||
               VPU::bitEnumContains(mode, VPU::DistributionMode::OVERLAPPED);
    };

    if (!isTiledMode(distributionMode)) {
        return mlir::success();
    }

    if (isTiledMode(distributionMode)) {
        if (distributedAttr.num_tiles() == nullptr || distributedAttr.num_clusters() == nullptr) {
            return printTo(emitError(), "Missing number of tiles and clusters.");
        }

        // Check for validity of tiling scheme
        const auto tilingScheme = parseIntArrayAttr<int64_t>(distributedAttr.num_tiles());
        const auto numClusters = distributedAttr.num_clusters().getInt();

        const auto isValidTile = [](auto dim) {
            return dim > 1;
        };

        if (llvm::count_if(tilingScheme, isValidTile) != 1) {
            return printTo(emitError(), "Currently supporting single axis cluster tiling.");
        }

        const auto axis = std::distance(tilingScheme.begin(), llvm::find_if(tilingScheme, isValidTile));

        if (tilingScheme[axis] != numClusters) {
            return printTo(emitError(), "Incompatibility between tiling scheme '{0}' and number of clusters '{1}'",
                           tilingScheme[axis], numClusters);
        }

        // Limitations on tiling axes
        if (VPU::bitEnumContains(distributionMode, VPU::DistributionMode::OVERLAPPED)) {
            if (axis != Dims4D::Act::H.ind() && axis != Dims4D::Act::W.ind()) {
                return printTo(emitError(), "Overlapped cluster tiling is only supported for dimensions H and W");
            }

            if (distributedAttr.alignment() != nullptr) {
                const auto alignment = parseIntArrayAttr<int64_t>(distributedAttr.alignment());
                if (alignment[axis] != 1) {
                    return printTo(
                            emitError(),
                            "Overlapped cluster tiling does not support alignment on the same axis used for tiling.");
                }
            }

            if (distributedAttr.kernel() == nullptr || distributedAttr.pads() == nullptr ||
                distributedAttr.strides() == nullptr) {
                return printTo(emitError(), "Overlapped cluster tiling requires kernel, pads and strides to be set");
            }
        }

        if (VPU::bitEnumContains(distributionMode, VPU::DistributionMode::SEGMENTED) &&
            !(axis == Dims4D::Act::H.ind() || axis == Dims4D::Act::C.ind() || axis == Dims4D::Act::W.ind() ||
              axis == Dims4D::Filter::OC.ind())) {
            return printTo(emitError(),
                           "Segmented cluster tiling is only supported for activation dimensions H, W and K "
                           "and kernel dimension K");
        }
    }

    if (distributedAttr.alignment() != nullptr) {
        const auto alignment = parseIntArrayAttr<int64_t>(distributedAttr.alignment());
        if (shape.size() != alignment.size()) {
            return printTo(emitError(), "Incompatibility in sizes between tensor shape '{0}' and alignment '{1}'",
                           shape.size(), alignment.size());
        }
    }

    if (distributedAttr.num_tiles() != nullptr) {
        const auto numTiles = parseIntArrayAttr<int64_t>(distributedAttr.num_tiles());
        if (shape.size() != numTiles.size()) {
            return printTo(emitError(), "Incompatibility in sizes between tensor shape '{0}' and tiling scheme '{1}'",
                           shape.size(), numTiles.size());
        }
    }

    if (distributedAttr.kernel() != nullptr) {
        const auto kernel = parseIntArrayAttr<int64_t>(distributedAttr.kernel());
        if (kernel.size() != 2) {
            return printTo(emitError(), "Expected kernel size to be 2. Got '{0}'", kernel.size());
        }
        const auto KY = kernel[Dims4D::Kernel::Y.ind()];
        const auto KX = kernel[Dims4D::Kernel::X.ind()];
        if (KY <= 0 || KX <= 0) {
            return printTo(emitError(), "Invalid kernel size: height '{0}', width '{1}'", KY, KX);
        }
    }

    if (distributedAttr.pads() != nullptr) {
        const auto padTop = distributedAttr.pads().top().getInt();
        const auto padBottom = distributedAttr.pads().bottom().getInt();
        const auto padLeft = distributedAttr.pads().left().getInt();
        const auto padRight = distributedAttr.pads().right().getInt();
        if (padTop < 0 || padBottom < 0 || padLeft < 0 || padRight < 0) {
            return printTo(emitError(), "Invalid pads: top '{0}', bottom '{1}', left '{2}', right '{3}'", padTop,
                           padBottom, padLeft, padRight);
        }
    }

    if (distributedAttr.strides() != nullptr) {
        const auto strides = parseIntArrayAttr<int64_t>(distributedAttr.strides());
        if (strides.size() != 2) {
            return printTo(emitError(), "Expected strides size to be 2. Got '{0}'", strides.size());
        }
        const auto SY = strides[Dims4D::Strides::Y.ind()];
        const auto SX = strides[Dims4D::Strides::X.ind()];
        if (SY <= 0 || SX <= 0) {
            return printTo(emitError(), "Invalid strides: height '{0}', width '{1}'", SY, SX);
        }
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::areDistributionModesCompatible(DistributionMode sourceMode,
                                                              DistributionMode targetMode) {
    // Consecutive distribution modes for a SOK chain or from HKSwitch to SOK
    if ((sourceMode == (DistributionMode::DUPLICATED | DistributionMode::SEGMENTED) ||
         sourceMode == (DistributionMode::MULTICASTED | DistributionMode::SEGMENTED)) &&
        targetMode == DistributionMode::DUPLICATED) {
        return mlir::success();
    }

    return mlir::failure();
}

mlir::LogicalResult vpux::VPU::areDistributionNumClustersCompatible(mlir::IntegerAttr sourceNumClusters,
                                                                    mlir::IntegerAttr targetNumClusters) {
    return mlir::success(sourceNumClusters.getInt() >= targetNumClusters.getInt());
}

mlir::LogicalResult vpux::VPU::areDistributionElementTypesCompatible(mlir::Type inType, mlir::Type outType) {
    if (inType != outType) {
        // allow different quantization parameters
        if (!inType.isa<mlir::quant::QuantizedType>() || !outType.isa<mlir::quant::QuantizedType>()) {
            return mlir::failure();
        }
        if (vpux::getElemTypeSize(inType) != vpux::getElemTypeSize(outType)) {
            return mlir::failure();
        }
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::areDistributionAttrsCompatible(DistributedTensorAttr sourceAttr,
                                                              DistributedTensorAttr targetAttr) {
    if (sourceAttr.alignment() != targetAttr.alignment()) {
        return mlir::failure();
    }

    const auto inDistributionMode = sourceAttr.mode().getValue();
    const auto outDistributionMode = targetAttr.mode().getValue();

    if (inDistributionMode != outDistributionMode) {
        if (VPU::areDistributionModesCompatible(inDistributionMode, outDistributionMode).failed()) {
            return mlir::failure();
        }
    }

    const auto inDistributionNumClusters = sourceAttr.num_clusters();
    const auto outDistributionNumClusters = targetAttr.num_clusters();

    if (VPU::areDistributionNumClustersCompatible(inDistributionNumClusters, outDistributionNumClusters).failed()) {
        return mlir::failure();
    }

    if ((inDistributionMode == VPU::DistributionMode::SEGMENTED) &&
        (outDistributionMode == VPU::DistributionMode::SEGMENTED)) {
        const auto inDistributionNumTiles = parseIntArrayAttr<int64_t>(sourceAttr.num_tiles());
        const auto outDistributionNumTiles = parseIntArrayAttr<int64_t>(targetAttr.num_tiles());
        if (inDistributionNumTiles != outDistributionNumTiles) {
            return mlir::failure();
        }
    }

    return mlir::success();
}

//
// Tiling utils
//

// Segmentation logic operates on schema and runtime assumption that a segmented tensor should be split equally
// across the axis, with the remainder cluster possibly having a smaller tile.
SmallVector<Shape> splitSegmentedShape(ArrayRef<int64_t> shape, ArrayRef<int64_t> tilingScheme,
                                       const int64_t numClusters, const int64_t axis,
                                       Optional<ArrayRef<int64_t>> alignment) {
    VPUX_THROW_UNLESS(axis < int64_t(shape.size()),
                      "An invalid split axis {0} specified, the shape tensor is {1} dimensional", axis, shape.size());
    VPUX_THROW_UNLESS(tilingScheme[axis] == numClusters,
                      "The number of tiles on axis {0} must be equal to the number of clusters specified for "
                      "compilation {1} but got {2}",
                      axis, tilingScheme[axis], numClusters);
    auto tiledShape = to_small_vector(shape);
    tiledShape[axis] = divUp(tiledShape[axis], tilingScheme[axis]);
    tiledShape = alignShape(tiledShape, alignment);

    // Last cluster will have the remainder and it doesn't have to be aligned
    auto remainderTileShape = to_small_vector(shape);
    remainderTileShape[axis] = shape[axis] - tiledShape[axis] * (tilingScheme[axis] - 1);
    VPUX_THROW_UNLESS(remainderTileShape[axis] > 0, "Improper split, '{0}' over '{1}' tiles", shape[axis],
                      tilingScheme[axis]);

    SmallVector<Shape> segmentedTiles(numClusters - 1, Shape(tiledShape));
    segmentedTiles.push_back(Shape(remainderTileShape));
    return segmentedTiles;
}

SmallVector<DimRange> getOverlappedInputTileDimRanges(ArrayRef<int64_t> shape, ArrayRef<int64_t> tilingScheme,
                                                      VPU::DistributedTensorAttr distributionAttr, const int64_t axis,
                                                      const int64_t numClusters) {
    const auto N = shape[Dims4D::Act::N.ind()];
    const auto C = shape[Dims4D::Act::C.ind()];
    const auto Y = shape[Dims4D::Act::H.ind()];
    const auto X = shape[Dims4D::Act::W.ind()];

    const auto kernel = parseIntArrayAttr<int64_t>(distributionAttr.kernel());
    const auto KY = kernel[Dims4D::Kernel::Y.ind()];
    const auto KX = kernel[Dims4D::Kernel::X.ind()];

    const auto pads = distributionAttr.pads();
    const auto padTop = pads.top().getInt();
    const auto padBottom = pads.bottom().getInt();
    const auto padLeft = pads.left().getInt();
    const auto padRight = pads.right().getInt();

    const auto strides = parseIntArrayAttr<int64_t>(distributionAttr.strides());
    const auto SY = strides[Dims4D::Strides::Y.ind()];
    const auto SX = strides[Dims4D::Strides::X.ind()];

    const auto outputHeight = (Y - KY + padTop + padBottom) / SY + 1;
    const auto outputWidth = (X - KX + padLeft + padRight) / SX + 1;
    const SmallVector<int64_t> outputShape{N, C, outputHeight, outputWidth};
    // Alignment should only be considered for final input shape,
    // not the intermediary output shape
    const auto outputTiles = splitSegmentedShape(outputShape, tilingScheme, numClusters, axis, None);

    int64_t offset = 0;
    SmallVector<DimRange> inputTileDimRanges;
    for (const auto& outputTile : outputTiles) {
        const auto dimSize = outputTile[Dim(axis)];
        const DimRange tileSize(offset, offset + dimSize);
        offset += dimSize;

        DimRange inputTile(0, 0);
        if (axis == Dims4D::Act::H.ind()) {
            std::tie(inputTile, std::ignore, std::ignore) =
                    vpux::inputForOutputDim(tileSize, KY, SY, {0, Y}, padTop, padBottom);
        } else if (axis == Dims4D::Act::W.ind()) {
            std::tie(inputTile, std::ignore, std::ignore) =
                    vpux::inputForOutputDim(tileSize, KX, SX, {0, X}, padLeft, padRight);
        } else {
            VPUX_THROW("Unsupported axis '{0}'", axis);
        }
        inputTileDimRanges.push_back(inputTile);
    }
    return inputTileDimRanges;
}

SmallVector<Shape> vpux::VPU::getPerClusterComputeShapes(ShapeRef shapeRef, DistributedTensorAttr distributionAttr) {
    auto shape = to_small_vector(shapeRef.raw());
    const auto distributionMode = distributionAttr.mode().getValue();

    const auto numClusters = distributionAttr.num_clusters().getInt();
    auto tiledComputeShapes = SmallVector<Shape>(numClusters);

    const auto isValidTile = [](auto dim) {
        return dim > 1;
    };

    Optional<ArrayRef<int64_t>> optionalAlignment = None;
    auto alignment = SmallVector<int64_t>(numClusters);
    if (distributionAttr.alignment() != nullptr) {
        alignment = parseIntArrayAttr<int64_t>(distributionAttr.alignment());
        optionalAlignment = Optional<ArrayRef<int64_t>>(alignment);
    }

    if (VPU::bitEnumContains(distributionMode, VPU::DistributionMode::SEGMENTED)) {
        const auto tilingScheme = parseIntArrayAttr<int64_t>(distributionAttr.num_tiles());
        const auto axis = std::distance(tilingScheme.begin(), llvm::find_if(tilingScheme, isValidTile));
        VPUX_THROW_UNLESS(axis < int64_t(tilingScheme.size()), "Segmented tiling scheme requires at least 1 dimension "
                                                               "to be segmented but the tiling schema is [1, 1, 1, 1]");
        tiledComputeShapes = splitSegmentedShape(shape, tilingScheme, numClusters, axis, optionalAlignment);
    } else if (VPU::bitEnumContains(distributionMode, VPU::DistributionMode::OVERLAPPED)) {
        const auto tilingScheme = parseIntArrayAttr<int64_t>(distributionAttr.num_tiles());
        const auto axis = std::distance(tilingScheme.begin(), llvm::find_if(tilingScheme, isValidTile));
        const auto inputTileDimRanges =
                getOverlappedInputTileDimRanges(shape, tilingScheme, distributionAttr, axis, numClusters);

        for (auto p : inputTileDimRanges | indexed) {
            const auto inputTile = p.value();
            const auto cluster = p.index();
            shape[axis] = inputTile.end - inputTile.begin;
            tiledComputeShapes[cluster] = Shape(alignShape(shape, optionalAlignment));
        }
    } else {
        std::fill_n(tiledComputeShapes.begin(), tiledComputeShapes.size(), Shape(alignShape(shape, optionalAlignment)));
    }

    return tiledComputeShapes;
}

SmallVector<Shape> vpux::VPU::getPerClusterComputeShapeOffsets(ShapeRef shapeRef,
                                                               DistributedTensorAttr distributionAttr) {
    const auto shape = to_small_vector(shapeRef.raw());
    const auto distributionMode = distributionAttr.mode().getValue();

    const auto numClusters = distributionAttr.num_clusters().getInt();
    auto tiledComputeShapeOffsets = SmallVector<Shape>(numClusters, Shape(shapeRef.size(), 0));

    const auto isValidTile = [](auto dim) {
        return dim > 1;
    };

    if (VPU::bitEnumContains(distributionMode, VPU::DistributionMode::SEGMENTED)) {
        const auto tiledComputeShapes = getPerClusterComputeShapes(shapeRef, distributionAttr);
        const auto tilingScheme = parseIntArrayAttr<int64_t>(distributionAttr.num_tiles());
        const auto axis = std::distance(tilingScheme.begin(), llvm::find_if(tilingScheme, isValidTile));
        int64_t offset = 0;
        for (int64_t idx = 0; idx < numClusters; idx++) {
            tiledComputeShapeOffsets[idx][Dim(axis)] = offset;
            offset += tiledComputeShapes[idx][Dim(axis)];
        }
    } else if (VPU::bitEnumContains(distributionMode, VPU::DistributionMode::OVERLAPPED)) {
        const auto tilingScheme = parseIntArrayAttr<int64_t>(distributionAttr.num_tiles());
        const auto axis = std::distance(tilingScheme.begin(), llvm::find_if(tilingScheme, isValidTile));
        const auto inputTileDimRanges =
                getOverlappedInputTileDimRanges(shape, tilingScheme, distributionAttr, axis, numClusters);

        for (auto p : inputTileDimRanges | indexed) {
            const auto inputTile = p.value();
            const auto cluster = p.index();
            tiledComputeShapeOffsets[cluster][Dim(axis)] = inputTile.begin;
        }
    }
    return tiledComputeShapeOffsets;
}

SmallVector<PadInfo> vpux::VPU::getPerClusterPadding(DistributedTensorAttr distributionAttr) {
    const auto mode = distributionAttr.mode().getValue();
    VPUX_THROW_UNLESS(mode == VPU::DistributionMode::OVERLAPPED,
                      "Currently getting per cluster padding is supported only for OVERLAPPED, mode - {0}",
                      VPU::stringifyDistributionMode(mode));

    SmallVector<PadInfo> perClusterPadInfo;
    const auto pads = distributionAttr.pads();
    const auto top = pads.top().getInt();
    const auto bottom = pads.bottom().getInt();
    const auto left = pads.left().getInt();
    const auto right = pads.right().getInt();

    perClusterPadInfo.push_back(PadInfo(left, right, top, 0));
    for (auto cluster = 1; cluster < distributionAttr.num_clusters().getInt() - 1; cluster++) {
        perClusterPadInfo.push_back(PadInfo(left, right, 0, 0));
    }
    perClusterPadInfo.push_back(PadInfo(left, right, 0, bottom));

    return perClusterPadInfo;
}

SmallVector<StridedShape> vpux::VPU::getPerClusterStridedShapes(ShapeRef shape, StridesRef strides, DimsOrder dimsOrder,
                                                                DistributedTensorAttr distributionAttr) {
    const auto distributionMode = distributionAttr.mode().getValue();
    const auto computeShapes = getPerClusterComputeShapes(shape, distributionAttr);

    SmallVector<StridedShape> stridedShapes;
    if (VPU::bitEnumContains(distributionMode, VPU::DistributionMode::DUPLICATED)) {
        for (const auto& computeShape : computeShapes) {
            stridedShapes.emplace_back(computeShape, strides);
        }
        return stridedShapes;
    }

    if (VPU::bitEnumContains(distributionMode, VPU::DistributionMode::SEGMENTED) ||
        VPU::bitEnumContains(distributionMode, VPU::DistributionMode::OVERLAPPED)) {
        const auto adaptedStrides = adaptStrides(shape, strides, computeShapes, dimsOrder);
        for (auto p : zip(computeShapes, adaptedStrides)) {
            stridedShapes.emplace_back(std::get<0>(p), std::get<1>(p));
        }
        return stridedShapes;
    }

    VPUX_THROW("Unsupported mode '{0}'", VPU::stringifyEnum(distributionMode));
}

//
// CompressionSchemeAttr
//

int64_t VPU::CompressionSchemeAttr::getTotalNumElems() const {
    if (getNumElems().empty()) {
        return 0;
    }
    auto numElems = getNumElems().getValues<int64_t>();
    return std::accumulate(numElems.begin(), numElems.end(), static_cast<int64_t>(0));
}

int64_t VPU::CompressionSchemeAttr::getNumElemsInRange(int64_t startIdx, int64_t size) const {
    const auto numElems = getNumElems().getValues<int64_t>();
    const auto startIt = numElems.begin() + startIdx;
    const auto endIt = startIt + size;
    return std::accumulate(startIt, endIt, static_cast<int64_t>(0));
}

Byte VPU::CompressionSchemeAttr::getAllocSize(mlir::Type elemType) const {
    const auto elemByteSize = getElemTypeSize(elemType).to<Byte>().count();
    const int64_t alignment = (getAlignment() != nullptr) ? getAlignment().getInt() : 1;
    const auto numElems = getNumElems().getValues<int64_t>();
    int64_t totalAllocSize = 0;
    for (auto num : numElems) {
        totalAllocSize += alignVal<int64_t>(num * elemByteSize, alignment);
    }
    return Byte(totalAllocSize);
}

VPU::CompressionSchemeAttr VPU::getCompressionSchemeAttr(mlir::Type type) {
    if (auto sparseType = type.dyn_cast_or_null<VPU::SparseTensorType>()) {
        return sparseType.getCompressionScheme();
    }
    return nullptr;
}

mlir::Type VPU::setCompressionSchemeAttr(mlir::Type type, VPU::CompressionSchemeAttr compressionSchemeAttr) {
    if (auto sparseType = type.dyn_cast_or_null<VPU::SparseTensorType>()) {
        return VPU::SparseTensorType::get(sparseType.getData(), sparseType.getSparsityMap(),
                                          sparseType.getStorageElementTable(), sparseType.getIsWeights(),
                                          compressionSchemeAttr);
    }
    return type;
}

VPU::CompressionSchemeAttr VPU::tileCompressionScheme(VPU::CompressionSchemeAttr compressionScheme,
                                                      ShapeRef tileOffsets, ShapeRef tileShape) {
    if (compressionScheme == nullptr) {
        return nullptr;
    }
    VPUX_THROW_UNLESS(compressionScheme.getAxis() != nullptr,
                      "Cannot tile compression scheme that is not over an axis");
    const size_t axis = compressionScheme.getAxis().getInt();
    VPUX_THROW_UNLESS(axis < tileOffsets.size() && axis < tileShape.size(),
                      "Axis {0} outside the range of tile dimensions: offsets size {1}, shape size {2}", axis,
                      tileOffsets.size(), tileShape.size());

    const auto numElems = compressionScheme.getNumElems().getValues<int64_t>();
    const auto dimOffset = tileOffsets[Dim(axis)];
    const auto dimShape = tileShape[Dim(axis)];

    const auto startIt = numElems.begin() + dimOffset;
    const auto endIt = startIt + dimShape;
    const auto tileNumElems = SmallVector<int64_t>(startIt, endIt);

    auto ctx = compressionScheme.getContext();
    const auto tileNumElemsType =
            mlir::RankedTensorType::get({static_cast<int64_t>(tileNumElems.size())}, getInt64Type(ctx));
    const auto tileNumElemsAttr = mlir::DenseElementsAttr::get(tileNumElemsType, makeArrayRef(tileNumElems));
    return VPU::CompressionSchemeAttr::get(ctx, compressionScheme.getAxis(), tileNumElemsAttr,
                                           compressionScheme.getAlignment());
}

//
// Generated
//

#include <vpux/compiler/dialect/VPU/generated/attributes/enums.cpp.inc>
#include <vpux/compiler/dialect/VPU/generated/attributes/structs.cpp.inc>
