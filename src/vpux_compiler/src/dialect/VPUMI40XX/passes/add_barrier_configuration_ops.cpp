//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/wlm_utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_ADDBARRIERCONFIGURATIONOPS
#define GEN_PASS_DEF_ADDBARRIERCONFIGURATIONOPS
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX

using namespace vpux;

namespace {

using BarrierConfig = SmallVector<uint32_t>;

// HW reg address for barrier fifo
constexpr uint32_t STRIDE = 0x20U;
constexpr uint8_t BARRIER_FIFO_DEPTH = 4;

struct BarrierDesc final {
    uint8_t producerCount;
    uint8_t producerInterrupt;
    uint8_t consumerCount;
    uint8_t consumerInterrupt;
    uint8_t isFinalBarrier;
    int64_t virtualId;
    int64_t wlmPage;

    BarrierDesc(uint8_t pCount, uint8_t pInterrupt, uint8_t cCount, uint8_t cInterrupt)
            : producerCount(pCount),
              producerInterrupt(pInterrupt),
              consumerCount(cCount),
              consumerInterrupt(cInterrupt),
              isFinalBarrier(0),
              virtualId(0),
              wlmPage(-1) {
    }
    ~BarrierDesc() = default;
};

uint32_t combineDescValues(uint8_t producerCount, uint8_t producerInterrupt, uint8_t consumerCount,
                           uint8_t consumerInterrupt) {
    uint32_t result = 0;

    result |= (static_cast<uint32_t>(producerCount) & 0xFF);
    result |= (static_cast<uint32_t>(producerInterrupt) & 0xFF) << 8;
    result |= (static_cast<uint32_t>(consumerCount) & 0xFF) << 16;
    result |= (static_cast<uint32_t>(consumerInterrupt) & 0xFF) << 24;

    return result;
}

class AddBarrierConfigurationOps : public VPUMI40XX::impl::AddBarrierConfigurationOpsBase<AddBarrierConfigurationOps> {
public:
    explicit AddBarrierConfigurationOps(
            const WorkloadManagementMode workloadManagementMode,
            const WorkloadManagementBarrierProgrammingMode workloadManagementBarrierProgrammingMode, Logger log)
            : _nBarrs(0),
              _barrierWithMaximumUsage(0),
              _numberOfDescriptorsPerBarrier(0),
              _workloadManagementBarrierProgrammingMode(workloadManagementBarrierProgrammingMode),
              _workloadManagementMode(workloadManagementMode),
              _disableAllInterrupts(workloadManagementBarrierProgrammingMode ==
                                    WorkloadManagementBarrierProgrammingMode::ALL_BARRIER_DMAS_SCHEDULED) {
        Base::initLogger(log, Base::getArgumentName());
    }

    void fillPhysicalBarrierUsage();
    BarrierConfig getBarrierConfig(std::ostringstream& logStream, VPUMI40XX::NNDMAOp barrierProgrammingDMAOp = nullptr);
    void createBarrierConfigurationStrideConstant(VPUMI40XX::MappedInferenceOp mpi, mlir::OpBuilder& builder,
                                                  mlir::Operation* cstInsertionPoint);
    void createRawBarrierConfigurationConstant(VPUMI40XX::MappedInferenceOp mpi, mlir::OpBuilder& builder,
                                               mlir::Operation* cstInsertionPoint);
    VPUMI40XX::NNDMAOp createDMAsToProgramAllBarriers(mlir::OpBuilder& builder, mlir::Operation* bufferInsertionPoint,
                                                      mlir::Operation* cstInsertionPoint,
                                                      mlir::Operation* dmaInsertionPoint);
    VPUMI40XX::NNDMAOp createBarrierProgrammingDmaOp(mlir::OpBuilder& builder, const BarrierConfig& barrierConfig,
                                                     mlir::Operation* cstInsertionPoint,
                                                     mlir::Operation* bufferInsertionPoint,
                                                     mlir::Operation* dmaInsertionPoint,
                                                     VPUMI40XX::NNDMAOp referenceDMAOp = nullptr);

    uint32_t getBarrierFifoAddr(size_t pid = 0);

private:
    void safeRunOnFunc() final;
    int64_t _nBarrs;
    int64_t _barrierWithMaximumUsage;
    int64_t _numberOfDescriptorsPerBarrier;
    uint32_t _barrierFIFOAddr = 0;

    // _barrierUsageIndex keeps track of how many configurations of each pid has been programmed
    // "Programmed" means how many configurations are part of a DMA
    // Index here represents the pid and value at index represents the usage index
    SmallVector<size_t> _barrierUsageIndex;
    WorkloadManagementBarrierProgrammingMode _workloadManagementBarrierProgrammingMode;
    WorkloadManagementMode _workloadManagementMode;
    SmallVector<uint32_t> _barrierProgrammingStrides;
    BarrierConfig _barrierConfigurationsRaw;
    SmallVector<SmallVector<BarrierDesc>> _physicalBarriersUsage;
    bool _disableAllInterrupts;
};

uint32_t AddBarrierConfigurationOps::getBarrierFifoAddr(size_t pid) {
    return _barrierFIFOAddr + (pid * STRIDE);
}

Const::DeclareOp createConstant(mlir::OpBuilder& builder, mlir::Operation* insertionPoint, ArrayRef<uint32_t> vals,
                                int64_t shapeSize) {
    const auto elemType = getUInt32Type(builder.getContext());
    const Shape valShape = {shapeSize};
    const auto dataStorageType = mlir::RankedTensorType::get(valShape.raw(), elemType);
    const auto dataAttr = mlir::DenseElementsAttr::get(dataStorageType, vals);

    auto memType = mlir::MemRefType::get(dataStorageType.getShape(), dataStorageType.getElementType());
    builder.setInsertionPoint(insertionPoint);
    auto configurationConstOp =
            builder.create<Const::DeclareOp>(builder.getUnknownLoc(), memType, Const::ContentAttr::get(dataAttr));

    return configurationConstOp;
}

VPUMI40XX::NNDMAOp AddBarrierConfigurationOps::createBarrierProgrammingDmaOp(
        mlir::OpBuilder& builder, const BarrierConfig& barrierConfig, mlir::Operation* cstInsertionPoint,
        mlir::Operation* bufferInsertionPoint, mlir::Operation* dmaInsertionPoint, VPUMI40XX::NNDMAOp referenceDMAOp) {
    auto physicalBarrierRangeAttr = referenceDMAOp != nullptr ? referenceDMAOp.getPhysicalBarrierRangeAttr() : nullptr;
    auto totalPidsToProgram = physicalBarrierRangeAttr != nullptr ? physicalBarrierRangeAttr.getPidCount() : _nBarrs;
    size_t firstPidInBuffer = physicalBarrierRangeAttr != nullptr ? physicalBarrierRangeAttr.getFirstPid() : 0;

    auto barrierConfigConstOp =
            createConstant(builder, cstInsertionPoint, barrierConfig, totalPidsToProgram * BARRIER_FIFO_DEPTH);

    const auto type = mlir::cast<vpux::NDTypeInterface>(barrierConfigConstOp.getOutput().getType());
    vpux::IndexedSymbolAttr memKindAttr =
            IndexedSymbolAttr::get(builder.getContext(), stringifyEnum(VPU::MemoryKind::Register));
    auto newType = type.changeMemSpace(memKindAttr);
    mlir::MemRefLayoutAttrInterface layout;
    auto memType =
            mlir::MemRefType::get(newType.getShape().raw(), newType.getElementType(), layout, newType.getMemSpace());

    builder.setInsertionPoint(bufferInsertionPoint);
    auto bufferOp = builder.create<VPURT::DeclareBufferOp>(
            builder.getUnknownLoc(), memType, VPURT::BufferSection::Register, getBarrierFifoAddr(firstPidInBuffer));

    auto ctx = builder.getContext();
    auto lengthAttr = vpux::getIntAttr(ctx, totalPidsToProgram * 16);
    auto zeroAttr = vpux::getIntAttr(ctx, 0);
    auto srcWidthAttr = vpux::getIntAttr(ctx, totalPidsToProgram * 16);
    auto dstWidthAttr = vpux::getIntAttr(ctx, 16);
    auto dstStrideAttr = vpux::getIntAttr(ctx, 32);

    // Can be anything, the prev DMA will define the index at reindexing stage
    auto indexAttr = referenceDMAOp != nullptr
                             ? mlir::cast<vpux::VPURegMapped::IndexType>(referenceDMAOp.getIndex().getType())
                             : VPURegMapped::IndexType::get(ctx, 0, 0, 0);

    auto dmaDescriptorAttr = VPUIP::DMADescriptorAttr::get(ctx, /*numPlane*/ zeroAttr, /*len*/ lengthAttr,
                                                           /*srcWidth*/ srcWidthAttr, /*srcStride*/ zeroAttr,
                                                           /*srcPlaneStride*/ zeroAttr, /*dstWidth*/ dstWidthAttr,
                                                           /*dstStride*/ dstStrideAttr, /*dstPlaneStride*/
                                                           zeroAttr);

    mlir::ValueRange waitBarriers = referenceDMAOp != nullptr ? referenceDMAOp.getWaitBarriers() : mlir::ValueRange();
    mlir::ValueRange updateBarriers =
            referenceDMAOp != nullptr ? referenceDMAOp.getUpdateBarriers() : mlir::ValueRange();
    mlir::Value previousTask = referenceDMAOp != nullptr ? referenceDMAOp.getPreviousTask() : nullptr;
    mlir::Value enqueueBarrier = referenceDMAOp != nullptr ? referenceDMAOp.getEnqueueBarrier() : nullptr;

    // Assign -1 to wlmPage if no reference DMA is provided, indicating the DMA is in bootstrap
    auto wlmPageAttr =
            referenceDMAOp != nullptr ? referenceDMAOp.getWlmPageAttr() : vpux::getIntAttr(builder.getContext(), -1);

    builder.setInsertionPoint(dmaInsertionPoint);
    return builder.create<VPUMI40XX::NNDMAOp>(
            builder.getUnknownLoc(), indexAttr, /*taskLocation*/ nullptr, barrierConfigConstOp.getOutput(),
            bufferOp.getBuffer(), previousTask, waitBarriers, updateBarriers,
            /*startAfter*/ 0,
            /*cleanAfter*/ 0, false, false, false, 0, VPUIP::DMAAccMode::DISABLE, nullptr, nullptr,
            /*transactionAttr*/ nullptr, dmaDescriptorAttr, nullptr, nullptr, false, nullptr, enqueueBarrier,
            wlmPageAttr);
}

void AddBarrierConfigurationOps::fillPhysicalBarrierUsage() {
    auto netFunc = getOperation();
    _physicalBarriersUsage.clear();
    _physicalBarriersUsage.resize(_nBarrs);
    _barrierUsageIndex.assign(_nBarrs, 0);

    // Find which pid has maximum usage as we will need to pad the rest
    DenseMap<int64_t, int64_t> barrierCount;

    auto barriers = vpux::to_small_vector(netFunc.getOps<VPUMI40XX::ConfigureBarrierOp>());
    for (auto barrierOp : barriers) {
        auto pid = barrierOp.getId();
        auto consumerInterrupt = _disableAllInterrupts ? 0 : 1;
        auto desc = BarrierDesc(barrierOp.getProducerCount().value_or(0), 0, barrierOp.getConsumerCount().value_or(0),
                                consumerInterrupt);

        // Used for debug trace
        desc.virtualId = barrierOp.getResult().getType().getValue();
        desc.wlmPage = barrierOp.getWlmPage().value_or(-1);

        if (barrierOp.getIsFinalBarrier()) {
            desc.consumerCount = 1;
            desc.producerInterrupt = 1;
            desc.consumerInterrupt = 0;
            desc.isFinalBarrier = 1;
        }
        _physicalBarriersUsage[pid].push_back(desc);
        ++barrierCount[pid];

        // Check if this pid has more ops than the current max
        if (barrierCount[pid] > _barrierWithMaximumUsage) {
            _barrierWithMaximumUsage = barrierCount[pid];
        }
    }
}

void AddBarrierConfigurationOps::createBarrierConfigurationStrideConstant(VPUMI40XX::MappedInferenceOp mpi,
                                                                          mlir::OpBuilder& builder,
                                                                          mlir::Operation* cstInsertionPoint) {
    _barrierProgrammingStrides.resize(_nBarrs, 0);
    for (int64_t pid = 0; pid < _nBarrs; ++pid) {
        _barrierProgrammingStrides[pid] = _physicalBarriersUsage[pid].size();
    }
    auto strideConstant = createConstant(builder, cstInsertionPoint, _barrierProgrammingStrides, _nBarrs);
    mpi.getNumOfBarrierReprogrammingsMutable().assign(strideConstant.getResult());
}

// For simplify preemption flow on FW side,
// compiler must to add extra zero descriptors for allow FW to restore FIFO using one DMA
// In worst case if preemption happens on the latest barrier, we need to have at least INITIAL_BARRIER_FIFO_DEPTH -
// 1 zero descriptors
void AddBarrierConfigurationOps::createRawBarrierConfigurationConstant(VPUMI40XX::MappedInferenceOp mpi,
                                                                       mlir::OpBuilder& builder,
                                                                       mlir::Operation* cstInsertionPoint) {
    auto maxBarrierReusage = std::max(_barrierWithMaximumUsage, static_cast<int64_t>(BARRIER_FIFO_DEPTH));
    _numberOfDescriptorsPerBarrier = maxBarrierReusage + BARRIER_FIFO_DEPTH - 1;
    auto totalAmountOfBarrierProgrammingDescs = _nBarrs * _numberOfDescriptorsPerBarrier;
    _barrierConfigurationsRaw.resize(totalAmountOfBarrierProgrammingDescs, 0);
    for (auto pid : irange(_nBarrs)) {
        auto barProgrammingDescVec = _physicalBarriersUsage[pid];
        for (auto i : irange(barProgrammingDescVec.size())) {
            auto desc = barProgrammingDescVec[i];
            _barrierConfigurationsRaw[pid * _numberOfDescriptorsPerBarrier + i] = combineDescValues(
                    desc.producerCount, desc.producerInterrupt, desc.consumerCount, desc.consumerInterrupt);
        }
    }

    // Create all barrier configuration constant
    auto barConfigurationConst =
            createConstant(builder, cstInsertionPoint, _barrierConfigurationsRaw, totalAmountOfBarrierProgrammingDescs);
    mpi.getBarrierConfigurationTasksMutable().assign(barConfigurationConst.getResult());
    mpi.setBarrierConfigurationTasksCountAttr(builder.getI64IntegerAttr(totalAmountOfBarrierProgrammingDescs));
}

// Creates the barrier configuration array for the requested page.
//
// - If barPDmaPage == 0: Returns configurations for the first four usages
//   of all available physical barriers (Bootstrap mode).
//
// - If barPDmaPage > 0: The configuration is determined based on whether the page is odd or even:
//     - Odd pages (barPDmaPage % 2 == 1): Configures only the first half of available barriers.
//     - Even pages (barPDmaPage % 2 == 0): Configures only the second half of available barriers.
//
// ## Details:
// - Each tile has 16 available PIDs, and each chunk has a size of
//   (availablePids × BARRIER_FIFO_DEPTH).
// - Barriers are assigned one of the following descriptor types:
//   NOTE: For ALL_BARRIER_DMAS_SCHEDULED cInterrupt is set to 1 for one barrier per page, required for heartbeat
//
//   1. Common Descriptor Partial WLM:
//      - Producer Count (pCount) = val
//      - Producer Interrupt (pInterrupt) = 0
//      - Consumer Count (cCount) = val
//      - Consumer Interrupt (cInterrupt) = 1
//
//   2. Common Descriptor Full WLM:
//      - Producer Count (pCount) = val
//      - Producer Interrupt (pInterrupt) = 0
//      - Consumer Count (cCount) = val
//      - Consumer Interrupt (cInterrupt) = 0
//
//   3. Special Descriptor for Heartbeat (Full WLM):
//      - Producer Count (pCount) = val
//      - Producer Interrupt (pInterrupt) = 0
//      - Consumer Count (cCount) = val
//      - Consumer Interrupt (cInterrupt) = 1
//
//   4. Final Barrier:
//      - Producer Count (pCount) = val
//      - Producer Interrupt (pInterrupt) = 1
//      - Consumer Count (cCount) = 1
//      - Consumer Interrupt (cInterrupt) = 0
//
//   5. Unused Barrier:
//      - Producer Count (pCount) = 0
//      - Producer Interrupt (pInterrupt) = 0
//      - Consumer Count (cCount) = 0
//      - Consumer Interrupt (cInterrupt) = 0
//
// The function updates _barrierUsageIndex to track progress and avoid reprogramming barriers unnecessarily.
BarrierConfig AddBarrierConfigurationOps::getBarrierConfig(std::ostringstream& logStream,
                                                           VPUMI40XX::NNDMAOp barrierProgrammingDMAOp) {
    BarrierConfig barrierConfig;
    // Clear before adding new logs
    logStream.str("");
    logStream.clear();

    // Default for bootstrap
    int64_t pidStart = 0;
    int64_t pidEnd = _nBarrs - 1;

    VPUIP::PhysicalBarrierRangeAttr physicalBarrierRangeAttr = nullptr;

    // We have a reference DMA with pid_start and pid_end defined
    if (barrierProgrammingDMAOp != nullptr) {
        physicalBarrierRangeAttr = barrierProgrammingDMAOp.getPhysicalBarrierRangeAttr();

        pidStart = physicalBarrierRangeAttr.getStart().getValue().getSExtValue();
        pidEnd = physicalBarrierRangeAttr.getEnd().getValue().getSExtValue();
    }

    logStream << "Programming barriers (" << pidStart << " to " << pidEnd << ") \n";
    for (int64_t pid = pidStart; pid <= pidEnd; ++pid) {
        auto& pidUsage = _physicalBarriersUsage[pid];
        size_t usageSize = pidUsage.size();
        // Use _barrierUsageIndex to track how many have been programmed
        size_t startIndex = _barrierUsageIndex[pid];

        logStream << "PID: " << pid << " ";

        for (size_t i = 0; i < BARRIER_FIFO_DEPTH; ++i) {
            if (startIndex < usageSize) {
                auto barrierDesc = pidUsage[startIndex];
                logStream << "  ViD: " << static_cast<int>(barrierDesc.virtualId)
                          << " | PCnt: " << static_cast<int>(barrierDesc.producerCount)
                          << " | Ccnt: " << static_cast<int>(barrierDesc.consumerCount)
                          << " | CInt: " << static_cast<int>(barrierDesc.consumerInterrupt)
                          << " | wlmPage: " << static_cast<int>(barrierDesc.wlmPage) << " ";

                barrierConfig.push_back(combineDescValues(barrierDesc.producerCount, barrierDesc.producerInterrupt,
                                                          barrierDesc.consumerCount, barrierDesc.consumerInterrupt));
                ++startIndex;
            } else {
                // Push an empty (unused) barrier entry
                barrierConfig.push_back(combineDescValues(0, 0, 0, 0));
            }
        }
        logStream << "\n";

        // Update _barrierUsageIndex to track progress
        _barrierUsageIndex[pid] = startIndex;
    }

    return barrierConfig;
}

// Created DMAs to program barriers and return first DMA which is used for re-indexing the DMAOps
VPUMI40XX::NNDMAOp AddBarrierConfigurationOps::createDMAsToProgramAllBarriers(mlir::OpBuilder& builder,
                                                                              mlir::Operation* bufferInsertionPoint,
                                                                              mlir::Operation* cstInsertionPoint,
                                                                              mlir::Operation* dmaInsertionPoint) {
    auto netFunc = getOperation();
    auto dmaTaskOps = netFunc.getOps<VPUMI40XX::NNDMAOp>();

    std::ostringstream logStream;

    // Step 1: Explicitly handle bootstrap programming
    _log.trace("Programming Bootstrap Barriers");
    VPUMI40XX::NNDMAOp bootstrapDMA;
    vpux::VPURegMapped::IndexType indexAttr;
    auto placeholderBarProgDMAs = llvm::to_vector(llvm::make_filter_range(dmaTaskOps, [](auto dma) {
        return dma.getPhysicalBarrierRangeAttr() != nullptr;
    }));

    // We can have this case for Non FWLM case, create explicit BarProgDMA at bootstrap
    if (placeholderBarProgDMAs.empty()) {
        auto bootstrapConfig = getBarrierConfig(logStream);
        bootstrapDMA = createBarrierProgrammingDmaOp(builder, bootstrapConfig, bufferInsertionPoint, cstInsertionPoint,
                                                     dmaInsertionPoint);

        indexAttr = mlir::cast<vpux::VPURegMapped::IndexType>(bootstrapDMA.getIndex().getType());
        _log.trace("DMA {0} {1}", indexAttr.getValue(), logStream.str());

        if (auto dmaTypeOp = llvm::dyn_cast<VPURegMapped::DMATypeOpInterface>(dmaInsertionPoint)) {
            dmaTypeOp.setPreviousTaskForOp(bootstrapDMA);
        }
    }

    // No need to go over all barriers in case of PWLM_V2_PAGES & INITIAL_BARRIER_DMAS_SCHEDULED
    if (_workloadManagementBarrierProgrammingMode !=
        WorkloadManagementBarrierProgrammingMode::ALL_BARRIER_DMAS_SCHEDULED) {
        return bootstrapDMA;
    }

    // Step 2: Process DMA tasks
    for (auto dmaOp : placeholderBarProgDMAs) {
        auto barrierConfig = getBarrierConfig(logStream, dmaOp);
        auto reprogrammingDMAOp =
                createBarrierProgrammingDmaOp(builder, barrierConfig, cstInsertionPoint, bufferInsertionPoint,
                                              /*dmaInsertionPoint*/ dmaOp, /*referenceDMAOp*/ dmaOp);
        // Need the first DMA for reindexList
        if (!bootstrapDMA) {
            bootstrapDMA = reprogrammingDMAOp;
        }
        indexAttr = mlir::cast<vpux::VPURegMapped::IndexType>(reprogrammingDMAOp.getIndex().getType());
        _log.trace("DMA {0} {1}", indexAttr.getValue(), logStream.str());

        dmaOp.getResult().replaceAllUsesWith(reprogrammingDMAOp.getResult());

        // Safe erase since iterator already advanced
        if (dmaOp->use_empty()) {
            dmaOp->erase();
        }
    }

    return bootstrapDMA;
}

void AddBarrierConfigurationOps::safeRunOnFunc() {
    if (workloadManagementBarrierProgrammingModeOpt.hasValue()) {
        _workloadManagementBarrierProgrammingMode = workloadManagementBarrierProgrammingModeOpt.getValue();
    }

    VPUX_THROW_WHEN(
            _workloadManagementMode == WorkloadManagementMode::PWLM_V1_BARRIER_FIFO &&
                    _workloadManagementBarrierProgrammingMode == WorkloadManagementBarrierProgrammingMode::LEGACY,
            "Unsupported Configuration WorkloadManagementMode:PWLM_V1_BARRIER_FIFO with "
            "WorkloadManagementBarrierProgrammingMode:LEGACY");

    VPUX_THROW_WHEN(_workloadManagementMode == WorkloadManagementMode::PWLM_V1_BARRIER_FIFO &&
                            _workloadManagementBarrierProgrammingMode ==
                                    WorkloadManagementBarrierProgrammingMode::ALL_BARRIER_DMAS_SCHEDULED,
                    "Unsupported Configuration WorkloadManagementMode:PWLM_V1_BARRIER_FIFO with "
                    "WorkloadManagementBarrierProgrammingMode:ALL_BARRIER_DMAS_SCHEDULED");

    auto netFunc = getOperation();
    auto mpi = VPUMI40XX::getMPI(netFunc);
    auto builder = mlir::OpBuilder(mpi.getOperation());
    _barrierFIFOAddr = config::getConstraint(netFunc, config::BARRIER_FIFO_ADDR);

    auto bufferOps = netFunc.getOps<VPURT::DeclareBufferOp>();
    auto bufferInsertionPoint = !bufferOps.empty() ? *bufferOps.begin() : &netFunc.getBody().front().front();

    auto declOps = netFunc.getOps<Const::DeclareOp>();
    auto cstInsertionPoint = !declOps.empty() ? *declOps.begin() : &netFunc.getBody().front().front();

    auto dmaTypeOps = netFunc.getOps<VPURegMapped::DMATypeOpInterface>();
    mlir::Operation* dmaInsertionPoint = !dmaTypeOps.empty() ? *dmaTypeOps.begin() : &netFunc.getBody().front().front();

    _nBarrs = VPUIP::getNumAvailableBarriers(netFunc);
    fillPhysicalBarrierUsage();

    switch (_workloadManagementBarrierProgrammingMode) {
    case WorkloadManagementBarrierProgrammingMode::NO_BARRIER_DMAS_SCHEDULED: {
        createBarrierConfigurationStrideConstant(mpi, builder, cstInsertionPoint);
        createRawBarrierConfigurationConstant(mpi, builder, cstInsertionPoint);
    } break;

    case WorkloadManagementBarrierProgrammingMode::INITIAL_BARRIER_DMAS_SCHEDULED: {
        createBarrierConfigurationStrideConstant(mpi, builder, cstInsertionPoint);
        createRawBarrierConfigurationConstant(mpi, builder, cstInsertionPoint);

        auto bootStrapDMAOp =
                createDMAsToProgramAllBarriers(builder, bufferInsertionPoint, cstInsertionPoint, dmaInsertionPoint);
        VPUMI40XX::reindexList<VPUMI40XX::NNDMAOp>(mpi, bootStrapDMAOp, 0, 0);
    } break;

    case WorkloadManagementBarrierProgrammingMode::ALL_BARRIER_DMAS_SCHEDULED: {
        auto firstDMAOp =
                createDMAsToProgramAllBarriers(builder, bufferInsertionPoint, cstInsertionPoint, dmaInsertionPoint);
        VPUMI40XX::reindexList<VPUMI40XX::NNDMAOp>(mpi, firstDMAOp, 0, 0);
    } break;
    default:
        VPUX_THROW("Unsupported Barrier Programing Mode: {0}", _workloadManagementBarrierProgrammingMode);
        break;
    }

    mpi.setWorkloadManagementBarrierProgrammingMode(static_cast<VPURegMapped::WorkloadManagementBarrierProgrammingMode>(
            _workloadManagementBarrierProgrammingMode));
}

}  // namespace

//
// createAddBarrierConfigurationOps
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createAddBarrierConfigurationOps(
        WorkloadManagementMode workloadManagementMode,
        WorkloadManagementBarrierProgrammingMode workloadManagementBarrierProgrammingMode, Logger log) {
    return std::make_unique<AddBarrierConfigurationOps>(workloadManagementMode,
                                                        workloadManagementBarrierProgrammingMode, log);
}
