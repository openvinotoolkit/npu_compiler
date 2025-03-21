//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/utils/passes.hpp"

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_ADDINITIALBARRIERCONFIGURATIONOPS
#define GEN_PASS_DEF_ADDINITIALBARRIERCONFIGURATIONOPS
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX

using namespace vpux;

static constexpr size_t INITIAL_BARRIER_FIFO_DEPTH = 4;

namespace {

class AddInitialBarrierConfigurationOps :
        public VPUMI40XX::impl::AddInitialBarrierConfigurationOpsBase<AddInitialBarrierConfigurationOps> {
public:
    explicit AddInitialBarrierConfigurationOps(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

struct BarrierDesc {
    uint8_t producerCount;
    uint8_t producerInterrupt;
    uint8_t consumerCount;
    uint8_t consumerInterrupt;

    BarrierDesc(uint8_t pCount, bool pInterrupt, uint8_t cCount, bool cInterrupt)
            : producerCount(pCount),
              producerInterrupt(pInterrupt),
              consumerCount(cCount),
              consumerInterrupt(cInterrupt) {
    }
};

uint32_t combineDescValues(uint8_t producerCount, uint8_t producerInterrupt, uint8_t consumerCount,
                           uint8_t consumerInterrupt) {
    uint32_t result = 0;

    result |= (static_cast<int32_t>(producerCount) & 0xFF);
    result |= (static_cast<int32_t>(producerInterrupt) & 0xFF) << 8;
    result |= (static_cast<int32_t>(consumerCount) & 0xFF) << 16;
    result |= (static_cast<int32_t>(consumerInterrupt) & 0xFF) << 24;

    return result;
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

void AddInitialBarrierConfigurationOps::safeRunOnFunc() {
    auto netFunc = getOperation();
    auto mpi = VPUMI40XX::getMPI(netFunc);
    auto builder = mlir::OpBuilder(mpi.getOperation());
    auto numberOfAvailablePhysicalBarriers = VPUIP::getNumAvailableBarriers(netFunc);
    SmallVector<SmallVector<BarrierDesc>> physicalBarriersUsageInDesc(numberOfAvailablePhysicalBarriers);

    // We have next values for descriptors:
    // Common descriptor: pCount = val, pInterrupt = 0, cCount = val, cInterrupt = 1
    // Final barrier:     pCount = val, pInterrupt = 1, cCount = 1,   cInterrupt = 0
    // Unused barrier:    pCount = 0,   pInterrupt = 0, cCount = 0,   cInterrupt = 0

    for (auto vBarrier : netFunc.getOps<VPUMI40XX::ConfigureBarrierOp>()) {
        auto pid = vBarrier.getId();
        auto desc = BarrierDesc(vBarrier.getProducerCount().value_or(0), 0, vBarrier.getConsumerCount().value_or(0), 1);
        if (vBarrier.getIsFinalBarrier()) {
            desc.consumerCount = 1;
            desc.producerInterrupt = 1;
            desc.consumerInterrupt = 0;
        }
        physicalBarriersUsageInDesc[pid].push_back(desc);
    }

    mlir::SmallVector<uint32_t> barrierProgrammingStrides(numberOfAvailablePhysicalBarriers, 0);
    size_t maxBarrierReusage = 0;
    for (auto pid : irange(numberOfAvailablePhysicalBarriers)) {
        barrierProgrammingStrides[pid] = physicalBarriersUsageInDesc[pid].size();
        maxBarrierReusage = std::max(maxBarrierReusage, physicalBarriersUsageInDesc[pid].size());
    }

    auto declOps = netFunc.getOps<Const::DeclareOp>();
    auto cstInsertionPoint = !declOps.empty() ? *declOps.begin() : &netFunc.getBody().front().front();

    auto strideConstant =
            createConstant(builder, cstInsertionPoint, barrierProgrammingStrides, numberOfAvailablePhysicalBarriers);

    mpi.getNumOfBarrierReprogrammingsMutable().assign(strideConstant.getResult());
    maxBarrierReusage = std::max(maxBarrierReusage, INITIAL_BARRIER_FIFO_DEPTH);

    // For simplify preemption flow on FW side,
    // compiler must to add extra zero descriptors for allow FW to restore FIFO using one DMA
    // In worst case if preemption happens on the latest barrier, we need to have at least INITIAL_BARRIER_FIFO_DEPTH -
    // 1 zero descriptors
    auto numberOfDescriptorsPerBarrier = maxBarrierReusage + INITIAL_BARRIER_FIFO_DEPTH - 1;

    auto totalAmountOfBarrierProgrammingDescs = numberOfAvailablePhysicalBarriers * numberOfDescriptorsPerBarrier;
    std::vector<uint32_t> barrierConfigurationsRaw(totalAmountOfBarrierProgrammingDescs, 0);
    for (auto pid : irange(numberOfAvailablePhysicalBarriers)) {
        auto barProgrammingDescVec = physicalBarriersUsageInDesc[pid];
        for (auto i : irange(barProgrammingDescVec.size())) {
            auto desc = barProgrammingDescVec[i];
            barrierConfigurationsRaw[pid * numberOfDescriptorsPerBarrier + i] = combineDescValues(
                    desc.producerCount, desc.producerInterrupt, desc.consumerCount, desc.consumerInterrupt);
        }
    }

    auto barConfigurationConst =
            createConstant(builder, cstInsertionPoint, barrierConfigurationsRaw, totalAmountOfBarrierProgrammingDescs);
    mpi.getBarrierConfigurationTasksMutable().assign(barConfigurationConst.getResult());
    mpi.setBarrierConfigurationTasksCountAttr(builder.getI64IntegerAttr(totalAmountOfBarrierProgrammingDescs));
}

}  // namespace

//
// createAddInitialBarrierConfigurationOps
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createAddInitialBarrierConfigurationOps(Logger log) {
    return std::make_unique<AddInitialBarrierConfigurationOps>(log);
}
