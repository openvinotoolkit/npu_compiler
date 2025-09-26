//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/dma_fusion_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux::VPUIP::arch40xx {
#define GEN_PASS_DECL_FUSESEGMENTEDDMA
#define GEN_PASS_DEF_FUSESEGMENTEDDMA
#include "vpux/compiler/NPU40XX/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP::arch40xx

using namespace vpux;

namespace {

std::optional<int64_t> getClusterId(mlir::Value val) {
    if (mlir::cast<vpux::NDTypeInterface>(val.getType()).getMemoryKind() != VPU::MemoryKind::CMX_NN) {
        return {};
    }
    auto bufOp = val.getDefiningOp<VPURT::DeclareBufferOp>();
    if (bufOp == nullptr) {
        return {};
    }
    if (bufOp.getSectionIndex().has_value()) {
        auto clusters = parseIntArrayAttr<int64_t>(bufOp.getSectionIndex().value());
        return clusters.front();
    }
    return {};
}

std::optional<int64_t> getClusterId(VPURT::TaskOp taskOp) {
    auto clusterId = getClusterId(VPUIP::getOutput(taskOp));
    if (clusterId.has_value()) {
        return clusterId;
    }
    return getClusterId(VPUIP::getInput(taskOp));
}

// clang-format off

// Prepare strides for DMAs inputs/outputs
// Stride depends on memory space of tensor.
// For DDR it's just distance between allocations. It can be major stride or
// some value obtained from declarations in case of view on minor dimension,
// see @FuseStridedBuffer2BufferDma test
// For 40XX CMX this stride grows exponentially starting from 2_MB.
// It happens because of 40XX addressing scheme, where tileId is encoded as bit mask, not regular address
// 000001_XXX..XXX - Tile 0
// 000010_XXX..XXX - Tile 1
// 000100_XXX..XXX - Tile 2 and so on
// As result of it, inter-cluster strides aren't same(2MB, 4MB, 8MB,...)
// and we can fuse only 2 DMAs together

// clang-format on

VPUIP::StrideInfo getStride(vpux::Logger log, SmallVector<VPURT::TaskOp> tasks, bool input) {
    auto firstDmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(tasks[0].getInnerTaskOp());
    auto secondDmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(tasks[1].getInnerTaskOp());
    // Check if some other pass broke DMAs compatibility
    if (!VPUIP::hasCompatibleTypes(firstDmaOp, secondDmaOp)) {
        log.trace("Incompatible DMAs");
        return {/*feasible=*/false, /*explicit=*/false, Byte(0)};
    }

    auto firstTaskVal = input ? VPUIP::getInput(tasks[0]) : VPUIP::getOutput(tasks[0]);
    auto secondTaskVal = input ? VPUIP::getInput(tasks[1]) : VPUIP::getOutput(tasks[1]);
    auto type = mlir::cast<vpux::NDTypeInterface>(firstTaskVal.getType());
    auto memKind = type.getMemoryKind();
    if (memKind == VPU::MemoryKind::CMX_NN) {
        auto maybeClusterId = getClusterId(firstTaskVal);
        SmallVector<Bit> INTERCLUSTER_STRIDES = {2_MB, 4_MB, 8_MB, 16_MB, 32_MB, 64_MB};
        VPUX_THROW_UNLESS(maybeClusterId.has_value(), "Can't get cluster");
        return {/*feasible=*/true, /*explicit=*/true, INTERCLUSTER_STRIDES[maybeClusterId.value()]};
    }
    VPUX_THROW_WHEN(memKind != VPU::MemoryKind::DDR, "Memory kind must be either DDR or CMX");
    // DMAs with large leading stride tends to have performance degradation, so disable them
    const Byte LARGE_STRIDE_THRESHOLD = 32_MB;
    if (Byte(type.getMemStrides()[MemDim(0)]) >= LARGE_STRIDE_THRESHOLD) {
        log.trace("Can't fuse copies with large stride");
        return {false, false, Byte(0)};
    }
    auto buf1 = firstTaskVal.getDefiningOp<VPURT::DeclareBufferOp>();
    auto buf2 = secondTaskVal.getDefiningOp<VPURT::DeclareBufferOp>();
    int64_t declarationsDistance = buf2.getByteOffset() - buf1.getByteOffset();
    if (declarationsDistance <= 0) {
        log.trace("Can't fuse copies with same base address");
        return {false, false, Byte(0)};
    }

    return {/*feasible=*/true, /*explicit=*/true, Byte(declarationsDistance)};
}

//
// FuseSegmentedDma
//

class FuseSegmentedDma final : public VPUIP::arch40xx::impl::FuseSegmentedDMABase<FuseSegmentedDma> {
public:
    explicit FuseSegmentedDma(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void FuseSegmentedDma::safeRunOnFunc() {
    auto funcOp = getOperation();

    const auto srcStrideProvider = [](vpux::Logger log, SmallVector<VPURT::TaskOp> tasks) -> VPUIP::StrideInfo {
        if (VPUIP::getCommonConstant(tasks) != nullptr) {
            return {/*feasible=*/true, false, Byte(0)};
        }
        return getStride(log, std::move(tasks), /*input=*/true);
    };

    const auto dstStrideProvider = [](vpux::Logger log, SmallVector<VPURT::TaskOp> tasks) -> VPUIP::StrideInfo {
        return getStride(log, std::move(tasks), /*input=*/false);
    };

    auto module = funcOp->getParentOfType<mlir::ModuleOp>();
    auto dmaOp = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    auto dmaPortCount = dmaOp.getCount();

    const auto getPort = [=](SmallVector<VPURT::TaskOp> tasks) {
        auto maybeClusterId = getClusterId(tasks.front());
        if (maybeClusterId.has_value()) {
            return (maybeClusterId.value() / 2) % dmaPortCount;
        }
        return static_cast<int64_t>(0);
    };

    VPUIP::handleDmaFusion(funcOp, _log, srcStrideProvider, dstStrideProvider, getPort);
};

};  // namespace

//
// createFuseSegmentedDmaPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::arch40xx::createFuseSegmentedDmaPass(Logger log) {
    return std::make_unique<FuseSegmentedDma>(log);
}
