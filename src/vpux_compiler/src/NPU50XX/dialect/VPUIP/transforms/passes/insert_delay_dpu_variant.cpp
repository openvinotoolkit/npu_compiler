//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/IR/native_attributes/padding_native.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/workload_utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::VPUIP::arch50xx {
#define GEN_PASS_DECL_INSERTDELAYDPUVARIANT
#define GEN_PASS_DEF_INSERTDELAYDPUVARIANT
#include "vpux/compiler/NPU50XX/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP::arch50xx

#include <mlir/IR/Builders.h>

//
// InsertDelayDPUVariant
//

class InsertDelayDPUVariant final : public VPUIP::arch50xx::impl::InsertDelayDPUVariantBase<InsertDelayDPUVariant> {
public:
    explicit InsertDelayDPUVariant(bool dpuProfilingEnabled, Logger log): _dpuProfilingEnabled(dpuProfilingEnabled) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final {
        if (mlir::failed(Base::initialize(ctx))) {
            return mlir::failure();
        }
        if (dpuProfilingEnabled.hasValue()) {
            _dpuProfilingEnabled = dpuProfilingEnabled.getValue();
        }
        return mlir::success();
    }

private:
    void safeRunOnFunc() final;

    bool _dpuProfilingEnabled;
};

void InsertDelayDPUVariant::safeRunOnFunc() {
    const auto usesODUAutopadAndHalo = [](VPUIP::NCEClusterTaskOp nceClusterTask) -> bool {
        auto& variantsBlock = nceClusterTask.getVariants().front();
        if (variantsBlock.empty()) {
            return false;
        }
        auto firstVariant = mlir::dyn_cast<VPUIP::DPUTaskOp>(variantsBlock.front());
        VPUX_THROW_WHEN(firstVariant == nullptr, "Unexpected variant operation: {0}", variantsBlock.front());
        auto outputType = mlir::cast<NDTypeInterface>(nceClusterTask.getOutput().getType());
        return VPU::outputCompatibleWithAutoPad(outputType) && firstVariant.getHaloRegionsAttr() != nullptr;
    };

    const auto createVariant = [](VPUIP::NCEClusterTaskOp nceClusterTask, bool atTheStart) {
        auto& variantsBlock = nceClusterTask.getVariants().front();
        auto builder = atTheStart ? mlir::OpBuilder::atBlockBegin(&variantsBlock)
                                  : mlir::OpBuilder::atBlockEnd(&variantsBlock);
        auto newVariant = mlir::cast<VPUIP::DPUTaskOp>(builder.clone(variantsBlock.front()));
        newVariant->setLoc(appendLoc(newVariant->getLoc(), "dummy"));

        const auto kernelSize = nceClusterTask.getKernelSizeAttr() != nullptr
                                        ? parseIntArrayAttr<int64_t>(nceClusterTask.getKernelSizeAttr())
                                        : SmallVector<int64_t>{};
        // Minimize channels if the variant is done at the start, as the data that will be overwritten by the following
        // variants. To note that eltwise tasks do not support splitting over the channel dimension, so their channel
        // dimension cannot be minimized
        const auto minimizeChannels = atTheStart && nceClusterTask.getTaskType() != VPUIP::NCETaskType::ELTWISE;
        auto newWorkload = VPUIP::minimizeWorkloadSize(
                VPUIP::WorkloadComponents{parseIntArrayAttr<int64_t>(newVariant.getInStartAttr()),
                                          parseIntArrayAttr<int64_t>(newVariant.getInEndAttr()),
                                          parseIntArrayAttr<int64_t>(newVariant.getOutStartAttr()),
                                          parseIntArrayAttr<int64_t>(newVariant.getOutEndAttr()),
                                          VPU::Padding::getClassFromAttr(newVariant.getPad())},
                kernelSize, minimizeChannels);
        const auto ctx = newVariant.getContext();
        newVariant.setInStartAttr(getIntArrayAttr(ctx, newWorkload.inStart));
        newVariant.setInEndAttr(getIntArrayAttr(ctx, newWorkload.inEnd));
        newVariant.setOutStartAttr(getIntArrayAttr(ctx, newWorkload.outStart));
        newVariant.setOutEndAttr(getIntArrayAttr(ctx, newWorkload.outEnd));
        newVariant.setPadAttr(VPU::Padding::getAttrFromClass(ctx, newWorkload.pad));

        if (!atTheStart) {
            // The variant is added at the end in order to delay the execution of the consumer tasks, so that the halo
            // region has enough time to write. This variant should not have halo regions of its own
            newVariant.setHaloRegionsAttr(nullptr);
        }
    };

    auto func = getOperation();
    func.walk([&](VPUIP::NCEClusterTaskOp nceClusterTask) {
        const auto usesAutopad = usesODUAutopadAndHalo(nceClusterTask);
        const auto usesLookupTable =
                nceClusterTask.getSprLookupTable() != nullptr || nceClusterTask.getPalletLookupTable() != nullptr;
        if (!usesAutopad && !usesLookupTable) {
            return;
        }

        const auto isInPlace = nceClusterTask.getIsInplace();
        VPUX_THROW_WHEN(isInPlace, "Cannot add dummy variant to in-place operation");

        _log.trace("Handling '{0}' op at '{1}'", nceClusterTask->getName(), nceClusterTask->getLoc());

        // Lookup tables are prefetched before all the barriers for the DPU are produced. Therefore, it is necessary to
        // ensure that the DMA tasks that bring the tables into CMX are completed once they are read. This is done by
        // adding an additional DPU variant at the start of the variants list, which exists purely for waiting until all
        // the barriers are produced.
        if (usesLookupTable) {
            createVariant(nceClusterTask, /*atTheStart=*/true);
        }

        // In case an operation writes in other clusters via ITI, there is no guarantee that the write
        // has been completed by the time the consumer tasks have started execution. This can result in incorrect data
        // being accessed by the consumers. To prevent this, a small variant is added at the end of the list of
        // variants which will produce a subset of the data as a previous variant, but without halo regions. This
        // represents a delay in execution, which should give the ITI writes enough time to finish the transaction.
        // Note: This is only done when ODU autopad is used by an operation, as these cases were discovered to lead to
        // a race condition. The likely cause is the unaligned memory accesses when the channels are not aligned to 16
        // bytes.
        if (usesAutopad) {
            // There is currently no per-variant HWP control, so the dummy variant that is added at the end would be
            // profiled. This would result in incorrect numbers, so avoid inserting this variant when profiling
            // E#160727: remove this skip once per-variant HWP is supported
            if (_dpuProfilingEnabled) {
                _log.nest().trace("Skip inserting variant at the end as DPU profiling is enabled");
                return;
            }
            createVariant(nceClusterTask, /*atTheStart=*/false);
        }
    });
}

//
// createInsertDelayDPUVariantPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::arch50xx::createInsertDelayDPUVariantPass(bool dpuProfilingEnabled,
                                                                                   Logger log) {
    return std::make_unique<InsertDelayDPUVariant>(dpuProfilingEnabled, log);
}
