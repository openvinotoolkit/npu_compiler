//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_writer.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/kernel_params_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/Support/FileSystem.h>

#include <iostream>
#include <vector>

using namespace vpux;

namespace {

//
// ConvertVPUIP2VPUIPRegMappedPass
//

class ConvertVPUIP2VPUIPRegMappedPass final : public ConvertVPUIP2VPUIPRegMappedBase<ConvertVPUIP2VPUIPRegMappedPass> {
public:
    explicit ConvertVPUIP2VPUIPRegMappedPass(Logger log): _log(log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;

    Logger _log;

    llvm::SmallVector<mlir::Value> unrollDistributedBuff(mlir::OpBuilder builder, mlir::Value output) {
        auto distributedOutput = output.getType().dyn_cast<VPUIP::DistributedBufferType>();
        if (!distributedOutput) {
            return {output};
        }

        llvm::SmallVector<mlir::Value> results;
        auto distribution = distributedOutput.getDistribution();
        auto outputMode =
                static_cast<std::underlying_type<VPU::DistributionMode>::type>(distribution.mode().getValue());
        auto duplicatedMode =
                static_cast<std::underlying_type<VPU::DistributionMode>::type>(VPU::DistributionMode::DUPLICATED);
        auto multicastedMode =
                static_cast<std::underlying_type<VPU::DistributionMode>::type>(VPU::DistributionMode::MULTICASTED);
        if ((outputMode & duplicatedMode) || (outputMode & multicastedMode)) {
            auto definingOp = mlir::cast<VPURT::DeclareBufferOp>(output.getDefiningOp());

            auto compactType = distributedOutput.getCompactType();

            auto totalClusters = distribution.num_clusters().getInt();

            auto byteOffset = definingOp.byteOffset();
            auto swizzlingKey = definingOp.swizzlingKey();
            auto buffSec = definingOp.getMemorySpace();

            for (int64_t cluster = 0; cluster < totalClusters; cluster++) {
                VPURT::DeclareBufferOp res;

                auto currMemLocation = compactType.getMemorySpace().cast<IndexedSymbolAttr>().getLeafNameAttr();
                auto newMemSpace = vpux::IndexedSymbolAttr::get(currMemLocation, static_cast<size_t>(cluster));
                auto memType = mlir::MemRefType::get(compactType.getShape(), compactType.getElementType(),
                                                     compactType.getLayout(), newMemSpace);
                if (swizzlingKey.hasValue()) {
                    res = builder.create<VPURT::DeclareBufferOp>(output.getLoc(), memType, buffSec, cluster, byteOffset,
                                                                 swizzlingKey.getValue());
                } else {
                    res = builder.create<VPURT::DeclareBufferOp>(output.getLoc(), memType, buffSec, cluster,
                                                                 byteOffset);
                }

                results.push_back(res.getResult());
            }
        } else {
            VPUX_THROW("Only distributed buffer with DUPLICATE is accepted as direct output of OP");
        }

        return results;
    }

    template <typename DMAType, typename CreatorFunc>
    void lowerDMA(CreatorFunc&& creator, VPURT::TaskOp taskOp, mlir::SmallVector<mlir::Value>& previousDMA,
                  mlir::SmallVector<int64_t>& dmaCount, bool& found) {
        for (auto op : llvm::make_early_inc_range(taskOp.body().getOps<DMAType>())) {
            found = true;
            mlir::OpBuilder builderBlk(taskOp);
            auto port = op.port();
            auto indexType = VPUIPRegMapped::IndexType::get(taskOp.getContext(), dmaCount[port]);

            auto waitBarriers = taskOp.waitBarriers();
            auto updateBarriers = taskOp.updateBarriers();

            auto trivialIndexType = VPUIPRegMapped::IndexType::get(taskOp.getContext(), 0);

            for (auto val : waitBarriers) {
                val.setType(trivialIndexType);
            }

            for (auto val : updateBarriers) {
                val.setType(trivialIndexType);
            }

            previousDMA[port] = creator(op, unrollDistributedBuff(builderBlk, op.output_buff()), previousDMA[port],
                                        indexType, waitBarriers, updateBarriers)
                                        ->getResult(0);

            ++dmaCount[port];
        }
    }

    void replaceVPURTTaskOpWithNNDMAOp(mlir::MLIRContext*, mlir::ModuleOp& moduleOp, mlir::FuncOp& funcOp,
                                       Logger _log) {
        _log.info("VPUIP_VPUIPRegMapped pass: replaceVPURTTaskOpWithNNDMAOp()");

        const auto dmaExecCount = IE::getAvailableExecutor(moduleOp, VPU::ExecutorKind::DMA_NN).count();

        llvm::SmallVector<mlir::Value> previousDMA(dmaExecCount, mlir::Value());
        mlir::SmallVector<int64_t> dmaCount(dmaExecCount, 0);

        for (auto taskOp : llvm::make_early_inc_range(funcOp.body().getOps<VPURT::TaskOp>())) {
            bool found = false;
            mlir::OpBuilder builderBlk(taskOp);

            lowerDMA<VPUIP::NNDMAOp>(
                    [&builderBlk, this](VPUIP::NNDMAOp dmaOp, llvm::SmallVector<mlir::Value> dmaResults,
                                        mlir::Value previousDMA, VPUIPRegMapped::IndexType indexType,
                                        mlir::ValueRange waitBarriers, mlir::ValueRange updateBarriers) {
                        return builderBlk.create<VPUIPRegMapped::NNDMAOp>(
                                builderBlk.getUnknownLoc(), indexType, dmaOp.input(), dmaResults, previousDMA,
                                mlir::ValueRange(waitBarriers), mlir::ValueRange(updateBarriers), false, 0, 0,
                                dmaOp.is_out_of_order(), dmaOp.is_critical(), dmaOp.port(), nullptr);
                    },
                    taskOp, previousDMA, dmaCount, found);

            if (found) {
                taskOp->erase();
                continue;
            }

            lowerDMA<VPUIP::PermuteDMAOp>(
                    [&builderBlk, this](VPUIP::PermuteDMAOp permuteDMAOp, llvm::SmallVector<mlir::Value> dmaResults,
                                        mlir::Value previousDMA, VPUIPRegMapped::IndexType indexType,
                                        mlir::ValueRange waitBarriers, mlir::ValueRange updateBarriers) {
                        const auto dataShape = getShape(permuteDMAOp.input());
                        VPUX_THROW_UNLESS(dataShape.size() == 2 || dataShape.size() == 3,
                                          "DMA op shape size should be 2 or 3. but got shape {0}", dataShape);

                        const auto dmaDescriptor = permuteDMAOp.dma_descriptor();
                        VPUX_THROW_UNLESS(dmaDescriptor.hasValue(), "DMA descriptor attr not found at '{0}'",
                                          permuteDMAOp->getLoc());
                        const auto dmaDescriptorValue = dmaDescriptor.getValue();

                        const auto numPlanes = checked_cast<uint32_t>(dmaDescriptorValue.numPlanes().getInt());
                        VPUX_THROW_UNLESS(numPlanes <= VPUIP::DMA_MAX_NUMBER_PLANES,
                                          "NUM PLANES should be less than or equal to {0}, but got {1}.",
                                          VPUIP::DMA_MAX_NUMBER_PLANES, numPlanes);

                        return builderBlk.create<VPUIPRegMapped::NNDMAOp>(
                                builderBlk.getUnknownLoc(), indexType, permuteDMAOp.input(), dmaResults, previousDMA,
                                mlir::ValueRange(waitBarriers), mlir::ValueRange(updateBarriers), false, 0, 0,
                                permuteDMAOp.is_out_of_order(), permuteDMAOp.is_critical(), permuteDMAOp.port(),
                                dmaDescriptorValue);
                    },
                    taskOp, previousDMA, dmaCount, found);

            if (found) {
                taskOp->erase();
                continue;
            }

            lowerDMA<VPUIP::UpsamplingDMAOp>(
                    [&builderBlk, this](VPUIP::UpsamplingDMAOp upsamplingDMAOp,
                                        llvm::SmallVector<mlir::Value> dmaResults, mlir::Value previousDMA,
                                        VPUIPRegMapped::IndexType indexType, mlir::ValueRange waitBarriers,
                                        mlir::ValueRange updateBarriers) {
                        const auto dmaDescriptor = upsamplingDMAOp.dma_descriptor();
                        VPUX_THROW_UNLESS(dmaDescriptor.hasValue(), "DMA descriptor attr not found at '{0}'",
                                          upsamplingDMAOp->getLoc());
                        const auto dmaDescriptorValue = dmaDescriptor.getValue();

                        const auto numPlanes = checked_cast<uint32_t>(dmaDescriptorValue.numPlanes().getInt());
                        VPUX_THROW_UNLESS(numPlanes <= VPUIP::DMA_MAX_NUMBER_PLANES,
                                          "NUM PLANES should be less than or equal to {0}, but got {1}.",
                                          VPUIP::DMA_MAX_NUMBER_PLANES, numPlanes);

                        return builderBlk.create<VPUIPRegMapped::NNDMAOp>(
                                builderBlk.getUnknownLoc(), indexType, upsamplingDMAOp.input(), dmaResults, previousDMA,
                                mlir::ValueRange(waitBarriers), mlir::ValueRange(updateBarriers), false, 0, 0, false,
                                true, upsamplingDMAOp.port(), dmaDescriptorValue);
                    },
                    taskOp, previousDMA, dmaCount, found);

            if (found) {
                taskOp->erase();
                continue;
            }

            lowerDMA<VPUIP::PerAxisTileDMAOp>(
                    [&builderBlk, this](VPUIP::PerAxisTileDMAOp perAxisTileDMAOp,
                                        llvm::SmallVector<mlir::Value> dmaResults, mlir::Value previousDMA,
                                        VPUIPRegMapped::IndexType indexType, mlir::ValueRange waitBarriers,
                                        mlir::ValueRange updateBarriers) {
                        const auto dmaDescriptor = perAxisTileDMAOp.dma_descriptor();
                        VPUX_THROW_UNLESS(dmaDescriptor.hasValue(), "DMA descriptor attr not found at '{0}'",
                                          perAxisTileDMAOp->getLoc());
                        const auto dmaDescriptorValue = dmaDescriptor.getValue();

                        const auto numPlanes = checked_cast<uint32_t>(dmaDescriptorValue.numPlanes().getInt());
                        VPUX_THROW_UNLESS(numPlanes <= VPUIP::DMA_MAX_NUMBER_PLANES,
                                          "NUM PLANES should be less than or equal to {0}, but got {1}.",
                                          VPUIP::DMA_MAX_NUMBER_PLANES, numPlanes);

                        return builderBlk.create<VPUIPRegMapped::NNDMAOp>(
                                builderBlk.getUnknownLoc(), indexType, perAxisTileDMAOp.input(), dmaResults,
                                previousDMA, mlir::ValueRange(waitBarriers), mlir::ValueRange(updateBarriers), false, 0,
                                0, false, true, perAxisTileDMAOp.port(), dmaDescriptorValue);
                    },
                    taskOp, previousDMA, dmaCount, found);

            if (found) {
                taskOp->erase();
                continue;
            }

            lowerDMA<VPUIP::CompressedDMAOp>(
                    [&builderBlk, this](VPUIP::CompressedDMAOp compressedDMAOp,
                                        llvm::SmallVector<mlir::Value> dmaResults, mlir::Value previousDMA,
                                        VPUIPRegMapped::IndexType indexType, mlir::ValueRange waitBarriers,
                                        mlir::ValueRange updateBarriers) {
                        return builderBlk.create<VPUIPRegMapped::NNDMAOp>(
                                builderBlk.getUnknownLoc(), indexType, compressedDMAOp.input(), dmaResults, previousDMA,
                                mlir::ValueRange(waitBarriers), mlir::ValueRange(updateBarriers), true, 0, 0,
                                compressedDMAOp.is_out_of_order(), compressedDMAOp.is_critical(),
                                compressedDMAOp.port(), nullptr);
                    },
                    taskOp, previousDMA, dmaCount, found);

            if (found) {
                taskOp->erase();
                continue;
            }

            lowerDMA<VPUIP::ExpandDMAOp>(
                    [&builderBlk, this](VPUIP::ExpandDMAOp expandDMAOp, llvm::SmallVector<mlir::Value> dmaResults,
                                        mlir::Value previousDMA, VPUIPRegMapped::IndexType indexType,
                                        mlir::ValueRange waitBarriers, mlir::ValueRange updateBarriers) {
                        return builderBlk.create<VPUIPRegMapped::NNDMAOp>(
                                builderBlk.getUnknownLoc(), indexType, expandDMAOp.input(), dmaResults, previousDMA,
                                mlir::ValueRange(waitBarriers), mlir::ValueRange(updateBarriers), false, 0, 0,
                                expandDMAOp.is_out_of_order(), expandDMAOp.is_critical(), expandDMAOp.port(),
                                expandDMAOp.dma_descriptor().getValue());
                    },
                    taskOp, previousDMA, dmaCount, found);

            if (found) {
                taskOp->erase();
                continue;
            }

            lowerDMA<VPUIP::SpaceToDepthDMAOp>(
                    [&builderBlk, this](VPUIP::SpaceToDepthDMAOp spaceToDepthDMAOp,
                                        llvm::SmallVector<mlir::Value> dmaResults, mlir::Value previousDMA,
                                        VPUIPRegMapped::IndexType indexType, mlir::ValueRange waitBarriers,
                                        mlir::ValueRange updateBarriers) {
                        return builderBlk.create<VPUIPRegMapped::NNDMAOp>(
                                builderBlk.getUnknownLoc(), indexType, spaceToDepthDMAOp.input(), dmaResults,
                                previousDMA, mlir::ValueRange(waitBarriers), mlir::ValueRange(updateBarriers), false, 0,
                                0, false, true, spaceToDepthDMAOp.port(),
                                spaceToDepthDMAOp.dma_descriptor().getValue());
                    },
                    taskOp, previousDMA, dmaCount, found);

            if (found) {
                taskOp->erase();
                continue;
            }

            lowerDMA<VPUIP::DepthToSpaceDMAOp>(
                    [&builderBlk, this](VPUIP::DepthToSpaceDMAOp depthToSpaceDMAOp,
                                        llvm::SmallVector<mlir::Value> dmaResults, mlir::Value previousDMA,
                                        VPUIPRegMapped::IndexType indexType, mlir::ValueRange waitBarriers,
                                        mlir::ValueRange updateBarriers) {
                        const auto inOrder = DimsOrder::fromValue(depthToSpaceDMAOp.input());
                        const auto outOrder = DimsOrder::fromValue(depthToSpaceDMAOp.output_buff());
                        auto isLegalType = (inOrder == DimsOrder::NHWC && outOrder == DimsOrder::NHWC);
                        VPUX_THROW_UNLESS(isLegalType, "DepthToSpaceDMAOp just support NHWC (NCHW TODO), but got {0}.",
                                          inOrder);

                        const auto dmaDescriptor = depthToSpaceDMAOp.dma_descriptor();
                        VPUX_THROW_UNLESS(dmaDescriptor.hasValue(), "DMA descriptor attr not found at '{0}'",
                                          depthToSpaceDMAOp->getLoc());
                        const auto dmaDescriptorValue = dmaDescriptor.getValue();

                        const auto numPlanes = checked_cast<uint32_t>(dmaDescriptorValue.numPlanes().getInt());
                        VPUX_THROW_UNLESS(numPlanes <= VPUIP::DMA_MAX_NUMBER_PLANES,
                                          "NUM PLANES should be less than or equal to {0}, but got {1}.",
                                          VPUIP::DMA_MAX_NUMBER_PLANES, numPlanes);

                        return builderBlk.create<VPUIPRegMapped::NNDMAOp>(
                                builderBlk.getUnknownLoc(), indexType, depthToSpaceDMAOp.input(), dmaResults,
                                previousDMA, mlir::ValueRange(waitBarriers), mlir::ValueRange(updateBarriers), false, 0,
                                0, false, true, depthToSpaceDMAOp.port(), dmaDescriptorValue);
                    },
                    taskOp, previousDMA, dmaCount, found);

            if (found) {
                taskOp->erase();
            }
        }

        _log.info("VPUIP_VPUIPRegMapped pass: replaceVPURTTaskOpWithNNDMAOp() -- end");
    }

    void replaceVPURTTaskOpWithKernelOps(mlir::MLIRContext* ctx, mlir::ModuleOp moduleOp, mlir::FuncOp funcOp,
                                         Logger _log) {
        _log.info("VPUIP_VPUIPRegMapped pass: replaceVPURTTaskOpWithKernelOps()");

        auto shave_task_count = 0;

        // Forever loop that runs until there are no more changes performed by
        //   the inner loop (so the algorithm has converged).

        for (auto taskOp : llvm::make_early_inc_range(funcOp.body().getOps<VPURT::TaskOp>())) {
            bool found = false;

            for (auto op : llvm::make_early_inc_range(taskOp.body().getOps<VPUIP::SwKernelOp>())) {
                found = true;
                mlir::OpBuilder builderBlk(taskOp);

                auto indexType = VPUIPRegMapped::IndexType::get(ctx, shave_task_count);

                auto wait_bars = taskOp.waitBarriers();
                auto update_bars = taskOp.updateBarriers();

                auto trivialIndexType = VPUIPRegMapped::IndexType::get(ctx, 0);

                for (auto val : wait_bars) {
                    val.setType(trivialIndexType);
                }

                for (auto val : update_bars) {
                    val.setType(trivialIndexType);
                }

                auto sw_kernel_symbol = op.kernelFunction();

                auto kernel_info_funcOp = moduleOp.lookupSymbol<mlir::FuncOp>(sw_kernel_symbol);

                auto kernel_elf =
                        std::string(kernel_info_funcOp->getAttrOfType<mlir::StringAttr>("VPU.kernel_entry").getValue());

                auto uint8Type = mlir::IntegerType::get(ctx, 8, mlir::IntegerType::SignednessSemantics::Unsigned);

                SmallVector<uint8_t> paramsVector =
                        vpux::VPUIPRegMapped::KernelParamsSerializer::createKernelParams(op);

                long int paramsSize = (long int)(paramsVector.size());

                auto kernelTextOp = builderBlk.create<VPUIPRegMapped::DeclareKernelTextOp>(
                        builderBlk.getUnknownLoc(), indexType, mlir::StringAttr::get(ctx, kernel_elf));

                auto kernelArgsOp = builderBlk.create<VPUIPRegMapped::DeclareKernelArgsOp>(
                        builderBlk.getUnknownLoc(), indexType, mlir::StringAttr::get(ctx, kernel_elf));

                auto kernelEntryOp = builderBlk.create<VPUIPRegMapped::DeclareKernelEntryOp>(
                        builderBlk.getUnknownLoc(), indexType, mlir::StringAttr::get(ctx, kernel_elf));

                auto kernelRangeOp = builderBlk.create<VPUIPRegMapped::ActKernelRangeOp>(
                        builderBlk.getUnknownLoc(), indexType, kernelTextOp, kernelArgsOp, kernelEntryOp);

                auto tileIndex = op.tileIndex().getValueOr(0);

                builderBlk.create<VPUIPRegMapped::ActKernelInvocationOp>(
                        builderBlk.getUnknownLoc(), indexType, mlir::ValueRange(wait_bars),
                        mlir::ValueRange(update_bars), kernelRangeOp.getResult(), /* tile= */ tileIndex,
                        /* start_after= */ 0, /* clean_after= */ 0);

                builderBlk.create<VPUIPRegMapped::KernelParamsOp>(
                        builderBlk.getUnknownLoc(), indexType, op.inputs(), op.output_buffs(),
                        mlir::StringAttr::get(ctx, kernel_elf),
                        mlir::DenseIntElementsAttr::get(mlir::VectorType::get({paramsSize}, uint8Type), paramsVector));

                shave_task_count++;
            }

            if (found) {
                taskOp->erase();
            }
        }
    }

    void replaceNCEClusterTaskOpWithDPUOps(mlir::MLIRContext* ctx, mlir::FuncOp funcOp, Logger _log) {
        int variant_task_count = 0;
        int invariant_task_count = 0;

        for (auto taskOp : llvm::make_early_inc_range(funcOp.getOps<VPURT::TaskOp>())) {
            bool found = false;

            _log.trace("replaceNCEClusterTaskOpWithDPUOps(): taskOp = {0}", taskOp);

            for (auto op : llvm::make_early_inc_range(taskOp.body().getOps<VPUIP::NCEClusterTaskOp>())) {
                found = true;
                mlir::OpBuilder builderBlk(taskOp);

                auto wait_barriers = taskOp.waitBarriers();
                auto update_barriers = taskOp.updateBarriers();

                auto trivialIndexType = VPUIPRegMapped::IndexType::get(ctx, 0);

                for (auto val : wait_barriers) {
                    val.setType(trivialIndexType);
                }

                for (auto val : update_barriers) {
                    val.setType(trivialIndexType);
                }

                const auto& dpuTasks = op.variants().getOps<VPUIP::DPUTaskOp>();
                VPUX_THROW_UNLESS(!dpuTasks.empty(), "Encountered op {} with empty dpu list", op);
                const auto& differentMPEModes = std::adjacent_find(dpuTasks.begin(), dpuTasks.end(),
                                                                   [](VPUIP::DPUTaskOp lhs, VPUIP::DPUTaskOp rhs) {
                                                                       return lhs.mpe_mode() != rhs.mpe_mode();
                                                                   });
                if (differentMPEModes != dpuTasks.end()) {
                    VPUIP::DPUTaskOp lhs = *differentMPEModes;
                    VPUIP::DPUTaskOp rhs = *std::next(differentMPEModes);
                    VPUX_THROW("Found dpu tasks {} and {} inside of {} which has different MPE modes {} and {} "
                               "accordingly, but only uniform MPE mode is supported by ELF",
                               lhs, rhs, op, lhs.mpe_mode(), rhs.mpe_mode());
                }

                VPUIP::DPUTaskOp first = *(dpuTasks.begin());
                auto mpe_freq_mode = VPU::MPEModeAttr::get(ctx, first.mpe_mode());
                auto invariantIndex = VPUIPRegMapped::IndexType::get(ctx, invariant_task_count);
                auto startAfterAttr = builderBlk.getIntegerAttr(builderBlk.getIntegerType(64, false), 0);
                auto cleanAfterAttr = builderBlk.getIntegerAttr(builderBlk.getIntegerType(64, false), 0);

                auto dpuResults = unrollDistributedBuff(builderBlk, op.output_buff());
                auto inv = builderBlk.create<VPUIPRegMapped::DPUInvariantOp>(
                        builderBlk.getUnknownLoc(), invariantIndex, op.input(), op.input_sparsity_map(),
                        op.input_storage_element_table(), op.weights(), op.weights_sparsity_map(), op.weight_table(),
                        op.parent_input(), op.parent_input_sparsity_map(), op.parent_input_storage_element_table(),
                        op.parent_output(), op.parent_output_sparsity_map(), dpuResults, op.output_sparsity_map_buff(),
                        op.profiling_data(), op.task_typeAttr(), mpe_freq_mode, op.kernel_sizeAttr(),
                        op.kernel_stridesAttr(), op.kernel_paddingAttr(), op.activation_window_channel_lengthAttr(),
                        op.is_continuedAttr(), op.cm_sp_patternAttr(), op.input_channels_compressionAttr(),
                        op.is_segmentedAttr(), op.out_channel_offsetAttr(), op.is_superdenseAttr(),
                        op.input_se_sizeAttr(), op.output_se_sizeAttr(), wait_barriers, update_barriers, startAfterAttr,
                        cleanAfterAttr);

                invariant_task_count++;

                for (auto dpuTaskOp : op.variants().getOps<VPUIP::DPUTaskOp>()) {
                    auto variantIndex = VPUIPRegMapped::IndexType::get(ctx, variant_task_count);
                    builderBlk.create<VPUIPRegMapped::DPUVariantOp>(
                            builderBlk.getUnknownLoc(), variantIndex, inv.getResult(), dpuTaskOp.outStartAttr(),
                            dpuTaskOp.outEndAttr(), dpuTaskOp.padAttr(), dpuTaskOp.mpe_modeAttr(),
                            dpuTaskOp.cluster_idAttr());
                    variant_task_count++;
                }

                if (op.ppe().hasOneBlock()) {
                    mlir::BlockAndValueMapping mapper;
                    op.ppe().cloneInto(&inv.ppe(), mapper);
                }
            }

            if (found) {
                taskOp->erase();
            }
        }
    }

    void setBarrierIndexValues(mlir::MLIRContext* ctx, mlir::FuncOp& funcOp, Logger _log) {
        auto barrier_count = 0;

        VPUX_UNUSED(_log);

        for (auto op : funcOp.getOps<VPUIPRegMapped::ConfigureBarrierOp>()) {
            auto indexType = VPUIPRegMapped::IndexType::get(ctx, barrier_count);

            op.getOperation()->getResult(0).setType(indexType);

            barrier_count++;
        }
    }

    template <typename TaskType>
    static bool noCond(TaskType i) {
        VPUX_UNUSED(i);
        return true;
    }

    template <typename TaskType, typename Condition = decltype(noCond<TaskType>)>
    size_t countTasksIf(mlir::FuncOp& funcOp, Condition&& condition = noCond) {
        auto tasks = funcOp.template getOps<TaskType>();
        return std::count_if(tasks.begin(), tasks.end(), std::forward<Condition>(condition));
    }

    template <typename TaskType, typename Condition = decltype(noCond<TaskType>)>
    mlir::Value findTaskIf(mlir::FuncOp& funcOp, Condition&& condition = noCond) {
        auto tasks = funcOp.template getOps<TaskType>();
        auto target = std::find_if(tasks.begin(), tasks.end(), std::forward<Condition>(condition));
        return target != tasks.end() ? (*target).getResult() : mlir::Value();
    }

    void createMappedInferenceOp(mlir::MLIRContext* ctx, mlir::ModuleOp& moduleOp, mlir::FuncOp& funcOp, Logger _log) {
        _log.info("VPUIP_VPUIPRegMapped pass: createMappedInferenceOp()");

        const auto dmaExecCount = IE::getAvailableExecutor(moduleOp, VPU::ExecutorKind::DMA_NN).count();

        mlir::SmallVector<mlir::Value> dmaTasks;
        mlir::Value invariantTasks;
        mlir::Value variantTasks;
        mlir::Value actKernelInvocations;
        mlir::Value actKernelRanges;
        mlir::Value barrierTasks;
        mlir::Value actShvRt;
        mlir::ValueRange actShaveStacks;

        mlir::SmallVector<int64_t> dmaCount(dmaExecCount, 0);
        int64_t barrierCount = 0;
        int64_t rangeCount = 0;
        int64_t invoCount = 0;
        int64_t invariantCount = 0;
        int64_t variantCount = 0;

        for (size_t listIdx = 0; listIdx < dmaCount.size(); ++listIdx) {
            auto dmaPortCond = [listIdx](auto op) {
                return op.port() == static_cast<int64_t>(listIdx);
            };
            dmaCount[listIdx] = countTasksIf<VPUIPRegMapped::NNDMAOp>(funcOp, dmaPortCond);
            auto dmaListHead = findTaskIf<VPUIPRegMapped::NNDMAOp>(funcOp, dmaPortCond);
            if (dmaListHead) {
                dmaTasks.push_back(dmaListHead);
            }
        }

        barrierCount = countTasksIf<VPUIPRegMapped::ConfigureBarrierOp>(funcOp);
        barrierTasks = findTaskIf<VPUIPRegMapped::ConfigureBarrierOp>(funcOp);
        rangeCount = countTasksIf<VPUIPRegMapped::ActKernelRangeOp>(funcOp);
        actKernelRanges = findTaskIf<VPUIPRegMapped::ActKernelRangeOp>(funcOp);
        invoCount = countTasksIf<VPUIPRegMapped::ActKernelInvocationOp>(funcOp);
        actKernelInvocations = findTaskIf<VPUIPRegMapped::ActKernelInvocationOp>(funcOp);
        invariantCount = countTasksIf<VPUIPRegMapped::DPUInvariantOp>(funcOp);
        invariantTasks = findTaskIf<VPUIPRegMapped::DPUInvariantOp>(funcOp);
        variantCount = countTasksIf<VPUIPRegMapped::DPUVariantOp>(funcOp);
        variantTasks = findTaskIf<VPUIPRegMapped::DPUVariantOp>(funcOp);

        auto trivialIndexType = VPUIPRegMapped::IndexType::get(ctx, 0);

        // create MappedInferenceOp
        mlir::OpBuilder builderFunc(&(funcOp.getBody().front().back()));

        builderFunc.create<VPUIPRegMapped::MappedInferenceOp>(
                mlir::UnknownLoc::get(ctx), trivialIndexType,
                dmaTasks,                                             // mlir::Value dmaTasks
                invariantTasks,                                       // mlir::Value invariantTasks
                variantTasks,                                         // mlir::Value variantTasks
                actKernelRanges,                                      // mlir::Value actKernelRanges
                actKernelInvocations,                                 // mlir::Value actKernelInvocations
                barrierTasks,                                         // mlir::Value barrierTasks
                actShvRt,                                             // mlir::Value actShaveRt
                actShaveStacks,                                       // mlir::ValueRange actShaveStacks
                builderFunc.getI64ArrayAttr(makeArrayRef(dmaCount)),  // mlir::ArrayAttr
                invariantCount,                                       // uint32_t invariantCount
                variantCount,                                         // uint32_t variantCount
                rangeCount,                                           // uint32_t rangeCount
                invoCount,                                            // uint32_t invoCount
                barrierCount                                          // uint32_t barrierCount
        );
    }
};  // namespace

class ConvertVPURTConfigureBarrierOp final : public mlir::OpRewritePattern<VPURT::ConfigureBarrierOp> {
public:
    ConvertVPURTConfigureBarrierOp(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPURT::ConfigureBarrierOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(VPURT::ConfigureBarrierOp origOp, mlir::PatternRewriter& rewriter) const {
        auto ctx = ConvertVPURTConfigureBarrierOp::getContext();

        auto trivialIndexType = VPUIPRegMapped::IndexType::get(ctx, 0);

        mlir::Value origOpResult = origOp.getResult();

        size_t producer_count = 0;
        size_t consumer_count = 0;

        // should use VPUIPRegMapped TaskOp interface
        for (auto user : origOpResult.getUsers()) {
            if (auto taskOp = mlir::dyn_cast<vpux::VPUIPRegMapped::ExecutableTaskOpInterface>(user)) {
                for (auto waitBar : taskOp.waitBarriers()) {
                    if (origOpResult == waitBar) {
                        consumer_count += taskOp.getBarrierHitsCount();
                    }
                }

                for (auto updateBar : taskOp.updateBarriers()) {
                    if (origOpResult == updateBar) {
                        producer_count += taskOp.getBarrierHitsCount();
                    }
                }
            }
        }

        mlir::IntegerType uint8Type = mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Unsigned);

        rewriter.replaceOpWithNewOp<VPUIPRegMapped::ConfigureBarrierOp>(
                origOp,
                trivialIndexType,                                   // setup all barriers with the trivial index (0)
                origOp.id(),                                        // real_id
                -1,                                                 // int32_t next_same_id()
                mlir::IntegerAttr::get(uint8Type, producer_count),  // origOp.producer_countAttr(),
                mlir::IntegerAttr::get(uint8Type, consumer_count)   // origOp.consumer_countAttr(),
        );
        barrier_count++;
        return mlir::success();
    }

private:
    Logger _log;
    mutable int barrier_count = 0;
};

void ConvertVPUIP2VPUIPRegMappedPass::safeRunOnModule() {
    auto ctx = &(getContext());
    auto moduleOp = getOperation();
    auto funcOp = *moduleOp.getOps<mlir::FuncOp>().begin();

    _log.trace("funcOp = {0}", funcOp);

    replaceVPURTTaskOpWithNNDMAOp(ctx, moduleOp, funcOp, _log);

    _log.trace("funcOp after replacing NNDMA Ops = {0}", funcOp);

    replaceVPURTTaskOpWithKernelOps(ctx, moduleOp, funcOp, _log);

    _log.trace("funcOp after replacing ActKernel Ops = {0}", funcOp);

    replaceNCEClusterTaskOpWithDPUOps(ctx, funcOp, _log);

    _log.trace("funcOp after replacing DPU Ops = {0}", funcOp);

    mlir::ConversionTarget target(*ctx);
    target.addLegalDialect<VPUIPRegMapped::VPUIPRegMappedDialect>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();
    target.addLegalOp<VPURT::DeclareBufferOp>();
    target.addLegalOp<VPUIP::PPETaskOp>();
    target.addLegalOp<VPUIP::GroupSparseBufferOp>();

    mlir::RewritePatternSet patterns(ctx);

    patterns.add<ConvertVPURTConfigureBarrierOp>(ctx, _log);

    if (mlir::failed(mlir::applyFullConversion(funcOp, target, std::move(patterns)))) {
        signalPassFailure();
    }

    _log.trace("funcOp after replacing Barrier Ops = {0}", funcOp);

    setBarrierIndexValues(ctx, funcOp, _log);

    _log.trace("funcOp after setting Barrier indexes = {0}", funcOp);

    createMappedInferenceOp(ctx, moduleOp, funcOp, _log);

    _log.trace("funcOp after generating MappedInferenceOp = {0}", funcOp);
}

}  // namespace

//
// createConvertVPUIP2VPUIPRegMappedPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertVPUIP2VPUIPRegMappedPass(Logger log) {
    return std::make_unique<ConvertVPUIP2VPUIPRegMappedPass>(log);
}
