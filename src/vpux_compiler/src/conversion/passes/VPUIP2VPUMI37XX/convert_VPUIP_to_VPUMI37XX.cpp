//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/core/bounded_buffer.hpp"
#include "vpux/compiler/core/profiling_metadata.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/kernel_params_utils.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/dma_limits.hpp"
#include "vpux/compiler/utils/llvm_to_binary.hpp"

#include <llvm/Support/FileSystem.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <vector>

namespace vpux {
#define GEN_PASS_DECL_CONVERTVPUIP2VPUMI37XX
#define GEN_PASS_DEF_CONVERTVPUIP2VPUMI37XX
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

namespace {

//
// ConvertVPUIP2VPUMI37XXPass
//

class ConvertVPUIP2VPUMI37XXPass final : public impl::ConvertVPUIP2VPUMI37XXBase<ConvertVPUIP2VPUMI37XXPass> {
public:
    explicit ConvertVPUIP2VPUMI37XXPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;

    llvm::SmallVector<mlir::Value> unrollDistributedBuff(mlir::OpBuilder builder, mlir::Value output) {
        auto distributedOutput = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(output.getType());
        if (!distributedOutput) {
            return {output};
        }

        llvm::SmallVector<mlir::Value> results;
        auto distribution = distributedOutput.getDistribution();
        auto outputMode =
                static_cast<std::underlying_type<VPU::DistributionMode>::type>(distribution.getMode().getValue());
        auto duplicatedMode =
                static_cast<std::underlying_type<VPU::DistributionMode>::type>(VPU::DistributionMode::DUPLICATED);
        auto multicastedMode =
                static_cast<std::underlying_type<VPU::DistributionMode>::type>(VPU::DistributionMode::MULTICASTED);
        if ((outputMode & duplicatedMode) || (outputMode & multicastedMode)) {
            auto definingOp = mlir::cast<VPURT::DeclareBufferOp>(output.getDefiningOp());

            auto compactType = distributedOutput.getCompactType();

            auto totalClusters = distribution.getNumClusters().getInt();

            auto byteOffset = definingOp.getByteOffset();
            auto swizzlingKey = definingOp.getSwizzlingKey();
            auto buffSec = definingOp.getMemorySpace();

            for (int64_t cluster = 0; cluster < totalClusters; cluster++) {
                VPURT::DeclareBufferOp res;

                auto currMemLocation =
                        mlir::cast<vpux::IndexedSymbolAttr>(compactType.getMemorySpace()).getLeafNameAttr();
                auto newMemSpace = vpux::IndexedSymbolAttr::get(currMemLocation, static_cast<size_t>(cluster));
                auto memType = mlir::MemRefType::get(compactType.getShape(), compactType.getElementType(),
                                                     compactType.getLayout(), newMemSpace);
                if (swizzlingKey.has_value()) {
                    res = builder.create<VPURT::DeclareBufferOp>(output.getLoc(), memType, buffSec, cluster, byteOffset,
                                                                 swizzlingKey.value());
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
    void lowerDMA(CreatorFunc&& creator, VPURT::TaskOp taskOp,
                  mlir::SmallVector<std::optional<VPUMI37XX::NNDMAOp>>& previousDMAs,
                  mlir::SmallVector<int64_t>& dmaCount, bool& found) {
        for (auto op : llvm::make_early_inc_range(taskOp.getBody().getOps<DMAType>())) {
            found = true;
            mlir::OpBuilder builderBlk(taskOp);

            const auto portValue = [op]() mutable {
                const auto port = op.getPort();
                VPUX_THROW_UNLESS(port.has_value(), "DMA port has not been set");
                return port.value();
            }();

            auto& previousDMA = previousDMAs[portValue];
            const auto indexType = VPURegMapped::IndexType::get(taskOp.getContext(), portValue, dmaCount[portValue]);

            auto waitBarriers = taskOp.getWaitBarriers();
            auto updateBarriers = taskOp.getUpdateBarriers();

            auto trivialIndexType = VPURegMapped::IndexType::get(taskOp.getContext(), 0);

            for (auto val : waitBarriers) {
                val.setType(trivialIndexType);
            }

            for (auto val : updateBarriers) {
                val.setType(trivialIndexType);
            }

            auto currentDMA = creator(op, unrollDistributedBuff(builderBlk, op.getOutputBuff()), indexType,
                                      waitBarriers, updateBarriers);
            if (previousDMA.has_value()) {
                previousDMA.value().getNextDMAIdxMutable().assign(currentDMA.getIndex());
            }

            previousDMA = currentDMA;

            ++dmaCount[portValue];
        }
    }

    void replaceVPURTTaskOpWithNNDMAOp(mlir::MLIRContext*, mlir::ModuleOp& moduleOp, mlir::func::FuncOp& funcOp,
                                       Logger _log) {
        _log.trace("VPUIP_VPUMI37XX pass: replaceVPURTTaskOpWithNNDMAOp()");

        const auto dmaExecCount = config::getAvailableExecutor(moduleOp, VPU::ExecutorKind::DMA_NN).getCount();
        const auto& dmaEngineLimits = VPUIP::DMA::getEngineLimits(config::getArch(moduleOp));
        const auto dmaMaxNumPlanes = dmaEngineLimits.getMaxNumPlanes();

        llvm::SmallVector<std::optional<VPUMI37XX::NNDMAOp>> previousDMA(dmaExecCount);
        mlir::SmallVector<int64_t> dmaCount(dmaExecCount, 0);

        mlir::OpBuilder builderBlk(&(funcOp.getBody().front().back()));
        builderBlk.setInsertionPoint((*(funcOp.getBody().getOps<mlir::func::ReturnOp>().begin())));
        for (auto taskOp : llvm::make_early_inc_range(funcOp.getBody().getOps<VPURT::TaskOp>())) {
            bool found = false;

            lowerDMA<VPUIP::NNDMAOp>(
                    [&](VPUIP::NNDMAOp dmaOp, llvm::SmallVector<mlir::Value> dmaResults,
                        VPURegMapped::IndexType indexType, mlir::ValueRange waitBarriers,
                        mlir::ValueRange updateBarriers) {
                        auto operation = builderBlk.create<VPUMI37XX::NNDMAOp>(
                                dmaOp->getLoc(), indexType, dmaOp.getInput(), dmaResults, nullptr,
                                mlir::ValueRange(waitBarriers), mlir::ValueRange(updateBarriers), /* start_after= */ 0,
                                /* clean_after= */ 0, dmaOp.getIsOutOfOrder(), dmaOp.getIsCritical(),
                                dmaOp.getPort().value(), VPUIP::DMAAccMode::DISABLE, /* dma_descriptor= */ nullptr);
                        builderBlk.setInsertionPoint(operation);
                        return operation;
                    },
                    taskOp, previousDMA, dmaCount, found);

            if (found) {
                taskOp->erase();
                continue;
            }

            lowerDMA<VPUIP::PermuteDMAOp>(
                    [&](VPUIP::PermuteDMAOp permuteDMAOp, llvm::SmallVector<mlir::Value> dmaResults,
                        VPURegMapped::IndexType indexType, mlir::ValueRange waitBarriers,
                        mlir::ValueRange updateBarriers) {
                        const auto dataShape = getShape(permuteDMAOp.getInput());
                        VPUX_THROW_UNLESS(dataShape.size() == 2 || dataShape.size() == 3,
                                          "DMA op shape size should be 2 or 3. but got shape {0}", dataShape);

                        const auto dmaDescriptor = permuteDMAOp.getDmaDescriptor();
                        VPUX_THROW_UNLESS(dmaDescriptor.has_value(), "DMA descriptor attr not found at '{0}'",
                                          permuteDMAOp->getLoc());
                        const auto dmaDescriptorValue = dmaDescriptor.value();

                        const auto numPlanes = checked_cast<uint32_t>(dmaDescriptorValue.getNumPlanes().getInt());
                        VPUX_THROW_UNLESS(numPlanes <= dmaMaxNumPlanes,
                                          "NUM PLANES should be less than or equal to {0}, but got {1}.",
                                          dmaMaxNumPlanes, numPlanes);

                        auto operation = builderBlk.create<VPUMI37XX::NNDMAOp>(
                                permuteDMAOp->getLoc(), indexType, permuteDMAOp.getInput(), dmaResults, nullptr,
                                mlir::ValueRange(waitBarriers), mlir::ValueRange(updateBarriers), 0, 0,
                                permuteDMAOp.getIsOutOfOrder(), permuteDMAOp.getIsCritical(),
                                permuteDMAOp.getPort().value(), VPUIP::DMAAccMode::DISABLE, dmaDescriptorValue);
                        builderBlk.setInsertionPoint(operation);
                        return operation;
                    },
                    taskOp, previousDMA, dmaCount, found);

            if (found) {
                taskOp->erase();
                continue;
            }

            lowerDMA<VPUIP::UpsamplingDMAOp>(
                    [&](VPUIP::UpsamplingDMAOp upsamplingDMAOp, llvm::SmallVector<mlir::Value> dmaResults,
                        VPURegMapped::IndexType indexType, mlir::ValueRange waitBarriers,
                        mlir::ValueRange updateBarriers) {
                        const auto dmaDescriptor = upsamplingDMAOp.getDmaDescriptor();
                        VPUX_THROW_UNLESS(dmaDescriptor.has_value(), "DMA descriptor attr not found at '{0}'",
                                          upsamplingDMAOp->getLoc());
                        const auto dmaDescriptorValue = dmaDescriptor.value();

                        const auto numPlanes = checked_cast<uint32_t>(dmaDescriptorValue.getNumPlanes().getInt());
                        VPUX_THROW_UNLESS(numPlanes <= dmaMaxNumPlanes,
                                          "NUM PLANES should be less than or equal to {0}, but got {1}.",
                                          dmaMaxNumPlanes, numPlanes);

                        auto operation = builderBlk.create<VPUMI37XX::NNDMAOp>(
                                upsamplingDMAOp->getLoc(), indexType, upsamplingDMAOp.getInput(), dmaResults, nullptr,
                                mlir::ValueRange(waitBarriers), mlir::ValueRange(updateBarriers), 0, 0,
                                upsamplingDMAOp.getIsOutOfOrder(), upsamplingDMAOp.getIsCritical(),
                                upsamplingDMAOp.getPort().value(), VPUIP::DMAAccMode::DISABLE, dmaDescriptorValue);
                        builderBlk.setInsertionPoint(operation);
                        return operation;
                    },
                    taskOp, previousDMA, dmaCount, found);

            if (found) {
                taskOp->erase();
                continue;
            }

            lowerDMA<VPUIP::PerAxisTileDMAOp>(
                    [&](VPUIP::PerAxisTileDMAOp perAxisTileDMAOp, llvm::SmallVector<mlir::Value> dmaResults,
                        VPURegMapped::IndexType indexType, mlir::ValueRange waitBarriers,
                        mlir::ValueRange updateBarriers) {
                        const auto dmaDescriptor = perAxisTileDMAOp.getDmaDescriptor();
                        VPUX_THROW_UNLESS(dmaDescriptor.has_value(), "DMA descriptor attr not found at '{0}'",
                                          perAxisTileDMAOp->getLoc());
                        const auto dmaDescriptorValue = dmaDescriptor.value();

                        const auto numPlanes = checked_cast<uint32_t>(dmaDescriptorValue.getNumPlanes().getInt());
                        VPUX_THROW_UNLESS(numPlanes <= dmaMaxNumPlanes,
                                          "NUM PLANES should be less than or equal to {0}, but got {1}.",
                                          dmaMaxNumPlanes, numPlanes);

                        auto operation = builderBlk.create<VPUMI37XX::NNDMAOp>(
                                perAxisTileDMAOp->getLoc(), indexType, perAxisTileDMAOp.getInput(), dmaResults, nullptr,
                                mlir::ValueRange(waitBarriers), mlir::ValueRange(updateBarriers), 0, 0,
                                perAxisTileDMAOp.getIsOutOfOrder(), perAxisTileDMAOp.getIsCritical(),
                                perAxisTileDMAOp.getPort().value(), VPUIP::DMAAccMode::DISABLE, dmaDescriptorValue);
                        builderBlk.setInsertionPoint(operation);
                        return operation;
                    },
                    taskOp, previousDMA, dmaCount, found);

            if (found) {
                taskOp->erase();
                continue;
            }

            lowerDMA<VPUIP::DecompressDMAOp>(
                    [&](VPUIP::DecompressDMAOp decompressDMAOp, llvm::SmallVector<mlir::Value> dmaResults,
                        VPURegMapped::IndexType indexType, mlir::ValueRange waitBarriers,
                        mlir::ValueRange updateBarriers) {
                        auto operation = builderBlk.create<VPUMI37XX::NNDMAOp>(
                                decompressDMAOp->getLoc(), indexType, decompressDMAOp.getInput(), dmaResults, nullptr,
                                mlir::ValueRange(waitBarriers), mlir::ValueRange(updateBarriers), 0, 0,
                                decompressDMAOp.getIsOutOfOrder(), decompressDMAOp.getIsCritical(),
                                decompressDMAOp.getPort().value(), VPUIP::DMAAccMode::DECOMPRESSION, nullptr);
                        builderBlk.setInsertionPoint(operation);
                        return operation;
                    },
                    taskOp, previousDMA, dmaCount, found);

            if (found) {
                taskOp->erase();
                continue;
            }

            lowerDMA<VPUIP::ExpandDMAOp>(
                    [&](VPUIP::ExpandDMAOp expandDMAOp, llvm::SmallVector<mlir::Value> dmaResults,
                        VPURegMapped::IndexType indexType, mlir::ValueRange waitBarriers,
                        mlir::ValueRange updateBarriers) {
                        auto operation = builderBlk.create<VPUMI37XX::NNDMAOp>(
                                expandDMAOp->getLoc(), indexType, expandDMAOp.getInput(), dmaResults, nullptr,
                                mlir::ValueRange(waitBarriers), mlir::ValueRange(updateBarriers), 0, 0,
                                expandDMAOp.getIsOutOfOrder(), expandDMAOp.getIsCritical(),
                                expandDMAOp.getPort().value(), VPUIP::DMAAccMode::DISABLE,
                                expandDMAOp.getDmaDescriptor().value());
                        builderBlk.setInsertionPoint(operation);
                        return operation;
                    },
                    taskOp, previousDMA, dmaCount, found);

            if (found) {
                taskOp->erase();
                continue;
            }

            lowerDMA<VPUIP::SpaceToDepthDMAOp>(
                    [&](VPUIP::SpaceToDepthDMAOp spaceToDepthDMAOp, llvm::SmallVector<mlir::Value> dmaResults,
                        VPURegMapped::IndexType indexType, mlir::ValueRange waitBarriers,
                        mlir::ValueRange updateBarriers) {
                        auto operation = builderBlk.create<VPUMI37XX::NNDMAOp>(
                                spaceToDepthDMAOp->getLoc(), indexType, spaceToDepthDMAOp.getInput(), dmaResults,
                                nullptr, mlir::ValueRange(waitBarriers), mlir::ValueRange(updateBarriers), 0, 0,
                                spaceToDepthDMAOp.getIsOutOfOrder(), spaceToDepthDMAOp.getIsCritical(),
                                spaceToDepthDMAOp.getPort().value(), VPUIP::DMAAccMode::DISABLE,
                                spaceToDepthDMAOp.getDmaDescriptor().value());
                        builderBlk.setInsertionPoint(operation);
                        return operation;
                    },
                    taskOp, previousDMA, dmaCount, found);

            if (found) {
                taskOp->erase();
                continue;
            }

            lowerDMA<VPUIP::DepthToSpaceDMAOp>(
                    [&](VPUIP::DepthToSpaceDMAOp depthToSpaceDMAOp, llvm::SmallVector<mlir::Value> dmaResults,
                        VPURegMapped::IndexType indexType, mlir::ValueRange waitBarriers,
                        mlir::ValueRange updateBarriers) {
                        const auto inOrder = DimsOrder::fromValue(depthToSpaceDMAOp.getInput());
                        const auto outOrder = DimsOrder::fromValue(depthToSpaceDMAOp.getOutputBuff());
                        auto isLegalType = (inOrder == DimsOrder::NHWC && outOrder == DimsOrder::NHWC);
                        VPUX_THROW_UNLESS(isLegalType, "DepthToSpaceDMAOp just support NHWC (NCHW TODO), but got {0}.",
                                          inOrder);

                        const auto dmaDescriptor = depthToSpaceDMAOp.getDmaDescriptor();
                        VPUX_THROW_UNLESS(dmaDescriptor.has_value(), "DMA descriptor attr not found at '{0}'",
                                          depthToSpaceDMAOp->getLoc());
                        const auto dmaDescriptorValue = dmaDescriptor.value();

                        const auto numPlanes = checked_cast<uint32_t>(dmaDescriptorValue.getNumPlanes().getInt());
                        VPUX_THROW_UNLESS(numPlanes <= dmaMaxNumPlanes,
                                          "NUM PLANES should be less than or equal to {0}, but got {1}.",
                                          dmaMaxNumPlanes, numPlanes);

                        auto operation = builderBlk.create<VPUMI37XX::NNDMAOp>(
                                depthToSpaceDMAOp->getLoc(), indexType, depthToSpaceDMAOp.getInput(), dmaResults,
                                nullptr, mlir::ValueRange(waitBarriers), mlir::ValueRange(updateBarriers), 0, 0,
                                depthToSpaceDMAOp.getIsOutOfOrder(), depthToSpaceDMAOp.getIsCritical(),
                                depthToSpaceDMAOp.getPort().value(), VPUIP::DMAAccMode::DISABLE, dmaDescriptorValue);
                        builderBlk.setInsertionPoint(operation);
                        return operation;
                    },
                    taskOp, previousDMA, dmaCount, found);

            if (found) {
                taskOp->erase();
            }
        }

        _log.trace("VPUIP_VPUMI37XX pass: replaceVPURTTaskOpWithNNDMAOp() -- end");
    }

    void createComputeOpSwKernel(
            mlir::MLIRContext* ctx, VPUIP::SwKernelOp op, mlir::OpBuilder builderBlk,
            mlir::func::FuncOp kernel_info_funcOp, const std::string& kernel_elf2, bool isCompiled,
            mlir::Operation::operand_range wait_bars, mlir::Operation::operand_range update_bars,
            VPURegMapped::IndexType indexType,
            llvm::DenseMap<mlir::StringAttr,
                           std::pair<VPUMI37XX::DeclareKernelTextOp, VPUMI37XX::DeclareKernelEntryOp>>&
                    kernelTextEntryMap) {
        auto findKernelTextAndEntryOps = [&](mlir::OpBuilder builderBlk, vpux::VPUIP::SwKernelOp op,
                                             VPURegMapped::IndexType indexType, mlir::StringAttr kernelElf)
                -> std::pair<VPUMI37XX::DeclareKernelTextOp, VPUMI37XX::DeclareKernelEntryOp> {
            if (kernelTextEntryMap.find(kernelElf) == kernelTextEntryMap.end()) {
                auto kernelTextOp =
                        builderBlk.create<VPUMI37XX::DeclareKernelTextOp>(op->getLoc(), indexType, kernelElf);
                auto kernelEntryOp =
                        builderBlk.create<VPUMI37XX::DeclareKernelEntryOp>(op->getLoc(), indexType, kernelElf);
                kernelTextEntryMap.insert(std::make_pair(kernelElf, std::make_pair(kernelTextOp, kernelEntryOp)));
            }

            return kernelTextEntryMap[kernelElf];
        };

        std::string kernel_elf;
        if (kernel_info_funcOp == nullptr) {
            kernel_elf = kernel_elf2;
            isCompiled = true;
        } else {
            kernel_elf =
                    std::string(kernel_info_funcOp->getAttrOfType<mlir::StringAttr>("VPU.kernel_entry").getValue());
            isCompiled = false;
        }

        SmallVector<uint8_t> paramsVector = vpux::VPUMI37XX::KernelParamsSerializer::createKernelParams(op);

        auto uint8Type = mlir::IntegerType::get(ctx, 8, mlir::IntegerType::SignednessSemantics::Unsigned);
        long int paramsSize = (long int)(paramsVector.size());

        auto [kernelTextOp, kernelEntryOp] =
                findKernelTextAndEntryOps(builderBlk, op, indexType, mlir::StringAttr::get(ctx, kernel_elf));

        auto kernelArgsOp = builderBlk.create<VPUMI37XX::DeclareKernelArgsOp>(op->getLoc(), indexType,
                                                                              mlir::StringAttr::get(ctx, kernel_elf));

        auto kernelRangeOp = builderBlk.create<VPUMI37XX::ActKernelRangeOp>(
                op->getLoc(), indexType, /*taskLocation*/ nullptr, kernelTextOp, kernelArgsOp, kernelEntryOp,
                mlir::SymbolRefAttr::get(ctx, VPU::stringifyActShaveTaskType(VPU::ActShaveTaskType::COMPUTE)));

        auto tileIndex = op.getTileIndex().value_or(0);

        const auto extractDynShapes = [](mlir::OperandRange dynShapes, std::optional<ArrayRef<int32_t>> dynShapesMap) {
            if (!dynShapesMap.has_value()) {
                return SmallVector<SmallVector<mlir::Value>>();
            }
            auto dynShapesBuffers = SmallVector<SmallVector<mlir::Value>>(dynShapesMap.value().size());

            const auto getDynShapeValueByIndex = [&](const auto index) {
                return (index == ABSENT_DIMS_FLAG) ? SmallVector<mlir::Value>{}
                                                   : SmallVector<mlir::Value>{dynShapes[index]};
            };
            llvm::transform(dynShapesMap.value(), dynShapesBuffers.begin(), getDynShapeValueByIndex);

            return dynShapesBuffers;
        };

        // mlir::ValueRange does not own data.
        // Therefore, the array to store dynamic shapes must be allocated first.
        // Then SmallVector<mlir::Value> must be casted to mlir::ValueRange.
        const auto toValueRange = [](ArrayRef<mlir::Value> range) -> mlir::ValueRange {
            return range;
        };

        auto dynInputShapes = extractDynShapes(op.getDynamicInputShapes(), op.getDynamicInputShapesMap());
        auto dynInputShapesRange = SmallVector<mlir::ValueRange>();
        llvm::transform(dynInputShapes, std::back_inserter(dynInputShapesRange), toValueRange);

        auto dynOutputShapes = extractDynShapes(op.getDynamicOutputShapeBuffs(), op.getDynamicOutputShapesMap());
        auto dynOutputShapesRange = SmallVector<mlir::ValueRange>();
        llvm::transform(dynOutputShapes, std::back_inserter(dynOutputShapesRange), toValueRange);

        auto kernelParams = builderBlk.create<VPUMI37XX::KernelParamsOp>(
                op->getLoc(), indexType, op.getInputs(), op.getOutputBuffs(), dynInputShapesRange, dynOutputShapesRange,
                mlir::StringAttr::get(ctx, kernel_elf),
                mlir::DenseIntElementsAttr::get(mlir::VectorType::get({paramsSize}, uint8Type), paramsVector),
                mlir::BoolAttr::get(ctx, isCompiled));

        builderBlk.create<VPUMI37XX::ActKernelInvocationOp>(op->getLoc(), indexType, /*taskLocation*/ nullptr,
                                                            mlir::ValueRange(wait_bars), mlir::ValueRange(update_bars),
                                                            kernelRangeOp.getResult(), kernelParams.getResult(),
                                                            op.getProfilingData(),
                                                            /* tile= */ tileIndex,
                                                            /* start_after= */ 0, /* clean_after= */ 0);
    }

    void createCacheOpSwKernel(mlir::MLIRContext* ctx, VPUIP::SwKernelOp op, mlir::OpBuilder builderBlk,
                               mlir::SymbolRefAttr kernelTaskType, bool isCompiled,
                               mlir::Operation::operand_range wait_bars, mlir::Operation::operand_range update_bars,
                               VPURegMapped::IndexType indexType) {
        auto taskTypeVal = VPU::symbolizeActShaveTaskType(kernelTaskType.getLeafReference().strref());
        VPUX_THROW_UNLESS(taskTypeVal.has_value(), "Operation '{0}' has invalid VPU.task_type attribute '{1}'",
                          op.getKernelFunction(), kernelTaskType.getLeafReference());

        auto kernel_type = std::string("cache_op");
        switch (taskTypeVal.value()) {
        case VPU::ActShaveTaskType::CACHE_FLUSH:
            kernel_type.append("_flush");
            break;
        case VPU::ActShaveTaskType::CACHE_INVALIDATE:
            kernel_type.append("_invalidate");
            break;
        case VPU::ActShaveTaskType::CACHE_FLUSH_INVALIDATE:
            kernel_type.append("_flush_invalidate");
            break;
        default:
            VPUX_THROW("Unrecognized Kernel Task Type '{0}'", kernelTaskType.getLeafReference());
        }

        auto kernelRangeOp = builderBlk.create<VPUMI37XX::ActKernelRangeOp>(
                op->getLoc(), indexType, /*taskLocation*/ nullptr, nullptr, nullptr, nullptr,
                mlir::SymbolRefAttr::get(ctx, VPU::stringifyActShaveTaskType(taskTypeVal.value())));

        auto tileIndex = op.getTileIndex().value_or(0);

        auto uint8Type = mlir::IntegerType::get(ctx, 8, mlir::IntegerType::SignednessSemantics::Unsigned);
        SmallVector<uint8_t> paramsVectorDummy = {0xFF};
        long int paramsSize = (long int)(paramsVectorDummy.size());

        SmallVector<mlir::ValueRange> dynInputShapes(0);
        SmallVector<mlir::ValueRange> dynOutputShapes(0);
        auto kernelParams = builderBlk.create<VPUMI37XX::KernelParamsOp>(
                op->getLoc(), indexType, mlir::ValueRange(), mlir::ValueRange(), dynInputShapes, dynOutputShapes,
                mlir::StringAttr::get(ctx, kernel_type),
                mlir::DenseIntElementsAttr::get(mlir::VectorType::get({paramsSize}, uint8Type), paramsVectorDummy),
                mlir::BoolAttr::get(ctx, isCompiled));

        builderBlk.create<VPUMI37XX::ActKernelInvocationOp>(
                op->getLoc(), indexType, /*taskLocation*/ nullptr, mlir::ValueRange(wait_bars),
                mlir::ValueRange(update_bars), kernelRangeOp.getResult(), kernelParams.getResult(), nullptr,
                /* tile= */ tileIndex,
                /* start_after= */ 0, /* clean_after= */ 0);
    }

    void replaceVPURTTaskOpWithKernelOps(mlir::MLIRContext* ctx, mlir::ModuleOp moduleOp, mlir::func::FuncOp funcOp,
                                         Logger _log) {
        _log.trace("VPUIP_VPUMI37XX pass: replaceVPURTTaskOpWithKernelOps()");

        auto shave_task_count = 0;
        llvm::DenseMap<mlir::StringAttr, std::pair<VPUMI37XX::DeclareKernelTextOp, VPUMI37XX::DeclareKernelEntryOp>>
                kernelTextEntryMap;

        // Forever loop that runs until there are no more changes performed by
        //   the inner loop (so the algorithm has converged).

        for (auto taskOp : llvm::make_early_inc_range(funcOp.getBody().getOps<VPURT::TaskOp>())) {
            bool found = false;

            for (auto op : llvm::make_early_inc_range(taskOp.getBodyRegion().getOps<VPUIP::SwKernelOp>())) {
                found = true;
                mlir::OpBuilder builderBlk(taskOp);

                auto indexType = VPURegMapped::IndexType::get(ctx, shave_task_count);

                auto wait_bars = taskOp.getWaitBarriers();
                auto update_bars = taskOp.getUpdateBarriers();

                auto trivialIndexType = VPURegMapped::IndexType::get(ctx, 0);

                for (auto val : wait_bars) {
                    val.setType(trivialIndexType);
                }

                for (auto val : update_bars) {
                    val.setType(trivialIndexType);
                }

                auto sw_kernel_symbol = op.getKernelFunction();

                auto kernel_info_funcOp = moduleOp.lookupSymbol<mlir::func::FuncOp>(sw_kernel_symbol);

                if (kernel_info_funcOp) {
                    const auto kernelTaskType = kernel_info_funcOp->getAttrOfType<mlir::SymbolRefAttr>("VPU.task_type");

                    // Check if the task is a Cache handling op
                    bool isCacheOp = VPUIP::isCacheOpTaskType(kernelTaskType);
                    if (!isCacheOp) {
                        std::string emptyString = std::string("");
                        createComputeOpSwKernel(ctx, op, builderBlk, kernel_info_funcOp, emptyString, false, wait_bars,
                                                update_bars, indexType, kernelTextEntryMap);
                    } else {
                        createCacheOpSwKernel(ctx, op, builderBlk, kernelTaskType, false, wait_bars, update_bars,
                                              indexType);
                    }
                } else {
                    // This is the compiled kernel case
                    auto kernel_info_llvmFuncOp = moduleOp.lookupSymbol<mlir::LLVM::LLVMFuncOp>(sw_kernel_symbol);

                    VPUX_THROW_UNLESS(kernel_info_llvmFuncOp != nullptr, "kernel_info_llvmFuncOp should be valid");

                    auto llvmFuncOpNameStr = kernel_info_llvmFuncOp.getName().str();

                    llvm::LLVMContext llvmContext;
                    auto llvmModule = vpux::translateToLLVMIR(moduleOp, sw_kernel_symbol, llvmContext);

                    createComputeOpSwKernel(ctx, op, builderBlk, nullptr, llvmFuncOpNameStr, true, wait_bars,
                                            update_bars, indexType, kernelTextEntryMap);

                    vpux::lowerLLVMToBinary(moduleOp, std::move(llvmModule), sw_kernel_symbol, _log);
                }

                shave_task_count++;
            }

            if (found) {
                taskOp->erase();
            }
        }
    }

    void replaceNCEClusterTaskOpWithDPUOps(mlir::MLIRContext* ctx, mlir::func::FuncOp funcOp, Logger _log) {
        int variant_task_count = 0;
        int invariant_task_count = 0;

        for (auto taskOp : llvm::make_early_inc_range(funcOp.getOps<VPURT::TaskOp>())) {
            bool found = false;

            _log.trace("replaceNCEClusterTaskOpWithDPUOps(): taskOp = {0}", taskOp);

            for (auto op : llvm::make_early_inc_range(taskOp.getBody().getOps<VPUIP::NCEClusterTaskOp>())) {
                found = true;
                mlir::OpBuilder builderBlk(taskOp);

                auto wait_barriers = taskOp.getWaitBarriers();
                auto update_barriers = taskOp.getUpdateBarriers();

                auto trivialIndexType = VPURegMapped::IndexType::get(ctx, 0);

                for (auto val : wait_barriers) {
                    val.setType(trivialIndexType);
                }

                for (auto val : update_barriers) {
                    val.setType(trivialIndexType);
                }

                const auto& dpuTasks = op.getVariants().getOps<VPUIP::DPUTaskOp>();
                VPUX_THROW_UNLESS(!dpuTasks.empty(), "Encountered op {} with empty dpu list", op);
                const auto& differentMPEModes = std::adjacent_find(dpuTasks.begin(), dpuTasks.end(),
                                                                   [](VPUIP::DPUTaskOp lhs, VPUIP::DPUTaskOp rhs) {
                                                                       return lhs.getMpeMode() != rhs.getMpeMode();
                                                                   });
                if (differentMPEModes != dpuTasks.end()) {
                    VPUIP::DPUTaskOp lhs = *differentMPEModes;
                    VPUIP::DPUTaskOp rhs = *std::next(differentMPEModes);
                    VPUX_THROW("Found dpu tasks {} and {} inside of {} which has different MPE "
                               "modes {} and {} "
                               "accordingly, but only uniform MPE mode is supported by ELF",
                               lhs, rhs, op, lhs.getMpeMode(), rhs.getMpeMode());
                }

                VPUIP::DPUTaskOp first = *(dpuTasks.begin());
                auto mpe_freq_mode = VPU::MPEModeAttr::get(ctx, first.getMpeMode());
                auto invariantIndex = VPURegMapped::IndexType::get(ctx, invariant_task_count);
                auto startAfterAttr = builderBlk.getIntegerAttr(builderBlk.getIntegerType(64, false), 0);
                auto cleanAfterAttr = builderBlk.getIntegerAttr(builderBlk.getIntegerType(64, false), 0);

                auto dpuResults = unrollDistributedBuff(builderBlk, op.getOutputBuff());
                auto invariant = builderBlk.create<VPUMI37XX::DPUInvariantOp>(
                        op->getLoc(), invariantIndex, /*taskLocation*/ nullptr, op.getInput(), op.getInputSparsityMap(),
                        op.getInputStorageElementTable(), op.getWeights(), op.getWeightsSparsityMap(),
                        op.getWeightTable(), op.getParentInput(), op.getParentInputSparsityMap(),
                        op.getParentInputStorageElementTable(), op.getParentOutput(), op.getParentOutputSparsityMap(),
                        dpuResults, op.getOutputSparsityMapBuff(), op.getProfilingData(), op.getTaskTypeAttr(),
                        op.getEltwiseTypeAttr(), mpe_freq_mode, op.getKernelSizeAttr(), op.getKernelStridesAttr(),
                        op.getKernelPaddingAttr(), op.getIsContinuedAttr(), op.getCmSpPatternAttr(),
                        op.getIsSegmentedAttr(), op.getInputChannelsCompressionAttr(),
                        op.getIsZeroOffsetWeightsTableAttr(), op.getOutChannelOffsetAttr(), op.getIsSuperdenseAttr(),
                        op.getIsInplaceAttr(), op.getInputSeSizeAttr(), op.getOutputSeSizeAttr(),
                        op.getIsPermuteQuantizeAttr(), wait_barriers, update_barriers, startAfterAttr, cleanAfterAttr);

                invariant_task_count++;

                for (auto dpuTaskOp : op.getVariants().getOps<VPUIP::DPUTaskOp>()) {
                    auto variantIndex = VPURegMapped::IndexType::get(ctx, variant_task_count);
                    builderBlk.create<VPUMI37XX::DPUVariantOp>(
                            dpuTaskOp->getLoc(), variantIndex, /*taskLocation*/ nullptr, invariant.getResult(),
                            dpuTaskOp.getOutStartAttr(), dpuTaskOp.getOutEndAttr(), dpuTaskOp.getPadAttr(),
                            dpuTaskOp.getMpeModeAttr(), dpuTaskOp.getClusterIdAttr(), dpuTaskOp.getWorkloadIdAttr());
                    variant_task_count++;
                }

                {
                    auto& ppeRegion = invariant.getPpe();
                    ppeRegion.emplaceBlock();

                    mlir::OpBuilder::InsertionGuard guard(builderBlk);
                    builderBlk.setInsertionPointToEnd(&ppeRegion.front());

                    for (auto ppe : op.getPpe().getOps<VPUIP::PPETaskOp>()) {
                        builderBlk.create<VPUMI37XX::PPETaskOp>(builderBlk.getUnknownLoc(), ppe->getResultTypes(),
                                                                ppe->getOperands(),
                                                                ppe->getAttrDictionary().getValue());
                    }
                }
            }

            if (found) {
                taskOp->erase();
            }
        }
    }

    void setBarrierIndexValues(mlir::MLIRContext* ctx, mlir::func::FuncOp& funcOp, Logger _log) {
        auto barrier_count = 0;

        VPUX_UNUSED(_log);

        for (auto op : funcOp.getOps<VPUMI37XX::ConfigureBarrierOp>()) {
            auto indexType = VPURegMapped::IndexType::get(ctx, barrier_count);

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
    size_t countTasksIf(mlir::func::FuncOp& funcOp, Condition&& condition = noCond) {
        auto tasks = funcOp.template getOps<TaskType>();
        return std::count_if(tasks.begin(), tasks.end(), std::forward<Condition>(condition));
    }

    template <typename TaskType, typename Condition = decltype(noCond<TaskType>)>
    mlir::Value findTaskIf(mlir::func::FuncOp& funcOp, Condition&& condition = noCond) {
        auto tasks = funcOp.template getOps<TaskType>();
        auto target = std::find_if(tasks.begin(), tasks.end(), std::forward<Condition>(condition));
        return target != tasks.end() ? (*target).getResult() : mlir::Value();
    }

    void createMappedInferenceOp(mlir::MLIRContext* ctx, mlir::ModuleOp& moduleOp, mlir::func::FuncOp& funcOp,
                                 Logger _log) {
        _log.trace("VPUIP_VPUMI37XX pass: createMappedInferenceOp()");

        const auto dmaExecCount = config::getAvailableExecutor(moduleOp, VPU::ExecutorKind::DMA_NN).getCount();

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

        mlir::SmallVector<std::optional<VPUMI37XX::NNDMAOp>> dmaHeadsOptionals(dmaCount.size());
        for (auto operation : funcOp.getOps<VPUMI37XX::NNDMAOp>()) {
            const auto port = checked_cast<std::size_t>(operation.getPort());
            dmaCount[port]++;

            if (mlir::cast<VPURegMapped::IndexType>(operation.getType()).getValue() == 0) {
                dmaHeadsOptionals[port] = operation;
            }
        }

        barrierCount = countTasksIf<VPUMI37XX::ConfigureBarrierOp>(funcOp);
        barrierTasks = findTaskIf<VPUMI37XX::ConfigureBarrierOp>(funcOp);
        rangeCount = countTasksIf<VPUMI37XX::ActKernelRangeOp>(funcOp);
        actKernelRanges = findTaskIf<VPUMI37XX::ActKernelRangeOp>(funcOp);
        invoCount = countTasksIf<VPUMI37XX::ActKernelInvocationOp>(funcOp);
        actKernelInvocations = findTaskIf<VPUMI37XX::ActKernelInvocationOp>(funcOp);
        invariantCount = countTasksIf<VPUMI37XX::DPUInvariantOp>(funcOp);
        invariantTasks = findTaskIf<VPUMI37XX::DPUInvariantOp>(funcOp);
        variantCount = countTasksIf<VPUMI37XX::DPUVariantOp>(funcOp);
        variantTasks = findTaskIf<VPUMI37XX::DPUVariantOp>(funcOp);

        auto trivialIndexType = VPURegMapped::IndexType::get(ctx, 0);

        // create MappedInferenceOp
        mlir::OpBuilder builderFunc(&(funcOp.getBody().front().back()));

        mlir::SmallVector<mlir::Value> dmaHeadsValues;
        for (auto optionalHead : dmaHeadsOptionals) {
            if (!optionalHead.has_value()) {
                continue;
            }
            dmaHeadsValues.push_back(optionalHead.value());
        }

        builderFunc.create<VPUMI37XX::MappedInferenceOp>(
                mlir::UnknownLoc::get(ctx), trivialIndexType,
                dmaHeadsValues,                                   // mlir::Value dmaTasks
                invariantTasks,                                   // mlir::Value invariantTasks
                variantTasks,                                     // mlir::Value variantTasks
                actKernelRanges,                                  // mlir::Value actKernelRanges
                actKernelInvocations,                             // mlir::Value actKernelInvocations
                barrierTasks,                                     // mlir::Value barrierTasks
                actShvRt,                                         // mlir::Value actShaveRt
                actShaveStacks,                                   // mlir::ValueRange actShaveStacks
                builderFunc.getI64ArrayAttr(ArrayRef(dmaCount)),  // mlir::ArrayAttr
                invariantCount,                                   // uint32_t invariantCount
                variantCount,                                     // uint32_t variantCount
                rangeCount,                                       // uint32_t rangeCount
                invoCount,                                        // uint32_t invoCount
                barrierCount                                      // uint32_t barrierCount
        );
    }

    void createProfilingMetadataOp(mlir::MLIRContext* ctx, mlir::ModuleOp moduleOp, mlir::func::FuncOp funcOp,
                                   Logger _log) {
        auto netOps = to_small_vector(moduleOp.getOps<net::NetworkInfoOp>());
        if (netOps.size() != 1) {
            return;
        }
        net::NetworkInfoOp netInfo = netOps.front();

        if (netInfo.getProfilingOutputsInfo().empty()) {
            return;
        }
        _log.trace("VPUIP_VPUMI37XX pass: createProfilingMetadataOp()");

        mlir::OpBuilder builderFunc(&(funcOp.getBody().front().back()));
        // Collect profiling metadata from all operations of function and pack them into FB, which will be serialized as
        // separate section. VPUMI37XX::ProfilingMetadataOp stores content of this section
        auto buffer = vpux::buildProfilingMetadataBuffer(netInfo, funcOp, _log);
        llvm::ArrayRef<char> rawMetadata{reinterpret_cast<const char*>(buffer.data()), buffer.size()};
        long int bufferSize = buffer.size();

        auto uint8Type = mlir::IntegerType::get(ctx, 8, mlir::IntegerType::SignednessSemantics::Unsigned);
        auto vectorType = mlir::VectorType::get({bufferSize}, uint8Type);
        const auto elemAttr = mlir::DenseElementsAttr::getFromRawBuffer(vectorType, rawMetadata);
        auto trivialIndexType = VPURegMapped::IndexType::get(ctx, 0);
        builderFunc.create<VPUMI37XX::ProfilingMetadataOp>(mlir::UnknownLoc::get(ctx), trivialIndexType, elemAttr);
    }

};  // namespace

class ConvertVPURTConfigureBarrierOp final : public mlir::OpRewritePattern<VPURT::ConfigureBarrierOp> {
public:
    ConvertVPURTConfigureBarrierOp(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPURT::ConfigureBarrierOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(VPURT::ConfigureBarrierOp origOp,
                                        mlir::PatternRewriter& rewriter) const override {
        auto ctx = ConvertVPURTConfigureBarrierOp::getContext();

        auto trivialIndexType = VPURegMapped::IndexType::get(ctx, 0);

        // E#105629: barrier variable may have incorrect type, so access result directly to avoid unsafe casting
        mlir::Value origOpResult = origOp->getResult(0);

        size_t producer_count = 0;
        size_t consumer_count = 0;

        // should use VPUMI37XX TaskOp interface
        for (auto user : origOpResult.getUsers()) {
            if (auto taskOp = mlir::dyn_cast<vpux::VPUMI37XX::ExecutableTaskOpInterface>(user)) {
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

        rewriter.replaceOpWithNewOp<VPUMI37XX::ConfigureBarrierOp>(
                origOp,
                trivialIndexType,                                   // setup all barriers with the trivial index (0)
                checked_cast<uint8_t>(origOp.getId()),              // real_id
                -1,                                                 // int64_t next_same_id()
                nullptr,                                            // mlir::Value previousSameId
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

void ConvertVPUIP2VPUMI37XXPass::safeRunOnModule() {
    auto ctx = &(getContext());
    auto moduleOp = getOperation();
    auto funcOpsRange = moduleOp.getOps<mlir::func::FuncOp>();
    VPUX_THROW_UNLESS(!funcOpsRange.empty(), "Empty FuncOp");
    auto funcOp = *funcOpsRange.begin();

    _log.trace("funcOp = {0}", funcOp);

    createProfilingMetadataOp(ctx, moduleOp, funcOp, _log);

    replaceVPURTTaskOpWithNNDMAOp(ctx, moduleOp, funcOp, _log);

    _log.trace("funcOp after replacing NNDMA Ops = {0}", funcOp);

    replaceVPURTTaskOpWithKernelOps(ctx, moduleOp, funcOp, _log);

    _log.trace("funcOp after replacing ActKernel Ops = {0}", funcOp);

    replaceNCEClusterTaskOpWithDPUOps(ctx, funcOp, _log);

    _log.trace("funcOp after replacing DPU Ops = {0}", funcOp);

    mlir::ConversionTarget target(*ctx);
    target.addLegalDialect<VPUMI37XX::VPUMI37XXDialect>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalOp<mlir::func::FuncOp, mlir::func::ReturnOp>();
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
// createConvertVPUIP2VPUMI37XXPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertVPUIP2VPUMI37XXPass(Logger log) {
    return std::make_unique<ConvertVPUIP2VPUMI37XXPass>(log);
}
