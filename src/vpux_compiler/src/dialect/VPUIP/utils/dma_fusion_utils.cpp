//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
#include "vpux/compiler/dialect/VPUIP/utils/dma_fusion_utils.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/utils/memref_attr_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {
mlir::Type normalize(mlir::Type type) {
    return mlir::cast<NDTypeInterface>(type).changeMemSpace(VPU::MemoryKind::DDR);
}

vpux::VPU::MemoryKind getMemKind(mlir::Type type) {
    return mlir::cast<NDTypeInterface>(type).getMemoryKind();
}

bool hasSameMemKind(mlir::Type src, mlir::Type dst) {
    auto srcMemory = getMemKind(src);
    auto dstMemory = getMemKind(dst);
    return srcMemory == dstMemory;
}

void eraseOp(mlir::Operation* op) {
    if (op->getUsers().empty()) {
        op->erase();
    }
}

template <class OpType>
OpType getCommonOp(SmallVector<VPURT::TaskOp> tasks, bool input) {
    OpType commonOp = nullptr;
    for (auto taskOp : tasks) {
        auto val = input ? VPUIP::getInput(taskOp) : VPUIP::getOutput(taskOp);
        auto op = val.getDefiningOp<OpType>();
        if (op == nullptr) {
            return nullptr;
        }
        if (commonOp == nullptr) {
            commonOp = op;
        }
    }
    return commonOp;
}

bool verifyPossibleType(SmallVector<int64_t> shape, SmallVector<Bit> strides, DimsOrder order) {
    const auto newOrder = inferNewDimsOrder(order, shape.size());

    const auto memShape = newOrder.toMemoryOrder(ShapeRef(shape));
    const auto memStrides = newOrder.toMemoryOrder(StridesRef(strides));

    const auto elemSize = 1_Bit;
    StrideReqs reqs;
    return reqs.checkStrides(memStrides, elemSize, memShape);
}

// Creates new type, which represents fusion.
vpux::NDTypeInterface createNewType(NDTypeInterface srcType, size_t newLeadingDim,
                                    const VPUIP::StrideInfo& strideInfo) {
    auto srcShape = srcType.getShape();
    SmallVector<int64_t> newShape(srcShape.begin(), srcShape.end());
    newShape.insert(newShape.begin(), newLeadingDim);
    NDTypeInterface newType;
    if (auto perAxisQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(srcType.getElementType())) {
        // We inserted new dim, so current qDim is shifted by 1
        auto newQuantDim = perAxisQType.getQuantizedDimension() + 1;
        auto newQType = mlir::quant::UniformQuantizedPerAxisType::get(
                perAxisQType.getFlags(), perAxisQType.getStorageType(), perAxisQType.getExpressedType(),
                perAxisQType.getScales(), perAxisQType.getZeroPoints(), newQuantDim, perAxisQType.getStorageTypeMin(),
                perAxisQType.getStorageTypeMax());

        newType = srcType.changeShapeElemType(ShapeRef(newShape), newQType);
    } else {
        newType = srcType.changeShape(ShapeRef(newShape));
    }

    if (strideInfo.isExplicit) {
        const auto strides = srcType.getStrides();
        SmallVector<Bit> newStrides(strides.begin(), strides.end());
        newStrides.insert(newStrides.begin(), strideInfo.value);
        if (!verifyPossibleType(std::move(newShape), newStrides, srcType.getDimsOrder())) {
            return nullptr;
        }
        newType = newType.changeStrides(StridesRef(newStrides));
    }

    return newType;
}

mlir::Value tryCreateConst(Const::DeclareOp cstOp, SmallVector<VPURT::TaskOp> tasks, vpux::Logger log) {
    auto srcType = cstOp.getType().dyn_cast<NDTypeInterface>();
    auto newInType =
            createNewType(srcType, /*newDim=*/tasks.size(), /*info=*/{});  // StridesInfo isn't required for constants
    if (newInType == nullptr) {
        log.trace("Can't infer input type, check StridesReq");
        return nullptr;
    }
    const auto origCstContentAttr = cstOp.getContentAttr();

    auto fakeBaseContent = origCstContentAttr.getBaseContent();
    auto newContentAttr = Const::ContentAttr::get(fakeBaseContent);
    std::vector<Const::ContentAttr> contentsToFuse;
    for (auto task : tasks) {
        auto contentAttr = VPUIP::getInput(task).getDefiningOp<Const::DeclareOp>().getContentAttr();
        contentsToFuse.push_back(contentAttr);
    }

    // newInType isn't inherited from RankedTensorType, because in VPUIP world it's MemrefType
    // Create RankedTensorType for content manually
    vpux::TensorAttr tensorAttr = nullptr;
    auto dimsOrder = newInType.getDimsOrder();
    if (!dimsOrder.isIdentity()) {
        tensorAttr = vpux::getTensorAttr(cstOp->getContext(), dimsOrder, nullptr);
    }
    auto contentType = mlir::RankedTensorType::get(newInType.getShape(), newInType.getElementType(), tensorAttr);
    auto fusedContentAttr = newContentAttr.transform().fuse(contentType, contentsToFuse).get();

    mlir::OpBuilder builder(cstOp);
    auto newCstOp = builder.create<Const::DeclareOp>(cstOp->getLoc(), newInType, std::move(fusedContentAttr));
    log.trace("Created input constants");
    return newCstOp;
}

mlir::Value createBufferDeclaration(SmallVector<VPURT::TaskOp> tasks, vpux::Logger log,
                                    const VPUIP::StrideInfo& strideInfo, bool isInput) {
    VPURT::DeclareBufferOp bufOp = getCommonBuffer(tasks, isInput);
    if (bufOp == nullptr) {
        return nullptr;
    }

    auto valType = bufOp.getType().dyn_cast<NDTypeInterface>();
    auto newType = createNewType(valType, /*newDim=*/tasks.size(), strideInfo);
    if (newType == nullptr) {
        log.trace("Can't infer buffer type, check StridesReq");
        return nullptr;
    }

    std::string locSuffix = isInput ? "src" : "dst";
    mlir::OpBuilder builder(bufOp);
    auto newBuff = builder.create<VPURT::DeclareBufferOp>(
            appendLoc(bufOp->getLoc(), "strided_buffer_alloc_{0}", locSuffix), newType, bufOp.getSectionAttr(),
            bufOp.getSectionIndexAttr(), bufOp.getByteOffsetAttr(), bufOp.getSwizzlingKeyAttr());
    log.trace("Created {0} buffer", locSuffix);
    return newBuff;
}

};  // namespace

bool vpux::VPUIP::hasCompatibleTypes(VPUIP::NNDMAOp currentDma, VPUIP::NNDMAOp nextDma) {
    // Disable CMX<->CMX and DDR<->DDR copies
    if (hasSameMemKind(currentDma.getInput().getType(), currentDma.getOutputBuff().getType())) {
        return false;
    }
    // We need to check that inputs and output are compatible(same data type, shape, strides). Because unrolling changes
    // CMX id we need to get rid of it to compare with == operator. Normalize change memSpace to DDR, so now they're
    // comparable
    return normalize(currentDma.getInput().getType()) == normalize(nextDma.getInput().getType()) &&
           normalize(currentDma.getOutputBuff().getType()) == normalize(nextDma.getOutputBuff().getType());
}

void vpux::VPUIP::handleDmaFusion(mlir::func::FuncOp funcOp, vpux::Logger log,
                                  const VPUIP::StrideProviderFunc& srcStrideProvider,
                                  const VPUIP::StrideProviderFunc& dstStrideProvider,
                                  const std::function<size_t(SmallVector<VPURT::TaskOp>)>& newPortProvider) {
    mlir::DenseMap<mlir::IntegerAttr, SmallVector<VPURT::TaskOp>> id2ops;
    std::vector<mlir::Operation*> toRemove;

    funcOp->walk([&](VPURT::TaskOp taskOp) {
        if (taskOp.getExecutorKind() != VPU::ExecutorKind::DMA_NN) {
            return;
        }

        auto currDmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(taskOp.getInnerTaskOp());
        if (currDmaOp == nullptr) {
            return;
        }
        VPUX_THROW_UNLESS(currDmaOp.getPort().has_value(), "DMA at '{0}' has no portId", currDmaOp->getLoc());

        mlir::IntegerAttr fusionId = currDmaOp.getFusionIdAttr();
        if (fusionId == nullptr) {
            return;
        }
        id2ops[fusionId].push_back(taskOp);
    });

    SmallVector<mlir::IntegerAttr> groupsToClean;
    for (auto& [fusionId, fusionCandidates] : id2ops) {
        if (fusionCandidates.size() < 2) {
            log.trace("Fusing group {0} is too small, skip", fusionId);
            groupsToClean.push_back(fusionId);
            continue;
        }

        std::sort(fusionCandidates.begin(), fusionCandidates.end(), [](VPURT::TaskOp firstOp, VPURT::TaskOp secondOp) {
            return firstOp->isBeforeInBlock(secondOp);
        });
        log.trace("Fusing group {0}", fusionId);
        auto nestedLog = log.nest();
        auto srcStride = srcStrideProvider(nestedLog, fusionCandidates);
        auto dstStride = dstStrideProvider(nestedLog, fusionCandidates);
        if (!srcStride.feasible || !dstStride.feasible) {
            nestedLog.trace("src/dst can't be fused");
            groupsToClean.push_back(fusionId);
            continue;
        }

        std::string dmaLocSuffix;
        mlir::Value newSrc = nullptr;
        if (auto cstOp = getCommonConstant(fusionCandidates)) {
            dmaLocSuffix = "cst2buf";
            newSrc = tryCreateConst(cstOp, fusionCandidates, nestedLog);
        } else {
            dmaLocSuffix = "buf2buf";
            newSrc = createBufferDeclaration(fusionCandidates, nestedLog, srcStride, /*isInput=*/true);
        }

        auto newDst = createBufferDeclaration(fusionCandidates, nestedLog, dstStride, /*isInput=*/false);
        if (newSrc == nullptr || newDst == nullptr) {
            nestedLog.trace("Can't create src/dst");
            groupsToClean.push_back(fusionId);
            continue;
        }

        auto newPort = newPortProvider(fusionCandidates);

        auto dmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(fusionCandidates.front().getInnerTaskOp());
        mlir::OpBuilder builder(dmaOp);
        auto newDmaOp = builder.create<VPUIP::NNDMAOp>(
                appendLoc(dmaOp->getLoc(), "fused_dma_{0}", dmaLocSuffix), newSrc, newDst,
                getIntAttr(dmaOp->getContext(), newPort), dmaOp.getIsOutOfOrderAttr(), dmaOp.getIsCriticalAttr(),
                dmaOp.getSpillIdAttr(), dmaOp.getCompressCandidateAttr(), /*dmaHwpId=*/nullptr,
                /*profilingMetadata=*/nullptr, /*splitCandidate=*/nullptr,
                /*profiling_buffer_mgmt=*/nullptr, /*fusionId=*/nullptr);
        if (dmaOp.getProfilingBufferMgmt()) {
            newDmaOp.setProfilingBufferMgmt(true);
        }
        fusionCandidates.front()->setLoc(newDmaOp->getLoc());

        // Remove all other DMAs with their tasks
        toRemove.insert(toRemove.end(), fusionCandidates.begin() + 1, fusionCandidates.end());
        // And DMA task from current task
        toRemove.push_back(dmaOp);
        nestedLog.trace("Fused group");
    }

    mlir::DenseSet<mlir::Operation*> secondRemoveGen;
    for (mlir::Operation* op : toRemove) {
        if (auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(op)) {
            secondRemoveGen.insert(getInput(taskOp).getDefiningOp());
            secondRemoveGen.insert(getOutput(taskOp).getDefiningOp());
        }
        if (auto dmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(op)) {
            secondRemoveGen.insert(dmaOp.getInput().getDefiningOp());
            secondRemoveGen.insert(dmaOp.getOutputBuff().getDefiningOp());
        }
        eraseOp(op);
    }
    for (mlir::Operation* op : secondRemoveGen) {
        eraseOp(op);
    }
    for (auto fusionId : groupsToClean) {
        for (auto taskOp : id2ops[fusionId]) {
            if (auto dmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(taskOp.getInnerTaskOp())) {
                dmaOp.setFusionIdAttr(nullptr);
            }
        }
    }

    log.trace("Done");
}

Const::DeclareOp VPUIP::getCommonConstant(SmallVector<VPURT::TaskOp> tasks) {
    return getCommonOp<Const::DeclareOp>(std::move(tasks), /*input=*/true);
}

VPURT::DeclareBufferOp VPUIP::getCommonBuffer(SmallVector<VPURT::TaskOp> tasks, bool input) {
    return getCommonOp<VPURT::DeclareBufferOp>(std::move(tasks), input);
}

mlir::Value VPUIP::getInput(VPURT::TaskOp taskOp) {
    auto dmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(taskOp.getInnerTaskOp());
    return dmaOp.getInput();
}

mlir::Value VPUIP::getOutput(VPURT::TaskOp taskOp) {
    auto dmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(taskOp.getInnerTaskOp());
    return dmaOp.getOutputBuff();
}
