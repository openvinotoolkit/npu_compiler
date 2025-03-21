//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/dpu_invariant_rewriter.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

namespace {

using namespace vpux;

mlir::Value extractValueForTile(mlir::ValueRange values, uint32_t tileIdx) {
    for (auto value : values) {
        if (auto declBuffOp = value.getDefiningOp<VPURT::DeclareBufferOp>()) {
            if (declBuffOp.getSection() == VPURT::BufferSection::MAC_Accumulators) {
                continue;
            }
        }
        if (value.getType().cast<NDTypeInterface>().getMemSpace().getIndex().value_or(0) == tileIdx) {
            return value;
        }
    }

    return nullptr;
}

}  // namespace

namespace vpux {
namespace vpumi40xx2vpuasm {

mlir::FailureOr<SymbolizationResult> DPUInvariantRewriter::symbolize(VPUMI40XX::DPUInvariantOp op, SymbolMapper&,
                                                                     mlir::ConversionPatternRewriter& rewriter) const {
    auto symName = findSym(op).getRootReference();
    auto taskLocation = findSym(op.getTaskLocation());

    auto optionalSym = [&](mlir::Value val) -> mlir::SymbolRefAttr {
        auto sym = val ? findSym(val) : nullptr;
        return sym;
    };

    auto inputSym = findSym(op.getInput());
    auto inputSparsityMapSym = optionalSym(op.getInputSparsityMap());
    auto inputSETableSym = optionalSym(op.getInputStorageElementTable());

    auto weightsSym = optionalSym(op.getWeights());
    auto weightsSparsityMapSym = optionalSym(op.getWeightsSparsityMap());
    auto weightTableSym = optionalSym(op.getWeightTable());
    auto weightTableDataPtrSym = optionalSym(op.getWeightTableDataPtr());
    auto weightTableSpPtrSym = optionalSym(op.getWeightTableSpPtr());
    auto weightTableScaleSym = optionalSym(op.getWeightTableScale());
    auto weightTableBiasSym = optionalSym(op.getWeightTableBias());
    auto weightZeroPointsSym = optionalSym(op.getWeightZeroPoints());

    auto sprLookupTableSym = optionalSym(op.getSprLookupTable());
    auto palletLookupTableSym = optionalSym(op.getPalletLookupTable());

    auto tileIdx = op.getIndex().getType().cast<VPURegMapped::IndexType>().getTileIdx();

    auto output = extractValueForTile(op.getOutputBuffs(), tileIdx);
    auto outputSym = optionalSym(output);

    mlir::Value outputSparsityMap;
    if (!op.getOutputSparsityMapBuffs().empty()) {
        outputSparsityMap = extractValueForTile(op.getOutputSparsityMapBuffs(), tileIdx);
        if (!outputSparsityMap) {
            _log.error("Output sparsity map buffer for tile#{0} not found in 'getOutputSparsityMapBuffs'", tileIdx);
            return mlir::failure();
        }
    }
    auto outputSparsityMapSym = optionalSym(outputSparsityMap);

    auto profilingDataSym = optionalSym(op.getProfilingData());

    auto maxPerXYSym = optionalSym(op.getMaxPerXy());
    auto minPerXYSym = optionalSym(op.getMinPerXy());

    mlir::ArrayAttr minMaxPerTensorAttr = nullptr;
    if (!op.getMinMaxPerTensor().empty()) {
        llvm::SmallVector<mlir::Attribute> minMaxPerTensorSyms(op.getMinMaxPerTensor().size());
        for (auto [index, minMaxTensor] : llvm::enumerate(op.getMinMaxPerTensor())) {
            auto minMaxSym = findSym(minMaxTensor);
            minMaxPerTensorSyms[index] = minMaxSym;
        }
        minMaxPerTensorAttr = mlir::ArrayAttr::get(rewriter.getContext(), minMaxPerTensorSyms);
    }

    auto waitAttr = vectorizeBarriers(op.getWaitBarriers());
    auto updateAttr = vectorizeBarriers(op.getUpdateBarriers());

    auto taskIdx = mlir::TypeAttr::get(op.getType());

    auto invariantUsers = op.getIndex().getUsers();
    llvm::SmallVector<mlir::Operation*> attachedVariants;
    std::copy_if(invariantUsers.begin(), invariantUsers.end(), std::back_inserter(attachedVariants),
                 [](mlir::Operation* op) {
                     return mlir::isa<VPUMI40XX::DPUVariantOp>(op);
                 });

    auto getTaskIndex = [](mlir::Operation* op) {
        auto variantOp = mlir::cast<VPUMI40XX::DPUVariantOp>(op);
        return variantOp.getIndexType().getValue();
    };

    auto firstVariant = std::min_element(attachedVariants.begin(), attachedVariants.end(),
                                         [&getTaskIndex](mlir::Operation* lhs, mlir::Operation* rhs) {
                                             return getTaskIndex(lhs) < getTaskIndex(rhs);
                                         });

    auto lastVariant = std::max_element(attachedVariants.begin(), attachedVariants.end(),
                                        [&getTaskIndex](mlir::Operation* lhs, mlir::Operation* rhs) {
                                            return getTaskIndex(lhs) < getTaskIndex(rhs);
                                        });

    auto variantsInGroupAttr = rewriter.getIntegerAttr(rewriter.getIntegerType(64, false), attachedVariants.size());
    auto firstVariantAttr =
            rewriter.getUI32IntegerAttr(mlir::cast<VPUMI40XX::DPUVariantOp>(*firstVariant).getIndexType().getValue());
    auto lastVariantAttr =
            rewriter.getUI32IntegerAttr(mlir::cast<VPUMI40XX::DPUVariantOp>(*lastVariant).getIndexType().getValue());

    mlir::TypeAttr outTypeContAttr = nullptr;
    if (op.getIsContinued()) {
        mlir::MLIRContext* ctx = rewriter.getContext();

        if (op.getOutputBuffs().size() != 1) {
            _log.error("Expected single output register buffer in case of continued convolution");
            return mlir::failure();
        }

        auto registerBuffer = mlir::cast<VPURT::DeclareBufferOp>(op.getOutputBuffs().front().getDefiningOp());

        auto section = registerBuffer.getSection();
        auto sectionIndexAttr = registerBuffer.getSectionIndex();
        auto sectionIndex =
                sectionIndexAttr.has_value() ? sectionIndexAttr.value()[0].cast<mlir::IntegerAttr>().getInt() : 0;
        auto byteOffset = registerBuffer.getByteOffset();

        auto location = VPUASM::MemLocationType::get(ctx, section, sectionIndex, byteOffset);
        auto memref = registerBuffer.getType().cast<mlir::MemRefType>();
        auto traits = VPUASM::BufferTraitsType::get(ctx, registerBuffer.getSwizzlingKey().value_or(0));

        auto buffType = VPUASM::BufferType::get(ctx, location, memref, traits);
        outTypeContAttr = mlir::TypeAttr::get(buffType);
    }

    auto invariant = rewriter.create<VPUASM::DPUInvariantOp>(
            op.getLoc(), symName, taskIdx, taskLocation, inputSym, inputSparsityMapSym, inputSETableSym, weightsSym,
            weightsSparsityMapSym, weightTableSym, weightTableDataPtrSym, weightTableSpPtrSym, weightTableScaleSym,
            weightTableBiasSym, weightZeroPointsSym, sprLookupTableSym, palletLookupTableSym, outputSym,
            outputSparsityMapSym, profilingDataSym, maxPerXYSym, minPerXYSym, minMaxPerTensorAttr, outTypeContAttr,
            waitAttr, updateAttr, op.getNceTaskTypeAttr(), op.getEltwiseTypeAttr(), op.getMpeFrequentModeAttr(),
            op.getMpeEngineAttr(), op.getKernelSizeAttr(), op.getKernelStridesAttr(), op.getKernelPaddingAttr(),
            op.getIsContinuedAttr(), op.getCmSpPatternAttr(), op.getInputChannelsCompressionAttr(),
            op.getIsZeroOffsetWeightsTableAttr(), op.getOutChannelOffsetAttr(), op.getIsSuperdenseAttr(),
            op.getIsInplaceAttr(), op.getInputSeSizeAttr(), op.getOutputSeSizeAttr(), op.getIsPermuteQuantizeAttr(),
            op.getIsSmallKernelOptimizedAttr(), op.getStartAfterAttr(), op.getCleanAfterAttr(), variantsInGroupAttr,
            firstVariantAttr, lastVariantAttr);
    {
        auto& ppeRegion = invariant.getPpe();
        ppeRegion.emplaceBlock();

        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToEnd(&ppeRegion.front());

        for (auto ppe : op.getPpe().getOps<VPUMI40XX::PPETaskOp>()) {
            rewriter.create<VPUASM::PPETaskOp>(rewriter.getUnknownLoc(), ppe->getResultTypes(), ppe->getOperands(),
                                               ppe->getAttrDictionary().getValue());
        }
    }

    rewriter.eraseOp(op);

    return SymbolizationResult(invariant);
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
