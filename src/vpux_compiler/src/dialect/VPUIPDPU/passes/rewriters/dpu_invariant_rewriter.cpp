//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIPDPU/rewriters/dpu_invariant_rewriter.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux::VPUIPDPU;

namespace {

template <typename CfgOp>
mlir::LogicalResult insertEntryBlock(mlir::OpBuilder& builder, mlir::Block* block, const mlir::Location& loc,
                                     const Logger& log) {
    builder.setInsertionPointToEnd(block);

    auto cfgOp = builder.create<CfgOp>(loc);
    auto& region = cfgOp.getOperation()->getRegion(0);
    auto entryBlock = builder.createBlock(&region);

    if (entryBlock == nullptr) {
        log.error("Error creating entry block for {0}", typeid(CfgOp).name());
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult insertInvBlockArgs(VPUASM::DPUInvariantOp op, const Logger& log, mlir::Block* invBlock,
                                       std::unordered_map<BlockArg, size_t>& invBlockArgsPos,
                                       ELF::SymbolReferenceMap& symRefMap) {
    // input activations
    if (op.getInput()) {
        auto inputType = getBufferType(symRefMap.lookupSymbol(op.getInput().value()));
        invBlock->addArgument(inputType, op.getLoc());
        invBlockArgsPos[BlockArg::ACT_IN] = invBlock->getNumArguments() - 1;
    }

    // input storage elements
    if (op.getInputStorageElementTable()) {
        auto inputSETableType = getBufferType(symRefMap.lookupSymbol(op.getInputStorageElementTable().value()));
        invBlock->addArgument(inputSETableType, op.getLoc());
        invBlockArgsPos[BlockArg::ACT_SE_IN] = invBlock->getNumArguments() - 1;
    }

    // input sparsity
    if (op.getInputSparsityMap()) {
        auto inputSparsityMapType = getBufferType(symRefMap.lookupSymbol(op.getInputSparsityMap().value()));
        invBlock->addArgument(inputSparsityMapType, op.getLoc());
        invBlockArgsPos[BlockArg::ACT_SPARSE_MAP_IN] = invBlock->getNumArguments() - 1;
    }

    // weights table
    if (op.getWeightTable()) {
        auto weightTableType = getBufferType(symRefMap.lookupSymbol(op.getWeightTable().value()));
        invBlock->addArgument(weightTableType, op.getLoc());
        invBlockArgsPos[BlockArg::WEIGHTS_TABLE] = invBlock->getNumArguments() - 1;
    }

    // weights table data ptr
    if (op.getWeightTableDataPtr()) {
        auto weightTableDataPtrType = getBufferType(symRefMap.lookupSymbol(op.getWeightTableDataPtr().value()));
        invBlock->addArgument(weightTableDataPtrType, op.getLoc());
        invBlockArgsPos[BlockArg::WEIGHTS_TABLE_DATA_PTR] = invBlock->getNumArguments() - 1;
    }

    // weights table sp ptr
    if (op.getWeightTableSpPtr()) {
        auto weightTableSpPtrType = getBufferType(symRefMap.lookupSymbol(op.getWeightTableSpPtr().value()));
        invBlock->addArgument(weightTableSpPtrType, op.getLoc());
        invBlockArgsPos[BlockArg::WEIGHTS_TABLE_SP_PTR] = invBlock->getNumArguments() - 1;
    }

    // weights table scale
    if (op.getWeightTableScale()) {
        auto weightTableScaleType = getBufferType(symRefMap.lookupSymbol(op.getWeightTableScale().value()));
        invBlock->addArgument(weightTableScaleType, op.getLoc());
        invBlockArgsPos[BlockArg::WEIGHTS_TABLE_SCALE] = invBlock->getNumArguments() - 1;
    }

    // weights table bias
    if (op.getWeightTableBias()) {
        auto weightTableBiasType = getBufferType(symRefMap.lookupSymbol(op.getWeightTableBias().value()));
        invBlock->addArgument(weightTableBiasType, op.getLoc());
        invBlockArgsPos[BlockArg::WEIGHTS_TABLE_BIAS] = invBlock->getNumArguments() - 1;
    }

    // weights table bias
    if (op.getWeightZeroPoints()) {
        auto weightZeroPointsType = getBufferType(symRefMap.lookupSymbol(op.getWeightZeroPoints().value()));
        invBlock->addArgument(weightZeroPointsType, op.getLoc());
        invBlockArgsPos[BlockArg::WEIGHTS_ZERO_POINTS] = invBlock->getNumArguments() - 1;
    }

    // weights
    if (op.getWeights()) {
        auto weightsType = getBufferType(symRefMap.lookupSymbol(op.getWeights().value()));
        invBlock->addArgument(weightsType, op.getLoc());
        invBlockArgsPos[BlockArg::WEIGHTS] = invBlock->getNumArguments() - 1;
    }

    // weights sparsity
    if (op.getWeightsSparsityMap()) {
        auto weightsSparsityMapType = getBufferType(symRefMap.lookupSymbol(op.getWeightsSparsityMap().value()));
        invBlock->addArgument(weightsSparsityMapType, op.getLoc());
        invBlockArgsPos[BlockArg::WEIGHTS_SPARSE_MAP] = invBlock->getNumArguments() - 1;
    }

    // spr lookup table
    if (op.getSprLookupTable()) {
        auto sprLookupTableType = getBufferType(symRefMap.lookupSymbol(op.getSprLookupTable().value()));
        invBlock->addArgument(sprLookupTableType, op.getLoc());
        invBlockArgsPos[BlockArg::SPR_LOOKUP_TABLE] = invBlock->getNumArguments() - 1;
    }

    // pallet lookup table
    if (op.getPalletLookupTable()) {
        auto palletLookupTableType = getBufferType(symRefMap.lookupSymbol(op.getPalletLookupTable().value()));
        invBlock->addArgument(palletLookupTableType, op.getLoc());
        invBlockArgsPos[BlockArg::PALLET_LOOKUP_TABLE] = invBlock->getNumArguments() - 1;
    }

    // output activations
    mlir::MemRefType outType;
    if (!op.getIsContinued() && op.getOutput()) {
        outType = getBufferType(symRefMap.lookupSymbol(op.getOutput().value()));
    } else if (op.getIsContinued() && op.getOutputTypeContinued()) {
        outType = op.getOutputTypeContinued().value().getMemref();
    } else {
        log.error("Expected either output buffer or output type for continued mode");
        return mlir::failure();
    }
    invBlock->addArgument(outType, op.getLoc());
    invBlockArgsPos[BlockArg::ACT_OUT] = invBlock->getNumArguments() - 1;

    // output sparsity
    if (op.getOutputSparsityMap()) {
        auto outputSparsityMapType = getBufferType(symRefMap.lookupSymbol(op.getOutputSparsityMap().value()));
        invBlock->addArgument(outputSparsityMapType, op.getLoc());
        invBlockArgsPos[BlockArg::ACT_SPARSE_MAP_OUT] = invBlock->getNumArguments() - 1;
    }

    // dynamic sequence length
    if (op.getDynamicSequenceLength()) {
        auto dynamicSequenceLength = getBufferType(symRefMap.lookupSymbol(op.getDynamicSequenceLength().value()));
        invBlock->addArgument(dynamicSequenceLength, op.getLoc());
        invBlockArgsPos[BlockArg::DYNAMIC_SEQUENCE_LENGTH] = invBlock->getNumArguments() - 1;
    }

    return mlir::success();
}

}  // namespace

namespace vpux {
namespace VPUIPDPU {
DPUInvariantRewriter::DPUInvariantRewriter(
        mlir::MLIRContext* ctx, Logger log, ELF::SymbolReferenceMap& symRefMap,
        VPURegMapped::NPU5PPEBackwardsCompatibilityMode npu5PPEBackwardsCompatibilityMode)
        : mlir::OpRewritePattern<VPUASM::DPUInvariantOp>(ctx),
          _log(log),
          _symRefMap(symRefMap),
          _npu5PPEBackwardsCompatibilityMode(npu5PPEBackwardsCompatibilityMode) {
    setDebugName("DPUInvariant_VPUIPDPURewriter");
}
mlir::LogicalResult DPUInvariantRewriter::matchAndRewrite(VPUASM::DPUInvariantOp op,
                                                          mlir::PatternRewriter& rewriter) const {
    auto inv = rewriter.create<VPUIPDPU::DPUInvariantOp>(
            op.getLoc(), op.getSymNameAttr(), op.getTaskIndexAttr(), op.getTaskLocationAttr(), op.getInputAttr(),
            op.getInputSparsityMapAttr(), op.getInputStorageElementTableAttr(), op.getWeightsAttr(),
            op.getWeightsSparsityMapAttr(), op.getWeightTableAttr(), op.getWeightTableDataPtrAttr(),
            op.getWeightTableScaleAttr(), op.getSprLookupTableAttr(), op.getPalletLookupTableAttr(), op.getOutputAttr(),
            op.getOutputSparsityMapAttr(), op.getProfilingDataAttr(), op.getIsZeroOffsetWeightsTableAttr(),
            op.getMaxPerXyAttr(), op.getMinPerXyAttr(), op.getMinMaxPerTensorAttr(), op.getDynamicSequenceLengthAttr(),
            op.getNceTaskTypeAttr(), op.getIsContinuedAttr());

    auto& invRegion = inv.getRegion();
    auto invBlock = rewriter.createBlock(&invRegion);
    std::unordered_map<BlockArg, size_t> invBlockArgsPos;

    auto dpuInvariantExpandIface = mlir::dyn_cast<VPUASM::DPUInvariantExpandOpInterface>(op.getOperation());
    if (dpuInvariantExpandIface == nullptr) {
        _log.error("Missing expand DPU invariant configuration interface for arch {0}",
                   stringifyArchKind(config::getArch(op)).str());
        return mlir::failure();
    }

    {
        mlir::OpBuilder::InsertionGuard guard(rewriter);

        if (insertInvBlockArgs(op, _log, invBlock, invBlockArgsPos, _symRefMap).failed()) {
            return mlir::failure();
        }

        if (dpuInvariantExpandIface.expandGeneralConfig(rewriter).failed()) {
            return mlir::failure();
        }

        if (insertEntryBlock<VPUIPDPU::IDUCfgOp>(rewriter, invBlock, op.getLoc(), _log).failed()) {
            return mlir::failure();
        }
        if (dpuInvariantExpandIface
                    .expandIDUConfig(rewriter, _log, invBlock, invBlockArgsPos, _symRefMap,
                                     _npu5PPEBackwardsCompatibilityMode)
                    .failed()) {
            return mlir::failure();
        }
        if (insertEntryBlock<VPUIPDPU::MPECfgOp>(rewriter, invBlock, op.getLoc(), _log).failed()) {
            return mlir::failure();
        }
        if (dpuInvariantExpandIface.expandMPEConfig(rewriter, _log, invBlock, invBlockArgsPos, _symRefMap).failed()) {
            return mlir::failure();
        }

        if (insertEntryBlock<VPUIPDPU::PPECfgOp>(rewriter, invBlock, op.getLoc(), _log).failed()) {
            return mlir::failure();
        }
        if (dpuInvariantExpandIface
                    .expandPPEConfig(rewriter, _log, invBlock, invBlockArgsPos, _symRefMap,
                                     _npu5PPEBackwardsCompatibilityMode)
                    .failed()) {
            return mlir::failure();
        }
        if (insertEntryBlock<VPUIPDPU::ODUCfgOp>(rewriter, invBlock, op.getLoc(), _log).failed()) {
            return mlir::failure();
        }
        if (dpuInvariantExpandIface.expandODUConfig(rewriter, _log, invBlock, invBlockArgsPos, _symRefMap).failed()) {
            return mlir::failure();
        }
    }

    rewriter.create<VPUIPDPU::BarrierCfgOp>(op.getLoc(), op.getWaitBarriers(), op.getUpdateBarriers(),
                                            op.getStartAfterAttr(), op.getCleanAfterAttr());

    rewriter.create<VPUIPDPU::DPUGroupOp>(op.getLoc(), op.getIndexType(), op.getVariantCount());

    rewriter.eraseOp(op);

    return mlir::success();
}

}  // namespace VPUIPDPU
}  // namespace vpux
