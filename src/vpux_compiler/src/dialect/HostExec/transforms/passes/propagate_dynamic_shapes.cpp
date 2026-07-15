//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/HostExec/IR/dialect.hpp"
#include "vpux/compiler/dialect/HostExec/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/core/IR/ops.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#include <optional>

namespace vpux {
namespace HostExec {
#define GEN_PASS_DECL_PROPAGATEDYNAMICSHAPES
#define GEN_PASS_DEF_PROPAGATEDYNAMICSHAPES
#include "vpux/compiler/dialect/HostExec/passes.hpp.inc"
}  // namespace HostExec
}  // namespace vpux

using namespace vpux;

namespace {

constexpr int32_t SHAPE_ELEMENT_WIDTH = 32;

// Shape buffers are modeled as static memref<rank x si32>.
bool isShapeMemRefType(mlir::MemRefType memrefType) {
    if (memrefType == nullptr) {
        return false;
    }

    auto* ctx = memrefType.getContext();
    const auto si32Type = mlir::IntegerType::get(ctx, SHAPE_ELEMENT_WIDTH, mlir::IntegerType::Signed);
    return memrefType.getRank() == 1 && memrefType.hasStaticShape() && memrefType.getElementType() == si32Type;
}

// Dynamic shape source is a direct dynamic memref passed to the call site.
std::optional<std::pair<mlir::Value, mlir::MemRefType>> getDynamicShapeSource(mlir::Value value) {
    const auto directType = mlir::dyn_cast<mlir::MemRefType>(value.getType());
    if (directType != nullptr && !directType.hasStaticShape()) {
        return std::make_pair(value, directType);
    }

    return std::nullopt;
}

void writeShapeToBuffer(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value source,
                        mlir::MemRefType sourceType, mlir::Value shapeBuffer, Logger log) {
    auto* ctx = rewriter.getContext();
    const auto shapeBufferType = mlir::cast<mlir::MemRefType>(shapeBuffer.getType());
    const auto shapeElemType = shapeBufferType.getElementType();
    const auto shapeElemIntType = mlir::dyn_cast<mlir::IntegerType>(shapeElemType);
    if (shapeElemIntType == nullptr) {
        log.trace("Skip writing shape: shape buffer element type is not integer: {0}", shapeElemType);
        return;
    }

    // arith.index_cast requires signless integer targets; materialize through signless type,
    // then adapt to the actual shape element type (e.g. si32) when needed.
    const auto signlessElemType = mlir::IntegerType::get(ctx, shapeElemIntType.getWidth());

    const auto rank = sourceType.getRank();
    const auto shapeBufferLen = shapeBufferType.getShape()[0];
    if (shapeBufferLen != rank) {
        log.trace("Skip writing shape: source rank {0} does not match shape buffer length {1}", rank, shapeBufferLen);
        return;
    }

    for (int64_t dimIdx = 0; dimIdx < rank; ++dimIdx) {
        auto storeIndex = rewriter.create<mlir::arith::ConstantIndexOp>(loc, dimIdx);
        const auto sourceDimIdx = rank - 1 - dimIdx;

        auto dimAsIndex = [&]() -> mlir::Value {
            auto sourceIndex = rewriter.create<mlir::arith::ConstantIndexOp>(loc, sourceDimIdx);
            if (sourceType.getShape()[sourceDimIdx] == mlir::ShapedType::kDynamic) {
                return rewriter.create<mlir::memref::DimOp>(loc, source, sourceIndex);
            }

            return rewriter.create<mlir::arith::ConstantIndexOp>(loc, sourceType.getShape()[sourceDimIdx]);
        }();

        auto dimSignless = rewriter.create<mlir::arith::IndexCastOp>(loc, signlessElemType, dimAsIndex);
        mlir::Value dimForStore = dimSignless;
        if (signlessElemType != shapeElemType) {
            dimForStore = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, shapeElemType, dimSignless.getResult())
                                  .getResult(0);
        }

        rewriter.create<mlir::memref::StoreOp>(loc, dimForStore, shapeBuffer, mlir::ValueRange{storeIndex});
    }
}

class PropagateDynamicShapesPass final : public HostExec::impl::PropagateDynamicShapesBase<PropagateDynamicShapesPass> {
public:
    explicit PropagateDynamicShapesPass(Logger log): _log(std::move(log)) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
    mlir::LogicalResult materializeInputShapesForNestedCall(Core::NestedCallOp nestedCall,
                                                            mlir::PatternRewriter& rewriter);

private:
    Logger _log;
};

// Purpose: Materialize dynamic input shapes at the host side before Core.NestedCall.
// The callee expects shape buffers (memref<rank x si32>) to be pre-filled with runtime values.
// Input shape buffers are populated in reverse dimension order. SW kernels expect shape data
// in this order and reverse the index internally when performing shape inference.
//
// Note: This uses memref.dim (similar to Reify/scf.for) to extract actual runtime extents
// from the input descriptor. After lowering to LLVM/UMD, shape buffers are passed as
// regular data operands to execute_graph with materialized runtime dimensions.
mlir::LogicalResult PropagateDynamicShapesPass::materializeInputShapesForNestedCall(Core::NestedCallOp nestedCall,
                                                                                    mlir::PatternRewriter& rewriter) {
    auto calleeFunc =
            mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(nestedCall, nestedCall.getCalleeAttr());
    if (calleeFunc == nullptr) {
        _log.trace("Failed to materialize input shapes: no valid callee func op was found for NestedCall '{0}'",
                   nestedCall.getCallee());
        return mlir::failure();
    }

    // NestedCall operands are split as: [input operands..., output operands...].
    // The number of outputs equals the number of results.
    const auto operands = nestedCall.getOperands();
    const auto numResults = nestedCall.getNumResults();
    if (operands.size() < numResults) {
        _log.trace("Failed to materialize input shapes for NestedCall '{0}': operand count {1} is smaller than "
                   "result count {2}",
                   nestedCall.getCallee(), operands.size(), numResults);
        return mlir::failure();
    }

    const auto numInputOperands = operands.size() - numResults;
    const auto inputOperands = operands.take_front(numInputOperands);

    if (calleeFunc.getNumArguments() < numInputOperands) {
        _log.trace("Failed to materialize input shapes for NestedCall '{0}': callee arg count {1} is smaller than "
                   "input operand count {2}",
                   nestedCall.getCallee(), calleeFunc.getNumArguments(), numInputOperands);
        return mlir::failure();
    }

    SmallVector<std::pair<mlir::Value, mlir::MemRefType>> dynamicInputSources;
    SmallVector<mlir::Value> inputShapeOperands;

    for (size_t i = 0; i < numInputOperands; ++i) {
        const auto calleeArgType = mlir::dyn_cast<mlir::MemRefType>(calleeFunc.getArgument(i).getType());
        if (isShapeMemRefType(calleeArgType)) {
            inputShapeOperands.push_back(inputOperands[i]);
            continue;
        }

        auto source = getDynamicShapeSource(inputOperands[i]);
        if (source.has_value()) {
            dynamicInputSources.push_back(source.value());
        }
    }

    if (inputShapeOperands.empty()) {
        _log.trace("Skip input shape materialization for NestedCall '{0}': no explicit input shape operands",
                   nestedCall.getCallee());
        return mlir::success();
    }

    if (dynamicInputSources.size() != inputShapeOperands.size()) {
        _log.trace("Failed to materialize input shapes for NestedCall '{0}': dynamic input source count {1} does "
                   "not match input shape operand count {2}",
                   nestedCall.getCallee(), dynamicInputSources.size(), inputShapeOperands.size());
        nestedCall.emitError("mismatched dynamic input/source pairing: expected the same number of dynamic input "
                             "sources and input shape operands, but got ")
                << dynamicInputSources.size() << " dynamic input source(s) and " << inputShapeOperands.size()
                << " input shape operand(s)";
        return mlir::failure();
    }
    const auto numPairs = dynamicInputSources.size();
    if (numPairs == 0) {
        _log.trace("No input shape buffers to materialize for NestedCall '{0}'", nestedCall.getCallee());
        return mlir::success();
    }

    rewriter.setInsertionPoint(nestedCall);
    for (size_t i = 0; i < numPairs; ++i) {
        const auto& [shapeSource, shapeSourceType] = dynamicInputSources[i];
        writeShapeToBuffer(rewriter, nestedCall.getLoc(), shapeSource, shapeSourceType, inputShapeOperands[i],
                           _log.nest(2));
    }

    _log.trace("Materialized {0} input shape buffer(s) for NestedCall '{1}'", numPairs, nestedCall.getCallee());
    return mlir::success();
}

void PropagateDynamicShapesPass::safeRunOnModule() {
    auto module = getOperation();

    _log.trace("Starting PropagateDynamicShapes pass (call-site-only mode)");

    const auto walkResult = module.walk([&](Core::NestedCallOp nestedCall) -> mlir::WalkResult {
        mlir::PatternRewriter rewriter(module.getContext());
        if (mlir::failed(materializeInputShapesForNestedCall(nestedCall, rewriter))) {
            nestedCall.emitError("failed to materialize input shapes for nested call");
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
    });

    if (walkResult.wasInterrupted()) {
        signalPassFailure();
        return;
    }

    _log.trace("Completed PropagateDynamicShapes pass");
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::HostExec::createPropagateDynamicShapesPass(Logger log) {
    return std::make_unique<PropagateDynamicShapesPass>(log);
}
