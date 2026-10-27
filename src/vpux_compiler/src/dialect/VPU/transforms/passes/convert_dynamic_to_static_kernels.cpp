//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_analysis_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/IRMapping.h>

#include "vpux/compiler/dialect/HostExec/params.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_CONVERTDYNAMICTOSTATICKERNELS
#define GEN_PASS_DEF_CONVERTDYNAMICTOSTATICKERNELS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

// Constants for container sizes
static constexpr size_t DEFAULT_OPERATION_SET_SIZE = 32;
static constexpr size_t DEFAULT_NEIGHBOR_SET_SIZE = 16;

//
// ConvertDynamicToStaticKernelsPass
//
class ConvertDynamicToStaticKernelsPass final :
        public VPU::impl::ConvertDynamicToStaticKernelsBase<ConvertDynamicToStaticKernelsPass> {
public:
    explicit ConvertDynamicToStaticKernelsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;

    bool handleExtractSliceOp(mlir::tensor::ExtractSliceOp extractSliceOp,
                              llvm::DenseMap<mlir::Value, mlir::RankedTensorType>& mapValueToStaticShape);
    bool handleCastOp(mlir::tensor::CastOp castOp,
                      llvm::DenseMap<mlir::Value, mlir::RankedTensorType>& mapValueToStaticShape,
                      SmallVector<mlir::tensor::CastOp>& castOpsToErase);
    void handleIndexSwitchOp(mlir::ModuleOp moduleOp, mlir::scf::IndexSwitchOp switchOp,
                             llvm::DenseMap<mlir::Value, mlir::RankedTensorType>& mapValueToStaticShape,
                             llvm::SmallPtrSet<mlir::Operation*, DEFAULT_OPERATION_SET_SIZE>& visitedCallOps);
    mlir::func::CallOp handleCallOp(mlir::ModuleOp moduleOp, mlir::OpBuilder& builder, mlir::func::CallOp callOp,
                                    llvm::DenseMap<mlir::Value, mlir::RankedTensorType>& mapValueToStaticShape);
    mlir::func::CallOp rewriteCallOpWithStaticType(
            mlir::func::CallOp callOp, mlir::ModuleOp moduleOp, mlir::OpBuilder& builder,
            const llvm::DenseMap<mlir::Value, mlir::RankedTensorType>& mapValueToStaticShape);
};

mlir::func::FuncOp createStaticFuncOp(mlir::func::FuncOp dynFuncOp, SmallVector<mlir::Type> newFuncResultTypes,
                                      mlir::ModuleOp moduleOp) {
    mlir::OpBuilder builder(moduleOp.getContext());
    auto newFuncName = dynFuncOp.getName().str() + "_static";
    assert(moduleOp.lookupSymbol<mlir::func::FuncOp>(newFuncName) == nullptr && "static function already exists");

    auto newFuncType = mlir::FunctionType::get(moduleOp->getContext(), mlir::TypeRange(newFuncResultTypes), {});
    return VPU::cloneFuncOp(dynFuncOp, newFuncName, newFuncType);
}

// Set dynamic shape attributes on function arguments and results. The attributes are used in AddNetInfoToModule pass
// to specify which network data info have dynamic strides.
static void setDynamicShapeAttributes(mlir::OpBuilder& builder, mlir::func::FuncOp oldFuncOp,
                                      mlir::func::FuncOp newFuncOp) {
    llvm::StringRef dynmicStridesAttrName = vpux::HostExec::HOST_EXEC_FUNC_ARG_DYNAMIC_STRIDES_ATTR_NAME;

    for (auto [idx, argType] : llvm::enumerate(oldFuncOp.getArgumentTypes())) {
        auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(argType);
        if (!tensorType.hasStaticShape()) {
            newFuncOp.setArgAttr(idx, dynmicStridesAttrName, builder.getBoolAttr(true));
        }
    }

    for (auto [idx, resultType] : llvm::enumerate(oldFuncOp.getResultTypes())) {
        auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(resultType);
        if (!tensorType.hasStaticShape()) {
            newFuncOp.setResultAttr(idx, dynmicStridesAttrName, builder.getBoolAttr(true));
        }
    }
}

mlir::func::CallOp ConvertDynamicToStaticKernelsPass::rewriteCallOpWithStaticType(
        mlir::func::CallOp callOp, mlir::ModuleOp moduleOp, mlir::OpBuilder& builder,
        const llvm::DenseMap<mlir::Value, mlir::RankedTensorType>& mapValueToStaticShape) {
    SmallVector<mlir::Type> staticOperands;
    SmallVector<mlir::Value> staticOperandsValues;
    for (auto operand : callOp.getOperands()) {
        if (IE::hasDynamicShape(operand)) {
            assert(mapValueToStaticShape.contains(operand) &&
                   "Expected static shape mapping for dynamic tensor operand");
            staticOperands.push_back(mapValueToStaticShape.at(operand));
        } else {
            staticOperands.push_back(operand.getType());
        }

        if (auto castOp = mlir::dyn_cast<mlir::tensor::CastOp>(operand.getDefiningOp())) {
            staticOperandsValues.push_back(castOp.getSource());
        } else {
            assert(!IE::hasDynamicShape(operand) && "Expected static tensor operand for function call");
            staticOperandsValues.push_back(operand);
        }
    }

    auto funcOp = moduleOp.lookupSymbol<mlir::func::FuncOp>(callOp.getCallee());
    mlir::func::FuncOp newFuncOp = nullptr;
    if (funcOp == nullptr) {
        auto staticFuncOpName = callOp.getCallee().str() + "_static";
        newFuncOp = moduleOp.lookupSymbol<mlir::func::FuncOp>(staticFuncOpName);
    } else {
        newFuncOp = createStaticFuncOp(funcOp, staticOperands, moduleOp);
        setDynamicShapeAttributes(builder, funcOp, newFuncOp);
    }
    assert(newFuncOp != nullptr && "Expected to find or create a static function");

    builder.setInsertionPoint(callOp);
    for (auto it : llvm::enumerate(llvm::zip(staticOperands, newFuncOp.getArgumentTypes()))) {
        auto idx = it.index();
        auto [staticType, newType] = it.value();
        if (staticType != newType) {
            if (idx >= staticOperandsValues.size()) {
                continue;  // Skip if index is out of bounds
            }

            auto extractSliceOp =
                    mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(staticOperandsValues[idx].getDefiningOp());
            assert(extractSliceOp != nullptr && "Expected extract_slice operation for dynamic tensor operand");

            auto rankedType = mlir::dyn_cast<mlir::RankedTensorType>(newType);
            SmallVector<mlir::OpFoldResult> newSizes = {};
            for (auto s : rankedType.getShape()) {
                newSizes.push_back(builder.getI64IntegerAttr(s));
            }
            auto newSliceOp = mlir::tensor::ExtractSliceOp::create(
                    builder, takeOpLoc(callOp, "extract_slice"), rankedType, staticOperandsValues[idx],
                    extractSliceOp.getMixedOffsets(), newSizes, extractSliceOp.getMixedStrides());
            staticOperandsValues[idx] = newSliceOp.getResult();
        }
    }

    // create a callOp with the new static function
    auto staticCallOp = mlir::func::CallOp::create(builder, callOp.getLoc(), newFuncOp.getName(),
                                                   newFuncOp.getFunctionType().getResults(), staticOperandsValues);

    callOp.replaceAllUsesWith(staticCallOp);
    if (callOp.use_empty()) {
        callOp.erase();
    }

    if (funcOp && funcOp.use_empty()) {
        funcOp.erase();
    }

    return staticCallOp;
}

void ConvertDynamicToStaticKernelsPass::handleIndexSwitchOp(
        mlir::ModuleOp moduleOp, mlir::scf::IndexSwitchOp switchOp,
        llvm::DenseMap<mlir::Value, mlir::RankedTensorType>& mapValueToStaticShape,
        llvm::SmallPtrSet<mlir::Operation*, DEFAULT_OPERATION_SET_SIZE>& visitedCallOps) {
    SmallVector<mlir::Type> returnTypes;
    for (auto& region : llvm::make_early_inc_range(switchOp->getRegions())) {
        for (auto& block : llvm::make_early_inc_range(region)) {
            llvm::DenseMap<mlir::Value, mlir::RankedTensorType> localMapValueToStaticShape = mapValueToStaticShape;
            for (auto& operation : llvm::make_early_inc_range(block)) {
                if (mlir::isa<mlir::func::CallOp>(operation)) {
                    auto callOp = mlir::cast<mlir::func::CallOp>(operation);
                    if (!visitedCallOps.contains(callOp)) {
                        mlir::OpBuilder blockBuilder(&block, block.begin());
                        auto newCallOp = handleCallOp(moduleOp, blockBuilder, callOp, localMapValueToStaticShape);
                        visitedCallOps.insert(newCallOp.getOperation());
                    }
                } else if (mlir::isa<mlir::tensor::ExtractSliceOp>(operation)) {
                    auto extractSliceOp = mlir::cast<mlir::tensor::ExtractSliceOp>(operation);
                    handleExtractSliceOp(extractSliceOp, localMapValueToStaticShape);
                } else if (mlir::isa<mlir::tensor::CastOp>(operation)) {
                    auto castOp = mlir::cast<mlir::tensor::CastOp>(operation);
                    SmallVector<mlir::tensor::CastOp> castOpsToErase;
                    handleCastOp(castOp, localMapValueToStaticShape, castOpsToErase);
                    for (auto op : castOpsToErase) {
                        if (op.use_empty()) {
                            op.erase();
                        }
                    }
                }
            }

            auto terminator = block.getTerminator();
            auto yieldOp = mlir::dyn_cast<mlir::scf::YieldOp>(terminator);
            assert(yieldOp != nullptr && "Expected yield operation as terminator of the case block");

            if (returnTypes.empty()) {
                for (auto operand : yieldOp.getOperands()) {
                    returnTypes.push_back(operand.getType());
                }
            }
        }
    }

    if (!returnTypes.empty()) {
        for (auto [idx, type] : llvm::enumerate(returnTypes)) {
            switchOp->getResult(idx).setType(type);
        }
    }
}

bool ConvertDynamicToStaticKernelsPass::handleExtractSliceOp(
        mlir::tensor::ExtractSliceOp extractSliceOp,
        llvm::DenseMap<mlir::Value, mlir::RankedTensorType>& mapValueToStaticShape) {
    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(extractSliceOp.getResultType());
    if (!resultType.hasStaticShape()) {
        _log.trace("Found dynamic tensor in extract_slice operation");
        return false;
    }

    auto newRankedTensorType = removeBoundsAttr(resultType);
    extractSliceOp.getResult().setType(newRankedTensorType);
    mapValueToStaticShape[extractSliceOp.getResult()] = newRankedTensorType;
    return true;
}

bool ConvertDynamicToStaticKernelsPass::handleCastOp(
        mlir::tensor::CastOp castOp, llvm::DenseMap<mlir::Value, mlir::RankedTensorType>& mapValueToStaticShape,
        SmallVector<mlir::tensor::CastOp>& castOpsToErase) {
    auto srcType = mlir::dyn_cast<mlir::RankedTensorType>(castOp.getSource().getType());
    if (!srcType.hasStaticShape()) {
        _log.trace("Found dynamic tensor in cast operation");
        return false;
    }

    auto newRankedTensorType = removeBoundsAttr(srcType);
    castOp.getResult().setType(newRankedTensorType);
    mapValueToStaticShape[castOp.getResult()] = newRankedTensorType;
    castOpsToErase.push_back(castOp);
    return true;
}

mlir::func::CallOp ConvertDynamicToStaticKernelsPass::handleCallOp(
        mlir::ModuleOp moduleOp, mlir::OpBuilder& builder, mlir::func::CallOp callOp,
        llvm::DenseMap<mlir::Value, mlir::RankedTensorType>& mapValueToStaticShape) {
    auto newCallOp = rewriteCallOpWithStaticType(callOp, moduleOp, builder, mapValueToStaticShape);
    for (auto newOperand : newCallOp.getResults()) {
        mapValueToStaticShape[newOperand] = mlir::dyn_cast<mlir::RankedTensorType>(newOperand.getType());
    }
    return newCallOp;
}

void ConvertDynamicToStaticKernelsPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    auto mainFuncOp = net::getMainFunc(moduleOp);
    mlir::OpBuilder builder(moduleOp.getContext());

    auto forOps = mainFuncOp.getOps<mlir::scf::ForOp>();
    if (forOps.empty()) {
        _log.trace("No scf.for operations found in the main function. Skipping convert_dynamic_to_static pass.");
        return;
    }

    auto defGetNeighbors =
            [&](mlir::Operation* op) -> llvm::SmallSetVector<mlir::Operation*, DEFAULT_NEIGHBOR_SET_SIZE> {
        llvm::SmallSetVector<mlir::Operation*, DEFAULT_NEIGHBOR_SET_SIZE> neighbors;
        for (auto operand : op->getOperands()) {
            if (auto definingOp = operand.getDefiningOp()) {
                neighbors.insert(definingOp);
            }
        }
        return neighbors;
    };

    auto getNeighbors = [&](mlir::Operation* op) -> llvm::SmallSetVector<mlir::Operation*, DEFAULT_NEIGHBOR_SET_SIZE> {
        llvm::SmallSetVector<mlir::Operation*, DEFAULT_NEIGHBOR_SET_SIZE> neighbors = {};
        if (auto insertSliceOp = mlir::dyn_cast<mlir::tensor::InsertSliceOp>(op)) {
            neighbors.insert(insertSliceOp.getSource().getDefiningOp());
        } else if (auto switchOp = mlir::dyn_cast<mlir::scf::IndexSwitchOp>(op)) {
            for (auto& region : switchOp.getCaseRegions()) {
                for (auto callOp : region.getOps<mlir::func::CallOp>()) {
                    for (auto operand : callOp.getOperands()) {
                        if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(operand.getType())) {
                            auto definingOp = operand.getDefiningOp();
                            if (!tensorType.hasStaticShape() && (definingOp != nullptr)) {
                                neighbors.insert(definingOp);
                            }
                        }
                    }
                }
            }
        } else {
            neighbors = defGetNeighbors(op);
        }

        return neighbors;
    };

    auto stopSearch = [&](mlir::Operation* op) {
        return mlir::isa<mlir::tensor::ExtractSliceOp>(op);
    };

    mlir::WalkResult walkResult = mainFuncOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::scf::ForOp forOp) {
        auto insertSliceOps = forOp.getOps<mlir::tensor::InsertSliceOp>();
        if (insertSliceOps.empty()) {
            return mlir::WalkResult::advance();
        }

        SmallVector<mlir::Operation*> startNodes;
        llvm::DenseMap<mlir::Operation*, mlir::RankedTensorType> insertSliceOrigInputTypes;
        for (auto sliceOp : insertSliceOps) {
            startNodes.push_back(sliceOp.getOperation());
            insertSliceOrigInputTypes[sliceOp.getOperation()] = sliceOp.getSourceType();
        }

        SmallVector<mlir::tensor::CastOp> castOps;
        auto sortedResults = vpux::VPU::collectOpsInTopologicalOrder(startNodes, getNeighbors, stopSearch);

        llvm::DenseMap<mlir::Value, mlir::RankedTensorType> mapValueToStaticShape;
        llvm::SmallPtrSet<mlir::Operation*, DEFAULT_OPERATION_SET_SIZE> visitedCallOps;
        for (auto op : sortedResults) {
            if (mlir::isa<mlir::tensor::ExtractSliceOp>(op)) {
                handleExtractSliceOp(mlir::cast<mlir::tensor::ExtractSliceOp>(op), mapValueToStaticShape);
            } else if (mlir::isa<mlir::tensor::InsertSliceOp>(op)) {
                auto sliceOp = mlir::cast<mlir::tensor::InsertSliceOp>(op);
                auto src = sliceOp.getSource();
                if (!mlir::isa<mlir::tensor::CastOp>(src.getDefiningOp())) {
                    mlir::OpBuilder castBuilder(sliceOp);
                    auto castOp = mlir::tensor::CastOp::create(castBuilder, sliceOp->getLoc(),
                                                               insertSliceOrigInputTypes[op], src);
                    sliceOp.setOperand(0, castOp.getResult());
                }
            } else if (mlir::isa<mlir::tensor::CastOp>(op)) {
                handleCastOp(mlir::cast<mlir::tensor::CastOp>(op), mapValueToStaticShape, castOps);
            } else if (mlir::isa<mlir::func::CallOp>(op)) {
                if (!visitedCallOps.contains(op)) {
                    auto newCallOp =
                            handleCallOp(moduleOp, builder, mlir::cast<mlir::func::CallOp>(op), mapValueToStaticShape);
                    visitedCallOps.insert(newCallOp.getOperation());
                }
            } else if (mlir::isa<mlir::scf::IndexSwitchOp>(op)) {
                auto switchOp = mlir::cast<mlir::scf::IndexSwitchOp>(op);
                handleIndexSwitchOp(moduleOp, switchOp, mapValueToStaticShape, visitedCallOps);
            }
        }

        // At this point all tensor types call operations in the chain should have static shapes
        // Intermediate cast operations that cast from static to dynamic can be removed
        for (auto castOp : llvm::make_early_inc_range(castOps)) {
            if (castOp.getSource().getType() == castOp.getResult().getType()) {
                castOp.getResult().replaceAllUsesWith(castOp.getSource());
            }

            if (castOp->use_empty()) {
                castOp.erase();
            }
        }
        return mlir::WalkResult::advance();
    });

    if (walkResult.wasInterrupted()) {
        signalPassFailure();
        return;
    }
}

}  // namespace

//
// createConvertDynamicToStaticKernelsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createConvertDynamicToStaticKernelsPass(Logger log) {
    return std::make_unique<ConvertDynamicToStaticKernelsPass>(log);
}
