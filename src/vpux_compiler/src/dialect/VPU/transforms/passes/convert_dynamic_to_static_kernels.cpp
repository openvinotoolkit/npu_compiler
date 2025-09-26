//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/IRMapping.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_CONVERTDYNAMICTOSTATICKERNELS
#define GEN_PASS_DEF_CONVERTDYNAMICTOSTATICKERNELS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

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
    bool adjustIndexIntoDynamicTensor(mlir::scf::ForOp forOp);

    auto isDynamicTensor(mlir::Value value) {
        if (auto rankedTensorType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType())) {
            return !rankedTensorType.hasStaticShape();
        }
        return false;
    };

    auto checkForDynamicTensor(mlir::Operation* op) {
        for (auto operand : op->getOperands()) {
            if (isDynamicTensor(operand)) {
                return true;
            }
        }

        for (auto result : op->getResults()) {
            if (isDynamicTensor(result)) {
                return true;
            }
        }

        return false;
    };

    bool isSupportedDynamicTensorType(mlir::Value value) {
        if (auto rankedTensorType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType())) {
            if (llvm::count(rankedTensorType.getShape(), mlir::ShapedType::kDynamic) == 1) {
                return true;
            } else {
                mlir::emitError(value.getLoc(), "Expected a ranked tensor type with exactly one dynamic dimension");
            }
        }
        return false;
    };

    void tryAndInferStaticShapes(mlir::Value value, int64_t staticDimSize);
    mlir::RankedTensorType inferStaticTensorType(mlir::Value value, int64_t staticDimSize, bool keepBounds = true);
    mlir::func::FuncOp createStaticFuncOp(mlir::func::FuncOp dynFuncOp, const mlir::FunctionType& staticFuncType,
                                          mlir::ModuleOp moduleOp, int64_t staticDimSize, bool eraseDynamicFunc = true);
    void addConversionCast(mlir::Operation* sliceOp, int64_t staticDimSize);
};

/**
 * @brief Adjusts indices into dynamic tensors within scf.for loops based on the current index and fixed step size.
 * When the remaining elements in the current iteration are fewer than the step size, this method backtracks the index
 * to ensure extraction of a static-shaped tensor.
 *
 * For example, given a dynamic tensor of shape <1x1x32x?xfp16> with bounds [1, 1, 32, 1000] and step size 100,
 * processing an input tensor of <1x1x32x250xfp16> would normally have the last iteration at index 200,
 * but only 50 elements would remain. This method adjusts the index to 150 (extracting a slice
 * from offset 150 with size 100), resulting in a static-shaped tensor <1x1x32x100xfp16>.
 */
bool ConvertDynamicToStaticKernelsPass::adjustIndexIntoDynamicTensor(mlir::scf::ForOp forOp) {
    auto affineMinOps = forOp.getOps<mlir::affine::AffineMinOp>();
    if (affineMinOps.empty()) {
        _log.error("affine.min operation not found in the scf.for loop body. Skipping conversion for this loop.");
        return false;
    }

    if (std::distance(affineMinOps.begin(), affineMinOps.end()) > 1) {
        _log.warning("Multiple affine.min operations found in the scf.for loop body !!");
    }

    auto affineMinOp = *affineMinOps.begin();
    mlir::OpBuilder builder(affineMinOp);
    builder.setInsertionPointAfter(affineMinOp);
    auto minValue = affineMinOp.getResult();
    auto cmpOp = builder.create<mlir::arith::CmpIOp>(forOp.getLoc(), mlir::arith::CmpIPredicate::ne, minValue,
                                                     forOp.getStep());
    auto ifOp = builder.create<mlir::scf::IfOp>(forOp.getLoc(), minValue.getType(), cmpOp.getResult(),
                                                /*withElseRegion=*/true);

    // Compute and return the adjusted index (backtrack index) in the 'then' region
    mlir::OpBuilder thenBuilder = ifOp.getThenBodyBuilder();
    auto stepDiff = thenBuilder.create<mlir::arith::SubIOp>(ifOp.getLoc(), forOp.getStep(), minValue);

    // Check whether we have enough elements to backtrack
    auto canBacktrack = thenBuilder.create<mlir::arith::CmpIOp>(stepDiff.getLoc(), mlir::arith::CmpIPredicate::sgt,
                                                                forOp.getInductionVar(), stepDiff);
    thenBuilder.create<mlir::cf::AssertOp>(canBacktrack.getLoc(), canBacktrack,
                                           "Not enough elements to backtrack in scf.for loop");

    auto adjustedIndex = thenBuilder.create<mlir::arith::SubIOp>(ifOp.getLoc(), forOp.getInductionVar(), stepDiff);
    thenBuilder.create<mlir::scf::YieldOp>(ifOp.getLoc(), mlir::ValueRange{adjustedIndex});

    // return the default induction variable as step size in the 'else' region
    mlir::OpBuilder elseBuilder = ifOp.getElseBodyBuilder();
    elseBuilder.create<mlir::scf::YieldOp>(ifOp.getLoc(), mlir::ValueRange{forOp.getInductionVar()});

    auto inductionVar = forOp.getInductionVar();
    inductionVar.replaceUsesWithIf(ifOp.getResult(0), [&](mlir::OpOperand& operand) {
        return llvm::isa<mlir::tensor::TensorDialect>(operand.getOwner()->getDialect());
    });
    minValue.replaceUsesWithIf(forOp.getStep(), [&](mlir::OpOperand& operand) {
        return llvm::isa<mlir::tensor::TensorDialect>(operand.getOwner()->getDialect());
    });

    return true;
}

/**
 * @brief This function traverses operations in reverse order, starting from the input operand,
 * and collects all operations that are parents of the operand tensors until it reaches slice operations.
 * It ensures that all operations contributing to dynamic tensors are captured for further processing.
 */
void ConvertDynamicToStaticKernelsPass::tryAndInferStaticShapes(mlir::Value value, int64_t staticDimSize) {
    SmallVector<mlir::Value> worklist = {value};
    llvm::DenseSet<mlir::Operation*> visited, dynamicTensorOps, sliceOps;
    while (!worklist.empty()) {
        auto op = worklist.pop_back_val();
        auto definingOp = op.getDefiningOp();
        if (definingOp == nullptr || !visited.insert(definingOp).second) {
            continue;
        }

        // Only traverse ops within the same scf.for region
        if (!definingOp->getParentOfType<mlir::scf::ForOp>()) {
            continue;
        }

        if (mlir::isa<mlir::tensor::ExtractSliceOp>(definingOp)) {
            sliceOps.insert(definingOp);
            continue;
        } else {
            dynamicTensorOps.insert(definingOp);
        }

        for (auto operand : definingOp->getOperands()) {
            if (isDynamicTensor(operand)) {
                worklist.push_back(operand);
            }
        }
    }

    VPUX_THROW_WHEN(sliceOps.empty(),
                    "No slice operations found in the scf.for loop. Cannot convert dynamic to static shapes");

    llvm::DenseSet<mlir::Operation*> sliceOpsWithConversionCast;
    for (auto sliceOp : sliceOps) {
        if (!sliceOpsWithConversionCast.insert(sliceOp).second) {
            continue;
        }
        addConversionCast(sliceOp, staticDimSize);
    }

    VPUX_THROW_WHEN(dynamicTensorOps.size() > 0, "Unsupported operations with dynamic tensors were found");
}

mlir::RankedTensorType ConvertDynamicToStaticKernelsPass::inferStaticTensorType(mlir::Value value,
                                                                                int64_t staticDimSize,
                                                                                bool keepBounds) {
    auto inputType = mlir::cast<mlir::RankedTensorType>(value.getType());
    VPUX_THROW_WHEN(inputType == nullptr, "Expected a ranked tensor type, but got: {0}", value.getType());
    auto shape = inputType.getShape();
    SmallVector<int64_t> staticShape(shape.begin(), shape.end());

    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == mlir::ShapedType::kDynamic) {
            staticShape[i] = staticDimSize;
        }
    }

    if (auto dictAttr = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(inputType.getEncoding())) {
        SmallVector<mlir::NamedAttribute> newAttrs;
        for (auto attr : dictAttr) {
            if (!keepBounds || attr.getName() != "bounds") {
                newAttrs.push_back(attr);
            }
        }
        // Rebuild the static type with all attributes except bounds
        return mlir::RankedTensorType::get(staticShape, inputType.getElementType(),
                                           mlir::DictionaryAttr::get(inputType.getContext(), newAttrs));
    }

    return mlir::RankedTensorType::get(staticShape, inputType.getElementType());
}

void ConvertDynamicToStaticKernelsPass::addConversionCast(mlir::Operation* sliceOp, int64_t staticDimSize) {
    auto staticTensorType = inferStaticTensorType(sliceOp->getResult(0), staticDimSize);
    VPUX_THROW_WHEN(staticTensorType == nullptr,
                    "Failed to infer static tensor type for slice operation {0} with static dimension size {1}",
                    sliceOp->getName(), staticDimSize);
    mlir::OpBuilder builder(sliceOp);
    builder.setInsertionPointAfter(sliceOp);
    auto castOp = builder.create<mlir::tensor::CastOp>(sliceOp->getLoc(), staticTensorType, sliceOp->getResult(0));
    for (auto user : llvm::make_early_inc_range(sliceOp->getUsers())) {
        if (user != castOp.getOperation()) {
            user->replaceUsesOfWith(sliceOp->getResult(0), castOp.getResult());
        }
    }
}

mlir::func::FuncOp ConvertDynamicToStaticKernelsPass::createStaticFuncOp(mlir::func::FuncOp dynFuncOp,
                                                                         const mlir::FunctionType& staticFuncType,
                                                                         mlir::ModuleOp moduleOp, int64_t staticDimSize,
                                                                         bool eraseDynamicFunc) {
    mlir::OpBuilder builder(moduleOp.getContext());
    auto staticFuncName = dynFuncOp.getName().str() + "_static";
    auto existingFuncOp = moduleOp.lookupSymbol<mlir::func::FuncOp>(staticFuncName);
    VPUX_THROW_WHEN(existingFuncOp, "Static function with name {0} already exists", staticFuncName);
    auto staticFuncOp = builder.create<mlir::func::FuncOp>(dynFuncOp.getLoc(), staticFuncName, staticFuncType);

    mlir::IRMapping valueMap;
    for (auto& oldBlock : dynFuncOp.getBody()) {
        auto* newBlock = staticFuncOp.addEntryBlock();

        // Map the block arguments from the old block to the new block
        for (auto argPair : llvm::zip(oldBlock.getArguments(), newBlock->getArguments())) {
            valueMap.map(std::get<0>(argPair), std::get<1>(argPair));
        }

        builder.setInsertionPointToStart(newBlock);
        for (auto& oldOp : oldBlock.getOperations()) {
            auto* newOp = builder.clone(oldOp, valueMap);
            for (auto it : llvm::enumerate(oldOp.getResults())) {
                size_t idx = it.index();
                mlir::Value result = it.value();
                if (isDynamicTensor(result)) {
                    auto staticResultType = inferStaticTensorType(result, staticDimSize);
                    newOp->getResult(idx).setType(staticResultType);
                    valueMap.map(result, newOp->getResult(idx));
                }
            }
        }
    }

    moduleOp.push_back(staticFuncOp);
    staticFuncOp->moveAfter(dynFuncOp);

    if (eraseDynamicFunc && dynFuncOp.use_empty()) {
        dynFuncOp.erase();
    }

    return staticFuncOp;
};

void ConvertDynamicToStaticKernelsPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    net::NetworkInfoOp netInfoOp;
    mlir::func::FuncOp mainFuncOp;
    net::NetworkInfoOp::getFromModule(moduleOp, netInfoOp, mainFuncOp);
    mlir::OpBuilder builder(moduleOp.getContext());

    mlir::WalkResult walkResult = mainFuncOp.walk([&](mlir::scf::ForOp forOp) {
        // Get the list of call operations inside the scf.for loop
        SmallVector<mlir::func::CallOp> callOpsWithDynamicTensors;
        for (auto op : forOp.getOps<mlir::func::CallOp>()) {
            // After extract_slice operations in scf tiling, resulting tensor slices have dynamic shapes without bounds
            // or dynamic_dims attributes. This causes hasDynamicTensors checks to fail. This additional check ensures
            // operations with dynamic tensors are correctly identified and processed.
            if (vpux::IE::hasDynamicTensors(op) || checkForDynamicTensor(op)) {
                callOpsWithDynamicTensors.push_back(op);
            }
        }

        if (callOpsWithDynamicTensors.empty()) {
            _log.info("No call operations with dynamic tensors found in the scf.for loop. Skipping conversion.");
            return mlir::WalkResult::advance();
        }

        // Get the step size from the scf.for loop
        auto stepValue = forOp.getStep();
        int64_t staticDimSize = 0;
        if (auto constOp = stepValue.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
            staticDimSize = constOp.value();
        }

        if (staticDimSize <= 0) {
            _log.error("Invalid static dimension size: {0}", staticDimSize);
            return mlir::WalkResult::interrupt();
        }

        if (!adjustIndexIntoDynamicTensor(forOp)) {
            _log.error("Failed to adjust index into dynamic tensor for scf.for loop.");
            return mlir::WalkResult::interrupt();
        }

        for (auto callOp : callOpsWithDynamicTensors) {
            for (auto operand : callOp.getOperands()) {
                if (isDynamicTensor(operand)) {
                    if (!isSupportedDynamicTensorType(operand)) {
                        return mlir::WalkResult::interrupt();
                    }
                    tryAndInferStaticShapes(operand, staticDimSize);
                }
            }

            SmallVector<mlir::Type> staticFuncInputTypes, staticFuncOutputTypes;
            for (auto result : callOp->getResults()) {
                if (isDynamicTensor(result)) {
                    if (!isSupportedDynamicTensorType(result)) {
                        return mlir::WalkResult::interrupt();
                    }
                    staticFuncOutputTypes.push_back(inferStaticTensorType(result, staticDimSize));
                } else {
                    staticFuncOutputTypes.push_back(result.getType());
                }
            }

            // Collect all input types for the static function type
            for (auto operand : callOp.getOperands()) {
                staticFuncInputTypes.push_back(operand.getType());
            }

            // Create the static function type
            auto staticFuncType =
                    mlir::FunctionType::get(callOp->getContext(), staticFuncInputTypes, staticFuncOutputTypes);

            // Look up the original function and create a static version
            auto dynFuncOp = moduleOp.lookupSymbol<mlir::func::FuncOp>(callOp.getCallee());
            if (dynFuncOp == nullptr) {
                // If the function is not found, it might be a dynamic function that needs to be created
                _log.error("Dynamic function {0} not found in module", callOp.getCallee());
                return mlir::WalkResult::interrupt();
            }

            // Create the new static function
            auto staticFuncOp = createStaticFuncOp(dynFuncOp, staticFuncType, moduleOp, staticDimSize);

            // Create a call to the new static function
            builder.setInsertionPoint(callOp);
            auto newCallOp = builder.create<mlir::func::CallOp>(callOp->getLoc(), staticFuncOp.getName(),
                                                                staticFuncOutputTypes, callOp.getOperands());

            // Handle all results from the call op
            for (auto resultPair : llvm::zip(callOp->getResults(), newCallOp->getResults())) {
                auto oldResult = std::get<0>(resultPair);
                auto newResult = std::get<1>(resultPair);

                // Create a cast operation to convert the static shaped tensor returned to dynamic tensor
                if (oldResult.getType() != newResult.getType()) {
                    if (!isDynamicTensor(oldResult)) {
                        _log.error("Type mismatch between old result {0} and new result {1}", oldResult.getType(),
                                   newResult.getType());
                        return mlir::WalkResult::interrupt();
                    }

                    auto castToDynamicTensor =
                            builder.create<mlir::tensor::CastOp>(newCallOp->getLoc(), oldResult.getType(), newResult);

                    for (auto user : llvm::make_early_inc_range(oldResult.getUsers())) {
                        if (user != castToDynamicTensor.getOperation()) {
                            if (mlir::isa<mlir::func::CallOp>(user)) {
                                user->replaceUsesOfWith(oldResult, newResult);
                            } else {
                                user->replaceUsesOfWith(oldResult, castToDynamicTensor.getResult());
                            }
                        }
                    }
                } else {
                    oldResult.replaceAllUsesWith(newResult);
                }
            }
            callOp->erase();
        }

        return mlir::WalkResult::advance();
    });

    if (walkResult.wasInterrupted()) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertDynamicToStaticKernelsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createConvertDynamicToStaticKernelsPass(Logger log) {
    return std::make_unique<ConvertDynamicToStaticKernelsPass>(log);
}
