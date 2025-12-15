//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/abstract_tree.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/outlining_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Visitors.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_SCFCOMPUTEOPSOUTLINING
#define GEN_PASS_DEF_SCFCOMPUTEOPSOUTLINING
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

struct ScfBlockData {
    mlir::Operation* op;
    llvm::SmallVector<llvm::SmallVector<mlir::Operation*>> computeBlockVec;
};
using ScfOpHierarchy = utils::AbstractTree<ScfBlockData>;

namespace ScfOutliner {

// Builds outlining functions for compute operations inside the scf blocks
class ScfBlockUpdater final : public ScfOpHierarchy::Visitor {
public:
    ScfBlockUpdater(const Logger& log, mlir::ModuleOp moduleOp, mlir::func::FuncOp funcOp)
            : _log(log), _moduleOp(moduleOp), _entryPointFuncOp(funcOp) {
    }

    bool visit(const Node& node) final {
        _log.trace("visiting {0}", node.data().op->getName());
        auto computeBlocks = node.data().computeBlockVec;
        llvm::SmallVector<mlir::Operation*> funcVec;
        for (auto computeBlock : computeBlocks) {
            if (!computeBlock.empty()) {
                auto blockData = getBlockContent(computeBlock);
                auto funcOp = buildComputeFunctionOp(blockData, _idx);
                buildCallOp(blockData, funcOp, _idx++);
            }
        }
        return true;
    }

    void endVisit(const Node& node) final {
        auto computeBlocks = node.data().computeBlockVec;
        for (auto computeBlock : computeBlocks) {
            for (auto op : llvm::reverse(computeBlock)) {
                if (!mlir::isa<Const::DeclareOp>(op)) {
                    assert(op->getUsers().empty() && "An op with users is not expected");
                    op->erase();
                }
            }
        }
        _log.trace("Finished visiting {0}", node.data().op->getName());
    }

private:
    Logger _log;
    mlir::ModuleOp _moduleOp;
    mlir::func::FuncOp _entryPointFuncOp;
    int _idx = 0;

    struct BlockIOAndOps {
        llvm::SetVector<mlir::Value> inputArgs;
        llvm::SetVector<mlir::Value> outputArgs;
        llvm::SetVector<mlir::Operation*> constInputOps;
        ArrayRef<mlir::Operation*> operations;
    };

    BlockIOAndOps getBlockContent(ArrayRef<mlir::Operation*> computeBlock);
    std::pair<llvm::SetVector<mlir::Value>, llvm::SetVector<mlir::Operation*>> getInputsForComputeBlock(
            ArrayRef<mlir::Operation*> computeBlock);
    llvm::SetVector<mlir::Value> getOutputsForComputeBlock(ArrayRef<mlir::Operation*> computeBlock);
    mlir::func::FuncOp buildComputeFunctionOp(const BlockIOAndOps& blockContents, size_t funcIndex);
    void buildCallOp(const BlockIOAndOps& blockContents, mlir::func::FuncOp funcOp, size_t funcIndex);
};

/**
 * @brief Capture all the data required to create a funcOp from slice
 * of operations
 */
ScfBlockUpdater::BlockIOAndOps ScfBlockUpdater::getBlockContent(ArrayRef<mlir::Operation*> computeBlock) {
    BlockIOAndOps blockData;
    std::tie(blockData.inputArgs, blockData.constInputOps) = getInputsForComputeBlock(computeBlock);
    blockData.outputArgs = getOutputsForComputeBlock(computeBlock);
    blockData.operations = computeBlock;

    return blockData;
}

/**
 * @brief Parse the results for the outlined operations and identify
 * the return arguments for outlined function
 */
llvm::SetVector<mlir::Value> ScfBlockUpdater::getOutputsForComputeBlock(ArrayRef<mlir::Operation*> computeBlock) {
    llvm::SetVector<mlir::Value> blockOutputs;

    for (auto computeOp : computeBlock) {
        for (auto result : computeOp->getResults()) {
            if (llvm::is_contained(blockOutputs, result)) {
                continue;
            }

            bool isOutput = llvm::any_of(result.getUsers(), [&](mlir::Operation* userOp) {
                if (mlir::isa<mlir::func::ReturnOp>(userOp)) {
                    return true;
                }
                if (!llvm::is_contained(computeBlock, userOp)) {
                    return true;
                }
                return false;
            });

            if (isOutput) {
                blockOutputs.insert(result);
            }
        }
    }
    return blockOutputs;
}

/**
 * @brief Parse each operand for the outlined operations and identify
 * the list of operands that need to be provided as arguments to the outlined function
 */
std::pair<llvm::SetVector<mlir::Value>, llvm::SetVector<mlir::Operation*>> ScfBlockUpdater::getInputsForComputeBlock(
        ArrayRef<mlir::Operation*> computeBlock) {
    llvm::SetVector<mlir::Value> blockInputs;
    llvm::SetVector<mlir::Operation*> constantOps;

    for (auto computeOp : computeBlock) {
        // Process inputs for function ops
        for (auto operand : computeOp->getOperands()) {
            if (llvm::is_contained(blockInputs, operand)) {
                continue;
            }

            auto parentOp = operand.getDefiningOp();
            if (!parentOp) {
                blockInputs.insert(operand);
                continue;
            }

            if (!llvm::is_contained(computeBlock, parentOp)) {
                if (VPU::isConstOperandOp(parentOp)) {
                    if (mlir::isa<VPU::GroupedViewLikeOpInterface>(parentOp)) {
                        for (auto op : parentOp->getOperands()) {
                            if (mlir::isa<mlir::BlockArgument>(op)) {
                                if (!llvm::is_contained(blockInputs, op)) {
                                    blockInputs.insert(op);
                                }
                            } else {
                                constantOps.insert(op.getDefiningOp());
                            }
                        }
                    }
                    constantOps.insert(parentOp);
                } else {
                    blockInputs.insert(operand);
                }
            }
        }
    }

    return {std::move(blockInputs), std::move(constantOps)};
}

mlir::func::FuncOp ScfBlockUpdater::buildComputeFunctionOp(const BlockIOAndOps& blockContents, size_t funcIndex) {
    auto builder = mlir::OpBuilder(_moduleOp.getBodyRegion());
    builder.setInsertionPoint(_entryPointFuncOp);

    mlir::TypeRange blockInputTypes(blockContents.inputArgs.getArrayRef());
    mlir::TypeRange blockOutputTypes(blockContents.outputArgs.getArrayRef());
    auto* ctx = _moduleOp.getContext();
    const auto funcType = mlir::FunctionType::get(ctx, blockInputTypes, blockOutputTypes);
    const auto funcLoc = appendLoc(_entryPointFuncOp.getLoc(), "_compute_group{0}", funcIndex);
    auto funcName = vpux::formatv("{0}_func{1}", _entryPointFuncOp.getName().str(), std::to_string(funcIndex));
    auto func = builder.create<mlir::func::FuncOp>(funcLoc, funcName.str(), funcType);
    func.setPrivate();

    OpBuilderLogger builderLog(_log.nest());
    builder = mlir::OpBuilder::atBlockEnd(func.addEntryBlock(), &builderLog);

    mlir::DenseMap<mlir::Value, mlir::Value> operandMapper;
    for (size_t i = 0; i < blockContents.inputArgs.size(); i++) {
        operandMapper[blockContents.inputArgs[i]] = func.getArgument(i);
    }

    llvm::SmallVector<mlir::Operation*> outlinedOps;
    outlinedOps.append(blockContents.constInputOps.begin(), blockContents.constInputOps.end());
    outlinedOps.append(blockContents.operations.begin(), blockContents.operations.end());
    for (const auto outlinedOp : outlinedOps) {
        mlir::IRMapping mapper;
        for (auto operand : outlinedOp->getOperands()) {
            mapper.map(operand, operandMapper[operand]);
        }
        auto clonedOp = builder.clone(*outlinedOp, mapper);
        for (size_t i = 0; i < clonedOp->getResults().size(); i++) {
            operandMapper[outlinedOp->getResult(i)] = clonedOp->getResult(i);
        }
    }

    SmallVector<mlir::Value> funcReturnOps;
    for (const auto output : blockContents.outputArgs) {
        funcReturnOps.push_back(operandMapper[output]);
    }

    const auto returnLoc = appendLoc(_entryPointFuncOp.getLoc(), "_part{0}_return", funcIndex);
    builder.create<mlir::func::ReturnOp>(returnLoc, funcReturnOps);
    return func;
}

void ScfBlockUpdater::buildCallOp(const BlockIOAndOps& blockContents, mlir::func::FuncOp funcOp, size_t funcIndex) {
    DenseMap<mlir::Value, mlir::Value> operandMapper;
    SmallVector<mlir::Value> funcInputOperands;
    for (const auto input : blockContents.inputArgs) {
        if (llvm::is_contained(operandMapper, input)) {
            funcInputOperands.push_back(operandMapper[input]);
        } else {
            funcInputOperands.push_back(input);
        }
    }

    OpBuilderLogger builderLog(_log.nest());
    auto builder = mlir::OpBuilder::atBlockBegin(&_entryPointFuncOp.getBody().front(), &builderLog);
    builder.setInsertionPoint(blockContents.operations.front());
    const auto callLoc = appendLoc(blockContents.operations.front()->getLoc(), "_part{0}_call", funcIndex);
    auto callOp = builder.create<mlir::func::CallOp>(callLoc, funcOp, funcInputOperands);
    for (auto outputPair : llvm::zip(blockContents.outputArgs, callOp.getResults())) {
        auto [oldOutput, newOutput] = outputPair;
        operandMapper[oldOutput] = newOutput;
        for (auto& use : llvm::make_early_inc_range(oldOutput.getUses())) {
            use.set(newOutput);
        }
    }
}
}  // namespace ScfOutliner

namespace {

//
// ScfComputeOpsOutliningPass
//
class ScfComputeOpsOutliningPass final : public VPU::impl::ScfComputeOpsOutliningBase<ScfComputeOpsOutliningPass> {
public:
    explicit ScfComputeOpsOutliningPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

/**
 * @brief Gather all NPU compute ops into seperate list
 * Split of ops is based whether there is a non VPU dialect op is present
 *
 * For example :
 *
 * scf.for {
 *   VPU.OP1
 *   tensor.extract
 *   VPU.OP2
 *   VPU.OP3
 * }
 *
 * will result in a split of [VPU.OP1], [VPU.OP2, VPU.OP3]
 */
llvm::SmallVector<llvm::SmallVector<mlir::Operation*>> getComputeOps(mlir::Operation* currentOp) {
    llvm::SmallVector<llvm::SmallVector<mlir::Operation*>> computeOpGroups;
    SmallVector<mlir::Operation*> currentGroup;

    auto appendToComputeBlocks = [&]() {
        if (!currentGroup.empty()) {
            computeOpGroups.push_back(currentGroup);
            currentGroup.clear();
        }
    };

    for (auto& reg : currentOp->getRegions()) {
        for (auto& block : reg.getBlocks()) {
            for (auto& innerOp : block) {
                if (mlir::isa<vpux::VPU::VPUDialect>(innerOp.getDialect())) {
                    if (!VPU::isConstOperandOp(&innerOp)) {
                        currentGroup.push_back(&innerOp);
                    }
                } else {
                    // Any non-VPU dialect operation will result in a new block of operations to ouline
                    appendToComputeBlocks();
                }
            }
            appendToComputeBlocks();
        }
    }

    return computeOpGroups;
}

// Note: this is here for debugging purposes only.
struct TreePrinter final : ScfOpHierarchy::Visitor {
    std::ostream& stream;
    size_t indentation = 0;

    TreePrinter(std::ostream& s): stream(s) {
    }

    bool visit(const Node& node) final {
        auto funcOp = node.data().op;
        auto computeNodes = node.data().computeBlockVec;
        constexpr StringLiteral prefix = "|- ";
        std::stringstream sstream;

        sstream << "[";
        for (auto ops : computeNodes) {
            sstream << "[";
            for (auto op : ops) {
                sstream << op->getName().getStringRef().str();
                sstream << " ,";
            }
            sstream << "]";
        }
        sstream << "]";

        stream << std::string(prefix.size() * indentation, ' ')
               << formatv("{0}{1} {2}\n", prefix, funcOp->getName().getStringRef().str(), sstream.str()).str();
        ++indentation;
        return true;
    }

    void endVisit(const Node&) final {
        --indentation;
    }
};

/**
 * @brief Find list of vpu operations inside the Scf block. A split is created
 * in the vpu ops for a new block if there is a non VPU dialect operation
 */
std::vector<ScfBlockData> findChildren(const ScfOpHierarchy::Node& node) {
    auto rootOp = node.data().op;
    VPUX_THROW_WHEN(!mlir::isa<mlir::func::FuncOp>(rootOp) && !mlir::isa<mlir::scf::SCFDialect>(rootOp->getDialect()),
                    "Each node should be a valid operation!!");

    std::vector<ScfBlockData> childNodes;
    rootOp->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* nestedOp) -> mlir::WalkResult {
        if (nestedOp == rootOp) {
            return mlir::WalkResult::advance();
        }
        if (mlir::isa<mlir::scf::SCFDialect>(nestedOp->getDialect()) && nestedOp->getNumRegions() > 0) {
            auto computeOpsVec = getComputeOps(nestedOp);
            childNodes.push_back(ScfBlockData{nestedOp, std::move(computeOpsVec)});
            // Nested scf blocks will be processed when the child node is visited
            // Skip after processing the current scf block
            return mlir::WalkResult::skip();
        }
        return mlir::WalkResult::advance();
    });
    return childNodes;
}

//
// safeRunOnModule
//
void ScfComputeOpsOutliningPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    net::NetworkInfoOp netInfo;
    mlir::func::FuncOp mainFuncOp;
    net::NetworkInfoOp::getFromModule(moduleOp, netInfo, mainFuncOp);

    // construct the tree
    auto computeOpsVec = getComputeOps(mainFuncOp);
    ScfOpHierarchy tree({ScfOpHierarchy::Node(ScfBlockData{mainFuncOp, std::move(computeOpsVec)}, {})}, findChildren);

    if (_log.isActive(LogLevel::Debug)) {
        std::stringstream stream;
        TreePrinter printer{stream};
        tree.apply(printer);
        _log.debug("The following operation tree is found:\n{0}\n", stream.str());
    }

    // Outline compute operations in scf block
    ScfOutliner::ScfBlockUpdater updater(_log, moduleOp, mainFuncOp);
    tree.apply(updater);

    // Const Declare operations are copied into each compute block where the operations are used
    // If there are no users of Const Declare Ops in main function, delete them from IR
    mainFuncOp.walk([&](mlir::Operation* op) {
        if (VPU::isConstOperandOp(op) && op->getUsers().empty()) {
            op->erase();
        }
    });
}

}  // namespace

//
// createScfComputeOpsOutliningPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createScfComputeOpsOutliningPass(Logger log) {
    return std::make_unique<ScfComputeOpsOutliningPass>(log);
}
