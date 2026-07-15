//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_analysis_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_analyzer.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/abstract_tree.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Visitors.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_SCFLOOPANALYSISANDDEBUG
#define GEN_PASS_DEF_SCFLOOPANALYSISANDDEBUG
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

struct ScfBlockInfo {
    mlir::Operation* op = nullptr;
    SmallVector<mlir::Operation*> dynamicTensorOps;
    ScfAnalysisInfo analysisInfo;
    ScfBlockAnalyzer blockAnalyzer;

    ScfBlockInfo(mlir::Operation* operation, ArrayRef<mlir::Operation*> dynamicTensorOperations,
                 ScfAnalysisInfo&& analysisInfo, ScfBlockAnalyzer&& blockAnalyzer)
            : op(operation),
              dynamicTensorOps(dynamicTensorOperations.begin(), dynamicTensorOperations.end()),
              analysisInfo(std::move(analysisInfo)),
              blockAnalyzer(std::move(blockAnalyzer)) {
    }

    // Delete copy operations to match ScfBlockAnalyzer
    ScfBlockInfo(const ScfBlockInfo&) = delete;
    ScfBlockInfo& operator=(const ScfBlockInfo&) = delete;

    ScfBlockInfo(ScfBlockInfo&&) noexcept = default;
    ScfBlockInfo& operator=(ScfBlockInfo&&) noexcept = default;

    ~ScfBlockInfo() {
    }

    auto isScfForOp() const -> bool {
        return op != nullptr && mlir::isa<mlir::scf::ForOp>(op);
    }

    auto getLoopParams() const -> std::tuple<size_t, size_t, size_t> {
        auto defaultParams = std::make_tuple(0, 0, 0);
        if (op == nullptr) {
            return defaultParams;
        }
        if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
            OpChainAnalysis analysis;
            return analysis.getLoopBoundsAndStep(forOp);
        }
        return defaultParams;
    }
};
using ScfOpHierarchy = utils::AbstractTree<ScfBlockInfo>;

namespace {

//
// SCFLoopAnalysisAndDebugPass
//
class SCFLoopAnalysisAndDebug final : public VPU::impl::SCFLoopAnalysisAndDebugBase<SCFLoopAnalysisAndDebug> {
public:
    explicit SCFLoopAnalysisAndDebug(Logger log): _log(log) {
        Base::initLogger(_log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
    vpux::Logger _log;
};

auto checkForOpsToBeAnalyzed(mlir::Operation* op) -> bool {
    if (mlir::isa<mlir::tensor::ExtractSliceOp, mlir::tensor::InsertSliceOp, mlir::tensor::PadOp,
                  mlir::scf::IndexSwitchOp>(op)) {
        return true;
    }

    // Check for affine / arith dialect ops with "analyze" attribute
    if (op->hasAttr(ANALYZE_ATTR) &&
        (mlir::isa<mlir::affine::AffineDialect, mlir::arith::ArithDialect>(op->getDialect()))) {
        return true;
    }

    return false;
}

llvm::SmallVector<mlir::Operation*> getOpsForAnalysis(mlir::scf::ForOp forOp) {
    SmallVector<mlir::Operation*> opsToBeAnalyzed;
    for (auto& op : forOp.getBody()->getOperations()) {
        if (mlir::isa<mlir::scf::ForOp>(op)) {
            continue;
        }

        if (checkForOpsToBeAnalyzed(&op)) {
            opsToBeAnalyzed.push_back(&op);
        }
    }
    return opsToBeAnalyzed;
}

// Note: this is here for debugging purposes only.
struct TreePrinter final : ScfOpHierarchy::Visitor {
    std::ostream& stream;
    size_t indentation = 0;

    TreePrinter(std::ostream& s): stream(s) {
    }

    bool visit(const Node& node) final {
        auto currentOp = node.data().op;
        auto computeNodes = node.data().dynamicTensorOps;
        constexpr StringLiteral prefix = "|- ";

        if (currentOp == nullptr) {
            stream << std::string(prefix.size() * indentation, ' ') << prefix.str() << "<null operation>\n";
            ++indentation;
            return true;
        }

        stream << std::string(prefix.size() * indentation, ' ')
               << formatv("{0}{1}", prefix, currentOp->getName().getStringRef().str()).str();

        if (node.data().isScfForOp()) {
            auto [lower, upper, step] = node.data().getLoopParams();
            stream << formatv(" -> lower={0}, upper={1}, step={2}", lower, upper, step).str();
        }

        stream << "\n";

        // Print ScfAnalysisInfo - all possible block sizes per dynamic dimension
        if (!computeNodes.empty()) {
            stream << std::string(prefix.size() * (indentation + 1), ' ')
                   << "=== ScfAnalysisInfo: All Possible Block Sizes ===\n";
            auto analysisInfo = node.data().analysisInfo;
            analysisInfo.print(stream, prefix.size() * (indentation + 1));
        }

        // Print Block Analysis results if available
        const auto& blockAnalyzer = node.data().blockAnalyzer;
        if (blockAnalyzer.getUniqueStates().size() > 0) {
            stream << "\n";
            stream << std::string(prefix.size() * (indentation + 1), ' ') << "=== Block Analysis ===\n";
            blockAnalyzer.printResults(stream, prefix.size() * (indentation + 1));
        }

        ++indentation;
        return true;
    }

    void endVisit(const Node&) final {
        --indentation;
    }
};

ScfAnalysisInfo runOpAnalysis(llvm::ArrayRef<mlir::Operation*> opsToBeAnalyzed, const Logger& log) {
    if (opsToBeAnalyzed.empty()) {
        return ScfAnalysisInfo();
    }

    auto op = opsToBeAnalyzed.front();
    ScfAnalysisInfo analysisInfo(opsToBeAnalyzed);

    auto parentForOps = analysisInfo.getParentForOps(op);
    auto inputRanges = analysisInfo.getIterationSpace(parentForOps);

    // Create analysis context and run unified analysis
    AnalysisContext context(inputRanges, log);
    analysisInfo.analyze(context);
    analysisInfo.setAttribute();

    return analysisInfo;
}

ScfBlockAnalyzer runScfBlockAnalysis(llvm::ArrayRef<mlir::Operation*> opsToBeAnalyzed, const Logger& log) {
    ScfBlockAnalyzer analyzer(opsToBeAnalyzed);

    if (opsToBeAnalyzed.empty()) {
        return analyzer;
    }

    auto op = opsToBeAnalyzed.front();
    ScfAnalysisInfo analysisInfo;
    auto parentForOps = analysisInfo.getParentForOps(op);
    auto inputRanges = analysisInfo.getIterationSpace(parentForOps);

    // Create analysis context and run unified analysis
    AnalysisContext analysisContext(inputRanges, log);
    analyzer.analyze(analysisContext);

    if (!parentForOps.empty()) {
        auto forOp = parentForOps.front();
        forOp->setAttr(UNIQUE_STATIC_BLOCKS_ATTR,
                       mlir::IntegerAttr::get(mlir::IntegerType::get(forOp->getContext(), 64),
                                              analyzer.getUniqueStateCount()));
    }

    return analyzer;
}

/**
 * @brief Find list of vpu operations inside the Scf block. A split is created
 * in the vpu ops for a new block if there is a non VPU dialect operation
 */
std::vector<ScfBlockInfo> findChildren(const ScfOpHierarchy::Node& node, const Logger& log) {
    auto rootOp = node.data().op;
    VPUX_THROW_WHEN(!mlir::isa<mlir::func::FuncOp>(rootOp) && !mlir::isa<mlir::scf::SCFDialect>(rootOp->getDialect()),
                    "Each node should be a valid operation!!");

    OpChainAnalysis opAnalysis;
    std::vector<ScfBlockInfo> childNodes;
    rootOp->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* nestedOp) -> mlir::WalkResult {
        if (nestedOp == rootOp) {
            return mlir::WalkResult::advance();
        }

        if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(nestedOp)) {
            auto opsToBeAnalyzed = getOpsForAnalysis(forOp);
            auto analysisInfo = runOpAnalysis(opsToBeAnalyzed, log.nest("scf-analysis"));

            // Run scf-based analysis
            auto blockAnalysis = runScfBlockAnalysis(opsToBeAnalyzed, log.nest("scf-analysis"));

            childNodes.emplace_back(nestedOp, std::move(opsToBeAnalyzed), std::move(analysisInfo),
                                    std::move(blockAnalysis));
            // Nested scf blocks will be processed when the child node is visited
            // Skip after processing the current scf block
            return mlir::WalkResult::skip();
        }
        return mlir::WalkResult::advance();
    });
    return childNodes;
}

void SCFLoopAnalysisAndDebug::safeRunOnModule() {
    auto moduleOp = getOperation();
    auto mainFuncOp = net::getMainFunc(moduleOp);

    // construct the tree
    std::vector<ScfOpHierarchy::Node> rootNodes;
    rootNodes.emplace_back(ScfBlockInfo(mainFuncOp, {}, ScfAnalysisInfo(), ScfBlockAnalyzer()),
                           std::vector<ScfOpHierarchy::Node>{});

    // Capture logger in lambda to pass through to findChildren
    auto findChildrenWithLogger = [this](const ScfOpHierarchy::Node& node) {
        return findChildren(node, _log);
    };

    ScfOpHierarchy tree(std::move(rootNodes), findChildrenWithLogger);

    if (_log.isActive(LogLevel::Debug)) {
        std::stringstream stream;
        TreePrinter printer{stream};
        tree.apply(printer);
        _log.debug("The following operation tree is found:\n{0}\n", stream.str());
    }
}

}  // namespace

//
// createSCFLoopAnalysisAndDebugPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createSCFLoopAnalysisAndDebugPass(Logger log) {
    return std::make_unique<SCFLoopAnalysisAndDebug>(log);
}
