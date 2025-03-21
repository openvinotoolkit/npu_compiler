//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/func_dialect.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/scope_exit.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/ValueRange.h>

#include <utility>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_LEGALIZEREPEATINGFUNCCALLS
#define GEN_PASS_DEF_LEGALIZEREPEATINGFUNCCALLS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

bool sameLengthRanges(ArrayRef<mlir::OperandRange> ranges) {
    VPUX_THROW_WHEN(ranges.size() <= 1, "Expected at least 2 ranges");
    const auto commonSize = ranges.front().size();
    return std::all_of(std::next(ranges.begin()), ranges.end(), [&](mlir::OperandRange range) {
        return range.size() == commonSize;
    });
}

/// Note: poor man's "range of ranges" implementation.
struct OperandRange2D {
    OperandRange2D(ArrayRef<mlir::OperandRange> ranges): _firstEnd(ranges.front().end()) {
        VPUX_THROW_WHEN((!sameLengthRanges(ranges)), "Ranges have different lengths");
        _begin = to_small_vector(ranges | transformed([](mlir::OperandRange range) {
                                     return range.begin();
                                 }));
    }

    static void advance(OperandRange2D& current) {
        for (auto& it : current._begin) {
            ++it;
        }
    }

    bool empty() const {
        // Note: this is enough since all inner ranges have the same length.
        return *_begin.begin() == _firstEnd;
    }

    ArrayRef<mlir::OperandRange::iterator> get() const {
        return _begin;
    }

private:
    SmallVector<mlir::OperandRange::iterator> _begin{};
    // Note: since all inner ranges must match (size-wise), using single end -
    // e.g. front(ends) - is sufficient.
    mlir::OperandRange::iterator _firstEnd;
};

mlir::Value getSingleRoot(const AliasesInfo& info, mlir::Value x) {
    auto roots = info.getRoots(x);
    VPUX_THROW_UNLESS(roots.size() == 1, "Value expected to have only one root. Got {1}", roots.size());
    return *roots.begin();
}

/// Argument equivalence-checker.
struct ArgEquivalenceChecker {
    ArgEquivalenceChecker(const AliasesInfo& info): _info(info) {
    }

    /// Returns whether two values are equivalent, i.e. they have the same root.
    bool operator()(mlir::Value x, mlir::Value y) const {
        return getSingleRoot(_info, x) == getSingleRoot(_info, y);
    }

    /// Returns whether two operand range iterators are equivalent.
    bool operator()(mlir::OperandRange::iterator x, mlir::OperandRange::iterator y) const {
        return operator()(*x, *y);
    }

private:
    const AliasesInfo& _info;
};

/// Traverses IR, locates repeating calls (if any), analyses which repeating
/// calls require updates and aggregates the ones that require an update into a
/// convenient data structure.
struct CallSiteAnalyzer {
    using EquivalentCallSites = SmallVector<mlir::func::CallOp>;

    /// Analysis information for the repeating call.
    struct RepeatingCallInfo {
        mlir::func::FuncOp func;                   // repeatedly called function.
        SmallVector<mlir::Value> correctFuncArgs;  // *correct* arguments to be used at each call-site. nullptr is used
                                                   // to signal that "no update" is required
        EquivalentCallSites callSites;             // all call-sites of the current function.
    };

    CallSiteAnalyzer(Logger& log, mlir::func::FuncOp topLevelFunc, mlir::OpBuilder& builder)
            : _log(log), _builder(builder), _aliasInfo(topLevelFunc) {
        CallSitesPerFunc allCallSites;
        topLevelFunc.walk([&](mlir::func::CallOp callOp) {
            auto funcOp = getCalledFunction(callOp);
            allCallSites[funcOp].push_back(callOp);
        });
        collectRepeatingCallInfos(std::move(allCallSites));
    }

    /// Returns analysis information for all repeating function calls that
    /// *require* an update.
    MutableArrayRef<RepeatingCallInfo> takeAllInfos() {
        return _infos;
    }

    const DenseMap<mlir::Value, mlir::Value>& getRootsForChangedOperands() const {
        return _rootsForChangedOperands;
    }

private:
    using CallSitesPerFunc = mlir::DenseMap<mlir::func::FuncOp, EquivalentCallSites>;
    /// Collects repeating calls information, filtering out "legal" cases.
    void collectRepeatingCallInfos(CallSitesPerFunc allCallSites) {
        const auto isRepeatingCall = [](std::pair<mlir::func::FuncOp, EquivalentCallSites>& maybeRepeating) -> bool {
            auto& [funcOp, callSites] = maybeRepeating;
            const bool funcHasArgs = funcOp.getNumArguments() > 0;
            return funcHasArgs && callSites.size() > 1;
        };

        for (auto& [funcOp, callSites] : allCallSites | filtered(isRepeatingCall)) {
            _log.trace("Analyzing repeating calls of '@{0}'", funcOp.getName());
            _log = _log.nest();
            VPUX_SCOPE_EXIT {
                _log = _log.unnest();
            };

            auto args = setupRepeatingCallArgs(funcOp, callSites);
            const bool thisRepeatingCallRequiresUpdate = !args.empty();
            if (thisRepeatingCallRequiresUpdate) {
                _infos.push_back({funcOp, std::move(args), std::move(callSites)});
            }

            _log.trace("Repeating calls of '@{0}' do{1}need an update", funcOp.getName(),
                       thisRepeatingCallRequiresUpdate ? " " : " NOT ");
        }
    }

    /// Returns function arguments to be used at each call-site. Returns empty
    /// arguments when no changes are required in any of the call-sites.
    SmallVector<mlir::Value> setupRepeatingCallArgs(mlir::func::FuncOp funcOp, ArrayRef<mlir::func::CallOp> callSites) {
        auto allArgs = to_small_vector(callSites | transformed([](mlir::func::CallOp call) {
                                           return call.getArgOperands();
                                       }));
        ensureValidRepeatingCall(funcOp, allArgs);

        OperandRange2D argsAcrossCallSites(allArgs);

        mlir::OpBuilder::InsertionGuard guard(_builder);
        _builder.setInsertionPoint(callSites.front());

        const size_t numArgs = funcOp.getNumArguments();
        // setup arguments for this repeating call: the ones that need an update
        // are allocated, others - used "as is" (nullptr is set as a stub).
        bool atLeastOneArgNeedsUpdate = false;
        SmallVector<mlir::Value> callArgs(numArgs, nullptr);
        for (size_t i = 0; !argsAcrossCallSites.empty(); OperandRange2D::advance(argsAcrossCallSites), ++i) {
            auto ithArgs = argsAcrossCallSites.get();
            const bool updateCurrentArg = !argsAreEquivalent(ithArgs);
            if (!updateCurrentArg) {
                _log.trace("Call-site operand #{0} is used \"as is\"", i);
                continue;
            }

            atLeastOneArgNeedsUpdate = true;
            rememberRootsOfUpdatedArgs(ithArgs);
            auto arg = makeNewCommonArg(funcOp, i);
            _log.trace("Call-site operand #{0} is created at {1}", i, arg.getLoc());
            callArgs[i] = std::move(arg);
        }
        if (!atLeastOneArgNeedsUpdate) {
            callArgs.clear();
        }

        return callArgs;
    }

    /// Returns whether given arguments are equivalent, that is, they all point
    /// to the same root.
    bool argsAreEquivalent(ArrayRef<mlir::OperandRange::iterator> args) const {
        const auto argsNotTheSame = std::not_fn(ArgEquivalenceChecker{_aliasInfo});
        auto it = std::adjacent_find(args.begin(), args.end(), argsNotTheSame);
        return it == args.end();
    }

    /// Returns a new allocation for the function argument at the specified
    /// index.
    mlir::Value makeNewCommonArg(mlir::func::FuncOp funcOp, size_t i) {
        const auto loc = appendLoc(funcOp.getLoc(), "_repeating_call_alloc");
        const auto type = mlir::dyn_cast<mlir::MemRefType>(funcOp.getArgument(i).getType());
        VPUX_THROW_WHEN(type == nullptr, "Function {0} has non-memref type argument at index {1}",
                        funcOp.getOperationName(), i);
        return _builder.create<mlir::memref::AllocOp>(loc, type).getResult();
    }

    /// Populates internal structure that keeps a mapping from a call-site
    /// operand to its root (allocation).
    void rememberRootsOfUpdatedArgs(ArrayRef<mlir::OperandRange::iterator> args) {
        for (auto argIt : args) {
            auto arg = *argIt;
            _rootsForChangedOperands[arg] = getSingleRoot(_aliasInfo, arg);
        }
    }

    /// Verifies that the IR satisfies the constraints of this pass.
    static void ensureValidRepeatingCall(mlir::func::FuncOp funcOp, ArrayRef<mlir::OperandRange> argsInAllCallSites) {
        const auto numInputs = VPUIP::getNumInputs(funcOp);
        OperandRange2D argsAcrossCallSites(argsInAllCallSites);
        for (size_t i = 0; i < numInputs; ++i) {
            // skip input arguments
            OperandRange2D::advance(argsAcrossCallSites);
        }

        // Note: this verification prohibits using the same value as output
        // argument in multiple call-sites. in theory, this restricts
        // allocation-related optimizations. ideally, there should be a separate
        // (verifier) pass that ensures that outputs are consumed (through
        // results) *before* they are being overwritten, then this more
        // restrictive check would not be needed
        mlir::DenseSet<mlir::Value> memorizedOutputs;
        memorizedOutputs.reserve(argsInAllCallSites.size() * VPUIP::getNumOutputs(funcOp));
        const auto visitValue = [&](mlir::OperandRange::iterator x) {
            const bool firstTimeSeenValue = memorizedOutputs.insert(*x).second;
            return !firstTimeSeenValue;
        };
        for (size_t i = 0; !argsAcrossCallSites.empty(); OperandRange2D::advance(argsAcrossCallSites), ++i) {
            const auto ithOutputs = argsAcrossCallSites.get();
            auto it = llvm::find_if(ithOutputs, visitValue);
            // this is currently not supported
            VPUX_THROW_WHEN((it != ithOutputs.end()), "Output operand '{0}' is used as output at multiple call-sites",
                            it->getBase()->get().getLoc());
        }
    }

    Logger& _log;
    mlir::OpBuilder& _builder;
    AliasesInfo _aliasInfo;
    SmallVector<RepeatingCallInfo> _infos;

    // A mapping from call-site operand to its root. Used by legalizer instead
    // of AliasesInfo (direct use is troublesome as legalizer updates the IR -
    // this breaks AliasesInfo's usage contract).
    DenseMap<mlir::Value, mlir::Value> _rootsForChangedOperands;
};

/// Fixes repeating calls for a particular function.
struct RepeatingCallLegalizer {
    RepeatingCallLegalizer(Logger& log, mlir::OpBuilder& builder, const DenseMap<mlir::Value, mlir::Value>& rootMapping)
            : _log(log), _builder(builder), _rootsForOperands(rootMapping) {
    }

    /// Legalizes all call-sites of the given repeating call.
    void legalize(CallSiteAnalyzer::RepeatingCallInfo&& info) {
        _log.trace("Legalizing function '@{0}'", info.func.getName());
        _log = _log.nest();
        VPUX_SCOPE_EXIT {
            _log = _log.unnest();
        };

        const auto numInputs = VPUIP::getNumInputs(info.func);
        for (auto callOp : info.callSites) {
            legalizeCallSite(numInputs, info.correctFuncArgs, callOp);
        }
    }

private:
    /// Legalizes a single call-site within the repeating call.
    void legalizeCallSite(size_t numInputs, ArrayRef<mlir::Value> commonArgs, mlir::func::CallOp callOp) {
        _log.trace("Processing call-site at {0}", callOp->getLoc());

        auto callSiteArgs = callOp.getArgOperands();
        VPUX_THROW_WHEN(commonArgs.size() != callSiteArgs.size(), "Call-site argument size is unexpected");

        auto commonArgsFirst = commonArgs.begin();
        auto callSiteArgsFirst = callSiteArgs.begin();
        const auto commonArgsInputLast = std::next(commonArgsFirst, numInputs);

        // update inputs
        for (size_t i = 0; commonArgsFirst != commonArgsInputLast; ++commonArgsFirst, ++callSiteArgsFirst, ++i) {
            replaceArgIfNecessary<true>(callOp, numInputs, i, *commonArgsFirst, *callSiteArgsFirst);
        }
        // update outputs
        for (size_t i = 0; commonArgsFirst != commonArgs.end(); ++commonArgsFirst, ++callSiteArgsFirst, ++i) {
            replaceArgIfNecessary<false>(callOp, numInputs, i, *commonArgsFirst, *callSiteArgsFirst);
        }
    }

    /// Top-level argument update dispatcher for input and output arguments.
    /// Note, this function does nothing when the call-site does not require
    /// i-th argument to be updated.
    template <bool IsInputArg>
    void replaceArgIfNecessary(mlir::func::CallOp callOp, size_t numInputs, size_t argIndex, mlir::Value commonArg,
                               mlir::Value callSiteArg) {
        const bool doUpdate = commonArg != nullptr;
        if (!doUpdate) {
            _log.trace("*{0}* argument '{1}' is NOT replaced", IsInputArg ? "Input" : "Output", callSiteArg);
            return;
        }

        if constexpr (IsInputArg) {
            std::ignore = numInputs;
            replaceInputArg(callOp, argIndex, commonArg, callSiteArg);
        } else {
            replaceOutputArg(callOp, numInputs, argIndex, commonArg, callSiteArg);
        }
    }

    /// Replaces source input argument with the specified alternative for the
    /// current call-site.
    void replaceInputArg(mlir::func::CallOp callOp, size_t inputIndex, mlir::Value dst, mlir::Value src) {
        _log.trace("Replacing *input* '{0}' with '{1}' at call-site {2}", src, dst, callOp.getLoc());

        mlir::OpBuilder::InsertionGuard guard(_builder);
        _builder.setInsertionPoint(callOp);
        auto loc = appendLoc(callOp.getLoc(), "_repeating_call_input");

        auto copy = _builder.create<VPUIP::CopyOp>(loc, src, dst).getResult();
        src.replaceUsesWithIf(copy, [&](mlir::OpOperand& use) {
            return isCurrentCallSite(callOp, use) && use.getOperandNumber() == inputIndex;
        });
    }

    /// Replaces source output argument with the specified alternative for the
    /// current call-cite. Unlike input argument replacement, this procedure
    /// optimizes certain edge cases.
    void replaceOutputArg(mlir::func::CallOp callOp, size_t numInputs, size_t resultIndex, mlir::Value dst,
                          mlir::Value src) {
        _log.trace("Replacing *output* '{0}' with '{1}' at call-site {2}", src, dst, callOp.getLoc());

        mlir::OpBuilder::InsertionGuard guard(_builder);
        _builder.setInsertionPointAfter(callOp);
        auto loc = appendLoc(callOp.getLoc(), "_repeating_call_output");

        const auto operandIndex = numInputs + resultIndex;
        src.replaceUsesWithIf(dst, [&](mlir::OpOperand& use) {
            return isCurrentCallSite(callOp, use) && use.getOperandNumber() == operandIndex;
        });
        auto result = callOp.getResults()[resultIndex];
        // Note: use root(src) here when creating a copy so that Copy's output
        // is always an allocation (not a result of some other operation) -
        // important for in-place functions.
        const auto srcRoot = getRootForChangedOperand(src);
        auto copy = _builder.create<VPUIP::CopyOp>(loc, result, srcRoot).getResult();
        result.replaceUsesWithIf(copy, [&](mlir::OpOperand& use) {
            const bool currentCopy = use.getOwner() == copy.getDefiningOp();

            // do not replace other call-sites of this repeating call with a
            // copy since:
            // 1. if this output is an input in another call (by result aliasing
            //    output), the copy is added in input argument update
            // 2. if this output is an output in another call (probably never
            //    happens, except in this pass), the data is overwritten anyway,
            //    no reason to copy
            return !currentCopy && !isCurrentRepeatingCall(callOp, use);
        });

        // Note: remove the no-uses copy manually since canonicalizer won't:
        // VPUIP operations have side-effects.
        if (copy.getUsers().empty()) {
            copy.getDefiningOp()->erase();
        }
    }

    /// Returns whether the given use matches the call operation.
    static bool isCurrentCallSite(mlir::func::CallOp callOp, mlir::OpOperand& use) {
        return use.getOwner() == callOp;
    }

    /// Returns whether the given use is a call operation with callee being the
    /// same as in the specified call operation.
    static bool isCurrentRepeatingCall(mlir::func::CallOp callOp, mlir::OpOperand& use) {
        const auto callee = callOp.getCallee();
        auto maybeCall = mlir::dyn_cast<mlir::func::CallOp>(use.getOwner());
        return maybeCall && maybeCall.getCallee() == callee;
    }

    mlir::Value getRootForChangedOperand(mlir::Value operand) {
        auto it = _rootsForOperands.find(operand);
        VPUX_THROW_WHEN(it == _rootsForOperands.end(), "No root found for operand '{0}'", operand);
        return it->second;
    }

    Logger& _log;
    mlir::OpBuilder& _builder;
    const DenseMap<mlir::Value, mlir::Value>& _rootsForOperands;
};

struct LegalizeRepeatingFuncCallsPass final :
        VPUIP::impl::LegalizeRepeatingFuncCallsBase<LegalizeRepeatingFuncCallsPass> {
    LegalizeRepeatingFuncCallsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void LegalizeRepeatingFuncCallsPass::safeRunOnFunc() {
    auto func = getOperation();
    _log.trace("Legalizing repeating calls inside function '@{0}' at {1}", func.getName(), func->getLoc());
    _log = _log.nest();
    VPUX_SCOPE_EXIT {
        _log = _log.unnest();
    };

    // rough high-level algorithm:
    // 1. (analysis) collect call sites per func by walking the IR
    // 2. (analysis) filter out legal cases. legal are:
    // 2.1. single call site
    // 2.1. adjacent_find(call sites) has no arg mismatches
    // 3. (legalization) update arguments of invalid cases

    mlir::OpBuilder builder(func);

    CallSiteAnalyzer analyzer(_log, func, builder);
    auto infos = analyzer.takeAllInfos();

    RepeatingCallLegalizer legalizer(_log, builder, analyzer.getRootsForChangedOperands());
    for (auto& info : infos) {
        legalizer.legalize(std::move(info));
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPUIP::createLegalizeRepeatingFuncCallsPass(Logger log) {
    return std::make_unique<LegalizeRepeatingFuncCallsPass>(log);
}
