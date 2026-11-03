//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/ShaveCodeGen/passes.hpp"
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/core/IR/memref_attr.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <array>
#include <iterator>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>

namespace vpux::ShaveCodeGen {
#define GEN_PASS_DECL_SPECIALIZESWKERNELS
#define GEN_PASS_DEF_SPECIALIZESWKERNELS
#include "vpux/compiler/ShaveCodeGen/passes.hpp.inc"
}  // namespace vpux::ShaveCodeGen

using namespace vpux;

namespace {

using RunPair = std::pair<VPUIP::SwKernelOp, VPUIP::SwKernelRun>;

// Returns the number of trailing iteration-space scalar args (index-typed) a kernel carries.
static unsigned numIterationSpaceScalars(mlir::func::FuncOp callee) {
    const auto kernelInfo = callee->getAttrOfType<VPU::KernelInfoAttr>(VPU::KernelInfoAttr::kFuncAttrName);
    return kernelInfo ? static_cast<unsigned>(2 * kernelInfo.getTilingAxes().size()) : 0;
}

// Read a callsite operand's element strides in memory order, along with its base offset in
// elements. Anything not known at compile time reads back as mlir::ShapedType::kDynamic.
static void getMemNDStridesAndOffset(mlir::MemRefType type, llvm::SmallVectorImpl<int64_t>& strides, int64_t& offset) {
    const auto ndType = mlir::cast<vpux::NDTypeInterface>(type);
    strides.clear();
    offset = 0;
    VPUX_THROW_WHEN(mlir::failed(type.getStridesAndOffset(strides, offset)),
                    "Cannot read strides and offset of operand type '{0}'", type);

    const auto dimsOrder = ndType.getDimsOrder();
    const llvm::SmallVector<int64_t> logicalStrides(strides.begin(), strides.end());
    for (size_t physPos = 0; physPos < logicalStrides.size(); ++physPos) {
        strides[physPos] = logicalStrides[dimsOrder.dimAt(physPos).ind()];
    }
}

// Per-callsite value of the iteration-space scalar stored at `attrIdx` in a run op's
// attribute list. Returns kDynamic if dynamic.
static int64_t iterationScalarValue(VPUIP::SwKernelRun runOp, unsigned attrIdx) {
    const auto attrs = runOp.getAttrs();
    if (!attrs.has_value() || attrIdx >= attrs->size()) {
        return mlir::ShapedType::kDynamic;
    }
    if (const auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>((*attrs)[attrIdx])) {
        return intAttr.getInt();
    }
    return mlir::ShapedType::kDynamic;
}

// ===========================================================================
// Partition model
// ---------------------------------------------------------------------------
// The specialization analysis half is constructed from an immutable per-callsite value
// snapshot (CallsiteValues), a lightweight callsite-subset view (Partition) and
// a heuristic (SplitHeuristic).
// Everything stores values in physical layout.
// ===========================================================================

// Physical-layout identity of one specializable quantity. physPos is the
// physical position of the dim, which makes
// values comparable across callsites regardless of each callsite's logical
// DimsOrder.
struct ParamKey {
    // For Dim/Stride/Offset this is the memref arg index. For IterScalar it is the
    // callee block-arg index of the scalar.
    unsigned argIdx = 0;
    // Physical position of the dim or stride this key refers to. Unused for Offset
    // and IterScalar, which are single scalars with no dim.
    unsigned physPos = 0;
    // We can specialize on dims and strides; Memref offsets and iteration-space scalars
    // are only propagated if uniform and never drive specialization,
    // however we track them here to share the same uniform-detection machinery.
    enum class Kind { Dim, Stride, Offset, IterScalar } kind = Kind::Dim;

    bool operator==(const ParamKey& other) const {
        return argIdx == other.argIdx && physPos == other.physPos && kind == other.kind;
    }

    // Comparator, used for binary search: sorted by argument index, kind and physPos.
    bool operator<(const ParamKey& other) const {
        if (argIdx != other.argIdx) {
            return argIdx < other.argIdx;
        }
        if (kind != other.kind) {
            return kind < other.kind;
        }
        return physPos > other.physPos;  // innermost (highest physPos) first
    }
};

// A set of distinct observed values for one parameter across some callsites.
// mlir::ShapedType::kDynamic encodes the "unknown" (dynamic) value.
using ValueSet = llvm::SmallVector<int64_t, 8>;

// One split decision: peel callsites whose `param` equals `value` into the matching
// side; everything else goes to the complement.
struct SplitDecision {
    ParamKey param;
    int64_t value = 0;
};

// Which side of the binary split a SplitDecision describes: the callsites whose param
// equals its value, or all the others.
enum class Side { Matching, Complement };

// Immutable physical snapshot of every callsite's specializable values, built
// once from the callee and callsites. After construction, nothing in the partition
// layer ever looks at the IR and all queries are answered from
// here. Parameters cover dynamic dims, strides, offsets, and iteration-space
// scalars.
class CallsiteValues {
public:
    // Builds the internal representation from the IR.
    static CallsiteValues build(mlir::func::FuncOp callee, llvm::ArrayRef<RunPair> callsites, unsigned firstOutputIdx);

    size_t numCallsites() const {
        return _numCallsites;
    }
    unsigned firstOutputIdx() const {
        return _firstOutputIdx;
    }
    unsigned numArgs() const {
        return _numArgs;
    }

    // Returns the physical-order type of the memref arg at the `argIdx` position or null for scalar/non-memref
    // args.
    mlir::MemRefType declaredArgType(unsigned argIdx) const {
        return _declaredArgTypes[argIdx];
    }

    // A pessimized sum of the arguments volumes in elements. Dynamic dims are counted as one.
    int64_t volume(size_t callsiteIdx) const {
        return _volumes[callsiteIdx];
    }

    // Every recorded parameter (dims, strides, offsets, iteration-space scalars), sorted.
    llvm::ArrayRef<ParamKey> params() const {
        return _paramKeys;
    }

    // The recorded params of one (memref arg, kind) group, innermost dim/stride
    // first.
    llvm::ArrayRef<ParamKey> paramsFor(unsigned argIdx, ParamKey::Kind kind) const;

    // Per-callsite values for `param`, indexed by global callsite index.
    // Always numCallsites() long: a param with nothing recorded reads as kDynamic
    // at every callsite.
    llvm::ArrayRef<int64_t> row(const ParamKey& param) const;

private:
    CallsiteValues() = default;

    struct ParamValues {
        ParamKey key;
        // Physical value at each global callsite (kDynamic represents unknown values).
        llvm::SmallVector<int64_t> perCallsite;
    };
    // Parameters for the callsites, sorted by the parameter keys.
    llvm::SmallVector<ParamValues> _params;
    // The keys of _params in the same order as _params.
    llvm::SmallVector<ParamKey> _paramKeys;
    size_t _numCallsites = 0;
    // Index of the first result argument of the kernel.
    unsigned _firstOutputIdx = 0;
    unsigned _numArgs = 0;
    // Declared physical-order type of each memref arg (null for scalar args).
    llvm::SmallVector<mlir::MemRefType> _declaredArgTypes;
    llvm::SmallVector<int64_t> _volumes;
    // kDynamic at every callsite; what row() returns for an unrecorded param, so that
    // it can always hand back a full-length row instead of an empty one.
    llvm::SmallVector<int64_t> _absentRow;
};

// A subset of callsites plus the value information induced by them. Cheap to
// copy. Every query is derived from the precomputed _values snapshot.
class Partition {
public:
    Partition(const CallsiteValues& values, llvm::SmallVector<size_t> callsites)
            : _values(&values), _callsites(std::move(callsites)) {
    }

    llvm::ArrayRef<size_t> callsites() const {
        return _callsites;
    }
    const CallsiteValues& values() const {
        return *_values;
    }

    // --- value queries over this partition ---
    // Distinct values of param over _callsites; includes mlir::ShapedType::kDynamic
    // iff some callsite is unknown.
    ValueSet possibleValues(const ParamKey& param) const;
    // The single concrete value shared by every callsite, or nullopt if the param
    // is non-uniform.
    std::optional<int64_t> uniformValue(const ParamKey& param) const;

    // Returns the specialized memref type given this partition.
    mlir::MemRefType specializeArgType(unsigned argIdx) const;

    // Create a new partition based on a specialization decision and a given side.
    Partition restrictTo(const SplitDecision& decision, Side side = Side::Matching) const;

    // Sum of per-callsite operand volumes over this partition's callsites. Weights
    // a split's execution saving in the heuristic. See CallsiteValues::volume.
    int64_t totalVolume() const;

    // Returns the possible values in the partition callsites of parameters with more
    // than one possible value if all possible values are known.
    // For such parameters we can pass additional runtime information to the optimizer.
    // using llvm.intr.assume.
    using AssumeCandidates = llvm::SmallVector<std::pair<ParamKey, llvm::SmallVector<int64_t>>>;
    AssumeCandidates assumeCandidates() const;

    // Returns a pair of (calleeArgIdx, value) for each iteration-space scalar arg uniform across this
    // partition's callsites.
    llvm::SmallVector<std::pair<unsigned, int64_t>> uniformIterScalars() const;

private:
    const CallsiteValues* _values;
    llvm::SmallVector<size_t> _callsites;
};

// Specialization policy interface.
class SplitHeuristic {
public:
    virtual ~SplitHeuristic() = default;

    // The parameter+value to split `partition` on next, or nullopt to make it a leaf.
    virtual std::optional<SplitDecision> chooseSplit(const Partition& partition) const = 0;
};

// 'never': every partition is a leaf and never split. Uniform values are still specialized by the
// transformation.
class NeverSplitHeuristic final : public SplitHeuristic {
public:
    explicit NeverSplitHeuristic(Logger log): _log(std::move(log)) {
    }
    std::optional<SplitDecision> chooseSplit(const Partition& partition) const override {
        _log.trace("leaf: {0} callsite(s), 'never' policy -- no split", partition.callsites().size());
        return std::nullopt;
    }

private:
    Logger _log;
};

// 'always': Creates the maximum number of specializations possible.
class AlwaysSplitHeuristic final : public SplitHeuristic {
public:
    explicit AlwaysSplitHeuristic(Logger log): _log(std::move(log)) {
    }
    std::optional<SplitDecision> chooseSplit(const Partition& partition) const override;

private:
    Logger _log;
};

// 'cost' (default): split only when the cost model says that savings on the matched
// callsites outweighs one extra kernel's cost.
class CostSplitHeuristic final : public SplitHeuristic {
public:
    explicit CostSplitHeuristic(Logger log): _log(std::move(log)) {
    }
    std::optional<SplitDecision> chooseSplit(const Partition& partition) const override;

private:
    // Innermost contiguous static element count of `child`'s specialized output type.
    int64_t contiguousN(const Partition& child) const;
    double penalty(const Partition& partition) const;

    // Cost model constants.
    static constexpr int64_t kIdealContiguous = 256;
    static constexpr int64_t kGranularity = 32;
    static constexpr int64_t kAlignmentBytes = 32;
    static constexpr double kUnalignedPenalty = 1.5;
    static constexpr int64_t kExtraKernelCost = 512;

    static constexpr double kIdealPenalty = 1.0;
    static_assert(kIdealContiguous % kGranularity == 0, "kIdealContiguous should be a multiple of kGranularity");
    static_assert(kUnalignedPenalty >= kIdealPenalty, "the alignment factor must be greater than 1.0");

    Logger _log;
};

// Splits callsites into specialization partitions given the chosen heuristic.
static llvm::SmallVector<Partition> planSpecializations(const CallsiteValues& values, const SplitHeuristic& heuristic);

// ---------------------------------------------------------------------------
// Implementations.
// ---------------------------------------------------------------------------

CallsiteValues CallsiteValues::build(mlir::func::FuncOp callee, llvm::ArrayRef<RunPair> callsites,
                                     unsigned firstOutputIdx) {
    CallsiteValues result;
    result._numCallsites = callsites.size();
    result._firstOutputIdx = firstOutputIdx;
    result._absentRow.assign(callsites.size(), mlir::ShapedType::kDynamic);

    auto firstRunOp = callsites.front().second;
    const unsigned numRunArgs = static_cast<unsigned>(firstRunOp.getArgs().size());
    result._numArgs = numRunArgs;

    const unsigned numCalleeArgs = callee.getNumArguments();
    const unsigned numIterScalars = numIterationSpaceScalars(callee);
    VPUX_THROW_UNLESS(numCalleeArgs >= numRunArgs + numIterScalars,
                      "SpecializeSwKernels: kernel '{0}' declares {1} iteration-space scalar(s) but has only "
                      "{2} arg(s) past its {3} memref arg(s)",
                      callee.getSymName(), numIterScalars, numCalleeArgs - numRunArgs, numRunArgs);

    // Record each memref arg's type; null for scalars.
    result._declaredArgTypes.reserve(numRunArgs);
    for (unsigned argIdx = 0; argIdx < numRunArgs; ++argIdx) {
        result._declaredArgTypes.push_back(mlir::dyn_cast<mlir::MemRefType>(callee.getArgument(argIdx).getType()));
    }

    // Per-callsite volume: total element count over the memref operands the body reads,
    // captured now while the raw run ops are still in hand.
    result._volumes.reserve(callsites.size());
    for (auto [_, runOp] : callsites) {
        int64_t callsiteVolume = 0;
        for (const auto& [argIdx, arg] : llvm::enumerate(runOp.getArgs())) {
            // An unused arg is free.
            if (callee.getArgument(static_cast<unsigned>(argIdx)).use_empty()) {
                continue;
            }
            const auto memref = mlir::dyn_cast<mlir::MemRefType>(arg.getType());
            if (!memref) {
                continue;
            }
            int64_t argVolume = 1;
            for (int64_t dimSize : memref.getShape()) {
                if (!mlir::ShapedType::isDynamic(dimSize)) {
                    argVolume *= dimSize;
                }
            }
            callsiteVolume += argVolume;
        }
        result._volumes.push_back(callsiteVolume);
    }

    for (unsigned argIdx = 0; argIdx < numRunArgs; ++argIdx) {
        // Dead args have no uses in the body and cannot benefit from specialization.
        if (callee.getArgument(argIdx).use_empty()) {
            continue;
        }
        // Only memref args carry shape/stride information.
        auto declaredType = mlir::dyn_cast<mlir::MemRefType>(callee.getArgument(argIdx).getType());
        if (!declaredType) {
            continue;
        }

        // Kernel args are always in memory order.
        const auto dimsOrder = mlir::cast<vpux::NDTypeInterface>(callee.getArgument(argIdx).getType()).getDimsOrder();
        VPUX_THROW_UNLESS(dimsOrder.isIdentity(),
                          "SpecializeSwKernels: arg #{0} of kernel '{1}' has non-identity order '{2}'; kernel args "
                          "are expected to be in memory order",
                          argIdx, callee.getSymName(), dimsOrder);
        const auto callsiteLogicalDimOf = [&](mlir::MemRefType callsiteType, unsigned physPos) -> unsigned {
            const auto callsiteDimsOrder = mlir::cast<vpux::NDTypeInterface>(callsiteType).getDimsOrder();
            return static_cast<unsigned>(callsiteDimsOrder.dimAt(physPos).ind());
        };

        // Construct dim params with values matched to the callsites.
        auto shape = declaredType.getShape();
        for (unsigned physPos = 0; physPos < static_cast<unsigned>(shape.size()); ++physPos) {
            if (!mlir::ShapedType::isDynamic(shape[physPos])) {
                continue;
            }
            ParamValues paramValues;
            paramValues.key = ParamKey{argIdx, physPos, ParamKey::Kind::Dim};
            paramValues.perCallsite.reserve(callsites.size());
            for (auto [_, runOp] : callsites) {
                auto callsiteType = mlir::cast<mlir::MemRefType>(runOp.getArgs()[argIdx].getType());
                paramValues.perCallsite.push_back(
                        callsiteType.getShape()[callsiteLogicalDimOf(callsiteType, paramValues.key.physPos)]);
            }
            result._params.push_back(std::move(paramValues));
        }

        // Construct offset and stride params with values matched to the callsites.
        if (!mlir::isa<mlir::StridedLayoutAttr>(declaredType.getLayout())) {
            continue;
        }
        int64_t declaredOffset = 0;
        llvm::SmallVector<int64_t> declaredStrides;
        if (mlir::failed(declaredType.getStridesAndOffset(declaredStrides, declaredOffset))) {
            continue;
        }
        for (unsigned physPos = 0; physPos < static_cast<unsigned>(declaredStrides.size()); ++physPos) {
            if (!mlir::ShapedType::isDynamic(declaredStrides[physPos])) {
                continue;
            }
            ParamValues paramValues;
            paramValues.key = ParamKey{argIdx, physPos, ParamKey::Kind::Stride};
            paramValues.perCallsite.reserve(callsites.size());
            for (auto [_, runOp] : callsites) {
                auto callsiteType = mlir::cast<mlir::MemRefType>(runOp.getArgs()[argIdx].getType());
                int64_t callsiteOffset = 0;
                llvm::SmallVector<int64_t> callsiteStrides;
                getMemNDStridesAndOffset(callsiteType, callsiteStrides, callsiteOffset);
                paramValues.perCallsite.push_back(callsiteStrides[paramValues.key.physPos]);
            }
            result._params.push_back(std::move(paramValues));
        }

        if (mlir::ShapedType::isDynamic(declaredOffset)) {
            ParamValues paramValues;
            paramValues.key = ParamKey{argIdx, 0, ParamKey::Kind::Offset};
            paramValues.perCallsite.reserve(callsites.size());
            for (auto [_, runOp] : callsites) {
                auto callsiteType = mlir::cast<mlir::MemRefType>(runOp.getArgs()[argIdx].getType());
                int64_t callsiteOffset = 0;
                llvm::SmallVector<int64_t> callsiteStrides;
                getMemNDStridesAndOffset(callsiteType, callsiteStrides, callsiteOffset);
                paramValues.perCallsite.push_back(callsiteOffset);
            }
            result._params.push_back(std::move(paramValues));
        }
    }

    // A KernelInfo-marked kernel carries, as its last (2 * numTilingAxes) args, the
    // iteration-space scalar arguments.
    if (numIterScalars > 0) {
        for (unsigned scalarIdx = 0; scalarIdx < numIterScalars; ++scalarIdx) {
            const unsigned calleeArgIdx = numCalleeArgs - numIterScalars + scalarIdx;
            if (callee.getArgument(calleeArgIdx).use_empty()) {
                // Dead args cannot benefit from specialization.
                continue;
            }
            const unsigned attrIdx = calleeArgIdx - numRunArgs;
            ParamValues paramValues;
            paramValues.key = ParamKey{calleeArgIdx, 0, ParamKey::Kind::IterScalar};
            paramValues.perCallsite.reserve(callsites.size());
            for (auto [_, runOp] : callsites) {
                paramValues.perCallsite.push_back(iterationScalarValue(runOp, attrIdx));
            }
            result._params.push_back(std::move(paramValues));
        }
    }

    VPUX_THROW_WHEN(result._declaredArgTypes[firstOutputIdx] == nullptr,
                    "SpecializeSwKernels: first output arg (#{0}) of kernel '{1}' is not a memref", firstOutputIdx,
                    callee.getSymName());

    // Sort _params by its keys to satisfy the class invariant.
    llvm::sort(result._params, [](const ParamValues& lhs, const ParamValues& rhs) {
        return lhs.key < rhs.key;
    });
    result._paramKeys.reserve(result._params.size());
    for (const auto& paramValues : result._params) {
        result._paramKeys.push_back(paramValues.key);
    }

    return result;
}

llvm::ArrayRef<ParamKey> CallsiteValues::paramsFor(unsigned argIdx, ParamKey::Kind kind) const {
    // Compare on the leading (argIdx, kind) in order to capture keys with
    // different physPos values. Relies on the existing ordering of keys (physPos
    // is compared last).
    const auto groupLess = [](const ParamKey& lhs, const ParamKey& rhs) {
        if (lhs.argIdx != rhs.argIdx) {
            return lhs.argIdx < rhs.argIdx;
        }
        return lhs.kind < rhs.kind;
    };
    const ParamKey probe{argIdx, 0, kind};
    const auto [first, last] = std::equal_range(_paramKeys.begin(), _paramKeys.end(), probe, groupLess);
    return llvm::ArrayRef<ParamKey>(_paramKeys)
            .slice(static_cast<size_t>(first - _paramKeys.begin()), static_cast<size_t>(last - first));
}

llvm::ArrayRef<int64_t> CallsiteValues::row(const ParamKey& param) const {
    // Linear on purpose since the number of keys is expected to be small.
    const auto it = llvm::find_if(_params, [&param](const ParamValues& paramValues) {
        return paramValues.key == param;
    });
    // nothing recorded: unknown at every callsite
    return it != _params.end() ? it->perCallsite : _absentRow;
}

ValueSet Partition::possibleValues(const ParamKey& param) const {
    const llvm::ArrayRef<int64_t> rowVals = _values->row(param);
    ValueSet distinctValues;
    for (size_t callsiteIdx : _callsites) {
        const int64_t value = rowVals[callsiteIdx];
        if (!llvm::is_contained(distinctValues, value)) {
            distinctValues.push_back(value);
        }
    }
    return distinctValues;
}

std::optional<int64_t> Partition::uniformValue(const ParamKey& param) const {
    if (_callsites.empty()) {
        return std::nullopt;
    }
    const llvm::ArrayRef<int64_t> rowVals = _values->row(param);
    const int64_t firstValue = rowVals[_callsites.front()];
    if (mlir::ShapedType::isDynamic(firstValue)) {
        return std::nullopt;
    }
    if (llvm::any_of(_callsites, [&rowVals, firstValue](size_t callsiteIdx) {
            return rowVals[callsiteIdx] != firstValue;
        })) {
        return std::nullopt;
    }
    return firstValue;
}

Partition Partition::restrictTo(const SplitDecision& decision, Side side) const {
    const llvm::ArrayRef<int64_t> rowVals = _values->row(decision.param);
    llvm::SmallVector<size_t> selected;
    for (size_t callsiteIdx : _callsites) {
        if ((rowVals[callsiteIdx] == decision.value) == (side == Side::Matching)) {
            selected.push_back(callsiteIdx);
        }
    }
    return Partition(*_values, std::move(selected));
}

Partition::AssumeCandidates Partition::assumeCandidates() const {
    AssumeCandidates result;
    for (const ParamKey& param : _values->params()) {
        // We only emit llvm.intr.assume for dims and iteration space args so skip
        // anything else.
        if (param.kind != ParamKey::Kind::Dim && param.kind != ParamKey::Kind::IterScalar) {
            continue;
        }
        const ValueSet distinctValues = possibleValues(param);
        if (distinctValues.size() > 1 && !llvm::is_contained(distinctValues, mlir::ShapedType::kDynamic)) {
            result.push_back({param, llvm::SmallVector<int64_t>(distinctValues.begin(), distinctValues.end())});
        }
    }
    return result;
}

llvm::SmallVector<std::pair<unsigned, int64_t>> Partition::uniformIterScalars() const {
    llvm::SmallVector<std::pair<unsigned, int64_t>> result;
    for (const ParamKey& param : _values->params()) {
        if (param.kind != ParamKey::Kind::IterScalar) {
            continue;
        }
        if (const std::optional<int64_t> value = uniformValue(param)) {
            result.push_back({param.argIdx, *value});
        }
    }
    return result;
}

int64_t Partition::totalVolume() const {
    int64_t sum = 0;
    for (size_t callsiteIdx : _callsites) {
        sum += _values->volume(callsiteIdx);
    }
    return sum;
}

mlir::MemRefType Partition::specializeArgType(unsigned argIdx) const {
    const mlir::MemRefType declaredType = _values->declaredArgType(argIdx);
    if (!declaredType) {
        return nullptr;
    }
    const unsigned rank = static_cast<unsigned>(declaredType.getRank());

    // Overlay this partition's uniform values onto the declared physical-order type.
    // physPos indexes the memref dims directly (callee args are physical order); a
    // param absent from the snapshot yields nullopt, leaving that slot dynamic. Dims
    // and strides that became uniform as a side effect of a split fold in here.
    llvm::SmallVector<int64_t> shape(declaredType.getShape().begin(), declaredType.getShape().end());
    for (unsigned physPos = 0; physPos < rank; ++physPos) {
        if (!mlir::ShapedType::isDynamic(shape[physPos])) {
            continue;
        }
        if (const std::optional<int64_t> uniformDim = uniformValue(ParamKey{argIdx, physPos, ParamKey::Kind::Dim})) {
            shape[physPos] = *uniformDim;
        }
    }

    // A packed identity layout recomputes its strides from the shape, so only the
    // dimensions matter.
    if (!mlir::isa<mlir::StridedLayoutAttr>(declaredType.getLayout())) {
        return mlir::MemRefType::get(shape, declaredType.getElementType(), declaredType.getLayout(),
                                     declaredType.getMemorySpace());
    }
    auto [strides, offset] = declaredType.getStridesAndOffset();  // StridedLayoutAttr always succeeds
    for (unsigned physPos = 0; physPos < rank; ++physPos) {
        if (mlir::ShapedType::isDynamic(strides[physPos])) {
            if (const std::optional<int64_t> uniformStride =
                        uniformValue(ParamKey{argIdx, physPos, ParamKey::Kind::Stride})) {
                strides[physPos] = *uniformStride;
            }
        }
    }
    if (mlir::ShapedType::isDynamic(offset)) {
        if (const std::optional<int64_t> uniformOffset = uniformValue(ParamKey{argIdx, 0, ParamKey::Kind::Offset})) {
            offset = *uniformOffset;
        }
    }
    return mlir::MemRefType::get(shape, declaredType.getElementType(),
                                 mlir::StridedLayoutAttr::get(declaredType.getContext(), offset, strides),
                                 declaredType.getMemorySpace());
}

int64_t CostSplitHeuristic::contiguousN(const Partition& child) const {
    mlir::MemRefType physType = child.specializeArgType(child.values().firstOutputIdx());
    const unsigned rank = static_cast<unsigned>(physType.getRank());

    // MemRefType finds how many trailing dims form a contiguous block (handling
    // unit dims and stride gaps); we take the product of their trailing static dimensions,
    // stopping at the first dynamic one.
    const int64_t numContig = physType.getNumContiguousTrailingDims();
    int64_t numElems = 1;
    for (int64_t physPos = static_cast<int64_t>(rank) - 1; physPos >= static_cast<int64_t>(rank) - numContig;
         --physPos) {
        if (mlir::ShapedType::isDynamic(physType.getShape()[physPos])) {
            break;
        }
        numElems *= physType.getShape()[physPos];
    }
    return numElems;
}

double CostSplitHeuristic::penalty(const Partition& partition) const {
    const int64_t contiguousElems = contiguousN(partition);

    int64_t maxBeneficial = std::min(contiguousElems, kIdealContiguous);
    maxBeneficial -= maxBeneficial % kGranularity;
    const double contiguityFactor = maxBeneficial >= kGranularity
                                            ? static_cast<double>(kIdealContiguous) / static_cast<double>(maxBeneficial)
                                            : static_cast<double>(kIdealContiguous);

    const auto outType = partition.values().declaredArgType(partition.values().firstOutputIdx());
    const int64_t elemBits = mlir::cast<vpux::NDTypeInterface>(outType).getElemTypeSize().count();
    const bool aligned = (contiguousElems * elemBits) % (kAlignmentBytes * 8) == 0;
    const double alignFactor = aligned ? 1.0 : kUnalignedPenalty;

    return contiguityFactor * alignFactor;
}

// Human-readable tag for a param's kind, used only in decision logging.
static llvm::StringRef paramKindName(ParamKey::Kind kind) {
    switch (kind) {
    case ParamKey::Kind::Dim:
        return "dim";
    case ParamKey::Kind::Stride:
        return "stride";
    case ParamKey::Kind::Offset:
        return "offset";
    case ParamKey::Kind::IterScalar:
        return "iter-scalar";
    }
    return "?";
}

// Get the specialization candidates from the first memref output.
// These will be ordered by physical position, from the innermost to the outermost, with
// a position's dim ahead of its stride.
// The idea behind this is that the input shapes are connected to the output shapes,
// so specializing on the output can likely properly specialize the inputs as well
// while avoiding a large space to explore.
static llvm::SmallVector<ParamKey, 16> firstOutputSpecializationCandidates(const Partition& partition) {
    const CallsiteValues& values = partition.values();
    const unsigned outIdx = values.firstOutputIdx();
    const llvm::ArrayRef<ParamKey> dims = values.paramsFor(outIdx, ParamKey::Kind::Dim);
    const llvm::ArrayRef<ParamKey> strides = values.paramsFor(outIdx, ParamKey::Kind::Stride);
    // Both groups already run innermost-first, which is this order restricted to one
    // kind, so merging them is enough. Kind::Dim < Kind::Stride breaks the tie
    // at a shared position.
    const auto innermostFirstDimBeforeStride = [](const ParamKey& lhs, const ParamKey& rhs) {
        if (lhs.physPos != rhs.physPos) {
            return lhs.physPos > rhs.physPos;
        }
        return lhs.kind < rhs.kind;
    };
    llvm::SmallVector<ParamKey, 16> ordered;
    ordered.reserve(dims.size() + strides.size());
    std::merge(dims.begin(), dims.end(), strides.begin(), strides.end(), std::back_inserter(ordered),
               innermostFirstDimBeforeStride);
    return ordered;
}

std::optional<SplitDecision> AlwaysSplitHeuristic::chooseSplit(const Partition& partition) const {
    // Peel the first non-uniform candidate that carries a concrete value.
    for (const ParamKey& param : firstOutputSpecializationCandidates(partition)) {
        const ValueSet values = partition.possibleValues(param);
        if (values.size() == 1) {
            continue;
        }
        for (int64_t value : values) {
            if (mlir::ShapedType::isDynamic(value)) {
                // We cannot specialize on an unknown value.
                continue;
            }
            _log.trace("split arg{0} {1} phys{2} = {3}: 'always' policy", param.argIdx, paramKindName(param.kind),
                       param.physPos, value);
            return SplitDecision{param, value};
        }
    }
    _log.trace("leaf: {0} callsite(s), 'always' policy -- nothing left to peel", partition.callsites().size());
    return std::nullopt;
}

std::optional<SplitDecision> CostSplitHeuristic::chooseSplit(const Partition& partition) const {
    // Cost of keeping `partition` as one shared kernel: its callsites run under the
    // partition's (still-generic) first-output type. Already ideal => nothing to gain, leaf.
    const double parentPenalty = penalty(partition);
    if (parentPenalty <= kIdealPenalty) {
        _log.trace("leaf: {0} callsite(s), penalty {1} already ideal -- no split", partition.callsites().size(),
                   parentPenalty);
        return std::nullopt;
    }

    // Walk candidates innermost-first; the first param yielding a split whose
    // saving on the matched callsites outweighs one extra kernel's cost
    // wins.
    double overallBestSaving = 0.0;
    for (const ParamKey& param : firstOutputSpecializationCandidates(partition)) {
        const ValueSet values = partition.possibleValues(param);
        if (values.size() == 1) {
            continue;
        }
        std::optional<SplitDecision> best;
        double bestSaving = 0.0;
        double bestChildPenalty = parentPenalty;
        int64_t bestVolume = 0;
        for (int64_t value : values) {
            if (mlir::ShapedType::isDynamic(value)) {
                continue;
            }
            const Partition matched = partition.restrictTo(SplitDecision{param, value});
            const double childPenalty = penalty(matched);
            const int64_t volume = matched.totalVolume();
            const double saving = (parentPenalty - childPenalty) * static_cast<double>(volume);
            if (saving > bestSaving) {
                best = SplitDecision{param, value};
                bestSaving = saving;
                bestChildPenalty = childPenalty;
                bestVolume = volume;
            }
        }
        overallBestSaving = std::max(overallBestSaving, bestSaving);
        if (best.has_value() && bestSaving > static_cast<double>(kExtraKernelCost)) {
            _log.trace("split arg{0} {1} phys{2} = {3}: penalty {4} -> {5}, volume {6}, saving {7} > cost {8}",
                       best->param.argIdx, paramKindName(best->param.kind), best->param.physPos, best->value,
                       parentPenalty, bestChildPenalty, bestVolume, bestSaving, kExtraKernelCost);
            return best;
        }
    }
    _log.trace("leaf: {0} callsite(s), best saving {1} <= cost {2} -- not worth specializing",
               partition.callsites().size(), overallBestSaving, kExtraKernelCost);
    return std::nullopt;
}

llvm::SmallVector<Partition> planSpecializations(const CallsiteValues& values, const SplitHeuristic& heuristic) {
    llvm::SmallVector<size_t> allCallsites;
    allCallsites.reserve(values.numCallsites());
    for (size_t callsiteIdx = 0; callsiteIdx < values.numCallsites(); ++callsiteIdx) {
        allCallsites.push_back(callsiteIdx);
    }

    llvm::SmallVector<Partition> leaves;
    llvm::SmallVector<Partition> worklist;
    worklist.emplace_back(values, std::move(allCallsites));

    while (!worklist.empty()) {
        Partition partition = worklist.pop_back_val();

        const std::optional<SplitDecision> decision = heuristic.chooseSplit(partition);
        if (!decision.has_value()) {
            leaves.push_back(std::move(partition));
            continue;
        }

        worklist.push_back(partition.restrictTo(*decision, Side::Matching));
        worklist.push_back(partition.restrictTo(*decision, Side::Complement));
    }
    return leaves;
}

// Named partitioning policies backing the `heuristic` pass option.
enum class HeuristicKind { Never, Always, Cost };

// Parses the `heuristic` pass option, defaulting to 'cost' for anything unrecognized.
static HeuristicKind parseHeuristicKind(llvm::StringRef name, Logger log) {
    const std::optional<HeuristicKind> kind = llvm::StringSwitch<std::optional<HeuristicKind>>(name)
                                                      .Case("never", HeuristicKind::Never)
                                                      .Case("always", HeuristicKind::Always)
                                                      .Case("cost", HeuristicKind::Cost)
                                                      .Default(std::nullopt);
    if (!kind.has_value()) {
        log.warning("[SpecializeSwKernels] unknown heuristic '{0}'; using 'cost'", name);
        return HeuristicKind::Cost;
    }
    return *kind;
}

// Construct the policy for `kind`.
static std::unique_ptr<SplitHeuristic> makeSplitHeuristic(HeuristicKind kind, Logger log) {
    switch (kind) {
    case HeuristicKind::Never:
        return std::make_unique<NeverSplitHeuristic>(std::move(log));
    case HeuristicKind::Always:
        return std::make_unique<AlwaysSplitHeuristic>(std::move(log));
    case HeuristicKind::Cost:
        return std::make_unique<CostSplitHeuristic>(std::move(log));
    }
    llvm_unreachable("unexpected HeuristicKind");
}

//
// SpecializeSwKernelsPass
//

class SpecializeSwKernelsPass final : public ShaveCodeGen::impl::SpecializeSwKernelsBase<SpecializeSwKernelsPass> {
public:
    explicit SpecializeSwKernelsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
    void runOnKernel(mlir::func::FuncOp callee, llvm::ArrayRef<RunPair> callsites, const SplitHeuristic& heuristic,
                     bool hasSkippedOps, const mlir::FrozenRewritePatternSet& propagationPatterns);
    // Produces the pattern set used by propagateTypes. Depends only on the context, so
    // it is built once per pass run and threaded down to every clone.
    static mlir::FrozenRewritePatternSet buildTypePropagationPatterns(mlir::MLIRContext* ctx);
    // Clone the callee once per non-trivial leaf partition, specialize that leaf's
    // uniform dims/strides/offsets into the clone's block arg types, re-infer
    // derived types, emit shape assumptions for dims, redirect the leaf's
    // callsites, and erase the original when every reference to it is gone.
    // `hasSkippedOps` reports if there are any untracked SwKernelOps for this callee.
    static void specializeKernel(mlir::func::FuncOp callee, llvm::ArrayRef<RunPair> callsites,
                                 llvm::ArrayRef<Partition> leaves, bool hasSkippedOps,
                                 const mlir::FrozenRewritePatternSet& propagationPatterns, Logger log);
    // Propagate the narrowed block-arg types through the clone body by running the
    // type propagation patterns. The propagation patterns include all canonicalization
    // patterns, so this will canonicalize as well.
    static void propagateTypesAndCanonicalize(mlir::func::FuncOp funcOp,
                                              const mlir::FrozenRewritePatternSet& propagationPatterns, Logger log);
};

void SpecializeSwKernelsPass::safeRunOnModule() {
    auto moduleOp = getOperation();

    llvm::MapVector<mlir::func::FuncOp, llvm::SmallVector<RunPair>> callsiteMap;
    // Callees with callsites that are ignored by this pass (with multiple run ops).
    llvm::DenseSet<mlir::Operation*> calleesWithSkippedOps;
    {
        // Caches each visited symbol table.
        mlir::SymbolTableCollection symbolTableCollection;

        moduleOp.walk([&](VPUIP::SwKernelOp kernelOp) {
            auto callee = symbolTableCollection.lookupNearestSymbolFrom<mlir::func::FuncOp>(
                    kernelOp, kernelOp.getKernelFunction());
            if (!callee) {
                return;
            }
            // Only specialize ShaveCodeGen kernels.
            if (callee.isDeclaration()) {
                return;
            }
            // We expect the pass to run after all SwKernelOps have been unrolled,
            // and to have a single SwKernelRun per op. Otherwise we would have to deal
            // with the fact that the callee symref needs to be the same on all child
            // SwKernelRun ops. If we encounter a case where this is not true we skip
            // specialization.
            const auto runOps = kernelOp.getBody().front().getOps<VPUIP::SwKernelRun>();
            const auto numRunOps = std::distance(runOps.begin(), runOps.end());
            if (numRunOps != 1) {
                _log.trace("[SpecializeSwKernels] skipping SwKernelOp at '{0}': {1} SW.Kernel.run op(s) share the one "
                           "kernel symbol of the enclosing op",
                           kernelOp->getLoc(), numRunOps);
                calleesWithSkippedOps.insert(callee.getOperation());
                return;
            }
            for (auto runOp : runOps) {
                callsiteMap[callee].emplace_back(kernelOp, runOp);
            }
        });
    }

    // Nothing to specialize: bail out before paying for the pattern set below.
    if (callsiteMap.empty()) {
        return;
    }

    // Build the partitioning policy once, chosen by the `heuristic` pass option.
    const HeuristicKind heuristicKind = parseHeuristicKind(heuristic, _log.nest());
    const std::unique_ptr<SplitHeuristic> splitHeuristic = makeSplitHeuristic(heuristicKind, _log.nest());

    const mlir::FrozenRewritePatternSet propagationPatterns = buildTypePropagationPatterns(moduleOp.getContext());

    for (auto& [callee, callsites] : callsiteMap) {
        runOnKernel(callee, callsites, *splitHeuristic, calleesWithSkippedOps.contains(callee.getOperation()),
                    propagationPatterns);
    }
}

void SpecializeSwKernelsPass::runOnKernel(mlir::func::FuncOp callee, llvm::ArrayRef<RunPair> callsites,
                                          const SplitHeuristic& heuristic, bool hasSkippedOps,
                                          const mlir::FrozenRewritePatternSet& propagationPatterns) {
    auto firstKernelOp = callsites.front().first;
    auto firstRunOp = callsites.front().second;

    // Determine the index of the first output memref.
    const auto numKernelInputs = firstKernelOp.getInputs().size();
    const auto runArgs = firstRunOp.getArgs();
    const auto firstOutputIt = llvm::find_if(runArgs, [&](mlir::Value arg) {
        auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(arg);
        VPUX_THROW_WHEN(blockArg == nullptr, "SpecializeSwKernels: SwKernelRun arg is not a block arg at '{0}'",
                        firstRunOp->getLoc());
        return blockArg.getArgNumber() >= numKernelInputs;
    });
    VPUX_THROW_WHEN(firstOutputIt == runArgs.end(), "SpecializeSwKernels: SwKernelRun at '{0}' has no output arg",
                    firstRunOp->getLoc());
    const unsigned firstOutputIdx = static_cast<unsigned>(std::distance(runArgs.begin(), firstOutputIt));

    // Build the callsite values snapshot once, then plan the partitioning.
    const CallsiteValues values = CallsiteValues::build(callee, callsites, firstOutputIdx);
    _log.trace("[SpecializeSwKernels] @{0}: {1} callsite(s)", callee.getSymName(), callsites.size());
    const llvm::SmallVector<Partition> leaves = planSpecializations(values, heuristic);

    _log.nest().trace("-> {0} partition(s)", leaves.size());

    specializeKernel(callee, callsites, leaves, hasSkippedOps, propagationPatterns, _log.nest(2));
}

static void insertShapeAssumptions(mlir::func::FuncOp clone,
                                   llvm::ArrayRef<std::pair<ParamKey, llvm::SmallVector<int64_t>>> assumeCandidates);

void SpecializeSwKernelsPass::specializeKernel(mlir::func::FuncOp callee, llvm::ArrayRef<RunPair> callsites,
                                               llvm::ArrayRef<Partition> leaves, bool hasSkippedOps,
                                               const mlir::FrozenRewritePatternSet& propagationPatterns, Logger log) {
    mlir::MLIRContext* ctx = callee.getContext();
    VPUIP::SwKernelOp firstKernelOp = callsites.front().first;
    const mlir::SymbolRefAttr origRef = firstKernelOp.getKernelFunction();
    size_t totalRedirected = 0;

    // Clones are added through the symbol table so that their names are uniqued against
    // the symbols already present, rather than silently redefining one.
    mlir::SymbolTable symbolTable(mlir::SymbolTable::getNearestSymbolTable(callee));

    for (size_t leafIdx = 0; leafIdx < leaves.size(); ++leafIdx) {
        const Partition& leaf = leaves[leafIdx];
        const CallsiteValues& values = leaf.values();
        const Partition::AssumeCandidates assumeCandidates = leaf.assumeCandidates();
        const llvm::SmallVector<std::pair<unsigned, int64_t>> uniformScalars = leaf.uniformIterScalars();

        // Build each memref arg's specialized type once (the single place clone arg
        // types are computed), keeping only the args whose type actually changes.
        llvm::SmallVector<std::pair<unsigned, mlir::MemRefType>> specializedTypes;
        for (unsigned argIdx = 0; argIdx < values.numArgs(); ++argIdx) {
            const mlir::MemRefType specialized = leaf.specializeArgType(argIdx);
            if (specialized && specialized != values.declaredArgType(argIdx)) {
                specializedTypes.push_back({argIdx, specialized});
            }
        }

        // Nothing to specialize for this leaf: cloning would produce a function
        // identical to the original, so leave its callsites on the original.
        if (specializedTypes.empty() && uniformScalars.empty() && assumeCandidates.empty()) {
            log.trace("partition {0}: {1} callsite(s), nothing to specialize -- left on @{2}", leafIdx,
                      leaf.callsites().size(), callee.getSymName());
            continue;
        }

        auto clone = mlir::cast<mlir::func::FuncOp>(callee->clone());
        clone.setSymName((callee.getSymName() + "_spec_" + llvm::Twine(leafIdx)).str());
        symbolTable.insert(clone, std::next(mlir::Block::iterator(callee)));

        // Narrow each specialized block arg to its specialized type, then cast the
        // arg back to its original type.
        if (!specializedTypes.empty()) {
            mlir::Block& entryBlock = clone.getBody().front();
            mlir::OpBuilder castBuilder(&entryBlock, entryBlock.begin());
            for (const auto& [argIdx, newType] : specializedTypes) {
                mlir::BlockArgument arg = clone.getArgument(argIdx);
                const mlir::Type origType = arg.getType();
                arg.setType(newType);
                auto castOp = mlir::memref::CastOp::create(castBuilder, clone.getLoc(), origType, arg);
                arg.replaceAllUsesExcept(castOp.getResult(), castOp);
            }
        }

        // Specialize uniform iteration-space scalars. Replace each block arg's uses with a
        // constant.
        if (!uniformScalars.empty()) {
            mlir::OpBuilder scalarBuilder(&clone.getBody().front(), clone.getBody().front().begin());
            for (const auto& [argIdx, value] : uniformScalars) {
                mlir::BlockArgument arg = clone.getArgument(argIdx);
                auto constOp = mlir::arith::ConstantIndexOp::create(scalarBuilder, clone.getLoc(), value);
                arg.replaceAllUsesWith(constOp.getResult());
            }
        }

        // Rebuild the function type to match the updated block arg types.
        llvm::SmallVector<mlir::Type> newArgTypes;
        for (mlir::BlockArgument arg : clone.getArguments()) {
            newArgTypes.push_back(arg.getType());
        }
        clone.setFunctionType(mlir::FunctionType::get(ctx, newArgTypes, clone.getFunctionType().getResults()));

        // Emit llvm.intr.assume for dims that stay dynamic but have a
        // set of known values across this leaf's callsites.
        insertShapeAssumptions(clone, assumeCandidates);

        // Propagate updated block arg types through the clone body.
        propagateTypesAndCanonicalize(clone, propagationPatterns, log);

        // Construct the redirect symbol reference. Replace only the leaf name
        // while preserving any root prefix.
        const mlir::StringAttr newLeaf = clone.getSymNameAttr();
        mlir::SymbolRefAttr newRef;
        if (origRef.getNestedReferences().empty()) {
            newRef = mlir::SymbolRefAttr::get(newLeaf);
        } else {
            auto nested = llvm::to_vector(origRef.getNestedReferences().drop_back());
            nested.push_back(mlir::FlatSymbolRefAttr::get(newLeaf));
            newRef = mlir::SymbolRefAttr::get(origRef.getRootReference(), nested);
        }

        // Redirect every callsite in this leaf to the specialized clone.
        for (size_t callsiteIdx : leaf.callsites()) {
            VPUIP::SwKernelOp kernelOp = callsites[callsiteIdx].first;  // non-const copy
            kernelOp.setKernelFunctionAttr(newRef);
        }
        totalRedirected += leaf.callsites().size();

        log.trace("partition {0}: {1} callsite(s) -> @{2} ({3} arg type(s) specialized, {4} scalar(s) specialized, "
                  "{5} assume candidate(s))",
                  leafIdx, leaf.callsites().size(), clone.getSymName(), specializedTypes.size(), uniformScalars.size(),
                  assumeCandidates.size());
    }

    // Erase the original callee once every reference to it is gone. We can only
    // do this if we've not skipped some callsites. This relies on a SwKernelOp's
    // kernelFunction being the only thing that ever names a kernel in @VPU.SW.
    if (totalRedirected == callsites.size() && !hasSkippedOps) {
        symbolTable.erase(callee);
    }
}

// Emit an llvm.intr.assume at the clone entry for each bounded quantity in
// `assumeCandidates` that specialization has not already resolved:
//   %v     = memref.dim %arg, physPos   (Dim)  |  %v = %arg  (IterScalar)
//   %eq0   = arith.cmpi eq, %v, <val0>
//   %eq1   = arith.cmpi eq, %v, <val1>
//   ...
//   %any   = arith.ori %eq0, %eq1   (chained)
//   llvm.intr.assume %any
// The possible callsite values need to all be static.
static void insertShapeAssumptions(mlir::func::FuncOp clone,
                                   llvm::ArrayRef<std::pair<ParamKey, llvm::SmallVector<int64_t>>> assumeCandidates) {
    if (assumeCandidates.empty()) {
        return;
    }
    mlir::Block& entryBlock = clone.getBody().front();
    mlir::OpBuilder builder(&entryBlock, entryBlock.begin());
    const mlir::Location loc = clone.getLoc();

    for (const auto& [param, values] : assumeCandidates) {
        mlir::BlockArgument arg = clone.getArgument(param.argIdx);
        if (arg.use_empty()) {
            // There is no point in inserting assumptions for dead arguments.
            continue;
        }

        mlir::Value observed;
        if (param.kind == ParamKey::Kind::IterScalar) {
            observed = arg;
        } else {
            if (!mlir::cast<mlir::MemRefType>(arg.getType()).isDynamicDim(param.physPos)) {
                continue;
            }
            observed = mlir::memref::DimOp::create(builder, loc, arg, static_cast<int64_t>(param.physPos)).getResult();
        }

        mlir::Value condition;
        for (int64_t val : values) {
            auto constOp = mlir::arith::ConstantIndexOp::create(builder, loc, val);
            auto cmpOp = mlir::arith::CmpIOp::create(builder, loc, mlir::arith::CmpIPredicate::eq, observed,
                                                     constOp.getResult());
            condition = condition ? mlir::arith::OrIOp::create(builder, loc, condition, cmpOp.getResult()).getResult()
                                  : cmpOp.getResult();
        }
        mlir::LLVM::AssumeOp::create(builder, loc, condition);
    }
}

// Propagate more-static type information into a memref.expand_shape.
//
// expand_shape "may not cast dynamicity" (see the op description): each source dim
// maps to one reassociation group of result dims whose sizes multiply back to it, and
// is dynamic iff its group holds a dynamic result dim. Narrowing the source breaks
// that agreement, and each group is adjusted one of two ways -- make its unknown
// result dims static when the source size determines them uniquely (exactly one
// unknown: srcSize / product(known siblings)), else relax the source dim back to
// dynamic. Both cases, given a memref<12x20xf32> cast to memref<?x?xf32>:
//
//     memref.expand_shape %s [[0, 1], [2, 3]] output_shape [3, %a, %b, %c]
//         : memref<?x?xf32> into memref<3x?x?x?xf32>
//  ->
//     %c = memref.cast %static : memref<12x20xf32> to memref<12x?xf32>
//     memref.expand_shape %c [[0, 1], [2, 3]] output_shape [3, 4, %b, %c]
//         : memref<12x?xf32> into memref<3x4x?x?xf32>        (+ cast back to the
//                                                             original result type)
//
//   [0, 1] = [3, ?] resolves to 12 / 3 = 4; [2, 3] = [?, ?] does not (20 = ? * ? has
//   no unique answer), so source dim 1 goes back to ?. That relaxation is what lets
//   dims 0 and 1 become static at all -- a static 20 under [?, ?] does not verify, so
//   the only alternative is to leave the op alone.
struct ExpandShapeOpCastFolder final : mlir::OpRewritePattern<mlir::memref::ExpandShapeOp> {
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::memref::ExpandShapeOp expandOp,
                                        mlir::PatternRewriter& rewriter) const override {
        const auto reassoc = expandOp.getReassociationIndices();
        const mlir::Value origSrc = expandOp.getSrc();

        // Peek through a foldable memref.cast to the most static source: only casts whose
        // source is more static are admitted, so folding one can only remove
        // dynamism. The source may still be partly dynamic.
        mlir::Value srcVal = origSrc;
        if (auto castOp = origSrc.getDefiningOp<mlir::memref::CastOp>()) {
            if (mlir::memref::CastOp::canFoldIntoConsumerOp(castOp)) {
                srcVal = castOp.getSource();
            }
        }
        const auto srcType = mlir::cast<mlir::MemRefType>(srcVal.getType());

        llvm::SmallVector<mlir::OpFoldResult> mixedOut = expandOp.getMixedOutputShape();

        // Propagate the static source sizes into the output dims, one group at a time.
        // Groups still holding an unknown output dim afterwards are relaxed below instead.
        for (size_t groupIdx = 0; groupIdx < reassoc.size(); ++groupIdx) {
            // Dynamic source dim: a dynamic result dim in this group is already legal.
            if (mlir::ShapedType::isDynamic(srcType.getShape()[groupIdx])) {
                continue;
            }
            // Resolvable only with exactly one unknown.
            const auto isUnknown = [&](int64_t idx) {
                return !mlir::getConstantIntValue(mixedOut[idx]).has_value();
            };
            if (llvm::count_if(reassoc[groupIdx], isUnknown) != 1) {
                continue;
            }
            const int64_t unresolvedPos = *llvm::find_if(reassoc[groupIdx], isUnknown);
            int64_t knownProduct = 1;
            for (int64_t idx : reassoc[groupIdx]) {
                if (auto constValue = mlir::getConstantIntValue(mixedOut[idx])) {
                    knownProduct *= *constValue;
                }
            }
            const int64_t srcSize = srcType.getShape()[groupIdx];
            if (knownProduct == 0) {
                // 0 == 0 * x holds for every x, so an empty source dim leaves the unknown
                // unresolvable. The relaxation below widens the source dim back to ?. A
                // non-empty source dim under a zero product has no solution.
                VPUX_THROW_WHEN(srcSize != 0,
                                "SpecializeSwKernels: inconsistent memref.expand_shape at "
                                "'{0}' -- source dim {1} (size {2}) is non-empty but its expanded "
                                "dims have a zero product; the callsite's static shapes breaks the "
                                "reshape size contract",
                                expandOp.getLoc(), groupIdx, srcSize);
                continue;
            }
            // A non-divisible product is latent UB in the input program.
            VPUX_THROW_WHEN(srcSize % knownProduct != 0,
                            "SpecializeSwKernels: inconsistent memref.expand_shape at "
                            "'{0}' -- source dim {1} (size {2}) is not divisible by the "
                            "known product {3} of its expanded dims; the callsite's "
                            "static shapes breaks the reshape size contract",
                            expandOp.getLoc(), groupIdx, srcSize, knownProduct);
            mixedOut[unresolvedPos] =
                    mlir::IntegerAttr::get(mlir::IndexType::get(expandOp->getContext()), srcSize / knownProduct);
        }

        // Fold arith.constant operands into
        // IntegerAttrs so the decomposition below classifies them as static.
        mlir::MLIRContext* ctx = expandOp->getContext();
        for (auto& outputSize : mixedOut) {
            if (auto constValue = mlir::getConstantIntValue(outputSize)) {
                outputSize = mlir::IntegerAttr::get(mlir::IndexType::get(ctx), *constValue);
            }
        }

        llvm::SmallVector<int64_t> newStatic;
        llvm::SmallVector<mlir::Value> newDynamic;
        std::tie(newStatic, newDynamic) = mlir::decomposeMixedValues(mixedOut);

        // Bring each source dim in line with what its group ended up needing: a source dim
        // has to be dynamic exactly when its group still holds an unknown, and static once
        // the whole group is known.
        llvm::SmallVector<int64_t> reqSrcShape(srcType.getShape().begin(), srcType.getShape().end());
        for (size_t groupIdx = 0; groupIdx < reassoc.size(); ++groupIdx) {
            const bool anyResultDyn = llvm::any_of(reassoc[groupIdx], [&](int64_t idx) {
                return mlir::ShapedType::isDynamic(newStatic[idx]);
            });
            if (anyResultDyn) {
                reqSrcShape[groupIdx] = mlir::ShapedType::kDynamic;
                continue;
            }
            int64_t resolvedProduct = 1;
            for (int64_t idx : reassoc[groupIdx]) {
                resolvedProduct *= newStatic[idx];
            }
            // A static source dim that disagrees with its own group is latent UB in the
            // input program, same as the non-divisible case above.
            VPUX_THROW_WHEN(
                    !mlir::ShapedType::isDynamic(reqSrcShape[groupIdx]) && reqSrcShape[groupIdx] != resolvedProduct,
                    "SpecializeSwKernels: inconsistent memref.expand_shape at '{0}' -- source dim {1} "
                    "(size {2}) does not match the product {3} of its expanded dims; the callsite's "
                    "static shapes breaks the reshape size contract",
                    expandOp.getLoc(), groupIdx, reqSrcShape[groupIdx], resolvedProduct);
            reqSrcShape[groupIdx] = resolvedProduct;
        }
        const bool needSrcCast = llvm::ArrayRef<int64_t>(reqSrcShape) != srcType.getShape();
        const mlir::MemRefType effectiveSrcType =
                needSrcCast ? mlir::MemRefType::get(reqSrcShape, srcType.getElementType(), srcType.getLayout(),
                                                    srcType.getMemorySpace())
                            : srcType;

        const auto newType = mlir::memref::ExpandShapeOp::computeExpandedType(effectiveSrcType, newStatic, reassoc);
        if (mlir::failed(newType)) {
            return mlir::failure();
        }

        // Bail out if we were already in the most-static form.
        const auto curSrcType = mlir::cast<mlir::MemRefType>(origSrc.getType());
        if (*newType == expandOp.getType() && effectiveSrcType == curSrcType &&
            llvm::ArrayRef<int64_t>(newStatic) == expandOp.getStaticOutputShape()) {
            return mlir::failure();
        }

        mlir::Value newSrc = srcVal;
        if (needSrcCast) {
            newSrc = mlir::memref::CastOp::create(rewriter, expandOp.getLoc(), effectiveSrcType, srcVal).getResult();
        }

        const llvm::SmallVector<mlir::OpFoldResult> newMixedOut = mlir::getMixedValues(newStatic, newDynamic, rewriter);
        auto newExpand = mlir::memref::ExpandShapeOp::create(rewriter, expandOp.getLoc(), *newType, newSrc, reassoc,
                                                             newMixedOut);
        if (*newType == expandOp.getType()) {
            rewriter.replaceOp(expandOp, newExpand.getResult());
        } else {
            rewriter.replaceOpWithNewOp<mlir::memref::CastOp>(expandOp, expandOp.getType(), newExpand);
        }
        return mlir::success();
    }
};

mlir::FrozenRewritePatternSet SpecializeSwKernelsPass::buildTypePropagationPatterns(mlir::MLIRContext* ctx) {
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<ExpandShapeOpCastFolder>(ctx);

    const std::array<mlir::Dialect*, 6> bodyDialects = {
            ctx->getLoadedDialect<mlir::math::MathDialect>(),     ctx->getLoadedDialect<mlir::arith::ArithDialect>(),
            ctx->getLoadedDialect<mlir::affine::AffineDialect>(), ctx->getLoadedDialect<mlir::scf::SCFDialect>(),
            ctx->getLoadedDialect<mlir::linalg::LinalgDialect>(), ctx->getLoadedDialect<mlir::memref::MemRefDialect>()};

    for (mlir::Dialect* dialect : bodyDialects) {
        if (dialect != nullptr) {
            dialect->getCanonicalizationPatterns(patterns);
        }
    }

    for (mlir::RegisteredOperationName op : ctx->getRegisteredOperations()) {
        if (llvm::is_contained(bodyDialects, &op.getDialect())) {
            op.getCanonicalizationPatterns(patterns, ctx);
        }
    }

    return mlir::FrozenRewritePatternSet(std::move(patterns));
}

void SpecializeSwKernelsPass::propagateTypesAndCanonicalize(mlir::func::FuncOp funcOp,
                                                            const mlir::FrozenRewritePatternSet& propagationPatterns,
                                                            Logger log) {
    if (mlir::failed(mlir::applyPatternsGreedily(funcOp, propagationPatterns))) {
        // Log non-convergence, however this does not affect correctness.
        log.trace("SpecializeSwKernels: type propagation did not converge for function @{0}", funcOp.getSymName());
    }
}

}  // namespace

//
// createSpecializeSwKernelsPass
//

std::unique_ptr<mlir::Pass> vpux::ShaveCodeGen::createSpecializeSwKernelsPass(Logger log) {
    return std::make_unique<SpecializeSwKernelsPass>(log);
}
