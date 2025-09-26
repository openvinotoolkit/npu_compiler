//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/abstract_tree.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/AnalysisManager.h"

namespace vpux::VPU {

/** @brief Internal prefix used for Init output

    This prefix is part of the contract between the compiler and the plugin in the context of weights separation.
    Be careful when changing it. "tw" stands for "transformed weights"
*/
constexpr const char* INIT_OUTPUT_PREFIX = "vpux_tw_";

using MemPermuteConversionAttributes =
        std::tuple<mlir::AffineMap /*identityLayout*/, MemShape /*inMemShape*/, mlir::AffineMap /*memPermute*/,
                   mlir::AffineMap /*dstOrder*/, ShapeRef /*outShape*/>;

/** @brief Returns the required permutation and shape for MemPermute conversion
    Used to help when recreating MemPermute transformation by using ShapeCast,
    LayoutCast and Transpose.
*/
MemPermuteConversionAttributes extractMemPermuteConversionAttributes(NDTypeInterface input,
                                                                     Const::MemPermuteAttr memPermuteAttr);

/** @brief Ensures function boundaries satisfy I/O requirements.

    Certain types (e.g. quantized) are not supported as network inputs /
    outputs. Thus, one has to adapt the values at the boundaries to ensure valid
    IR is produced. This utility provides generic dispatch to handle boundary
    adaptation.
*/
struct IoBoundaryAdapter {
    /** @brief Stores quantization-related type information necessary to perform
               boundary adaptation.

        @note Unfortunately, this type information could not be encapsulated inside
              the adaptation function as it varies from value to value. While
              restoring storage type from quantized type is possible, the inverse is
              impossible as zero-points and scales are lost.
     */
    struct TypeInfo {
        mlir::Type quantizedType = nullptr;
        mlir::Type storageType = nullptr;

        /** @brief Returns whether this type info is valid.

            Invalid info means that there's no information regarding the
            quantization parameters and thus nothing can be done.
        */
        bool valid() const {
            return quantizedType != nullptr && storageType != nullptr;
        }
    };

    //! @brief Forwards the specified value without modifying it.
    static mlir::Value identity(mlir::OpBuilder&, mlir::Location, mlir::Value value, const TypeInfo&) {
        return value;
    }

    using ConvertFunc = mlir::Value (*)(mlir::OpBuilder&, mlir::Location, mlir::Value, const TypeInfo&);

    //! @brief Wraps function input as specified.
    ConvertFunc wrapInput = nullptr;
    //! @brief Wraps function result as specified.
    ConvertFunc wrapOutput = nullptr;
};

/** @brief Defines the schedule kind.
 */
enum class WeightsSeparationSchedule { Init, Main };

using CallChainData = std::pair<mlir::func::CallOp, mlir::func::FuncOp>;
using CallChainTree = utils::AbstractTree<CallChainData>;

/** @brief Returns a "call chain" tree constructed from the starting function.

    Returns a weights-separation-specific tree that represents the outlining
    structure. An example of such a tree is:
    ```
    |- {nullptr, main}
       |- {"call foo1", foo1}
          |- {"call foo2", foo2}
       |- {"call foo3", foo3}
    ```
    where "call fooX" is a CallOp operation inside the respective function and
    fooX is a standalone function produced by the outlining.

    @note This tree is the basic data structure used by weights separation to
    construct init and main schedules.
*/
CallChainTree getOutliningRepresentation(mlir::func::FuncOp startFunc);

/** @brief Splits constant transformations into Init and Main schedule parts.

    The list of transformations of a particular DeclareOp can be split into two
    parts: One Part inside the init function and the other part inside the main
    function.

    For example, it is undesirable to perform a SubView operation as the last
    transformation inside of init. Taking two SubViews of the same input value
    produces 2 output values in init. This unnecessarily increases the required
    IO bandwidth between main and init. It would be better to delay the SubView
    by performing it as part of main. Then we only have to transfer a single
    output value from init to main.
*/
class TransformationsSplit {
    mlir::Location _loc;
    Const::ContentAttr _contentAttr;
    // Note: these ArrayRefs are valid as long as the underlying attribute
    // exists, which is until the context is destroyed.
    ArrayRef<Const::TransformAttrInterface> _inInitTransformations;
    ArrayRef<Const::TransformAttrInterface> _postInitTransformations;
    IoBoundaryAdapter::TypeInfo _ioTypeInfo;

    //! @brief Returns the base type of the constant.
    NDTypeInterface getBaseType() const;
    //! @brief Returns the "boundary" type: result of init / input of main.
    NDTypeInterface getBoundaryType() const;

public:
    TransformationsSplit(Const::DeclareOp declareOp);

    //! @brief Returns location of the associated constant.
    mlir::Location getLoc() const;
    //! @brief Returns content attribute associated with this split.
    Const::ContentAttr getContentAttr() const;

    /** @brief A schedule-independent "slice" of the transformation split.

        Defines the general set of inputs necessary to convert constant
        transformations to IR form.
    */
    struct Projection {
        Const::ContentAttr contentAttr;  // current content attribute
        NDTypeInterface argType;         // argument type to be used in function signature
        ArrayRef<Const::TransformAttrInterface> precedingTransformations;  // transformations that are already applied
        ArrayRef<Const::TransformAttrInterface> transformations;           // transformations to convert to IR
        IoBoundaryAdapter::TypeInfo ioTypeInfo;  // I/O metadata associated with this projection
    };

    //! @brief Returns a "projection" of the split according to the schedule.
    Projection take(WeightsSeparationSchedule schedule) const;
};

//! @brief A stable operator<(). Stability is achieved by relying on a
//! combination of constant's name and stable transformation hashes.
bool operator<(const TransformationsSplit& x, const TransformationsSplit& y);

namespace detail {
// Semi-private utility used in implementation and in tests.
vpux::Byte getResultBufferSizeForInit(const TransformationsSplit& x);
}  // namespace detail

//! @brief Specifies whether a given constant is "trivial" within the scope of
//! weights separation (e.g. has only view-like transformations).
//!
//! @note Used by isSuitableForWeightsSeparation().
bool isTrivialForWeightsSeparation(Const::DeclareOp constOp);

//! @brief Specifies whether a given constant is "move-worthy" within the scope
//! of weights separation.
//!
//! @note Used internally in collectMoveWorthyConstants().
bool isSuitableForWeightsSeparation(Const::DeclareOp constOp);

//! @brief Collects all constant operations that are worth moving to the Init
//! schedule.
std::vector<Const::DeclareOp> collectMoveWorthyConstants(const Logger& log, mlir::func::FuncOp mainFunc);

/** @brief Slices the collected transformations splits according to the
    threshold.

    Applies a slicing algorithm to the specified constant operations
    (represented as transformations splits), using memory limit as a hint. The
    memory limit is a rough estimate of how much combined memory (input buffers
    size + output buffers size) each slice should use. The produced slices could
    be used independently of one another.

    The algorithm ensures that no base content is shared across different
    slices. That is, even if a given slice exceeds the memory limit, but that
    slice only contains transformations for a single base content, the slice
    would remain intact (i.e. not split into more slices). This is done
    intentionally to maintain the following property: when transformations in
    the slice are applied to a constant, the constant (by def. of this
    algorithm) is guaranteed to no longer be needed (by any other slice).

    @note The algorithm requires transformation splits to be sorted according to
    the operator<(const TransformationsSplit&, const TransformationsSplit&).
*/
std::vector<std::vector<TransformationsSplit>> sliceAccordingToMemoryLimit(const Logger& log,
                                                                           ArrayRef<TransformationsSplit> splits,
                                                                           vpux::Byte memoryLimit);

//! @brief Represents a weights-separation specific argument cache entry.
struct ConstArg {
    mlir::DenseResourceElementsAttr content;
    ArrayRef<Const::TransformAttrInterface> transformations;

    ConstArg(mlir::DenseResourceElementsAttr c, ArrayRef<Const::TransformAttrInterface> ts)
            : content(c), transformations(ts) {
        assert(content != nullptr && "Only dense_resource<> base constant become ConstArgs");
    }

    ConstArg(const TransformationsSplit::Projection& proj)
            : ConstArg(mlir::dyn_cast<mlir::DenseResourceElementsAttr>(proj.contentAttr.getBaseContent()),
                       proj.precedingTransformations) {
    }

    //! @brief Returns a unique name derived from the ConstArg's components.
    std::string getUniqueName() const;

    // Note: equality comparison is rather slow due to the need to compare
    // transformation arrays.
    friend bool operator==(const ConstArg& x, const ConstArg& y) {
        return x.content == y.content && x.transformations == y.transformations;
    }
    friend bool operator!=(const ConstArg& x, const ConstArg& y) {
        return !(x == y);
    }
};

/** @brief Converts a particular sequence of transformations to IR operations.

    @note The class is capable of generating both Init and Main schedule
    operations generically.
 */
class ConstOpConverter {
    mlir::func::FuncOp _func;
    OpBuilderLogger _builderLogger;
    mlir::OpBuilder _opBuilder;
    Logger _log;
    // Note: operation cache is pimpl-ed to not expose it as a public class.
    class OperationCache;
    std::unique_ptr<OperationCache> _operationCache;

    // IR building function with internal caching semantics.
    std::tuple<ArrayRef<Const::TransformAttrInterface>, mlir::Value> createMatchingOperation(
            mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input,
            ArrayRef<Const::TransformAttrInterface> transformations, WeightsSeparationSchedule scheduleKind);

public:
    ConstOpConverter(mlir::func::FuncOp func, const Logger& log);
    ~ConstOpConverter();
    ConstOpConverter(const ConstOpConverter&) = delete;
    ConstOpConverter(ConstOpConverter&&) = delete;
    ConstOpConverter& operator=(const ConstOpConverter&) = delete;
    ConstOpConverter& operator=(ConstOpConverter&&) = delete;

    mlir::func::FuncOp getFunction() const {
        return _func;
    }

    //! @brief Returns a result of the chain of IR operations, created from the
    //! corresponding const transformations.
    mlir::Value convertToIrForm(mlir::Location baseLoc, const VPU::TransformationsSplit::Projection& split,
                                mlir::BlockArgument arg, const IoBoundaryAdapter& ioAdaptor,
                                WeightsSeparationSchedule scheduleKind);
};

/** @brief A callable that tells whether a particular FuncOp was already
           visited.
*/
class FuncOpVisitor {
    mlir::DenseSet<mlir::func::FuncOp> _cache;

public:
    // Returns whether the function was already seen.
    bool operator()(mlir::func::FuncOp op) {
        const bool firstOccurrence = _cache.insert(op).second;
        return !firstOccurrence;
    }
};

/** @brief An analysis object that holds meta information.
 */
struct WeightsSeparationInfo {
    //! @brief Creates the analysis object.
    WeightsSeparationInfo(mlir::ModuleOp moduleOp);

    /** @brief Returns whether the analysis object must be invalidated.

        This is an analysis "interface" function that tells the analysis
        management system whether the object has to be invalidated (destroyed).
        The function is discovered statically via type traits and used
        implicitly by mlir::detail::AnalysisConcept::invalidate().

        The function returns `false` by default signifying that the analysis
        must be always preserved.
    */
    bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses&);

    /** @brief Marks current analysis object as invalidated.

        Changes the internal state of the analysis so that
        WeightsSeparationInfo::isInvalidated() returns `true`.
     */
    void invalidate();

    /** @brief Returns transformation splits collected from IR.

        Returns cached transformation splits constructed from weights found in
        original IR during analysis.
     */
    const std::vector<TransformationsSplit>& getCollectedSplits() const;

private:
    std::vector<TransformationsSplit> _splits;

    bool _preserved{true};  // participates in invalidation mechanism
};

using CreateSliceOpFunc = FuncRef<mlir::Operation*(mlir::OpBuilder& builder, mlir::Location l, mlir::Value input,
                                                   ArrayRef<int64_t> offsets, ArrayRef<int64_t> sizes)>;
/** @brief Converts specific inputs of function into an obfuscated "blob".

    Replaces given function inputs - specified by indices - with a "blob" value
    (tensor<Nxi8>). The "blob" is sliced within the function and the individual
    values are type-restored to maintain valid IR.

    Relies on Core::ReinterpretCast operation for type de-obfuscation.
 */
void obfuscateInputs(const Logger& log, mlir::Location loc, mlir::func::FuncOp funcOp, ArrayRef<size_t> indices,
                     CreateSliceOpFunc createSlice);

using CreateConcatOpFunc = FuncRef<mlir::Operation*(mlir::OpBuilder& builder, mlir::Location l,
                                                    ArrayRef<mlir::Value> inputs, int64_t axis)>;
/** @brief Converts specific outputs of function into an obfuscated "blob".

    Replaces given function outputs - specified by indices - with a "blob" value
    (tensor<Nxi8>). The "blob" is constructed by concatenating type-obfuscated
    outputs to maintain valid IR.

    Relies on Core::ReinterpretCast operation for type obfuscation.
 */
void obfuscateOutputs(const Logger& log, mlir::Location loc, mlir::func::FuncOp funcOp, ArrayRef<size_t> indices,
                      CreateConcatOpFunc createConcat);

}  // namespace vpux::VPU

namespace llvm {
template <>
struct DenseMapInfo<vpux::VPU::ConstArg> {
    static vpux::VPU::ConstArg getEmptyKey();
    static vpux::VPU::ConstArg getTombstoneKey();
    static unsigned getHashValue(const vpux::VPU::ConstArg& x);
    static bool isEqual(const vpux::VPU::ConstArg& x, const vpux::VPU::ConstArg& y);
};

template <>
struct format_provider<vpux::VPU::ConstArg> {
    static void format(const vpux::VPU::ConstArg& arg, llvm::raw_ostream& stream, StringRef style) {
        stream << "WsConstArg<";
        llvm::support::detail::build_format_adapter(arg.content).format(stream, style);
        stream << ", ";
        llvm::support::detail::build_format_adapter(arg.transformations).format(stream, style);
        stream << ">";
    }
};
}  // namespace llvm
