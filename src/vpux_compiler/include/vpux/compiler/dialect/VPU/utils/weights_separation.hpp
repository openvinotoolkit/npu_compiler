//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace vpux::VPU {

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
    Const::DeclareOp _declareOp;
    // These ArrayRefs are valid as long as the underlying attribute exists, which is until the context is destroyed.
    ArrayRef<Const::TransformAttrInterface> _inInitTransformations;
    ArrayRef<Const::TransformAttrInterface> _postInitTransformations;
    IoBoundaryAdapter::TypeInfo _ioTypeInfo;

    //! @brief Returns the base type of the constant.
    NDTypeInterface getBaseType() const;
    //! @brief Returns the "boundary" type: result of init / input of main.
    NDTypeInterface getBoundaryType() const;

public:
    TransformationsSplit(Const::DeclareOp declareOp);

    /** @brief A schedule-independent "slice" of the transformation split.

        Defines the general set of inputs necessary to convert constant
        transformations to IR form.
    */
    struct Projection {
        Const::DeclareOp declareOp;  // current constant operation
        NDTypeInterface argType;     // argument type to be used in function signature
        ArrayRef<Const::TransformAttrInterface> precedingTransformations;  // transformations that are already applied
        ArrayRef<Const::TransformAttrInterface> transformations;           // transformations to convert to IR
        IoBoundaryAdapter::TypeInfo ioTypeInfo;  // I/O metadata associated with this projection
    };

    //! @brief Returns a "projection" of the split according to the schedule.
    Projection take(WeightsSeparationSchedule schedule) const;

    Const::DeclareOp declareOp() const {
        return _declareOp;
    }
};

//! @brief Collects all constant operations that are worth moving to the Init
//! schedule, and returns them as transformation splits.
SmallVector<TransformationsSplit> collectMoveWorthyTransformationSplits(mlir::func::FuncOp mainFunc);

//! @brief Represents a weights-separation specific argument cache entry.
struct ConstArg {
    mlir::DenseResourceElementsAttr content;
    ArrayRef<Const::TransformAttrInterface> transformations;

    ConstArg(mlir::DenseResourceElementsAttr c, ArrayRef<Const::TransformAttrInterface> ts)
            : content(c), transformations(ts) {
        assert(content != nullptr && "Only dense_resource<> base constant become ConstArgs");
    }

    ConstArg(const TransformationsSplit::Projection& proj)
            : ConstArg(
                      mlir::dyn_cast<mlir::DenseResourceElementsAttr>(proj.declareOp.getContentAttr().getBaseContent()),
                      proj.precedingTransformations) {
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
    std::vector<std::tuple<mlir::Value, Const::ContentAttr>> _convertedConsts;

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

    ArrayRef<std::tuple<mlir::Value, Const::ContentAttr>> getConvertedConsts() const {
        return _convertedConsts;
    }

    //! @brief Returns a result of the chain of IR operations, created from the
    //! corresponding const transformations.
    mlir::Value convertToIrForm(mlir::Location baseLoc, const VPU::TransformationsSplit::Projection& split,
                                mlir::BlockArgument arg, const IoBoundaryAdapter& ioAdaptor,
                                WeightsSeparationSchedule scheduleKind);
};

}  // namespace vpux::VPU

namespace llvm {
template <>
struct DenseMapInfo<vpux::VPU::ConstArg> {
    static vpux::VPU::ConstArg getEmptyKey();
    static vpux::VPU::ConstArg getTombstoneKey();
    static unsigned getHashValue(const vpux::VPU::ConstArg& x);
    static bool isEqual(const vpux::VPU::ConstArg& x, const vpux::VPU::ConstArg& y);
};
}  // namespace llvm
