//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/batch.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"

#include <unordered_set>

namespace vpux::IE {
#define GEN_PASS_DECL_DEBATCHER
#define GEN_PASS_DEF_DEBATCHER
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

namespace detail {
struct DowncastedTypeDescription {
    mlir::Type downcastedType;
    bool isDowncasted;
    Shape originalShape;
};

DowncastedTypeDescription getDowncastedTypeIfApplicable(mlir::Value operand,
                                                        const DebatchCoeffDescription& downcastOp) {
    auto type = mlir::cast<vpux::NDTypeInterface>(operand.getType());
    auto originShape = type.getShape();
    auto batchDowncastedShape = downcastOp.apply(originShape);
    if (originShape == batchDowncastedShape) {
        return DowncastedTypeDescription{type, false, originShape.raw()};
    }
    type = type.changeShape(batchDowncastedShape);

    // If we had encountered a dynamic batch before we downcasted it,
    // it means that old dynamic properties as bounds and masks are still attached to it,
    if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(type); boundedType != nullptr) {
        auto shapeBounds = Shape{boundedType.getBounds()};
        if (!shapeBounds.empty()) {
            shapeBounds = downcastOp.apply(shapeBounds);
            auto staticBatchBounds = Bounds{shapeBounds.begin(), shapeBounds.end()};
            type = boundedType.changeBounds(staticBatchBounds);
        }
    }
    if (auto dynamicType = mlir::dyn_cast<Core::DynamicDimsMaskTensorType>(type); dynamicType != nullptr) {
        auto dynamicMask = Shape{dynamicType.getDynamicDimsMask()};
        if (!dynamicMask.empty()) {
            dynamicMask = downcastOp.apply(dynamicMask);
            auto staticBatchMask = DynamicDimsMask{dynamicMask.begin(), dynamicMask.end()};
            type = dynamicType.changeDynamicDimsMask(staticBatchMask);
        }
    }
    if (type.getShape().isStatic()) {
        type = vpux::getTensorType(type.getShape(), type.getElementType(), type.getDimsOrder(), type.getMemSpace());
    }
    return DowncastedTypeDescription{type, true, originShape.raw()};
}

std::list<mlir::Value> getOperandsToDebatch(mlir::Operation* op,
                                            std::unordered_set<mlir::Operation*>& activationOperationCache) {
    if (op == nullptr || mlir::isa<vpux::Const::DeclareOp>(op)) {
        return {};
    }

    std::list<mlir::Value> ret;
    auto operands = op->getOperands();
    for (auto o : operands) {
        auto parentOp = o.getDefiningOp();
        if (parentOp == nullptr) {
            // doesn't ascend to function arguments or it's a first operation in list
            ret.push_back(o);
            continue;
        }

        if (activationOperationCache.find(parentOp) == activationOperationCache.end() &&
            getOperandsToDebatch(parentOp, activationOperationCache).empty()) {
            // the producer is not an activation kind or its grandparents are not either
            continue;
        }

        ret.push_back(o);
    }

    // put the operation in cache if it belongs to an activation kind
    if (!ret.empty()) {
        activationOperationCache.insert(op);
    }
    return ret;
}

template <class Integer>
SmallVector<Integer> consumeArrayAttrAsIntegerArray(mlir::Operation* op, std::string_view attName) {
    VPUX_THROW_UNLESS(op != nullptr, "consumeArrayAttrAsIntegerArray failed on nullptr");
    auto attr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(op->getAttr(attName));
    VPUX_THROW_UNLESS(attr != nullptr, "Unexpected type for \"{0}\", only \"mlir::ArrayAttr\" supported", attName);

    return parseIntArrayAttr<Integer>(attr);
}

struct ConversionDescription {
    Shape from;
    DebatchCoeffDescription coeff;
};

void debatchOpAttribute(mlir::Operation* op, const StringRef attrName, const ConversionDescription& operandDescr,
                        const Logger& log) {
    log.trace("Additional processing required for an operation: {0} as it has a special attribute: \"{1}\"",
              op->getName().getStringRef(), attrName);
    auto currResultshapeValue = Shape(consumeArrayAttrAsIntegerArray<int64_t>(op, attrName));
    log.trace("Attribute \"{0}\" value: {1}, original operand value: {2}, casted coeff {3}", attrName,
              currResultshapeValue, operandDescr.from, operandDescr.coeff.to_string());
    auto expectedResultshapeValue =
            operandDescr.coeff.applyProportionFromShape(operandDescr.from, currResultshapeValue);

    log.trace("Attribute \"{0}\" now has value: {1}", attrName, expectedResultshapeValue);
    auto downcastedAttr = getIntArrayAttr(op->getContext(), expectedResultshapeValue);
    VPUX_THROW_UNLESS(downcastedAttr != nullptr, "Cannot create downcasted attribute \"{0}\"", attrName);
    op->setAttr(attrName, downcastedAttr);
}

bool hasOnlyBatchDynamicAttribute(mlir::Operation* op, const StringRef attrName, const DebatchCoeffDescription& coeff) {
    auto currResultshapeValue = Shape(consumeArrayAttrAsIntegerArray<int64_t>(op, attrName));
    for (size_t i = 0; i < currResultshapeValue.size(); i++) {
        if (i != static_cast<size_t>(coeff.batchPositionIndex.ind()) &&
            currResultshapeValue[Dim{i}] == mlir::ShapedType::kDynamic) {
            return false;
        }
    }
    return true;
}

struct OpConverter {
    virtual ~OpConverter() = default;
    virtual bool isApplicable(mlir::Operation* op) const = 0;
    virtual void apply(mlir::Operation* op, const std::vector<detail::ConversionDescription>& opArguments) const = 0;
    virtual void refineResults(mlir::Operation* op, const std::vector<ConversionDescription>& inOperands,
                               DenseMap<mlir::OpResult, DebatchCoeffDescription>& inOutResults) const = 0;
};

struct DefaultOpConverter : public OpConverter {
    DefaultOpConverter(Logger& log): _log(log) {
    }

    bool isApplicable(mlir::Operation*) const override {
        return true;
    }
    void apply(mlir::Operation*, const std::vector<detail::ConversionDescription>&) const override {
    }
    void refineResults(mlir::Operation* op, const std::vector<ConversionDescription>& inOperands,
                       DenseMap<mlir::OpResult, DebatchCoeffDescription>& inOutResults) const override {
        VPUX_THROW_UNLESS(inOutResults.size() == 0, "DefaultOpConverter expected empty results to override, got: {0}",
                          inOutResults.size());
        auto opResults = op->getResults();
        for (auto result : opResults) {
            auto resultShape = inOperands[0].coeff.applyProportionFromShape(inOperands[0].from, vpux::getShape(result));
            auto correctionCoeff = DebatchCoeffDescription::createFromShapes(vpux::getShape(result), resultShape);
            inOutResults[result] = correctionCoeff;
            _log.debug("Update inOutResults for result: {0}, coeff: {1}", result, correctionCoeff.to_string());
        }
    }

private:
    Logger _log;
};

struct ShapeValueAttrOpConverter : public OpConverter {
    ShapeValueAttrOpConverter(Logger& log): _log(log) {
    }

    static const StringRef getShapeValueAttrName() {
        return "shape_value";
    };

    bool isApplicable(mlir::Operation* op) const override {
        return op->hasAttr(getShapeValueAttrName());
    }

    void apply(mlir::Operation* op, const std::vector<detail::ConversionDescription>& operands) const override {
        debatchOpAttribute(op, getShapeValueAttrName(), operands[0], _log);
    }

    void refineResults(mlir::Operation*, const std::vector<ConversionDescription>&,
                       DenseMap<mlir::OpResult, DebatchCoeffDescription>& results) const override {
        VPUX_THROW_UNLESS(results.size() == 1,
                          "Operation attributed by \"{0}\" is supposed to produce only one result, got: {1}",
                          getShapeValueAttrName(), results.size());
    }

private:
    Logger _log;
};

struct DynamicReshapeAttrOpConverter : public OpConverter {
    DynamicReshapeAttrOpConverter(Logger& log): _log(log) {
    }

    static const StringRef getShapeValueAttrName() {
        return "output_shape";
    };

    bool isApplicable(mlir::Operation* op) const override {
        return op->hasAttr(getShapeValueAttrName());
    }

    void apply(mlir::Operation* op, const std::vector<detail::ConversionDescription>& operands) const override {
        if (!hasOnlyBatchDynamicAttribute(op, getShapeValueAttrName(), operands[0].coeff)) {
            debatchOpAttribute(op, getShapeValueAttrName(), operands[0], _log);
            if (op->hasAttr("output_bounds")) {
                debatchOpAttribute(op, "output_bounds", operands[0], _log);
            }
            return;
        }

        if (IE::DynamicReshapeOp dynamicReshapeOp = mlir::dyn_cast_or_null<IE::DynamicReshapeOp>(op);
            dynamicReshapeOp != nullptr) {
            mlir::OpBuilder builder(dynamicReshapeOp);

            // Having only dynamic N debatched, we will get the operation with completely static shapes.
            // Dynamic operation with static shapes will not pass verification
            // Let's substitute DynamicReshapeOp with ReshapeOp and debatch the latter op
            mlir::Location loc = appendLoc(op->getLoc(), "_dynReshape_substitution");
            auto currResultshapeValue =
                    Shape(consumeArrayAttrAsIntegerArray<int64_t>(dynamicReshapeOp, getShapeValueAttrName()));
            mlir::Operation* reshapeSubstitutionOp =
                    builder.create<IE::ReshapeOp>(loc, dynamicReshapeOp.getInput(), nullptr, false,
                                                  getIntArrayAttr(builder.getContext(), currResultshapeValue));
            ShapeValueAttrOpConverter{const_cast<Logger&>(_log)}.apply(reshapeSubstitutionOp, operands);
            auto origType = mlir::cast<vpux::NDTypeInterface>(dynamicReshapeOp.getResult().getType());
            auto originShape = origType.getShape();

            auto substitutedType = mlir::cast<vpux::NDTypeInterface>(reshapeSubstitutionOp->getResults()[0].getType());
            substitutedType = substitutedType.changeShape(originShape);
            reshapeSubstitutionOp->getResults()[0].setType(substitutedType);

            dynamicReshapeOp.replaceAllUsesWith(reshapeSubstitutionOp);
        }
    }

    void refineResults(mlir::Operation* op, const std::vector<ConversionDescription>&,
                       DenseMap<mlir::OpResult, DebatchCoeffDescription>& results) const override {
        VPUX_THROW_UNLESS(results.size() == 1,
                          "Operation attributed by \"{0}\" is supposed to produce only one result, got: {1}",
                          getShapeValueAttrName(), results.size());
        // Dynamic shape specific operation must be excluded from further debatching manipulations in the graph,
        // only if it has been substituted by an non-dynamic analog.
        // The current chain of graph terminates here, because the result of that operation
        // is not used anymore due to the substitution.
        // Although theoretically we don't have to do anything more here,
        // the MLIR will call verification routine for each operation even though
        // it is not used anymore, which means we must guarantee that the operation
        // remains valid. In particular, this DynamicReshapeOp must has at least one dynamic tensor.
        // As we changed operands already, let's keep the result dynamic:
        // all we need it to return desiredBatchValue as kDynamic to bypass
        // the type transformation in the graph.
        // In this case the existing batch and the desired batch are the same and
        // debatcher algo skips it
        for (auto& [res, coeff] : results) {
            if (res.getUses().empty()) {
                coeff.desiredBatchValue = mlir::ShapedType::kDynamic;
                _log.debug("An operation: {0} result: {1} is set to be excluded from debatching", op->getName(), res);
            }
        }
    }

private:
    Logger _log;
};

struct SizesAttrOpConverter : public OpConverter {
    SizesAttrOpConverter(Logger& log): _log(log) {
    }

    static const StringRef getShapeValueAttrName() {
        return "sizes_attr";
    };

    bool isApplicable(mlir::Operation* op) const override {
        return op->hasAttr(getShapeValueAttrName());
    }

    void apply(mlir::Operation* op, const std::vector<detail::ConversionDescription>& operands) const override {
        _log.trace("Additional processing required for an operation: {0} as it has a special attribute: \"{1}\"",
                   op->getName().getStringRef(), getShapeValueAttrName());
        static constexpr std::string_view axesAttrName("axes_attr");
        VPUX_THROW_UNLESS(op->hasAttr(axesAttrName), "SizesAttrOpConverter requires additional \"{0}\" for processing",
                          axesAttrName);

        std::optional<Dim> batchAxisIndex;
        auto axisValues = consumeArrayAttrAsIntegerArray<int64_t>(op, axesAttrName);
        for (const auto& [dimIndex, axisValue] : axisValues | indexed) {
            if (Dim(axisValue) == operands[0].coeff.batchPositionIndex) {
                batchAxisIndex = Dim(dimIndex);
            }
        }
        if (!batchAxisIndex.has_value()) {
            _log.trace("Attribute \"{0}\" has no N dimenstion, skip conversion step", axesAttrName);
            return;
        }

        // downcast attribute by a batch size in N dimensition
        auto to = operands[0].coeff.apply(operands[0].from);
        auto expectedResultshapeValue = Shape(consumeArrayAttrAsIntegerArray<int64_t>(op, getShapeValueAttrName()));
        _log.trace("Attribute \"{0}\" value: {1}, original operand value: {2}, casted operand value: {3}",
                   getShapeValueAttrName(), expectedResultshapeValue, operands[0].from, to);
        expectedResultshapeValue[batchAxisIndex.value()] /=
                (operands[0].from[operands[0].coeff.batchPositionIndex] / to[operands[0].coeff.batchPositionIndex]);

        auto downcastedAttr = getIntArrayAttr(op->getContext(), expectedResultshapeValue);
        VPUX_THROW_UNLESS(downcastedAttr != nullptr, "Cannot create downcasted attribute \"{0}\"",
                          getShapeValueAttrName());
        op->setAttr(getShapeValueAttrName(), downcastedAttr);
    }

    void refineResults(mlir::Operation*, const std::vector<ConversionDescription>&,
                       DenseMap<mlir::OpResult, DebatchCoeffDescription>& results) const override {
        VPUX_THROW_UNLESS(results.size() == 1,
                          "Operation attributed by \"{0}\" is supposed to produce only one result, got: {1}",
                          getShapeValueAttrName(), results.size());
    }

private:
    Logger _log;
};

struct DimensionLimiterConverter : public OpConverter {
    // This converter is used to limit the expansion of N (batch) dimension of the constant value with respect to the
    // result shape. Currently only supported on BroadcastOp.
    // Details in BroadcastOp implementation below.
    DimensionLimiterConverter(Logger& log): _log(log) {
    }

    bool isApplicable(mlir::Operation* op) const override {
        // Currently only applies to BroadcastOp
        if (mlir::isa<IE::BroadcastOp>(op)) {
            return true;
        } else {
            return false;
        }
    }

    template <typename T>
    void setConstValuesToNDim(vpux::Const::DeclareOp origConstDeclareOp, const mlir::RankedTensorType scaleShape,
                              const vpux::Const::Content& origConstContent,
                              const detail::ConversionDescription operandChange) const {
        // When BroadcastOp is encountered, there are two cases (assume the operands are Const, and BroadcastOp result
        // shape should be (1,)):
        // - Case 1: Operand 0 shape needs to be modified
        //      - Operand 0: shape(3,) value[2, 2, 2]
        //      - Operand 1: shape(1,) value[1]
        //      - Result:    shape(3,) value[2, 2, 2]
        //          - Solution: Operand 0 shape needs to be limited to shape(1,) value[2], so results will be shape(1,)
        // - Case 2: Operand 1 shape needs to be modified
        //      - Operand 0: shape(1,) value[2]
        //      - Operand 1: shape(1,) value[3]
        //      - Result:    shape(3,) value[2, 2, 2]
        //          - Solution: Operand 1 value needs to be change to value[1], so results will be shape(1,)
        //                      (based on OpenVINO broadcasting rules)
        auto origConstVal = origConstContent.getValues<T>();
        SmallVector<T> scaleValue(origConstVal);

        // Checks that the values are consistent across all batches
        // Consistent Tensor (batch size 3)
        //      Input scale value = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        //      Spltting based on span size -> [1, 2, 3], [1, 2, 3], [1, 2, 3]
        //                                  -> All batches are consistent
        // Non consistent Tensor (batch size 3)
        //      Input scale value = [1, 2, 3, 4, 5, 6, 1, 2, 3]
        //      Spltting based on span size -> [1, 2, 3], [4, 5, 6], [1, 2, 3]
        //                                  -> 2nd batch is not consistent

        bool isConsistent = true;
        auto to = operandChange.coeff.apply(operandChange.from);
        auto batchPositionIndex = operandChange.coeff.batchPositionIndex.ind();
        auto batchSize = to[operandChange.coeff.batchPositionIndex];
        size_t spanSize = scaleValue.size() / batchSize;
        for (int64_t batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
            // Compare the span of scaleValue for the current batch with the first batch
            if (batchIndex != batchPositionIndex &&
                memcmp(scaleValue.data() + batchPositionIndex * spanSize, scaleValue.data() + batchIndex * spanSize,
                       spanSize * sizeof(T)) != 0) {
                isConsistent = false;
                break;
            }
        }

        if (isConsistent) {
            // Handling scaleValue size not the same as result (arg.to) total size
            // Truncate scaleValue to match arg.to's total size (likely only affects operand 0 will have different shape
            // as arg.to) Else, change constant scale value to match arg.to's N dimension
            if (scaleValue.size() != static_cast<size_t>(to.totalSize())) {
                scaleValue.resize(to.totalSize());
                _log.trace("[DimLimiter] - BroadcastOp | Truncated scale value to match arg.to's total size: {0}",
                           scaleValue.size());
            } else if (operandChange.from.totalSize() != to.totalSize()) {
                // Change constant scale value to match arg.to's N dimension
                scaleValue[batchPositionIndex] = batchSize;
                _log.trace("[DimLimiter] - BroadcastOp | Changing constant value to match arg.to's N dimension: {0}",
                           scaleValue[batchPositionIndex]);
            }

            // Create the constant
            mlir::OpBuilder builder(origConstDeclareOp);
            auto newScaleConstantOperand =
                    Const::createConst(builder, origConstDeclareOp.getLoc(), scaleShape, ArrayRef(scaleValue));
            _log.trace("[DimLimiter] - BroadcastOp | New Scale Constant Operand: {0}", newScaleConstantOperand);
            origConstDeclareOp.replaceAllUsesWith(newScaleConstantOperand);
        } else {
            VPUX_THROW("Const values are not all the same or are not consistent: Values : {0}", scaleValue);
        }
    }

    void apply(mlir::Operation* op, const std::vector<detail::ConversionDescription>& inOperands) const override {
        VPUX_THROW_UNLESS(op->getResults().size() == inOperands.size(),
                          "Op results size mismatch with inOperands size, expected: {0}, got: {1}",
                          op->getResults().size(), inOperands.size());

        // Currently only BroadcastOp is supported
        if (mlir::isa<IE::BroadcastOp>(op)) {
            broadcastConstModifier(op, inOperands);
        } else {
            VPUX_THROW("Unsupported 'DimensionLimiterConverter' for \"{0}\"", op->getName());
        }
    }

    void broadcastConstModifier(mlir::Operation* op,
                                const std::vector<detail::ConversionDescription>& inOperands) const {
        // Modifies values / limits shape of Const from influencing the result shape of BroadcastOp
        // Mainly encountered when debatching, when inferred result shape did not match the calculated shape, and
        // traversing up the IR.
        auto arg = inOperands[0];
        // Guard clause to ensure we only rewrite constant if BroadcastOp shape has changed through inOperands
        auto to = arg.coeff.apply(arg.from);
        if (arg.from == to) {
            return;
        }
        _log.trace("[DimLimiter] - BroadcastOp | Result change From: {0} To: {1}", arg.from, to);

        // Get a list of Const operands
        std::list<int> constOperandIndices;
        for (unsigned int operandIndex = 0; operandIndex < op->getNumOperands(); ++operandIndex) {
            auto operand = op->getOperand(operandIndex);
            if (auto constDeclareOp = operand.getDefiningOp<Const::DeclareOp>()) {
                // For Operand 0, if the operand shape and the arg.to shape is the same, skip adding to the list
                auto operandShape = getShape(operand);
                if (operandIndex == 0 && operandShape == to) {
                    continue;
                }
                // Save the operand's index if its defining op is a Const::DeclareOp
                constOperandIndices.push_back(operandIndex);
            }
        }
        _log.trace("[DimLimiter] - BroadcastOp | Const Operand Indices: {0}", constOperandIndices);

        for (auto constOperandIndex : constOperandIndices) {
            auto origConstDeclareOp = op->getOperand(constOperandIndex).getDefiningOp<Const::DeclareOp>();
            // mlir::OpBuilder builder(origConstDeclareOp);

            // 1. Get original constant data type
            auto constType =
                    mlir::cast<vpux::NDTypeInterface>(op->getOperand(constOperandIndex).getType()).getElementType();

            // 2. Set to debatched shape, 'arg.to', following data type of the original constant
            const auto scaleShape = mlir::RankedTensorType::get(ArrayRef(to.raw().data(), to.raw().size()),  // Shape
                                                                constType);  // Data type
            _log.trace("[DimLimiter] - BroadcastOp | Const Type: {0}, Scale Shape: {1}", constType, scaleShape);

            // 3. Set the correct data
            auto origConstContent = origConstDeclareOp.getContent();

            if (constType.isF32()) {
                setConstValuesToNDim<float>(origConstDeclareOp, scaleShape, origConstContent, arg);
            } else if (constType.isF64()) {
                setConstValuesToNDim<double>(origConstDeclareOp, scaleShape, origConstContent, arg);
            } else if (constType.isInteger(32)) {
                setConstValuesToNDim<int32_t>(origConstDeclareOp, scaleShape, origConstContent, arg);
            } else {
                VPUX_THROW("[DimLimiter] - BroadcastOp | Unsupported constant data type: {0}", constType);
            }
        }
    }

    void refineResults(mlir::Operation*, const std::vector<ConversionDescription>&,
                       DenseMap<mlir::OpResult, DebatchCoeffDescription>&) const override {
        // Does nothing
        return;
    }

private:
    Logger _log;
};

struct AttributedConstOpConverter : public OpConverter {
    AttributedConstOpConverter(Logger& log): _log(log) {
    }

    bool isApplicable(mlir::Operation* op) const override {
        if (mlir::isa<vpux::Const::DeclareOp>(op)) {
            return mlir::dyn_cast<vpux::Const::DeclareOp>(op);
        }
        return false;
    }

    void apply(mlir::Operation* op, const std::vector<detail::ConversionDescription>&) const override {
        _log.trace("Additional processing required for an operation: {0} as it has a special content attribute",
                   op->getName().getStringRef());
        auto constDeclareOp = mlir::dyn_cast<vpux::Const::DeclareOp>(op);
        VPUX_THROW_UNLESS(constDeclareOp != nullptr, "Expected vpux::Const::DeclareOp, got: {0}",
                          op->getName().getStringRef());
        auto attrType = constDeclareOp.getContentAttr().getType();
        const auto opType = mlir::cast<vpux::NDTypeInterface>(constDeclareOp.getType());

        // For type with swizzling skip the shape check as the content
        // might have been flattened to accomodate swizzled buffer.
        if (!vpux::getSwizzlingSchemeAttr(opType)) {
            if (opType.getShape() != attrType.getShape()) {
                /* We need to debatch Constant here, canonize it (so that all unrealized_cast will be put at the block
                 * beginning) and use it as a function arguments in the similar way as debatched `main` arguments are
                 * passed to a function Need to teach outliner to recognize such types of unrealized_cast's Until that
                 * we will throw an exception here. When done, uncomment the further block of code
                 */

                mlir::OpBuilder builder(constDeclareOp);
                auto operand = op->getResults()[0];
                builder.setInsertionPointAfterValue(operand);
                const auto debatchedArgLoc = appendLoc(operand.getLoc(), "_debatched_const");
                auto unrealized_cast =
                        builder.create<mlir::UnrealizedConversionCastOp>(debatchedArgLoc, opType, operand);
                operand.replaceUsesWithIf(unrealized_cast.getResult(0), [&](mlir::OpOperand& opOperand) {
                    return opOperand.getOwner() != unrealized_cast;
                });
                operand.setType(attrType);

                // As a const DeclareOp is being debatched, we need to move the original
                // DeclareOp on top of an enclosing block to imitate func_args behavior.
                // Respective UnrealizedConversionCastOp must be placed after first existing UnrealizedConversionCastOp
                // in the block of operations. Such separation of operations resembles canonization approach, where we
                // move constants on top. This canonization allows us to outline operations encompassed between the
                // first and last unrealized_cast easily and treat these const DeclareOps in the same way as we treat
                // main-function arguments without additional processing
                auto curBlock = constDeclareOp->getBlock();
                VPUX_THROW_WHEN(curBlock == nullptr, "Operation {0} must be a part of a block", *constDeclareOp);
                VPUX_THROW_WHEN(curBlock->getOperations().empty(), "Operation {0} enclosure block must not be empty",
                                *constDeclareOp);
                constDeclareOp->moveBefore(&(*curBlock->getOperations().begin()));
                for (auto& blockOp : curBlock->getOperations()) {
                    if (mlir::isa<mlir::UnrealizedConversionCastOp>(blockOp)) {
                        unrealized_cast->moveAfter(&blockOp);
                        break;
                    }
                }
            }
        }
    }

    void refineResults(mlir::Operation*, const std::vector<ConversionDescription>&,
                       DenseMap<mlir::OpResult, DebatchCoeffDescription>&) const override {
        // Does nothing
        return;
    }

private:
    Logger _log;
};

struct OpCastVisitor {
    OpCastVisitor(Logger& log): _log(log.nest()) {
        converters.push_back(std::make_unique<DefaultOpConverter>(log));
        converters.push_back(std::make_unique<ShapeValueAttrOpConverter>(log));
        converters.push_back(std::make_unique<DynamicReshapeAttrOpConverter>(log));
        converters.push_back(std::make_unique<SizesAttrOpConverter>(log));
        converters.push_back(std::make_unique<DimensionLimiterConverter>(log));
        converters.push_back(std::make_unique<AttributedConstOpConverter>(log));
    }

    OpCastVisitor(const OpCastVisitor&) = default;
    OpCastVisitor(OpCastVisitor&&) = default;
    OpCastVisitor& operator=(const OpCastVisitor&) = default;
    OpCastVisitor& operator=(OpCastVisitor&&) = default;

    ~OpCastVisitor() noexcept {
        // it appears that std::stingstream operator<<() might throw std::bad_cast due to locale issues. Providing that
        // it's a bad practice to rethrow exception from dtors, we must just catch everything to stop static
        // analysators complaining
        try {
            _log.debug("Statistic: {0}", stat.to_string());
        } catch (...) {
        }
    }

    struct Statistic {
        size_t ops_total_count = 0;
        size_t ops_debatched_count = 0;
        size_t applied_converters_count = 0;
        std::string to_string() const {
            std::stringstream sstream;
            sstream << "debatched ops count: (" << ops_debatched_count << "/" << ops_total_count << ")"
                    << ", total transformations: " << applied_converters_count << std::endl;
            return sstream.str();
        }
    } stat;

    template <class Converter, class... Args>
    void addConverter(Args&&... args) {
        converters.push_back(std::make_unique<Converter>(_log, std::forward<Args>(args)...));
    }

    DenseMap<mlir::OpResult, DebatchCoeffDescription> runConverters(
            mlir::Operation* op, std::vector<detail::ConversionDescription> operationArguments) {
        DenseMap<mlir::OpResult, DebatchCoeffDescription> deductedResultShapes;
        bool conversions_applied = false;
        stat.ops_total_count++;
        for (const auto& c : converters) {
            if (c->isApplicable(op)) {
                c->apply(op, operationArguments);
                c->refineResults(op, operationArguments, deductedResultShapes);
                conversions_applied = true;
                stat.applied_converters_count++;
            }
        }
        if (conversions_applied) {
            stat.ops_debatched_count++;
        }
        _log.debug("deductedResultShapes: {0}", deductedResultShapes.size());
        return deductedResultShapes;
    }

    DenseMap<mlir::OpResult, DebatchCoeffDescription> visit(
            mlir::Operation* op, const std::list<mlir::Value>& operands,
            DenseMap<mlir::Value, detail::ConversionDescription>& operandsConversionDescription) {
        // NOTE: operands = operandsToDebatch
        // NOTE: operandsConversionDescription = debatchedOperandStorage
        VPUX_THROW_UNLESS(op != nullptr, "Empty operation");
        std::vector<detail::ConversionDescription> operationArguments;
        operationArguments.reserve(operands.size());
        for (mlir::Value operand : operands) {
            VPUX_THROW_UNLESS(
                    operandsConversionDescription.contains(operand),
                    "operandsConversionDescription with size: {0} doesn't contain operand {1} conversion info",
                    operandsConversionDescription.size(), operand);
            operationArguments.push_back(operandsConversionDescription.at(operand));
        }

        DenseMap<mlir::OpResult, DebatchCoeffDescription> deductedResultShapes;
        SmallVector<mlir::Type> predictedResultTypes;
        if (mlir::isa<vpux::IE::ReshapeOp, vpux::IE::AffineReshapeOp>(op)) {
            // If operation is AffineReshape, get predicted shape first
            // -- In AffineReshape, result shape is calculated based on shape_value attribute
            // -- Running converters first will alter shape_value attribute
            // -- The inferred shape will not be reflective on the input operand in AffineReshape
            // -- Effectively, this means the inferred result shape is the correct shape, but the
            //    input operand shape is wrong
            predictedResultTypes = getPredictedResult(op);
            deductedResultShapes = this->runConverters(op, std::move(operationArguments));
        } else {
            deductedResultShapes = this->runConverters(op, std::move(operationArguments));
            predictedResultTypes = getPredictedResult(op);
        }

        // Lambda function to check if we can debatch the operand
        // Checking is based on operandsConversionDescription
        auto needDowncast =
                [](const mlir::Value& operand,
                   const llvm::DenseMap<mlir::Value, ConversionDescription>& operandsConversionDescription) -> bool {
            auto it = operandsConversionDescription.find(operand);
            if (it == operandsConversionDescription.end()) {
                return true;
            }
            auto to = it->second.coeff.apply(it->second.from);
            if (it->second.from == to) {
                // Operand is not in the list, or 'from' and 'to' are the same, we can safely debatch it
                return true;
            }
            return false;
        };

        if (!predictedResultTypes.empty()) {
            for (auto resultPair : zip(deductedResultShapes, predictedResultTypes)) {
                auto [opResult, debachCoeff] = std::get<0>(resultPair);
                auto predictedResultType = mlir::dyn_cast<vpux::NDTypeInterface>(std::get<1>(resultPair));
                VPUX_THROW_UNLESS(predictedResultType != nullptr,
                                  "predictedResultType has non vpux::NDTypeInterface type '{0}'",
                                  std::get<1>(resultPair));

                unsigned numOperands = op->getNumOperands();
                unsigned operandIndex = 0;

                auto calculatedShape = debachCoeff.apply(vpux::getShape(opResult));
                // Continuously modify the IR tree until the calculated shape is equal to the predicted shape
                while (predictedResultType.getShape() != calculatedShape && operandIndex < numOperands) {
                    // Initialize as empty (as input on next visit call), to be filled with operands that needs to be
                    // debatched from the defining Operation
                    std::list<mlir::Value> definingOpOperandsToDebatch;
                    auto oneOperand = op->getOperand(operandIndex);
                    _log.trace("Predicted shape: {0}, Calculated shape: {1}, Operand Index: {2}",
                               predictedResultType.getShape(), calculatedShape, operandIndex);

                    if (needDowncast(oneOperand, operandsConversionDescription)) {
                        // If we need to debatch the operand, add it to nonDebatchedOperands
                        auto correctOperandShape = debachCoeff.applyProportionFromShape(predictedResultType.getShape(),
                                                                                        vpux::getShape(oneOperand));
                        auto correctionCoeff = DebatchCoeffDescription::createFromShapes(vpux::getShape(oneOperand),
                                                                                         correctOperandShape);

                        auto downcastedType = getDowncastedTypeIfApplicable(oneOperand, correctionCoeff);
                        _log.trace("Current Shape: {0}, Correct Shape: {1}, Downcasted Type: {2}, coeff: {3}",
                                   downcastedType.originalShape, correctOperandShape, downcastedType.downcastedType,
                                   correctionCoeff.to_string());
                        if (downcastedType.isDowncasted) {
                            // Set the downcast type as new operand shape
                            oneOperand.setType(downcastedType.downcastedType);

                            mlir::Operation* operandDefiningOp = oneOperand.getDefiningOp();
                            if (operandDefiningOp == nullptr) {
                                continue;
                            }
                            definingOpOperandsToDebatch.push_back(oneOperand);

                            // Add the debatched operand into operandsConversionDescription (to be used in next visit
                            // call)
                            operandsConversionDescription[oneOperand] =
                                    detail::ConversionDescription{downcastedType.originalShape, correctionCoeff};

                            // Visit the operandDefiningOp of the operand that we just downcasted
                            this->visit(operandDefiningOp, definingOpOperandsToDebatch, operandsConversionDescription);
                            predictedResultTypes = getPredictedResult(op);
                            predictedResultType = mlir::dyn_cast<vpux::NDTypeInterface>(predictedResultTypes[0]);
                        }
                    }
                    // Go to the next operand in the operation
                    operandIndex++;
                }
            }
        }
        return deductedResultShapes;
    }

private:
    SmallVector<mlir::Type> getPredictedResult(mlir::Operation* op) {
        SmallVector<mlir::Type> predictedResultTypes;
        auto iface = mlir::dyn_cast<mlir::InferTypeOpInterface>(op);
        if (iface) {
            VPUX_THROW_WHEN(
                    iface.inferReturnTypes(op->getContext(), op->getLoc(), op->getOperands(), op->getAttrDictionary(),
                                           op->getPropertiesStorage(), op->getRegions(), predictedResultTypes)
                            .failed(),
                    "Failed to infer return types for operation '{0}'", op->getName());
        }
        return predictedResultTypes;
    }

    std::vector<std::unique_ptr<OpConverter>> converters;
    Logger _log;
};
}  // namespace detail

//
// DebatcherPass
//

class DebatcherPass final : public IE::impl::DebatcherBase<DebatcherPass> {
public:
    explicit DebatcherPass(const DebatcherOptions& options, Logger log) {
        Base::initLogger(log, Base::getArgumentName());
        Base::copyOptionValuesFrom(options);
        _log.debug("Create {0}", getName());
    }

    mlir::LogicalResult initializeOptions(
            StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) final;

private:
    void safeRunOnFunc() final;
};

mlir::LogicalResult DebatcherPass::initializeOptions(
        StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) {
    if (mlir::failed(Base::initializeOptions(options, errorHandler))) {
        return mlir::failure();
    }
    _log.debug("{0}: {1}", debatcherIntputCoeffPartitions.getArgStr(), debatcherIntputCoeffPartitions.getValue());
    _log.trace("initializing of {0} succeeded", getName());
    return mlir::success();
}

//
// safeRunOnModule
//

void DebatcherPass::safeRunOnFunc() {
    _log.trace("{0}::safeRunOnModule", getName());

    auto main = getOperation();
    mlir::ModuleOp module = main->getParentOfType<mlir::ModuleOp>();

    // Initialize an auxiliary activation operations cache.
    // We will gather such operations gradually during our main-function traversing.
    // This approach gives us an opportunity to make decision whether or not a particular operation
    // should be debatched by the following rule: we must not debatch operands of an operation
    // which producers are constant operations or simple declarations (without strong needs),
    // consequently we must debatch only operands which ascend to arguments of the main-function,
    // hence if an operand of the current operation is produced by
    // a parent operation which is an activation operation then the operand must be debatched.
    // Having this cache determined, we leverage an optimization here: every time once an operation
    // is discerned as an activation-operation we will store it in the cache so that
    // we could consider this stored operation as the activation parent operation for
    // next operation from traversing graph list.
    std::unordered_set<mlir::Operation*> activationOperations;
    activationOperations.insert(main);
    // remember first generation relatives as activation operations
    llvm::for_each(main.getArguments(), [&activationOperations](auto blockArg) {
        llvm::for_each(blockArg.getUsers(), [&activationOperations](auto userOp) {
            activationOperations.insert(userOp);
        });
    });

    // Storage for all operands which have been downcasted during graph traversing.
    // This prevents an operand being downcasted second time, as operands of
    // a typical operation are also can be represented by results of a previous operation,
    // debatched on a previous step.
    // An operation is considered "debatched" when the conditions have met:
    //  1. is has all operands debatched
    //  2. it has all its results debatched either.
    DenseMap<mlir::Value, detail::ConversionDescription> debatchedOperandStorage;

    // This is a storage for debatched results of a previously debatched operation,
    // which might as well be appeared as operands of a current operation to debatch.
    // If an operand exists in this storage, but not in `debatchedOperandStorage`,
    // then the operation must be debatched and its result must be remembered
    // in `debatchedOperandStorage` as well, to avoid double downcasting when it
    // being appeared as another operand of another operation
    DenseMap<mlir::Value, DebatchCoeffDescription> opResultsToDebatch;

    // The operation is considered "debatched" only when it has its operands in
    // `debatchedOperandStorage` and all its results in `opResultsToDebatch`
    // During this graph traversing, the algorithm does the following:
    //  1. applies individual per operands downcasting rules, described in DebatchCoefficients,
    // when these operands are not in `debatchedOperandStorage`;
    //  2. deducts what a rule for an operation result will be after downcasting,
    // taking into account those operand downcasting rules in DebatchCoefficients;
    //  3. remembers the result rule as DebatchCoefficient per the result in `opResultsToDebatch`;
    //  4. pumps these rules from `opResultsToDebatch` into `debatchedOperandStorage` when done
    // to avoid them being downcasted further at 1 step
    _log.debug("Use an option value \"{0}\": {1}", debatcherIntputCoeffPartitions.getArgStr(),
               debatcherIntputCoeffPartitions.getValue());
    auto debatchingCoefficients = DebatchCoefficients::create(debatcherIntputCoeffPartitions.getValue());
    if (debatchingCoefficients.has_value()) {
        for (size_t i = 0; i < debatchingCoefficients->size(); ++i) {
            auto coeffValues = debatchingCoefficients ? debatchingCoefficients->getCoefficient(i) : std::nullopt;
            VPUX_THROW_UNLESS(coeffValues->batchPositionIndex == vpux::Dim(0),
                              "DebatchCoeffDescription expects the batch position to be 0, got: {0} in {1}",
                              coeffValues->batchPositionIndex, coeffValues->to_string());
        }
    }
    size_t argIndex = 0;
    for (const auto& arg : main.getArguments()) {
        DebatchCoeffDescription debatchCoeffForArg;
        if (debatchingCoefficients.has_value()) {
            auto coeffCandidate = debatchingCoefficients->getCoefficient(argIndex);
            if (coeffCandidate.has_value()) {
                debatchCoeffForArg = coeffCandidate.value();
            }
        }
        // Put args of main in `debatchedOperandStorage` as if they have already been debatched,
        // which prevents the following algorithm from downcasting all these arguments
        // as it will have done for other operands of a typical function during the graph traversing.
        // We need to employ such an exception for args of main, as these arguments
        // must be downcasted by `unrealized cast` instead of usual type conversion.
        // Thus from the graph traversing perspective these arguments, once they
        // have unrealize-cast'ed, must be put into `opResultsToDebatch` and `debatchedOperandStorage`
        // Let's remember them right now, as we are already gathering coefficients for them
        auto originalShape = Shape{vpux::getShape(arg)};
        debatchedOperandStorage[arg] = detail::ConversionDescription{originalShape, debatchCoeffForArg};
        opResultsToDebatch[arg] = debatchedOperandStorage[arg].coeff;
        _log.trace("Func arg num: {0}, original shape: {1}, desired transformation: {2}", argIndex, originalShape,
                   opResultsToDebatch[arg].to_string());
        argIndex++;
    }

    _log.trace("create builder for inserting an `unrealize_converion_cast` at region boundaries");
    mlir::OpBuilder builder(main);
    builder.setInsertionPointAfter(main);
    auto mainArgs = main.getArguments();
    _log.trace("Enforce cast for `main` arguments count: {0} ", mainArgs.size());
    bool needDebatch = false;
    llvm::for_each(mainArgs, [&needDebatch, &builder, &debatchedOperandStorage, &opResultsToDebatch, this](auto& arg) {
        auto descr = detail::getDowncastedTypeIfApplicable(arg, opResultsToDebatch[arg]);
        if (!descr.isDowncasted) {
            // UnrealizedConversionCastOp doesn't distinguish whether a type is same.
            // skip it explicitly if type hasn't been changed
            return;
        }
        needDebatch = true;
        builder.setInsertionPointAfterValue(arg);
        const auto debatchedArgLoc = appendLoc(arg.getLoc(), "_debatched_arg");
        auto unrealized_cast =
                builder.create<mlir::UnrealizedConversionCastOp>(debatchedArgLoc, descr.downcastedType, arg);
        _log.trace("apply unrealized_cast");
        arg.replaceUsesWithIf(unrealized_cast.getResult(0), [&](mlir::OpOperand& opOperand) {
            return opOperand.getOwner() != unrealized_cast;
        });

        // Since the new operand as an opResult has been injected,
        // we must remember it in debatchedOperandStorage with the only exception
        debatchedOperandStorage[unrealized_cast.getResult(0)] = {debatchedOperandStorage[arg].from,
                                                                 opResultsToDebatch[arg]};
    });

    if (!needDebatch) {
        _log.debug("Debatching is not required");
        return;
    }
    setCompileMethodDebatch(module);
    _log.trace("Walk through `main` region and debatch all operations");
    detail::OpCastVisitor transformation(_log);

    main.walk([this, &activationOperations, &debatchedOperandStorage, &opResultsToDebatch,
               &transformation](mlir::Operation* op) {
        // Do not debatch non-activation operations
        // They might be debatched as a part of a particular operation consolidation,
        // as a backward graph traversing routine. Basically, this can happen if a "batch"
        // is "hardcoded" as a constanct inside a model
        if (mlir::isa<vpux::Const::DeclareOp, mlir::func::ReturnOp, mlir::UnrealizedConversionCastOp>(op)) {
            mlir::OperationName name = op->getName();
            _log.trace("skip op by name: {0}, Identifier: {1} ", name.getStringRef(), name.getIdentifier());
            return;
        }

        // Check if operation is suitable to debatch.
        // The condition is met when at least one operand of the operations
        // requires for debatching
        auto operandsToDebatch = detail::getOperandsToDebatch(op, activationOperations);
        _log.trace("Operation: {0} - gathered operands as debatching candidates: ({1}/{2})",
                   op->getName().getStringRef(), operandsToDebatch.size(), op->getOperands().size());
        if (!operandsToDebatch.empty()) {
            for (mlir::Value operand : operandsToDebatch) {
                _log.nest().trace("Operation: {0}, operand: {1} ", op->getName().getStringRef(), operand);
                if (debatchedOperandStorage.contains(operand)) {
                    _log.nest().trace(
                            "Skip operand conversion from: {0}, to: {1} - which was debatched already as an opResult",
                            debatchedOperandStorage[operand].from, debatchedOperandStorage[operand].coeff.to_string());
                    continue;
                }
                _log.nest().trace("Operand: {0} for debatching found, desired transformation: {1}", operand,
                                  opResultsToDebatch[operand].to_string());
                auto descr = detail::getDowncastedTypeIfApplicable(operand, opResultsToDebatch[operand]);
                operand.setType(descr.downcastedType);
                debatchedOperandStorage[operand] = {descr.originalShape, opResultsToDebatch[operand]};
                _log.nest().trace("operand debatched from: {0}, to: {1}", debatchedOperandStorage[operand].from,
                                  debatchedOperandStorage[operand].coeff.to_string());
            }

            // Having types of operands of the operation downcasted,
            // try to approach the operation consolidation phase: where we
            // debatch the operation, which includes its attributes correction
            // as well as deducting correct operation results types.
            // This consolidation routine determines which batch values these results
            // must have got, providing that DebatchCoefficients have applied to the operation
            // input operands. Some operation may change N of input tensors and produce
            // batched result even it the initial tensors were non-batched. To settle this down,
            // the consolidation routine traverse the graph in the backward direction starting from the
            // the unsettled operation to find out a source of this discrepancy,
            // which might be a constant having a hardcoded value of N.
            // A typical example is a `Broadcast(tensor, N)` which makes initially
            // non-batched tensor batched by N. If that found, then the such constant will be debatched
            // accordingly
            DenseMap<mlir::OpResult, DebatchCoeffDescription> possibleResultShapes =
                    transformation.visit(op, operandsToDebatch, debatchedOperandStorage);
            // Remember debatching coefficients for these results, deducted
            // individually for every operation. These debatched coefficients will be used
            // later as downcasting ratio for next operations which consumes those results
            // as operands for these operations
            for (auto val : possibleResultShapes) {
                opResultsToDebatch[val.first] = val.second;
            }

            // Change type of downcasted results and pump coefficients over `debatchedOperandStorage`
            // to avoid double downcasting them on next step
            auto opResults = op->getResults();
            _log.trace("Operation: {0} - has OpResults count: {1}", op->getName().getStringRef(), opResults.size());
            llvm::for_each(opResults, [&debatchedOperandStorage, &opResultsToDebatch, this, op](mlir::OpResult r) {
                auto descr = detail::getDowncastedTypeIfApplicable(r, opResultsToDebatch[r]);
                r.setType(descr.downcastedType);
                if (!debatchedOperandStorage.contains(r)) {
                    debatchedOperandStorage[r] = {descr.originalShape, opResultsToDebatch[r]};
                    _log.nest().trace("remember debatched result from: {1}, to: {2}", op->getName().getStringRef(),
                                      debatchedOperandStorage[r].from, debatchedOperandStorage[r].coeff.to_string());
                }
            });
        }
    });

    _log.trace("restoration of original ReturnOps args of 'main'");
    auto resultOriginalTypes = main.getResultTypes();
    main.walk([&builder, &resultOriginalTypes](mlir::func::ReturnOp op) {
        auto operands = op->getOperands();
        builder.setInsertionPoint(op);
        for (auto resultOpDescriptor : zip(operands, resultOriginalTypes)) {
            auto& [operand, originalOpType] = resultOpDescriptor;
            auto castedType = mlir::cast<vpux::NDTypeInterface>(operand.getType());
            if (castedType == originalOpType) {
                continue;
            }
            castedType.changeShape(mlir::cast<vpux::NDTypeInterface>(originalOpType).getShape());
            const auto debatchedResLoc = appendLoc(operand.getLoc(), "_debatched_arg");
            auto unrealizedCast =
                    builder.create<mlir::UnrealizedConversionCastOp>(debatchedResLoc, originalOpType, operand);
            operand.replaceUsesWithIf(unrealizedCast.getResult(0), [&](mlir::OpOperand& opOperand) {
                return opOperand.getOwner() == op;
            });
        }
    });
}
}  // namespace

//
// createDebatcherPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createDebatcherPass(const vpux::DebatcherOptions& options, Logger log) {
    return std::make_unique<DebatcherPass>(options, log);
}

std::unique_ptr<mlir::Pass> vpux::IE::createDebatcherPass(Logger log) {
    return createDebatcherPass(vpux::DebatcherOptions{}, log);
}
