//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <stack>
#include <unordered_set>

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/utils/core/format.hpp"

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

DowncastedTypeDescription getDowncastedTypeIfApplicable(mlir::Value operand, const Shape& desiredShape) {
    auto type = operand.getType().template cast<vpux::NDTypeInterface>();
    auto originShape = type.getShape();
    bool debatched = false;
    if (desiredShape[Dims4D::Act::N] == 0 || originShape[Dims4D::Act::N] == 1) {
        return DowncastedTypeDescription{type, debatched, originShape.raw()};
    }
    type = type.changeShape(desiredShape);
    debatched = true;

    return DowncastedTypeDescription{type, debatched, originShape.raw()};
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
    auto attr = op->getAttr(attName).dyn_cast_or_null<mlir::ArrayAttr>();
    VPUX_THROW_UNLESS(attr != nullptr, "Unexpected type for \"{0}\", only \"mlir::ArrayAttr\" supported", attName);

    return parseIntArrayAttr<Integer>(attr);
}

struct ConversionDescription {
    Shape from;
    Shape to;
};

struct OpConverter {
    virtual ~OpConverter() = default;
    virtual bool isApplicable(mlir::Operation* op) const = 0;
    virtual void apply(mlir::Operation* op, const std::vector<detail::ConversionDescription>& opArguments) const = 0;
    virtual void refineResults(mlir::Operation* op, const std::vector<ConversionDescription>& inOperands,
                               DenseMap<mlir::OpResult, Shape>& inOutResults) const = 0;
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
                       DenseMap<mlir::OpResult, Shape>& inOutResults) const override {
        VPUX_THROW_UNLESS(inOutResults.size() == 0, "DefaultOpConverter expected empty results to override, got: {0}",
                          inOutResults.size());
        auto opResults = op->getResults();
        for (auto result : opResults) {
            int64_t batchDenominator = inOperands[0].from[Dims4D::Act::N] / inOperands[0].to[Dims4D::Act::N];
            auto resultShape = Shape{vpux::getShape(result).raw()};
            // Prevents dimension N from becoming 0 (fractional division by batchDenominator)
            if (resultShape[Dims4D::Act::N] >= batchDenominator) {
                VPUX_THROW_WHEN(resultShape[Dims4D::Act::N] % batchDenominator != 0,
                                "Cannot divide N dimension by {0} for result: {1}", batchDenominator, result);
                resultShape[Dims4D::Act::N] /= batchDenominator;
            }
            inOutResults[result] = std::move(resultShape);
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
        _log.trace("Additional processing requires for an operation: {0} as it has a special attribute: \"{1}\"",
                   op->getName().getStringRef(), getShapeValueAttrName());
        auto expectedResultshapeValue = Shape(consumeArrayAttrAsIntegerArray<int64_t>(op, getShapeValueAttrName()));
        _log.trace("Attribute \"{0}\" value: {1}, original operand value: {2}, casted operand value: {3}",
                   getShapeValueAttrName(), expectedResultshapeValue, operands[0].from, operands[0].to);

        // downcast attribute by a batch size in N dimension
        expectedResultshapeValue[Dims4D::Act::N] /= (operands[0].from[Dims4D::Act::N] / operands[0].to[Dims4D::Act::N]);

        auto downcastedAttr = getIntArrayAttr(op->getContext(), expectedResultshapeValue);
        VPUX_THROW_UNLESS(downcastedAttr != nullptr, "Cannot create downcasted attribute \"{0}\"",
                          getShapeValueAttrName());
        op->setAttr(getShapeValueAttrName(), downcastedAttr);
    }

    void refineResults(mlir::Operation*, const std::vector<ConversionDescription>&,
                       DenseMap<mlir::OpResult, Shape>& results) const override {
        VPUX_THROW_UNLESS(results.size() == 1,
                          "Operation attributed by \"{0}\" is supposed to produce only one result, got: {1}",
                          getShapeValueAttrName(), results.size());
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
        _log.trace("Additional processing requires for an operation: {0} as it has a special attribute: \"{1}\"",
                   op->getName().getStringRef(), getShapeValueAttrName());
        static constexpr std::string_view axesAttrName("axes_attr");
        VPUX_THROW_UNLESS(op->hasAttr(axesAttrName), "SizesAttrOpConverter requires additional \"{0}\" for processing",
                          axesAttrName);

        std::optional<Dim> batchAxisIndex;
        auto axisValues = consumeArrayAttrAsIntegerArray<int64_t>(op, axesAttrName);
        for (const auto& [dimIndex, axisValue] : axisValues | indexed) {
            if (Dim(axisValue) == Dims4D::Act::N) {
                batchAxisIndex = Dim(dimIndex);
            }
        }
        if (!batchAxisIndex.has_value()) {
            _log.trace("Attribute \"{0}\" has no N dimenstion, skip conversion step", axesAttrName);
            return;
        }

        // downcast attribute by a batch size in N dimensition
        auto expectedResultshapeValue = Shape(consumeArrayAttrAsIntegerArray<int64_t>(op, getShapeValueAttrName()));
        _log.trace("Attribute \"{0}\" value: {1}, original operand value: {2}, casted operand value: {3}",
                   getShapeValueAttrName(), expectedResultshapeValue, operands[0].from, operands[0].to);
        expectedResultshapeValue[batchAxisIndex.value()] /=
                (operands[0].from[Dims4D::Act::N] / operands[0].to[Dims4D::Act::N]);

        auto downcastedAttr = getIntArrayAttr(op->getContext(), expectedResultshapeValue);
        VPUX_THROW_UNLESS(downcastedAttr != nullptr, "Cannot create downcasted attribute \"{0}\"",
                          getShapeValueAttrName());
        op->setAttr(getShapeValueAttrName(), downcastedAttr);
    }

    void refineResults(mlir::Operation*, const std::vector<ConversionDescription>&,
                       DenseMap<mlir::OpResult, Shape>& results) const override {
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
        size_t batchSize = operandChange.to[Dims4D::Act::N];
        size_t spanSize = scaleValue.size() / batchSize;
        for (size_t batchIndex = 1; batchIndex < batchSize; ++batchIndex) {
            // Compare the span of scaleValue for the current batch with the first batch
            if (memcmp(scaleValue.data(), scaleValue.data() + batchIndex * spanSize, spanSize * sizeof(T)) != 0) {
                isConsistent = false;
                break;
            }
        }

        if (isConsistent) {
            // Handling scaleValue size not the same as result (arg.to) total size
            // Truncate scaleValue to match arg.to's total size (likely only affects operand 0 will have different shape
            // as arg.to) Else, change constant scale value to match arg.to's N dimension
            if (scaleValue.size() != static_cast<size_t>(operandChange.to.totalSize())) {
                scaleValue.resize(operandChange.to.totalSize());
                _log.trace("[DimLimiter] - BroadcastOp | Truncated scale value to match arg.to's total size: {0}",
                           scaleValue.size());
            } else if (operandChange.from.totalSize() != operandChange.to.totalSize()) {
                // Change constant scale value to match arg.to's N dimension
                scaleValue[0] = operandChange.to[Dims4D::Act::N];
                _log.trace("[DimLimiter] - BroadcastOp | Changing constant value to match arg.to's N dimension: {0}",
                           scaleValue[0]);
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
        if (arg.from == arg.to) {
            return;
        }
        _log.trace("[DimLimiter] - BroadcastOp | Result change From: {0} To: {1}", arg.from, arg.to);

        // Get a list of Const operands
        std::list<int> constOperandIndices;
        for (unsigned int operandIndex = 0; operandIndex < op->getNumOperands(); ++operandIndex) {
            auto operand = op->getOperand(operandIndex);
            if (auto constDeclareOp = operand.getDefiningOp<Const::DeclareOp>()) {
                // For Operand 0, if the operand shape and the arg.to shape is the same, skip adding to the list
                auto operandShape = getShape(operand);
                if (operandIndex == 0 && operandShape == arg.to) {
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
            auto constType = op->getOperand(constOperandIndex).getType().cast<NDTypeInterface>().getElementType();

            // 2. Set to debatched shape, 'arg.to', following data type of the original constant
            const auto scaleShape =
                    mlir::RankedTensorType::get(ArrayRef(arg.to.raw().data(), arg.to.raw().size()),  // Shape
                                                constType);                                          // Data type
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
                       DenseMap<mlir::OpResult, Shape>&) const override {
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
        converters.push_back(std::make_unique<SizesAttrOpConverter>(log));
        converters.push_back(std::make_unique<DimensionLimiterConverter>(log));
    }

    template <class Converter, class... Args>
    void addConverter(Args&&... args) {
        converters.push_back(std::make_unique<Converter>(_log, std::forward<Args>(args)...));
    }

    DenseMap<mlir::OpResult, Shape> runConverters(mlir::Operation* op,
                                                  std::vector<detail::ConversionDescription> operationArguments) {
        DenseMap<mlir::OpResult, Shape> deductedResultShapes;
        for (const auto& c : converters) {
            if (c->isApplicable(op)) {
                c->apply(op, operationArguments);
                c->refineResults(op, operationArguments, deductedResultShapes);
            }
        }
        return deductedResultShapes;
    }

    DenseMap<mlir::OpResult, Shape> visit(
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

        DenseMap<mlir::OpResult, Shape> deductedResultShapes;
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
                [&](const mlir::Value& operand,
                    const llvm::DenseMap<mlir::Value, ConversionDescription>& operandsConversionDescription) -> bool {
            if (auto it = operandsConversionDescription.find(operand);
                it == operandsConversionDescription.end() || it->second.from == it->second.to) {
                // Operand is not in the list, or 'from' and 'to' are the same, we can safely debatch it
                return true;
            }
            return false;
        };

        if (!predictedResultTypes.empty()) {
            for (auto resultPair : zip(deductedResultShapes, predictedResultTypes)) {
                auto [opResult, calculatedShape] = std::get<0>(resultPair);
                auto predictedResultType = std::get<1>(resultPair).dyn_cast<vpux::NDTypeInterface>();
                (void)opResult;
                VPUX_THROW_UNLESS(predictedResultType != nullptr,
                                  "predictedResultType has non vpux::NDTypeInterface type '{0}'",
                                  std::get<1>(resultPair));

                unsigned numOperands = op->getNumOperands();
                unsigned operandIndex = 0;

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
                        auto currentOperandShape = Shape{vpux::getShape(oneOperand).raw()};
                        auto correctOperandShape = currentOperandShape;
                        int64_t batchDenominator =
                                predictedResultType.getShape()[Dims4D::Act::N] / calculatedShape[Dims4D::Act::N];
                        if (correctOperandShape[Dims4D::Act::N] >= batchDenominator) {
                            VPUX_THROW_WHEN(correctOperandShape[Dims4D::Act::N] % batchDenominator != 0,
                                            "Cannot divide N dimension by {0} for operand: {1}", batchDenominator,
                                            oneOperand);
                            correctOperandShape[Dims4D::Act::N] /= batchDenominator;
                        }

                        auto downcastedType = getDowncastedTypeIfApplicable(oneOperand, correctOperandShape);
                        _log.trace("Current Shape: {0}, Correct Shape: {1}, Downcasted Type: {2}", currentOperandShape,
                                   correctOperandShape, downcastedType.downcastedType);
                        if (downcastedType.isDowncasted) {
                            auto downcastedOperandShape =
                                    downcastedType.downcastedType.cast<vpux::NDTypeInterface>().getShape();
                            // Set the downcast type as new operand shape
                            oneOperand.setType(downcastedType.downcastedType);

                            mlir::Operation* operandDefiningOp = oneOperand.getDefiningOp();
                            if (operandDefiningOp == nullptr) {
                                continue;
                            }
                            definingOpOperandsToDebatch.push_back(oneOperand);

                            // Add the debatched operand into operandsConversionDescription (to be used in next visit
                            // call)
                            operandsConversionDescription[oneOperand] = detail::ConversionDescription{
                                    downcastedType.originalShape, Shape(downcastedOperandShape)};

                            // Visit the operandDefiningOp of the operand that we just downcasted
                            this->visit(operandDefiningOp, definingOpOperandsToDebatch, operandsConversionDescription);
                            predictedResultTypes = getPredictedResult(op);
                            predictedResultType = predictedResultTypes[0].dyn_cast<vpux::NDTypeInterface>();
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

    mlir::LogicalResult initializeOptions(StringRef options) final;

private:
    void safeRunOnFunc() final;
};

mlir::LogicalResult DebatcherPass::initializeOptions(StringRef options) {
    if (mlir::failed(Base::initializeOptions(options))) {
        return mlir::failure();
    }
    _log.trace("{0}: {1}", extraArgs.getArgStr(), extraArgs.getValue());
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
    // which producers are constant operations or simple declarations, consequently
    // we must debatch only operands which ascend to argumens of the main-function,
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

    DenseMap<mlir::Value, detail::ConversionDescription> debatchedOperandStorage;
    DenseMap<mlir::Value, Shape> opResultsToDebatch;
    for (const auto& arg : main.getArguments()) {
        auto originalShape = Shape{vpux::getShape(arg)};
        auto desiredShape = originalShape;
        desiredShape[Dims4D::Act::N] = 1;
        // Put args of main in debatchedOperandStorage as they have already been debatched.
        // It's not true yet, though they will be "unrealized_cast'ed" later.
        // Given that assumption stated, we will leverage a generic algorithm for traversing through
        // operations and debatching operands from opResultsToDebatch collection only.
        // The reason why we had added args on main debatchedOperandStorage is tricky:
        // we shall not debatch operands of a first operation in the body of `main`
        // which operands ascent to main args. Otherwise, we will change types on main
        // inadvertently which we must not.
        debatchedOperandStorage[arg] = detail::ConversionDescription{originalShape, desiredShape};
        opResultsToDebatch[arg] = desiredShape;
        _log.trace("arg: {0}, desired shape: {1}", arg, opResultsToDebatch[arg]);
    }

    _log.trace("create builder for inserting an `unrealize_converion_cast` at region boundaries");
    mlir::OpBuilder builder(main);
    builder.setInsertionPointAfter(main);
    auto ctx = module.getContext();
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
        auto unrealized_cast =
                builder.create<mlir::UnrealizedConversionCastOp>(arg.getLoc(), descr.downcastedType, arg);
        _log.trace("apply unrealized_cast");
        arg.replaceUsesWithIf(unrealized_cast.getResult(0), [&](mlir::OpOperand& opOperand) {
            return opOperand.getOwner() != unrealized_cast;
        });

        // Since the new operand as an opResult has been injected,
        // we must remember it in debatchedOperandStorage with the only exception
        debatchedOperandStorage[unrealized_cast.getResult(0)] = {debatchedOperandStorage[arg].from,
                                                                 opResultsToDebatch[arg]};
    });

    if (needDebatch) {
        setCompileMethodDebatch(module);
    }
    _log.trace("Walk through `main` region and debatch all operations");
    detail::OpCastVisitor transformation(_log);

    main.walk([this, &activationOperations, &debatchedOperandStorage, &opResultsToDebatch,
               &transformation](mlir::Operation* op) {
        // Do not debatch non-activation operations
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
                            debatchedOperandStorage[operand].from, debatchedOperandStorage[operand].to);
                    continue;
                }
                _log.nest().trace("Operand: {0} for debatching found, desired shape: {1}", operand,
                                  opResultsToDebatch[operand]);
                auto descr = detail::getDowncastedTypeIfApplicable(operand, opResultsToDebatch[operand]);
                operand.setType(descr.downcastedType);
                debatchedOperandStorage[operand] = {descr.originalShape, opResultsToDebatch[operand]};
                _log.nest().trace("operand debatched from: {0}, to: {1}", debatchedOperandStorage[operand].from,
                                  debatchedOperandStorage[operand].to);
            }

            // apply args cast on operation and remember deducted result shapes in todo-queue
            DenseMap<mlir::OpResult, Shape> possibleResultShapes =
                    transformation.visit(op, operandsToDebatch, debatchedOperandStorage);
            for (auto val : possibleResultShapes) {
                opResultsToDebatch[val.first] = val.second;
            }

            auto opResults = op->getResults();
            _log.trace("Operation: {0} - has OpResults count: {1}", op->getName().getStringRef(), opResults.size());
            llvm::for_each(opResults, [&debatchedOperandStorage, &opResultsToDebatch, this, op](mlir::OpResult r) {
                auto descr = detail::getDowncastedTypeIfApplicable(r, opResultsToDebatch[r]);
                r.setType(descr.downcastedType);
                if (!debatchedOperandStorage.contains(r)) {
                    debatchedOperandStorage[r] = {descr.originalShape, opResultsToDebatch[r]};
                    _log.nest().trace("remember debatched result from: {1}, to: {2}", op->getName().getStringRef(),
                                      debatchedOperandStorage[r].from, debatchedOperandStorage[r].to);
                }
            });
        }
    });

    _log.trace("restoration of original ReturnOps args of 'main'");
    auto resultOriginalTypes = main.getResultTypes();
    main.walk([&builder, &ctx, &resultOriginalTypes](mlir::func::ReturnOp op) {
        auto operands = op->getOperands();
        builder.setInsertionPoint(op);
        for (auto resultOpDescriptor : zip(operands, resultOriginalTypes)) {
            auto& [operand, originalOpType] = resultOpDescriptor;
            auto casted_type = operand.getType().template cast<vpux::NDTypeInterface>();
            if (casted_type == originalOpType) {
                continue;
            }
            casted_type.changeShape(originalOpType.template cast<vpux::NDTypeInterface>().getShape().toValues());
            auto unrealized_cast = builder.create<mlir::UnrealizedConversionCastOp>(mlir::UnknownLoc::get(ctx),
                                                                                    originalOpType, operand);
            operand.replaceUsesWithIf(unrealized_cast.getResult(0), [&](mlir::OpOperand& opOperand) {
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
