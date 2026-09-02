//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/scf/dialect_processors.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Attributes.h>
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/ADT/bit.h>

namespace vpux::VPU {

namespace {

constexpr int64_t MAX_SHIFT = 64;

}  // namespace

std::optional<int64_t> getIntValueFromDimOp(mlir::tensor::DimOp dimOp, const Logger& log) {
    auto dimIndex = mlir::getConstantIntValue(dimOp.getIndex());
    if (!dimIndex.has_value()) {
        log.warning("Dim index is not a constant!");
        return std::nullopt;
    }

    if (auto rankedType = mlir::dyn_cast<mlir::RankedTensorType>(dimOp.getSource().getType())) {
        const auto dimIndexValue = dimIndex.value();
        if (rankedType.hasStaticShape() && dimIndexValue < rankedType.getRank()) {
            return rankedType.getShape()[dimIndexValue];
        }

        if (auto boundedType = mlir::dyn_cast<vpux::Core::BoundedTensorType>(rankedType)) {
            auto bounds = boundedType.getBounds().raw();
            if (dimIndexValue < static_cast<int64_t>(bounds.size())) {
                return bounds[dimIndexValue];
            }
        }
    }

    return std::nullopt;
}

mlir::scf::YieldOp getYieldOperation(mlir::Block* block) {
    if (!block) {
        return nullptr;
    }

    if (auto terminator = block->getTerminator()) {
        if (auto yieldOp = mlir::dyn_cast<mlir::scf::YieldOp>(terminator)) {
            return yieldOp;
        }
    }

    for (auto& op : block->getOperations()) {
        if (auto yieldOp = mlir::dyn_cast<mlir::scf::YieldOp>(&op)) {
            return yieldOp;
        }
    }

    return nullptr;
}

mlir::Operation* getBlockTerminator(mlir::Block* block) {
    if (!block) {
        return nullptr;
    }
    return block->getTerminator();
}

namespace {

std::optional<int64_t> checkAndGetValueFromConstOrDimOp(mlir::Value operand, const Logger& log) {
    auto integerValue = mlir::getConstantIntValue(operand);
    if (integerValue.has_value()) {
        return integerValue;
    }

    if (auto dimOp = mlir::dyn_cast_or_null<mlir::tensor::DimOp>(operand.getDefiningOp())) {
        auto dimensionValue = getIntValueFromDimOp(dimOp, log);
        if (dimensionValue.has_value()) {
            return dimensionValue;
        }
    }
    return std::nullopt;
}

}  // namespace

void DialectProcessorRegistry::registerProcessor(std::unique_ptr<IDialectProcessor> processor) {
    _processors.push_back(std::move(processor));
    _dialectCache.clear();
}

IDialectProcessor* DialectProcessorRegistry::getProcessor(mlir::Operation* op) const {
    auto* dialect = op->getDialect();

    auto cacheIterator = _dialectCache.find(dialect);
    if (cacheIterator != _dialectCache.end()) {
        return cacheIterator->second;
    }

    for (const auto& processor : _processors) {
        if (processor->canProcess(op)) {
            _dialectCache[dialect] = processor.get();
            return processor.get();
        }
    }

    _dialectCache[dialect] = nullptr;
    return nullptr;
}

bool DialectProcessorRegistry::hasProcessor(mlir::Operation* op) const {
    return getProcessor(op) != nullptr;
}

std::unique_ptr<DialectProcessorRegistry> DialectProcessorRegistry::createDefault(
        TensorBoundResolver tensorBoundResolver) {
    auto registry = std::make_unique<DialectProcessorRegistry>();
    registry->registerProcessor(
            std::make_unique<AffineDialectProcessor>(Logger::global().nest("affine-dialect-processor")));
    registry->registerProcessor(
            std::make_unique<ArithmeticDialectProcessor>(Logger::global().nest("arith-dialect-processor")));
    registry->registerProcessor(std::make_unique<SCFDialectProcessor>(Logger::global().nest("scf-dialect-processor")));
    registry->registerProcessor(std::make_unique<TensorDialectProcessor>(
            Logger::global().nest("tensor-dialect-processor"), std::move(tensorBoundResolver)));
    return registry;
}

bool AffineDialectProcessor::canProcess(mlir::Operation* op) const {
    return mlir::isa<mlir::affine::AffineDialect>(op->getDialect());
}

bool AffineDialectProcessor::processOperation(mlir::Operation* op, ScalarValueMap& valueMap,
                                              BlockProcessor blockProcessor) const {
    (void)blockProcessor;
    auto [affineMap, mapOperands] = getAffineMapAndOperands(op);
    SmallVector<mlir::Attribute> operandAttrs;
    bool success = true;
    for (auto operand : mapOperands) {
        int64_t operandValue = 0;
        auto valueIterator = valueMap.find(operand);
        if (valueIterator != valueMap.end()) {
            operandValue = valueIterator->second;
        } else {
            auto value = checkAndGetValueFromConstOrDimOp(operand, _log);
            if (value.has_value()) {
                operandValue = value.value();
            } else {
                _log.warning("Missing operand value for affine operation: {0}", op->getName());
                success = false;
                break;
            }
        }
        operandAttrs.push_back(mlir::IntegerAttr::get(operand.getType(), operandValue));
    }

    if (!success) {
        return false;
    }

    SmallVector<mlir::Attribute> resultsAttrs;
    if (affineMap.constantFold(operandAttrs, resultsAttrs).failed()) {
        return false;
    }

    SmallVector<int64_t> results;
    for (auto attr : resultsAttrs) {
        results.push_back(mlir::cast<mlir::IntegerAttr>(attr).getInt());
    }

    int64_t result = getAffineResult(op, results);
    valueMap[op->getResult(0)] = result;
    return true;
}

std::pair<mlir::AffineMap, mlir::ValueRange> AffineDialectProcessor::getAffineMapAndOperands(
        mlir::Operation* op) const {
    if (auto affineOp = mlir::dyn_cast<mlir::affine::AffineMinOp>(op)) {
        return {affineOp.getAffineMap(), affineOp.getOperands()};
    }
    if (auto affineOp = mlir::dyn_cast<mlir::affine::AffineMaxOp>(op)) {
        return {affineOp.getAffineMap(), affineOp.getOperands()};
    }
    if (auto applyOp = mlir::dyn_cast<mlir::affine::AffineApplyOp>(op)) {
        return {applyOp.getAffineMap(), applyOp.getOperands()};
    }

    VPUX_THROW("Unsupported affine operation type: {0}", op->getName());
}

int64_t AffineDialectProcessor::getAffineResult(mlir::Operation* op, llvm::ArrayRef<int64_t> results) const {
    if (results.empty()) {
        VPUX_THROW("Empty results array for operation: {0}", op->getName());
    }

    if (mlir::isa<mlir::affine::AffineMinOp>(op)) {
        return *llvm::min_element(results);
    }
    if (mlir::isa<mlir::affine::AffineMaxOp>(op)) {
        return *llvm::max_element(results);
    }
    if (mlir::isa<mlir::affine::AffineApplyOp>(op)) {
        return results[0];
    }

    VPUX_THROW("Unsupported affine operation type: {0}", op->getName());
}

bool ArithmeticDialectProcessor::canProcess(mlir::Operation* op) const {
    return mlir::isa<mlir::arith::ArithDialect>(op->getDialect());
}

bool ArithmeticDialectProcessor::processOperation(mlir::Operation* op, ScalarValueMap& valueMap,
                                                  BlockProcessor blockProcessor) const {
    (void)blockProcessor;
    if (auto constOp = mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValueAttr())) {
            valueMap[op->getResult(0)] = intAttr.getInt();
            return true;
        }
        if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(constOp.getValueAttr())) {
            // Scalar float constant — store as double bit pattern for float ops (e.g. arith.mulf).
            valueMap[op->getResult(0)] = llvm::bit_cast<int64_t>(floatAttr.getValueAsDouble());
            return true;
        }
        // Non-scalar constant (e.g. dense<[8.0, 4.0]> : tensor<2xf16>): the value cannot be
        // represented as a scalar int64_t in the value map. TensorDialectProcessor handles
        // such constants directly by reading the defining-op attribute (bypassing valueMap),
        // so the chain evaluation can safely continue.
        _log.trace("Skipping non-scalar arith.constant (result type: {0})", op->getResult(0).getType());
        return true;
    }

    for (auto operand : op->getOperands()) {
        if (!valueMap.contains(operand)) {
            auto intValue = checkAndGetValueFromConstOrDimOp(operand, _log);
            if (intValue.has_value()) {
                valueMap[operand] = intValue.value();
                continue;
            }

            _log.warning("Missing operand value for arith operation: {0}", op->getName());
        }
    }

    return llvm::TypeSwitch<mlir::Operation*, bool>(op)
            .Case<mlir::arith::MinUIOp>([&](auto minOp) {
                auto lhs = checked_cast<uint64_t>(valueMap[minOp.getLhs()]);
                auto rhs = checked_cast<uint64_t>(valueMap[minOp.getRhs()]);
                auto resultValue = std::min(lhs, rhs);
                valueMap[op->getResult(0)] = checked_cast<int64_t>(resultValue);
                return true;
            })
            .Case<mlir::arith::AddIOp>([&](auto addOp) {
                auto resultValue = valueMap[addOp.getLhs()] + valueMap[addOp.getRhs()];
                valueMap[op->getResult(0)] = resultValue;
                return true;
            })
            .Case<mlir::arith::SubIOp>([&](auto subOp) {
                auto resultValue = valueMap[subOp.getLhs()] - valueMap[subOp.getRhs()];
                valueMap[op->getResult(0)] = resultValue;
                return true;
            })
            .Case<mlir::arith::MulIOp>([&](auto mulOp) {
                auto resultValue = valueMap[mulOp.getLhs()] * valueMap[mulOp.getRhs()];
                valueMap[op->getResult(0)] = resultValue;
                return true;
            })
            .Case<mlir::arith::DivSIOp>([&](auto divOp) {
                if (valueMap[divOp.getRhs()] == 0) {
                    _log.warning("Division by zero in DivSIOp");
                    return false;
                }
                auto resultValue = valueMap[divOp.getLhs()] / valueMap[divOp.getRhs()];
                valueMap[op->getResult(0)] = resultValue;
                return true;
            })
            .Case<mlir::arith::DivUIOp>([&](auto divOp) {
                if (valueMap[divOp.getRhs()] == 0) {
                    _log.warning("Division by zero in DivUIOp");
                    return false;
                }
                auto resultValue = valueMap[divOp.getLhs()] / valueMap[divOp.getRhs()];
                valueMap[op->getResult(0)] = resultValue;
                return true;
            })
            .Case<mlir::arith::CmpIOp>([&](auto cmpOp) {
                auto lhs = valueMap[cmpOp.getLhs()];
                auto rhs = valueMap[cmpOp.getRhs()];
                auto predicate = cmpOp.getPredicate();
                bool result = false;
                switch (predicate) {
                case mlir::arith::CmpIPredicate::eq:
                    result = (lhs == rhs);
                    break;
                case mlir::arith::CmpIPredicate::ne:
                    result = (lhs != rhs);
                    break;
                case mlir::arith::CmpIPredicate::slt:
                    result = (lhs < rhs);
                    break;
                case mlir::arith::CmpIPredicate::sle:
                    result = (lhs <= rhs);
                    break;
                case mlir::arith::CmpIPredicate::sgt:
                    result = (lhs > rhs);
                    break;
                case mlir::arith::CmpIPredicate::sge:
                    result = (lhs >= rhs);
                    break;
                case mlir::arith::CmpIPredicate::ult:
                    result = (static_cast<uint64_t>(lhs) < static_cast<uint64_t>(rhs));
                    break;
                case mlir::arith::CmpIPredicate::ule:
                    result = (static_cast<uint64_t>(lhs) <= static_cast<uint64_t>(rhs));
                    break;
                case mlir::arith::CmpIPredicate::ugt:
                    result = (static_cast<uint64_t>(lhs) > static_cast<uint64_t>(rhs));
                    break;
                case mlir::arith::CmpIPredicate::uge:
                    result = (static_cast<uint64_t>(lhs) >= static_cast<uint64_t>(rhs));
                    break;
                }
                valueMap[op->getResult(0)] = result ? 1 : 0;
                return true;
            })
            .Case<mlir::arith::RemSIOp>([&](auto remOp) {
                if (valueMap[remOp.getRhs()] == 0) {
                    _log.warning("Division by zero in RemSIOp");
                    return false;
                }
                auto resultValue = valueMap[remOp.getLhs()] % valueMap[remOp.getRhs()];
                valueMap[op->getResult(0)] = resultValue;
                return true;
            })
            .Case<mlir::arith::RemUIOp>([&](auto remOp) {
                if (valueMap[remOp.getRhs()] == 0) {
                    _log.warning("Division by zero in RemUIOp");
                    return false;
                }
                auto lhs = static_cast<uint64_t>(valueMap[remOp.getLhs()]);
                auto rhs = static_cast<uint64_t>(valueMap[remOp.getRhs()]);
                auto resultValue = static_cast<int64_t>(lhs % rhs);
                valueMap[op->getResult(0)] = resultValue;
                return true;
            })
            .Case<mlir::arith::SelectOp>([&](auto selectOp) {
                auto condition = valueMap[selectOp.getCondition()];
                auto trueValue = valueMap[selectOp.getTrueValue()];
                auto falseValue = valueMap[selectOp.getFalseValue()];
                auto resultValue = (condition != 0) ? trueValue : falseValue;
                valueMap[op->getResult(0)] = resultValue;
                return true;
            })
            .Case<mlir::arith::ShLIOp>([&](auto shiftOp) {
                auto lhs = valueMap[shiftOp.getLhs()];
                auto rhs = valueMap[shiftOp.getRhs()];
                if (rhs < 0 || rhs >= MAX_SHIFT) {
                    _log.warning("Invalid shift amount in ShLIOp: {0}", rhs);
                    return false;
                }
                auto resultValue = lhs << rhs;
                valueMap[op->getResult(0)] = resultValue;
                return true;
            })
            .Case<mlir::arith::ShRSIOp>([&](auto shiftOp) {
                auto lhs = valueMap[shiftOp.getLhs()];
                auto rhs = valueMap[shiftOp.getRhs()];
                if (rhs < 0 || rhs >= MAX_SHIFT) {
                    _log.warning("Invalid shift amount in ShRSIOp: {0}", rhs);
                    return false;
                }
                auto resultValue = lhs >> rhs;
                valueMap[op->getResult(0)] = resultValue;
                return true;
            })
            .Case<mlir::arith::ShRUIOp>([&](auto shiftOp) {
                auto lhs = valueMap[shiftOp.getLhs()];
                auto rhs = valueMap[shiftOp.getRhs()];
                if (rhs < 0 || rhs >= MAX_SHIFT) {
                    _log.warning("Invalid shift amount in ShRUIOp: {0}", rhs);
                    return false;
                }
                auto unsignedLhs = static_cast<uint64_t>(lhs);
                auto resultValue = static_cast<int64_t>(unsignedLhs >> rhs);
                valueMap[op->getResult(0)] = resultValue;
                return true;
            })
            .Case<mlir::arith::OrIOp>([&](auto orOp) {
                auto lhs = valueMap[orOp.getLhs()];
                auto rhs = valueMap[orOp.getRhs()];
                auto resultValue = lhs | rhs;
                valueMap[op->getResult(0)] = resultValue;
                return true;
            })
            .Case<mlir::arith::AndIOp>([&](auto andOp) {
                auto lhs = valueMap[andOp.getLhs()];
                auto rhs = valueMap[andOp.getRhs()];
                auto resultValue = lhs & rhs;
                valueMap[op->getResult(0)] = resultValue;
                return true;
            })
            .Case<mlir::arith::IndexCastOp>([&](auto castOp) {
                valueMap[op->getResult(0)] = valueMap[castOp.getIn()];
                return true;
            })
            .Case<mlir::arith::SIToFPOp>([&](auto siToFpOp) {
                auto intVal = valueMap[siToFpOp.getIn()];
                valueMap[op->getResult(0)] = llvm::bit_cast<int64_t>(static_cast<double>(intVal));
                return true;
            })
            .Case<mlir::arith::FPToSIOp>([&](auto fpToSiOp) {
                auto floatBits = valueMap[fpToSiOp.getIn()];
                auto floatVal = llvm::bit_cast<double>(floatBits);
                valueMap[op->getResult(0)] = static_cast<int64_t>(floatVal);
                return true;
            })
            .Case<mlir::arith::MulFOp>([&](auto mulFOp) {
                auto lhsBits = valueMap[mulFOp.getLhs()];
                auto rhsBits = valueMap[mulFOp.getRhs()];
                auto lhs = llvm::bit_cast<double>(lhsBits);
                auto rhs = llvm::bit_cast<double>(rhsBits);
                valueMap[op->getResult(0)] = llvm::bit_cast<int64_t>(lhs * rhs);
                return true;
            })
            .Case<mlir::arith::AddFOp>([&](auto addFOp) {
                auto lhsBits = valueMap[addFOp.getLhs()];
                auto rhsBits = valueMap[addFOp.getRhs()];
                auto lhs = llvm::bit_cast<double>(lhsBits);
                auto rhs = llvm::bit_cast<double>(rhsBits);
                valueMap[op->getResult(0)] = llvm::bit_cast<int64_t>(lhs + rhs);
                return true;
            })
            .Case<mlir::arith::DivFOp>([&](auto divFOp) {
                auto lhsBits = valueMap[divFOp.getLhs()];
                auto rhsBits = valueMap[divFOp.getRhs()];
                auto lhs = llvm::bit_cast<double>(lhsBits);
                auto rhs = llvm::bit_cast<double>(rhsBits);
                if (rhs == 0.0) {
                    _log.warning("Division by zero in DivFOp");
                    return false;
                }
                valueMap[op->getResult(0)] = llvm::bit_cast<int64_t>(lhs / rhs);
                return true;
            })
            .Case<mlir::arith::ExtFOp>([&](auto extFOp) {
                auto it = valueMap.find(extFOp.getIn());
                if (it == valueMap.end()) {
                    _log.trace("Missing input value for arith.extf: operand not in value map");
                    return false;
                }
                valueMap[op->getResult(0)] = it->second;
                return true;
            })
            .Default([&](mlir::Operation*) {
                _log.trace("Unsupported arith operation: {0}", op->getName());
                return false;
            });
}

bool SCFDialectProcessor::canProcess(mlir::Operation* op) const {
    return mlir::isa<mlir::scf::SCFDialect>(op->getDialect());
}

bool SCFDialectProcessor::processOperation(mlir::Operation* op, ScalarValueMap& valueMap,
                                           BlockProcessor blockProcessor) const {
    return llvm::TypeSwitch<mlir::Operation*, bool>(op)
            .Case<mlir::scf::IfOp>([&](auto ifOp) {
                return processIfOp(ifOp, valueMap, blockProcessor);
            })
            .Case<mlir::scf::YieldOp>([&](auto yieldOp) {
                for (auto [idx, operand] : llvm::enumerate(yieldOp.getOperands())) {
                    if (idx < op->getNumResults()) {
                        auto it = valueMap.find(operand);
                        if (it != valueMap.end()) {
                            valueMap[op->getResult(idx)] = it->second;
                        } else {
                            auto result = checkAndGetValueFromConstOrDimOp(operand, _log);
                            if (result.has_value()) {
                                valueMap[op->getResult(idx)] = result.value();
                            } else {
                                _log.warning("Failed to get integer value for yield operand: {0}", operand);
                                return false;
                            }
                        }
                    }
                }
                return true;
            })
            .Default([&](mlir::Operation*) {
                _log.trace("Unsupported scf operation: {0}", op->getName());
                return false;
            });
}

bool SCFDialectProcessor::processIfOp(mlir::scf::IfOp ifOp, ScalarValueMap& valueMap,
                                      const BlockProcessor& blockProcessor) const {
    auto condition = ifOp.getCondition();
    if (!valueMap.contains(condition)) {
        _log.warning("Condition value not found in value map for scf.if operation {0}", condition);
        return false;
    }

    bool condResult = valueMap[condition] != 0;
    mlir::Block* activeBlock = condResult ? ifOp.thenBlock() : ifOp.elseBlock();

    if (!activeBlock) {
        return false;
    }

    blockProcessor(activeBlock, valueMap);

    auto terminator = activeBlock->getTerminator();
    for (auto [idx, result] : llvm::enumerate(ifOp.getResults())) {
        auto valueIterator = valueMap.find(terminator->getOperand(idx));
        if (valueIterator != valueMap.end()) {
            valueMap[result] = valueIterator->second;
        } else {
            auto termResult = checkAndGetValueFromConstOrDimOp(terminator->getOperand(idx), _log);
            if (termResult.has_value()) {
                valueMap[result] = termResult.value();
            } else {
                _log.warning("Failed to get integer value for if terminator operand: {0}", terminator->getOperand(idx));
                return false;
            }
        }
    }

    return true;
}

bool TensorDialectProcessor::canProcess(mlir::Operation* op) const {
    return mlir::isa<mlir::tensor::TensorDialect>(op->getDialect());
}

bool TensorDialectProcessor::processExtractOp(mlir::tensor::ExtractOp extractOp, ScalarValueMap& valueMap) const {
    // Use a pre-seeded tensor bound if available.
    auto tensorIt = valueMap.find(extractOp.getTensor());
    if (tensorIt != valueMap.end()) {
        valueMap[extractOp.getResult()] = tensorIt->second;
        return true;
    }

    auto indices = extractOp.getIndices();
    if (indices.size() != 1) {
        _log.trace("tensor.extract with {0}-dimensional index not supported", indices.size());
        return false;
    }

    int64_t indexVal = 0;
    auto indexIt = valueMap.find(indices[0]);
    if (indexIt != valueMap.end()) {
        indexVal = indexIt->second;
    } else {
        auto constIdx = mlir::getConstantIntValue(indices[0]);
        if (!constIdx.has_value()) {
            _log.trace("tensor.extract: index is not statically known");
            return false;
        }
        indexVal = constIdx.value();
    }

    auto* defOp = extractOp.getTensor().getDefiningOp();
    auto constOp = defOp ? mlir::dyn_cast<mlir::arith::ConstantOp>(defOp) : nullptr;
    if (!constOp) {
        // The source is not a constant, so the extracted element cannot be read directly. Ask the injected
        // resolver whether a bound can be recovered for this tensor instead.
        if (_boundResolver != nullptr) {
            if (auto bound = _boundResolver(extractOp.getTensor())) {
                valueMap[extractOp.getResult()] = bound.value();
                return true;
            }
        }
        _log.trace("tensor.extract: non-constant source without a pre-seeded or recovered bound");
        return false;
    }

    auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue());
    if (!denseAttr) {
        _log.trace("tensor.extract: constant is not a DenseElementsAttr");
        return false;
    }

    auto numElements = denseAttr.getNumElements();
    if (indexVal < 0 || indexVal >= numElements) {
        _log.trace("tensor.extract: index {0} is out of bounds (size {1})", indexVal, numElements);
        return false;
    }

    if (mlir::isa<mlir::FloatType>(extractOp.getResult().getType())) {
        auto fpValues = denseAttr.template getValues<llvm::APFloat>();
        auto fpValIt = fpValues.begin();
        std::advance(fpValIt, indexVal);
        llvm::APFloat fpVal(*fpValIt);
        bool lossy = false;
        fpVal.convert(llvm::APFloat::IEEEdouble(), llvm::APFloat::rmNearestTiesToEven, &lossy);
        valueMap[extractOp.getResult()] = llvm::bit_cast<int64_t>(fpVal.convertToDouble());
    } else {
        auto intValues = denseAttr.template getValues<llvm::APInt>();
        auto intValIt = intValues.begin();
        std::advance(intValIt, indexVal);
        valueMap[extractOp.getResult()] = (*intValIt).getSExtValue();
    }
    return true;
}

bool TensorDialectProcessor::processOperation(mlir::Operation* op, ScalarValueMap& valueMap, BlockProcessor) const {
    return llvm::TypeSwitch<mlir::Operation*, bool>(op)
            .Case<mlir::tensor::DimOp>([&](auto dimOp) {
                auto dimValue = getIntValueFromDimOp(dimOp, _log);
                if (!dimValue.has_value()) {
                    _log.trace("Could not resolve tensor.dim to a static value");
                    return false;
                }
                valueMap[op->getResult(0)] = dimValue.value();
                return true;
            })
            .Case<mlir::tensor::ExtractOp>([&](auto extractOp) {
                return processExtractOp(extractOp, valueMap);
            })
            .Default([&](mlir::Operation*) {
                _log.trace("Unsupported tensor operation: {0}", op->getName());
                return false;
            });
}

}  // namespace vpux::VPU
