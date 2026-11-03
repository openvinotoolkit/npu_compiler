//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <cstdint>
#include <vpux/utils/core/format.hpp>
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/Shave/IR/attributes.hpp"
#include "vpux/compiler/dialect/Shave/IR/ops/meta-ops.hpp"
#include "vpux/compiler/dialect/Shave/custom_layer_config.hpp"
#include "vpux/compiler/dialect/Shave/transforms/passes.hpp"
#include "vpux/compiler/dialect/Shave/utils/config_ir_utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

namespace vpux::Shave {
#define GEN_PASS_DECL_APPLYEXTERNALKERNELCONFIGS
#define GEN_PASS_DEF_APPLYEXTERNALKERNELCONFIGS
#include "vpux/compiler/dialect/Shave/passes.hpp.inc"
}  // namespace vpux::Shave

using namespace vpux;

namespace {

//
// E#232091: Attempt to reuse existing similar utilities
//
Shave::TensorDataType getDataTypeFromMlirType(mlir::Type type) {
    if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
        auto typeWidth = floatType.getWidth();
        switch (typeWidth) {
        case 64:
            return Shave::TensorDataType::FP64;
        case 32:
            return Shave::TensorDataType::FP32;
        case 16:
            if (type.isBF16()) {
                return Shave::TensorDataType::BF16;
            }
            return Shave::TensorDataType::FP16;
        case 8:
            if (mlir::isa<mlir::Float8E4M3FNType>(floatType)) {
                return Shave::TensorDataType::HF8;
            } else if (mlir::isa<mlir::Float8E5M2Type>(floatType)) {
                return Shave::TensorDataType::BF8;
            }
            break;
        }
    } else if (auto integerType = mlir::dyn_cast<mlir::IntegerType>(type)) {
        if (integerType.isSigned()) {
            auto typeWidth = integerType.getWidth();
            switch (typeWidth) {
            case 64:
                return Shave::TensorDataType::I64;
            case 32:
                return Shave::TensorDataType::I32;
            case 16:
                return Shave::TensorDataType::I16;
            case 8:
                return Shave::TensorDataType::I8;
            case 4:
                return Shave::TensorDataType::I4;
            case 2:
                return Shave::TensorDataType::I2;
            case 1:
                return Shave::TensorDataType::BIN;
            }
        } else if (integerType.isUnsigned()) {
            auto typeWidth = integerType.getWidth();
            switch (typeWidth) {
            case 64:
                return Shave::TensorDataType::U64;
            case 32:
                return Shave::TensorDataType::U32;
            case 16:
                return Shave::TensorDataType::U16;
            case 8:
                return Shave::TensorDataType::U8;
            case 4:
                return Shave::TensorDataType::U4;
            case 2:
                return Shave::TensorDataType::U2;
            case 1:
                return Shave::TensorDataType::BIN;
            }
        } else if (integerType.isSignless()) {
            auto typeWidth = integerType.getWidth();
            switch (typeWidth) {
            case 64:
                return Shave::TensorDataType::I64;
            case 32:
                return Shave::TensorDataType::I32;
            case 16:
                return Shave::TensorDataType::I16;
            case 8:
                return Shave::TensorDataType::I8;
            case 4:
                return Shave::TensorDataType::I4;
            case 2:
                return Shave::TensorDataType::I2;
            case 1:
                return Shave::TensorDataType::BIN;
            }
        }
    }
    VPUX_THROW("Conversion to Shave::TensorDataType failed for {0}", type);
    return Shave::TensorDataType::UNDEFINED;
}

mlir::Type getMlirTypeFromDataType(Shave::TensorDataType type, mlir::MLIRContext* ctx) {
    switch (type) {
    case Shave::TensorDataType::FP64:
        return mlir::Float64Type::get(ctx);
    case Shave::TensorDataType::FP32:
        return mlir::Float32Type::get(ctx);
    case Shave::TensorDataType::FP16:
        return mlir::Float16Type::get(ctx);
    case Shave::TensorDataType::BF16:
        return mlir::BFloat16Type::get(ctx);
    case Shave::TensorDataType::HF8:
        return mlir::Float8E4M3FNType::get(ctx);
    case Shave::TensorDataType::BF8:
        return mlir::Float8E5M2Type::get(ctx);
    case Shave::TensorDataType::I64:
        return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed);
    case Shave::TensorDataType::I32:
        return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);
    case Shave::TensorDataType::I16:
        return mlir::IntegerType::get(ctx, 16, mlir::IntegerType::Signed);
    case Shave::TensorDataType::I8:
        return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Signed);
    case Shave::TensorDataType::I4:
        return mlir::IntegerType::get(ctx, 4, mlir::IntegerType::Signed);
    case Shave::TensorDataType::I2:
        return mlir::IntegerType::get(ctx, 2, mlir::IntegerType::Signed);
    case Shave::TensorDataType::U64:
        return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
    case Shave::TensorDataType::U32:
        return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Unsigned);
    case Shave::TensorDataType::U16:
        return mlir::IntegerType::get(ctx, 16, mlir::IntegerType::Unsigned);
    case Shave::TensorDataType::U8:
        return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Unsigned);
    case Shave::TensorDataType::U4:
        return mlir::IntegerType::get(ctx, 4, mlir::IntegerType::Unsigned);
    case Shave::TensorDataType::U2:
        return mlir::IntegerType::get(ctx, 2, mlir::IntegerType::Unsigned);
    case Shave::TensorDataType::BIN:
        return mlir::IntegerType::get(ctx, 1, mlir::IntegerType::Unsigned);
    default:
        VPUX_THROW("Conversion from Shave::TensorDataType failed for {0}", type);
    }
}

class ConfigApplicator {
private:
    mlir::Operation* _extKernelOp = nullptr;

    struct ConfigApplicationTracker {
        bool rankConfigApplied = false;
        bool layoutConfigApplied = false;
        bool dataTypeConfigApplied = false;

        // Used to store the type post the rank config, but pre-layout config
        // to correctly compensate for layout changes without being affected by rank changes
        vpux::NDTypeInterface typePostRankConfig = nullptr;
    };

    // The new operands and result types for the ExternalKernelOp after applying config-driven transformations.
    std::vector<mlir::Value> _newOperands{};
    std::vector<mlir::Type> _newResultTypes{};

    // The new result values after creating the new ExternalKernelOp, used to track compensation op insertion and final
    // SSA replacement.
    std::vector<mlir::Value> _newResults{};

    // Tracker for each result to record which types of config-driven transformations have been applied, used to
    // determine necessary compensation ops
    std::vector<ConfigApplicationTracker> _resultModificationTracker{};

public:
    ConfigApplicator() = delete;
    explicit ConfigApplicator(mlir::Operation* extKernelOp): _extKernelOp(extKernelOp) {
        // Start with original operands and result types, and apply transformations on them based on config
        // Collect only the regular I/O. Scratch buffers are added later by MaterializeScratchBuffers pass
        auto inputsRange = mlir::cast<Shave::ExternalKernelOp>(extKernelOp).getInputs();
        _newOperands.insert(_newOperands.end(), inputsRange.begin(), inputsRange.end());
        _newResultTypes.insert(_newResultTypes.end(), extKernelOp->result_type_begin(), extKernelOp->result_type_end());

        _newResults.reserve(_newResultTypes.size());

        // Initialize tracker for each result type
        _resultModificationTracker.resize(_newResultTypes.size());
    }

    // Applies I/O configs in order: rank -> layout -> data type
    // Input operands transformations are applied in-place, generating the necessary ops JIT and populating _newOperands
    // Result type transformations are applied, generating _newResultTypes, but not adding any new ops at this time
    mlir::LogicalResult applyIOConfig(const std::vector<Shave::CustomLayerConfig::IOTensorConfig>& inputConfigs,
                                      const std::vector<Shave::CustomLayerConfig::IOTensorConfig>& outputConfigs) {
        // Apply Rank config
        if (mlir::failed(applyRankConfig(inputConfigs, outputConfigs))) {
            return mlir::failure();
        }

        // Apply Layout config
        if (mlir::failed(applyLayoutConfig(inputConfigs, outputConfigs))) {
            return mlir::failure();
        }

        // Apply Data Type config
        if (mlir::failed(applyDataTypeConfig(inputConfigs, outputConfigs))) {
            return mlir::failure();
        }
        return mlir::success();
    }

    // Finalizes transformation by creating the new ExternalKernelOp with transformed operands and result types,
    // inserting necessary compensation ops, and replacing all uses of original results with the final compensated
    // values.
    mlir::Operation* finalize() {
        return finalizeOpModification();
    }

private:
    // Example IR transformation (config: enforce FP16, rank-4, NCHW layout on a rank-3 FP32 op):
    //
    //   Before:
    //     %0 = ExternalKernelOp(%in)  : (tensor<1x3x224xf32>) -> tensor<1x3x224xf32>
    //
    //   After:
    //     // pre-ops inserted before the new ExternalKernelOp:
    //     %shaped     = IE.Reshape(%in)            : tensor<1x3x224xf32>   -> tensor<1x1x3x224xf32>
    //     %reordered  = IE.Reorder(%shaped)        : tensor<1x1x3x224xf32> -> tensor<1x1x224x3xf32>
    //     %converted  = IE.Convert(%reordered)     : tensor<1x1x224x3xf32> -> tensor<1x1x224x3xf16>
    //     %new0 = ExternalKernelOp(%converted)     : (tensor<1x1x224x3xf16>) -> tensor<1x1x224x3xf16>
    //     // compensation ops inserted after, restoring original types for downstream users:
    //     %conv_back  = IE.Convert(%new0)          : tensor<1x1x224x3xf16> -> tensor<1x1x224x3xf32>
    //     %reord_back = IE.Reorder(%conv_back)     : tensor<1x1x224x3xf32> -> tensor<1x1x3x224xf32>
    //     %shape_back = IE.Reshape(%reord_back)    : tensor<1x1x3x224xf32> -> tensor<1x3x224xf32>
    //     // all original users of %0 now use %shape_back (same type as %0)
    //
    mlir::Operation* finalizeOpModification() {
        mlir::OpBuilder builder(_extKernelOp);
        builder.setInsertionPoint(_extKernelOp);
        auto oldExtKernelOp = mlir::cast<Shave::ExternalKernelOp>(_extKernelOp);

        // Scratch buffers are intentionally empty here; added later in VPU pipeline.
        auto newOp = Shave::ExternalKernelOp::create(
                builder, _extKernelOp->getLoc(), _newResultTypes, _newOperands,
                /*scratch_buffers=*/mlir::ValueRange{}, oldExtKernelOp.getConfig().value_or(nullptr),
                oldExtKernelOp.getKernelAttributesAttr(), oldExtKernelOp.getUniqueIdAttr(), oldExtKernelOp.getUsesDma(),
                oldExtKernelOp.getUsesDpu(), nullptr);

        _newResults.insert(_newResults.end(), newOp.result_begin(), newOp.result_end());

        auto numResults = newOp.getNumResults();

        // Apply compensation ops in order: dtype -> layout -> rank.
        // When a result has multiple modifications, each subsequent compensation op
        // must be inserted after the previous one to preserve SSA def-use order.
        // lastCompOp[i] tracks the tail of the compensation chain for result i.
        std::vector<mlir::Operation*> lastCompOp(numResults, newOp.getOperation());

        // Step 1 - data-type compensation
        for (size_t idx = 0; idx < numResults; idx++) {
            auto originalResult = _extKernelOp->getResult(idx);
            if (_resultModificationTracker[idx].dataTypeConfigApplied) {
                mlir::OpBuilder::InsertionGuard guard(builder);
                builder.setInsertionPointAfter(lastCompOp[idx]);
                auto originalNDType = mlir::cast<vpux::NDTypeInterface>(originalResult.getType());
                auto locSuff = vpux::printToString("convert_back_dtype_{0}", idx);
                auto convertOp = IE::ConvertOp::create(builder, vpux::takeOpLoc(newOp, locSuff), _newResults[idx],
                                                       originalNDType.getElementType());
                _newResults[idx] = convertOp.getResult();
                lastCompOp[idx] = convertOp.getOperation();
            }
        }

        // Step 2 - layout compensation
        for (size_t idx = 0; idx < numResults; idx++) {
            if (_resultModificationTracker[idx].layoutConfigApplied) {
                mlir::OpBuilder::InsertionGuard guard(builder);
                builder.setInsertionPointAfter(lastCompOp[idx]);

                VPUX_THROW_UNLESS(_resultModificationTracker[idx].typePostRankConfig != nullptr,
                                  "Expected type pre-layout enforcement to be set for layout compensation");
                auto originalNDType = _resultModificationTracker[idx].typePostRankConfig;
                auto locSuff = vpux::printToString("reorder_back_layout_{0}", idx);
                auto reorderOp = IE::ReorderOp::create(builder, vpux::takeOpLoc(newOp, locSuff), _newResults[idx],
                                                       originalNDType.getDimsOrder().toAffineMap(builder.getContext()));
                _newResults[idx] = reorderOp.getResult();
                lastCompOp[idx] = reorderOp.getOperation();
            }
        }

        // Step 3 - rank (shape) compensation
        for (size_t idx = 0; idx < numResults; idx++) {
            auto originalResult = _extKernelOp->getResult(idx);
            if (_resultModificationTracker[idx].rankConfigApplied) {
                mlir::OpBuilder::InsertionGuard guard(builder);
                builder.setInsertionPointAfter(lastCompOp[idx]);
                auto originalNDType = mlir::cast<vpux::NDTypeInterface>(originalResult.getType());
                auto targetShapeAttr = vpux::getIntArrayAttr(builder.getContext(), originalNDType.getShape().raw());
                auto locSuff = vpux::printToString("reshape_back_shape_{0}", idx);
                auto reshapeOp = IE::ReshapeOp::create(builder, vpux::takeOpLoc(newOp, locSuff), _newResults[idx],
                                                       targetShapeAttr);
                _newResults[idx] = reshapeOp.getResult();
                lastCompOp[idx] = reshapeOp.getOperation();
            }
        }

        _extKernelOp->replaceAllUsesWith(_newResults);
        _extKernelOp->erase();
        return newOp.getOperation();
    }

    mlir::LogicalResult applyRankConfig(const std::vector<Shave::CustomLayerConfig::IOTensorConfig>& inputConfigs,
                                        const std::vector<Shave::CustomLayerConfig::IOTensorConfig>& outputConfigs) {
        mlir::OpBuilder builder(_extKernelOp);
        builder.setInsertionPoint(_extKernelOp);
        const auto& ctx = _extKernelOp->getContext();

        for (auto [idx, inputOperand, inputConfig] : llvm::enumerate(_newOperands, inputConfigs)) {
            if (!inputConfig.enforcedRank.has_value()) {
                continue;  // No enforced rank for this input, skip
            } else {
                auto enforcedRank = inputConfig.enforcedRank.value();
                auto operandNDType = mlir::cast<vpux::NDTypeInterface>(inputOperand.getType());
                auto operandRank = operandNDType.getRank();
                if (operandRank == static_cast<int64_t>(enforcedRank) || enforcedRank == Shave::TensorRank::_ND) {
                    continue;  // Rank already matches enforced rank, no action needed
                } else {
                    auto enforcedRankValue = static_cast<int64_t>(enforcedRank);
                    if (enforcedRankValue < operandRank) {
                        return _extKernelOp->emitError("Rank config violation: input operand rank (")
                               << operandRank << ") is greater than enforced rank (" << enforcedRankValue << ")";
                    }
                    auto inputShape = operandNDType.getShape();

                    VPUX_THROW_UNLESS(inputShape.isStatic(),
                                      "Only static shapes are supported now for External Kernels");
                    auto shapeVec = to_small_vector(inputShape);
                    auto shapeDiff = enforcedRankValue - operandRank;
                    for (int64_t i = 0; i < shapeDiff; i++) {
                        shapeVec.insert(shapeVec.begin(), 1);  // Insert leading dimensions of size 1
                    }
                    auto newShape = vpux::ShapeRef(shapeVec);
                    const auto targetShapeAttr = vpux::getIntArrayAttr(ctx, newShape.raw());
                    auto locSuff = vpux::printToString("reshape_{0}", idx);
                    auto newOperand = IE::ReshapeOp::create(builder, vpux::takeOpLoc(_extKernelOp, locSuff),
                                                            inputOperand, targetShapeAttr)
                                              .getResult();
                    _newOperands[idx] = newOperand;
                }
            }
        }

        for (auto [idx, outputResultType, outputConfig] : llvm::enumerate(_newResultTypes, outputConfigs)) {
            if (!outputConfig.enforcedRank.has_value()) {
                continue;  // No enforced rank for this output, skip
            } else {
                auto enforcedRank = outputConfig.enforcedRank.value();
                auto resNDType = mlir::cast<vpux::NDTypeInterface>(outputResultType);
                auto resRank = resNDType.getRank();
                if (resRank == static_cast<int64_t>(enforcedRank) || enforcedRank == Shave::TensorRank::_ND) {
                    continue;  // Rank already matches enforced rank, no action needed
                } else {
                    auto enforcedRankValue = static_cast<int64_t>(enforcedRank);
                    if (enforcedRankValue < resRank) {
                        return _extKernelOp->emitError("Rank config violation: output result rank (")
                               << resRank << ") is greater than enforced rank (" << enforcedRankValue << ")";
                    }
                    auto outputShape = resNDType.getShape();

                    VPUX_THROW_UNLESS(outputShape.isStatic(),
                                      "Only static shapes are supported now for External Kernels");
                    auto shapeVec = to_small_vector(outputShape);
                    auto shapeDiff = enforcedRankValue - resRank;
                    for (int64_t i = 0; i < shapeDiff; i++) {
                        shapeVec.insert(shapeVec.begin(), 1);  // Insert leading dimensions of size 1
                    }
                    auto newShape = vpux::ShapeRef(shapeVec);
                    const auto newType = resNDType.changeShape(newShape);
                    _newResultTypes[idx] = newType;
                    _resultModificationTracker[idx].rankConfigApplied = true;
                }
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult applyLayoutConfig(const std::vector<Shave::CustomLayerConfig::IOTensorConfig>& inputConfigs,
                                          const std::vector<Shave::CustomLayerConfig::IOTensorConfig>& outputConfigs) {
        mlir::OpBuilder builder(_extKernelOp);
        builder.setInsertionPoint(_extKernelOp);
        const auto& ctx = _extKernelOp->getContext();

        // Apply input layout configs
        for (auto [idx, inputOperand, inputConfig] : llvm::enumerate(_newOperands, inputConfigs)) {
            if (!inputConfig.enforcedLayout.has_value()) {
                continue;  // No enforced config for this tensor, skip
            } else {
                auto enforcedLayout = inputConfig.enforcedLayout.value();
                auto operandNDType = mlir::cast<vpux::NDTypeInterface>(inputOperand.getType());
                auto operandLayout = operandNDType.getDimsOrder().invertedCode();
                if (operandLayout == static_cast<uint64_t>(enforcedLayout) ||
                    enforcedLayout == Shave::TensorLayout::DEFAULT) {
                    continue;  // Value already matches enforced config, no action needed
                } else {
                    auto dummyLayout = vpux::DimsOrder::fromCode(static_cast<uint64_t>(enforcedLayout));

                    auto enforcedDimsOrder = vpux::DimsOrder::fromCode(dummyLayout.invertedCode());
                    auto inputOperandRank = static_cast<uint64_t>(operandNDType.getRank());
                    if (enforcedDimsOrder.numDims() != inputOperandRank) {
                        return _extKernelOp->emitError("Layout config violation: input operand rank (")
                               << inputOperandRank << ") is different from config layout rank ("
                               << enforcedDimsOrder.numDims() << ")";
                    }
                    auto locSuff = vpux::printToString("reorder_{0}", idx);
                    auto reorderOp = IE::ReorderOp::create(builder, vpux::takeOpLoc(_extKernelOp, locSuff),
                                                           inputOperand, enforcedDimsOrder.toAffineMap(ctx))
                                             .getResult();
                    _newOperands[idx] = reorderOp;
                }
            }
        }

        // Apply output layout configs
        for (auto [idx, outputResultType, outputConfig] : llvm::enumerate(_newResultTypes, outputConfigs)) {
            if (!outputConfig.enforcedLayout.has_value()) {
                continue;  // No enforced config for this tensor, skip
            } else {
                auto enforcedLayout = outputConfig.enforcedLayout.value();
                auto operandNDType = mlir::cast<vpux::NDTypeInterface>(outputResultType);
                auto operandLayout = operandNDType.getDimsOrder().invertedCode();
                if (operandLayout == static_cast<uint64_t>(enforcedLayout) ||
                    enforcedLayout == Shave::TensorLayout::DEFAULT) {
                    continue;  // Value already matches enforced config, no action needed
                } else {
                    auto dummyLayout = vpux::DimsOrder::fromCode(static_cast<uint64_t>(enforcedLayout));

                    auto enforcedDimsOrder = vpux::DimsOrder::fromCode(dummyLayout.invertedCode());
                    auto operandRank = static_cast<uint64_t>(operandNDType.getRank());
                    if (enforcedDimsOrder.numDims() != operandRank) {
                        return _extKernelOp->emitError("Layout config violation: output result rank (")
                               << operandRank << ") is different from config layout rank ("
                               << enforcedDimsOrder.numDims() << ")";
                    }

                    // Cache the type pre-layout change, but post any potential rank changes to ensure correct
                    // compensation later
                    _resultModificationTracker[idx].typePostRankConfig = operandNDType;

                    auto newType = operandNDType.changeDimsOrder(enforcedDimsOrder);
                    _newResultTypes[idx] = newType;
                    _resultModificationTracker[idx].layoutConfigApplied = true;
                }
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult applyDataTypeConfig(
            const std::vector<Shave::CustomLayerConfig::IOTensorConfig>& inputConfigs,
            const std::vector<Shave::CustomLayerConfig::IOTensorConfig>& outputConfigs) {
        mlir::OpBuilder builder(_extKernelOp);
        builder.setInsertionPoint(_extKernelOp);
        const auto& ctx = _extKernelOp->getContext();

        for (auto [idx, inputOperand, inputConfig] : llvm::enumerate(_newOperands, inputConfigs)) {
            if (!inputConfig.enforcedDataType.has_value()) {
                continue;  // No enforced config for this tensor, skip
            } else {
                auto enforcedDataType = inputConfig.enforcedDataType.value();
                auto operandNDType = mlir::cast<vpux::NDTypeInterface>(inputOperand.getType());
                auto operandDataType = operandNDType.getElementType();
                if (getDataTypeFromMlirType(operandDataType) == enforcedDataType ||
                    enforcedDataType == Shave::TensorDataType::UNDEFINED) {
                    continue;  // Value already matches enforced config, no action needed
                } else {
                    auto locSuff = vpux::printToString("convert_{0}", idx);
                    auto newOperand = IE::ConvertOp::create(builder, takeOpLoc(_extKernelOp, locSuff), inputOperand,
                                                            getMlirTypeFromDataType(enforcedDataType, ctx))
                                              .getResult();
                    _newOperands[idx] = newOperand;
                }
            }
        }

        for (auto [idx, outputResultType, outputConfig] : llvm::enumerate(_newResultTypes, outputConfigs)) {
            if (!outputConfig.enforcedDataType.has_value()) {
                continue;  // No enforced config for this tensor, skip
            } else {
                auto enforcedDataType = outputConfig.enforcedDataType.value();
                auto operandNDType = mlir::cast<vpux::NDTypeInterface>(outputResultType);
                auto operandDataType = operandNDType.getElementType();
                if (getDataTypeFromMlirType(operandDataType) == enforcedDataType ||
                    enforcedDataType == Shave::TensorDataType::UNDEFINED) {
                    continue;  // Value already matches enforced config, no action needed
                } else {
                    auto newType = operandNDType.changeElemType(getMlirTypeFromDataType(enforcedDataType, ctx));
                    _newResultTypes[idx] = newType;
                    _resultModificationTracker[idx].dataTypeConfigApplied = true;
                }
            }
        }
        return mlir::success();
    }
};

// Helper: split IOTensorConfigs into input/output vectors sorted by argIndex.
// Used by both validateTensorConfig and enforceTensorIoConfig.
static std::pair<std::vector<Shave::CustomLayerConfig::IOTensorConfig>,
                 std::vector<Shave::CustomLayerConfig::IOTensorConfig>>
splitAndSortIOConfigs(const Shave::CustomLayerConfig::IOConfig& ioConfig) {
    std::vector<Shave::CustomLayerConfig::IOTensorConfig> inputConfigs;
    std::vector<Shave::CustomLayerConfig::IOTensorConfig> outputConfigs;
    std::copy_if(ioConfig.ioTensorConfigs.begin(), ioConfig.ioTensorConfigs.end(), std::back_inserter(inputConfigs),
                 [](const auto& c) {
                     return c.ioType == Shave::IOType::Input;
                 });
    std::copy_if(ioConfig.ioTensorConfigs.begin(), ioConfig.ioTensorConfigs.end(), std::back_inserter(outputConfigs),
                 [](const auto& c) {
                     return c.ioType == Shave::IOType::Output;
                 });
    auto byArgIndex = [](const auto& a, const auto& b) {
        return a.argIndex < b.argIndex;
    };
    std::sort(inputConfigs.begin(), inputConfigs.end(), byArgIndex);
    std::sort(outputConfigs.begin(), outputConfigs.end(), byArgIndex);
    return {std::move(inputConfigs), std::move(outputConfigs)};
}

mlir::LogicalResult validateTensorConfig(Shave::ExternalKernelOp extKernelOp,
                                         const Shave::CustomLayerConfig::IOConfig& tensorConfig) {
    auto totalConfigIoCount = tensorConfig.ioTensorConfigs.size();

    // Count only regular $inputs, not $scratch_buffers (added later in VPU pipeline)
    auto opInputCount = extKernelOp.getInputs().size();
    auto opOutputCount = extKernelOp.getNumResults();
    auto opTotalIoCount = opInputCount + opOutputCount;

    if (totalConfigIoCount != opTotalIoCount) {
        return extKernelOp.emitError("IO Config mismatch: total IO configs provided (")
               << totalConfigIoCount << ") does not match total number of inputs and outputs (" << opTotalIoCount
               << ")";
    }

    auto [inputConfigs, outputConfigs] = splitAndSortIOConfigs(tensorConfig);

    auto configInputCount = inputConfigs.size();
    auto configOutputCount = outputConfigs.size();

    if (configInputCount != opInputCount) {
        return extKernelOp.emitError("IO Config mismatch: number of input configs provided (")
               << configInputCount << ") does not match number of inputs (" << opInputCount << ")";
    }
    if (configOutputCount != opOutputCount) {
        return extKernelOp.emitError("IO Config mismatch: number of output configs provided (")
               << configOutputCount << ") does not match number of outputs (" << opOutputCount << ")";
    }

    // Lambda to validate a single tensor config against its corresponding value
    auto validateSingleTensorConfig = [&extKernelOp](const vpux::NDTypeInterface& tensorType,
                                                     const Shave::CustomLayerConfig::IOTensorConfig& config, size_t idx,
                                                     size_t expectedArgIndex,
                                                     const char* tensorKind) -> mlir::LogicalResult {
        if (config.argIndex != expectedArgIndex) {
            return extKernelOp.emitError("IO Config mismatch: ")
                   << tensorKind << " config arg_index (" << config.argIndex << ") does not match " << tensorKind
                   << " index (" << expectedArgIndex << ")";
        }

        // Check data type: use the enforced value when set, otherwise use the actual tensor type.
        // If supported-data-types is non-empty, the chosen value must appear in the list.
        if (!config.supportedDataTypes.empty()) {
            auto tensorDType = config.enforcedDataType.has_value()
                                       ? config.enforcedDataType.value()
                                       : getDataTypeFromMlirType(tensorType.getElementType());
            if (std::find(config.supportedDataTypes.begin(), config.supportedDataTypes.end(), tensorDType) ==
                config.supportedDataTypes.end()) {
                return extKernelOp.emitError("IO Config mismatch: ")
                       << tensorKind << " config data type (" << static_cast<uint64_t>(tensorDType)
                       << ") is not supported for " << tensorKind << " " << idx;
            }
        }

        // Check Rank: same pattern — prefer the enforced value when set.
        if (!config.supportedRanks.empty()) {
            auto enumRank = config.enforcedRank.has_value() ? config.enforcedRank.value()
                                                            : static_cast<Shave::TensorRank>(tensorType.getRank());
            if (std::find(config.supportedRanks.begin(), config.supportedRanks.end(), enumRank) ==
                config.supportedRanks.end()) {
                return extKernelOp.emitError("IO Config mismatch: ")
                       << tensorKind << " config rank (" << static_cast<uint64_t>(enumRank) << ") is not supported for "
                       << tensorKind << " " << idx;
            }
        }

        return mlir::success();
    };

    // Validate input configs against operands (use getInputs(), not getOperands(), to exclude
    // scratch_buffers which share the same operand list via AttrSizedOperandSegments)
    for (auto [idx, inputOperand, inputConfig] : llvm::enumerate(extKernelOp.getInputs(), inputConfigs)) {
        auto operandNDType = mlir::cast<vpux::NDTypeInterface>(inputOperand.getType());
        if (mlir::failed(validateSingleTensorConfig(operandNDType, inputConfig, idx, idx, "input"))) {
            return mlir::failure();
        }
    }

    // Validate output configs against results.
    // Output arg-index values are independently 0-based (0, 1, 2, ...),
    // separate from the input arg-index space.
    for (auto [idx, outputResult, outputConfig] : llvm::enumerate(extKernelOp->getResults(), outputConfigs)) {
        auto resultNDType = mlir::cast<vpux::NDTypeInterface>(outputResult.getType());
        if (mlir::failed(validateSingleTensorConfig(resultNDType, outputConfig, idx, idx, "output"))) {
            return mlir::failure();
        }
    }

    return mlir::success();
}

//
// Apply IO configs
//

mlir::FailureOr<Shave::ExternalKernelOp> enforceTensorIoConfig(Shave::ExternalKernelOp extKernelOp,
                                                               const Shave::CustomLayerConfig::IOConfig& tensorConfig) {
    //
    // Handle IO preference configs
    //

    auto [inputConfigs, outputConfigs] = splitAndSortIOConfigs(tensorConfig);

    ConfigApplicator applicator(extKernelOp);
    if (mlir::failed(applicator.applyIOConfig(inputConfigs, outputConfigs))) {
        return mlir::failure();
    }

    auto updatedOp = applicator.finalize();
    return mlir::cast<Shave::ExternalKernelOp>(updatedOp);
}

class ApplyExternalKernelConfigs final :
        public Shave::impl::ApplyExternalKernelConfigsBase<ApplyExternalKernelConfigs> {
public:
    explicit ApplyExternalKernelConfigs(Logger log): _log(std::move(log)) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() override;
    mlir::FailureOr<Shave::ExternalKernelOp> applyConfigs(Shave::ExternalKernelOp extKernelOp,
                                                          Shave::CustomOpConfigOp configOp);

    mlir::FailureOr<Shave::ExternalKernelOp> applyIOConfig(Shave::ExternalKernelOp extKernelOp,
                                                           Shave::CustomOpIOConfigOp ioConfigOp);
    mlir::FailureOr<Shave::ExternalKernelOp> applyKernelConfig(Shave::ExternalKernelOp extKernelOp,
                                                               Shave::CustomOpKernelConfigOp kernelConfigOp);

private:
    Logger _log;
};

mlir::FailureOr<Shave::ExternalKernelOp> ApplyExternalKernelConfigs::applyConfigs(Shave::ExternalKernelOp extKernelOp,
                                                                                  Shave::CustomOpConfigOp configOp) {
    _log.info("Applying configuration for External Kernel: {0}", extKernelOp.getUniqueId());

    // Apply IO Config (optional)
    auto ioConfigOps = configOp.getOps<Shave::CustomOpIOConfigOp>();
    VPUX_THROW_UNLESS(ioConfigOps.empty() || std::distance(ioConfigOps.begin(), ioConfigOps.end()) == 1,
                      "Multiple IO configurations found for ExternalKernel {0}", extKernelOp.getUniqueId());
    if (!ioConfigOps.empty()) {
        auto r = applyIOConfig(extKernelOp, *ioConfigOps.begin());
        if (mlir::failed(r)) {
            return mlir::failure();
        }
        extKernelOp = *r;
    }

    // Apply Kernel Config (optional)
    auto kernelConfigOps = configOp.getOps<Shave::CustomOpKernelConfigOp>();
    VPUX_THROW_UNLESS(kernelConfigOps.empty() || std::distance(kernelConfigOps.begin(), kernelConfigOps.end()) == 1,
                      "Multiple Kernel configurations found for ExternalKernel {0}", extKernelOp.getUniqueId());
    if (!kernelConfigOps.empty()) {
        auto r = applyKernelConfig(extKernelOp, *kernelConfigOps.begin());
        if (mlir::failed(r)) {
            return mlir::failure();
        }
        extKernelOp = *r;
    }

    return extKernelOp;
}

//
// IO Config Validation and Application
//

mlir::FailureOr<Shave::ExternalKernelOp> ApplyExternalKernelConfigs::applyIOConfig(
        Shave::ExternalKernelOp extKernelOp, Shave::CustomOpIOConfigOp ioConfigOp) {
    _log.debug("Applying IO configuration for External Kernel: {0}", extKernelOp.getUniqueId());
    auto ioConfig = Shave::parseIOConfig(ioConfigOp);

    _log.debug("Step 1: Validate IO Tensor Configs");
    if (mlir::failed(validateTensorConfig(extKernelOp, ioConfig))) {
        return mlir::failure();
    }

    _log.debug("Step 2: Apply IO preference configs");
    auto enforceResult = enforceTensorIoConfig(extKernelOp, ioConfig);
    if (mlir::failed(enforceResult)) {
        return mlir::failure();
    }
    extKernelOp = *enforceResult;

    // NOTE: Scratch buffer materialization (VPU::EmptyOp -> scratch_buffers operand)
    // is intentionally deferred to Shave::MaterializeScratchBuffers, which runs at
    // the start of the VPU pipeline after type/layout enforcement is complete here.

    _log.debug("Configs for External Kernel {0} were validated and applied successfully", extKernelOp->getLoc());
    return extKernelOp;
}

mlir::FailureOr<Shave::ExternalKernelOp> ApplyExternalKernelConfigs::applyKernelConfig(
        Shave::ExternalKernelOp extKernelOp, Shave::CustomOpKernelConfigOp kernelConfigOp) {
    _log.debug("Applying Kernel configuration for External Kernel: {0}", extKernelOp.getUniqueId());

    // Propagate uses-dma and uses-dpu flags as UnitAttrs on ExternalKernelOp.
    // These are carried through the VPU pipeline and into VPUIP::SwKernelOp
    // by the bufferization pass (bufferizeSWLayerOp), enabling DMA/DPU-specific
    // compiler paths: LegalizeShaveSubmitDMAs, cache handling bypass, barrier
    // pages split, WLM graph split, etc.
    if (kernelConfigOp.getUsesDmaAttr()) {
        extKernelOp.setUsesDmaAttr(mlir::UnitAttr::get(extKernelOp->getContext()));
    }
    if (kernelConfigOp.getUsesDpuAttr()) {
        extKernelOp.setUsesDpuAttr(mlir::UnitAttr::get(extKernelOp->getContext()));
    }
    return extKernelOp;
}

void ApplyExternalKernelConfigs::safeRunOnFunc() {
    auto func = getOperation();

    auto extKernelOps = func.getOps<Shave::ExternalKernelOp>();
    if (extKernelOps.empty()) {
        _log.debug("No ExternalKernels encountered - applying external kernel configs can be skipped.");
        return;
    }

    // Collect (op, configOp) pairs before modifying IR to avoid iterator
    // invalidation caused by op replacement inside applyConfigs.
    using WorkItem = std::pair<Shave::ExternalKernelOp, Shave::CustomOpConfigOp>;
    std::vector<WorkItem> workList;
    for (auto extKernelOp : extKernelOps) {
        auto configOp = extKernelOp.getCustomOpConfig();
        if (!configOp) {
            _log.debug("No config link for ExternalKernelOp {0} - leaving it unchanged", extKernelOp.getUniqueId());
            continue;
        }
        workList.emplace_back(extKernelOp, configOp);
    }

    for (auto [extKernelOp, configOp] : workList) {
        if (mlir::failed(applyConfigs(extKernelOp, configOp))) {
            signalPassFailure();
            return;
        }
    }
}

}  // namespace

//
// createApplyExternalKernelConfigsPass
//
std::unique_ptr<mlir::Pass> vpux::Shave::createApplyExternalKernelConfigsPass(Logger log) {
    return std::make_unique<ApplyExternalKernelConfigs>(log);
}
