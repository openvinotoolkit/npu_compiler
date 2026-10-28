//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include <vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp>
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_legalization.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/utils/core/range.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Support/LLVM.h>
#include <algorithm>
#include <cmath>

using namespace vpux;

namespace {
mlir::DictionaryAttr mergeClampAttrs(mlir::DictionaryAttr currentClampAttr, IE::ClampOp clampOp) {
    auto maxAttrName = clampOp.getMaxAttrName();
    auto minAttrName = clampOp.getMinAttrName();

    assert(currentClampAttr.contains(maxAttrName) && currentClampAttr.contains(minAttrName) &&
           "The current clamp attribute is expected contain min and max values.");

    auto currentClampMin = mlir::cast<mlir::FloatAttr>(currentClampAttr.get(minAttrName)).getValueAsDouble();
    auto currentClampMax = mlir::cast<mlir::FloatAttr>(currentClampAttr.get(maxAttrName)).getValueAsDouble();

    const auto minClampOp = clampOp.getMinAttr().getValueAsDouble();
    const auto maxClampOp = clampOp.getMaxAttr().getValueAsDouble();

    const auto newMin = std::max(currentClampMin, minClampOp);
    const auto newMax = std::min(currentClampMax, maxClampOp);
    const auto newMinAttr = getFPAttr(clampOp.getContext(), newMin);
    const auto newMaxAttr = getFPAttr(clampOp.getContext(), newMax);

    SmallVector<mlir::NamedAttribute> newClampAttr;
    newClampAttr.emplace_back(maxAttrName, newMaxAttr);
    newClampAttr.emplace_back(minAttrName, newMinAttr);

    return mlir::DictionaryAttr::get(clampOp.getContext(), newClampAttr);
}

constexpr auto activationRelu = [](double val) {
    return std::max(val, 0.0);
};
constexpr auto activationTanh = [](double val) {
    return std::tanh(val);
};
constexpr auto activationSigmoid = [](double val) {
    return 1.0 / (1.0 + std::exp(-val));
};
constexpr auto activationExp = [](double val) {
    return std::exp(val);
};

}  // namespace

//
// LayerOpInterface
//

mlir::LogicalResult vpux::IE::verifyLayer(mlir::Operation* op) {
    if (op->getOperands().empty()) {
        return errorAt(op, "Layer Operation has no operands");
    }
    if (op->getResults().empty()) {
        return errorAt(op, "Layer Operation has no results");
    }

    const auto verifyType = [&](mlir::Type type, StringRef name, unsigned ind) {
        if (mlir::isa<mlir::MemRefType>(type)) {
            return errorAt(op, "Layer Operation has MemRef {0} #{1}", name, ind);
        }

        if (auto mainType = mlir::dyn_cast<vpux::NDTypeInterface>(type)) {
            if (validateQuantElemType(op->getLoc(), mainType).failed()) {
                return mlir::failure();
            }
        }

        return mlir::success();
    };

    for (auto& arg : op->getOpOperands()) {
        if (verifyType(arg.get().getType(), "operand", arg.getOperandNumber()).failed()) {
            return mlir::failure();
        }
    }
    for (auto res : op->getOpResults()) {
        if (verifyType(res.getType(), "result", res.getResultNumber()).failed()) {
            return mlir::failure();
        }
    }

    return mlir::success();
}

bool vpux::IE::hasToVerifyShape(mlir::Operation* op) {
    if (op == nullptr) {
        return false;
    }

    const auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (moduleOp == nullptr) {
        return false;
    }

    const auto verificationEnabledAttr = moduleOp->getAttr(IE::SHAPE_VERIFICATION_ENABLED);
    if (verificationEnabledAttr == nullptr) {
        return false;
    }

    return mlir::isa<mlir::UnitAttr>(verificationEnabledAttr);
}

//
// ShapeInfoOpInterface
//

mlir::LogicalResult vpux::IE::verifyInputIs4D(mlir::Value input, bool allowNull) {
    VPUX_THROW_WHEN(input == nullptr && !allowNull, "Input value shouldn't be null.");
    if (input == nullptr) {
        return mlir::success();
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    if (inputType.getRank() != 4) {
        return errorAt(input.getLoc(), "Expected 4D input, got rank {0}", inputType.getRank());
    }

    return mlir::success();
}

mlir::LogicalResult vpux::IE::verifyShapeInfo(mlir::Operation* op) {
    if (!hasToVerifyShape(op)) {
        return mlir::success();
    }

    auto shapeInfoOp = mlir::cast<IE::ShapeInfoOpInterface>(op);
    if (mlir::failed(shapeInfoOp.verifyShapeInfo())) {
        return mlir::failure();
    }

    for (const auto result : op->getResults()) {
        const auto resultType = mlir::cast<vpux::NDTypeInterface>(result.getType());
        if (resultType.getRank() != 4) {
            return errorAt(op, "ShapeInfoOpInterface requires NDTypeInterface results with rank 4");
        }
    }

    return mlir::success();
}

mlir::LogicalResult vpux::IE::inferTensorTypes(InferTypeComponentsCb componentsCb, mlir::MLIRContext* ctx,
                                               std::optional<mlir::Location> loc, mlir::ValueRange operands,
                                               mlir::DictionaryAttr attrs, mlir::OpaqueProperties props,
                                               mlir::RegionRange regions, SmallVectorImpl<mlir::Type>& inferredTypes) {
    SmallVector<mlir::ShapedTypeComponents> components;
    if (mlir::failed(componentsCb(ctx, loc, operands, attrs, props, regions, components))) {
        return mlir::failure();
    }

    for (const auto& desc : components) {
        VPUX_THROW_UNLESS(desc.hasRank(), "Unranked TensorType is not supported");

        const auto type = mlir::RankedTensorType::get(desc.getDims(), desc.getElementType(), desc.getAttribute());
        inferredTypes.push_back(type);
    }

    return mlir::success();
}

//
// LayerWithPostOpInterface
//

IE::PostOpAttr vpux::IE::attributizePostOp(mlir::Operation* postOp) {
    if (postOp == nullptr) {
        return nullptr;
    }

    return llvm::TypeSwitch<mlir::Operation*, IE::PostOpAttr>(postOp)
            .Case<IE::ReLUOp>([](auto reluOp) {
                return IE::ReluAttr::get(reluOp.getContext());
            })
            .Case<IE::LeakyReluOp>([](auto leakyReluOp) {
                return IE::LeakyReluAttr::get(leakyReluOp.getContext(), leakyReluOp.getNegativeSlopeAttr());
            })
            .Case<IE::PReluOp>([](auto pReluOp) -> IE::PostOpAttr {
                const auto ctx = pReluOp.getContext();
                const auto slopesConst = pReluOp.getNegativeSlope().template getDefiningOp<Const::DeclareOp>();
                if (slopesConst == nullptr) {
                    return nullptr;
                }

                const auto slopesContent = slopesConst.getContentAttr().fold();
                if (slopesContent.isSplat()) {
                    const auto slope = slopesContent.template getSplatValue<double>();
                    return IE::LeakyReluAttr::get(ctx, getFPAttr(ctx, slope));
                }

                const auto slopes = slopesContent.template getValues<double>();
                return IE::PReluAttr::get(ctx, getFPArrayAttr(ctx, slopes));
            })
            .Case<IE::TanhOp>([](auto tanhOp) {
                return IE::TanhAttr::get(tanhOp.getContext());
            })
            .Case<IE::SigmoidOp>([](auto sigmoidOp) {
                return IE::SigmoidAttr::get(sigmoidOp.getContext());
            })
            .Case<IE::SwishOp>([](auto swishOp) -> IE::PostOpAttr {
                const auto ctx = swishOp.getContext();
                auto beta = swishOp.getBetaValueAttr();
                if (beta == nullptr) {
                    if (swishOp.getBeta() != nullptr) {
                        // Non-constant beta, see the SwishOp canonicalizer
                        return nullptr;
                    }
                    beta = getFPAttr(ctx, 1.0);
                }
                return IE::SwishAttr::get(ctx, beta);
            })
            .Case<IE::GeluOp>([](auto geluOp) {
                return IE::GeluAttr::get(geluOp.getContext());
            })
            .Case<IE::ExpOp>([](auto expOp) {
                return IE::ExpAttr::get(expOp.getContext());
            })
            .Case<IE::HSwishOp>([](auto hswishOp) {
                return IE::HSwishAttr::get(hswishOp.getContext());
            })
            .Default([](auto) -> IE::PostOpAttr {
                return nullptr;
            });
}

bool vpux::IE::isMonotonicActivationFunctionNonDecreasing(mlir::Operation* postOp) {
    return isMonotonicActivationFunctionNonDecreasing(attributizePostOp(postOp));
}

bool vpux::IE::isMonotonicActivationFunctionNonDecreasing(IE::PostOpAttr postOpAttr) {
    // LeakyRelu is only monotonically increasing if slope is non-negative
    if (auto attr = mlir::dyn_cast<IE::LeakyReluAttr>(postOpAttr)) {
        const auto slope = attr.getNegativeSlope().getValueAsDouble();
        return slope >= 0.0;
    }

    return mlir::isa_and_nonnull<IE::ReluAttr, IE::TanhAttr, IE::SigmoidAttr, IE::ExpAttr>(postOpAttr);
}

bool vpux::IE::applyMonotonicActivationFunction(IE::PostOpAttr postOpAttr, IE::ActivationFunctionCb callback) {
    if (postOpAttr == nullptr) {
        return false;
    }

    return llvm::TypeSwitch<mlir::Attribute, bool>(postOpAttr)
            .Case<IE::ReluAttr>([&](IE::ReluAttr) {
                callback(activationRelu);
                return true;
            })
            .Case<IE::LeakyReluAttr>([&](IE::LeakyReluAttr attr) {
                const auto slope = attr.getNegativeSlope().getValueAsDouble();
                if (slope < 0.0) {
                    // A negative slope makes LeakyReLU non-monotonic
                    return false;
                }
                auto activationLeakyRelu = [slope](double val) {
                    return val < 0.0 ? val * slope : val;
                };
                callback(activationLeakyRelu);
                return true;
            })
            .Case<IE::TanhAttr>([&](IE::TanhAttr) {
                callback(activationTanh);
                return true;
            })
            .Case<IE::SigmoidAttr>([&](IE::SigmoidAttr) {
                callback(activationSigmoid);
                return true;
            })
            .Case<IE::ExpAttr>([&](IE::ExpAttr) {
                callback(activationExp);
                return true;
            })
            .Default([](mlir::Attribute) {
                // Swish, Gelu, HSwish and per-channel PRelu are not monotonically non-decreasing
                return false;
            });
}

bool vpux::IE::applyMonotonicActivationFunction(mlir::Operation* postOp, IE::ActivationFunctionCb callback) {
    if (postOp == nullptr) {
        return false;
    }

    auto postOpAttr = attributizePostOp(postOp);
    if (!isMonotonicActivationFunctionNonDecreasing(postOpAttr)) {
        return false;
    }

    return applyMonotonicActivationFunction(postOpAttr, callback);
}

mlir::DictionaryAttr vpux::IE::remapClampThroughActivation(mlir::DictionaryAttr clampAttr, IE::PostOpAttr postOpAttr) {
    if (clampAttr == nullptr) {
        return nullptr;
    }

    if (postOpAttr == nullptr) {
        return clampAttr;
    }

    mlir::DictionaryAttr result = nullptr;
    auto applied = applyMonotonicActivationFunction(postOpAttr, [&](IE::ActivationFunction activationFunc) {
        const auto min = activationFunc(clampAttr.getAs<mlir::FloatAttr>("min").getValueAsDouble());
        const auto max = activationFunc(clampAttr.getAs<mlir::FloatAttr>("max").getValueAsDouble());
        result = createClampAttr(clampAttr.getContext(), min, max);
    });

    return applied ? result : nullptr;
}

void vpux::IE::setClampOp(mlir::Operation* mainOp, mlir::Operation* maybeClampOp) {
    auto clampOp = mlir::dyn_cast<IE::ClampOp>(maybeClampOp);
    VPUX_THROW_WHEN(clampOp == nullptr, "Not ClampOp provided at {0}", maybeClampOp->getLoc());

    mlir::DictionaryAttr clampOpInfo;
    if (auto mainClampAttr =
                mlir::dyn_cast_or_null<mlir::DictionaryAttr>(mainOp->getAttr(vpux::OperationAttrName::CLAMP))) {
        clampOpInfo = mergeClampAttrs(mainClampAttr, clampOp);
    } else {
        clampOpInfo = clampOp->getAttrDictionary();
    }
    mainOp->setAttr(vpux::OperationAttrName::CLAMP, clampOpInfo);
}

mlir::DictionaryAttr vpux::IE::createClampAttr(mlir::MLIRContext* ctx, double min, double max) {
    const auto minAttr = mlir::NamedAttribute(mlir::StringAttr::get(ctx, "min"), getFPAttr(ctx, min));
    const auto maxAttr = mlir::NamedAttribute(mlir::StringAttr::get(ctx, "max"), getFPAttr(ctx, max));
    return mlir::DictionaryAttr::get(ctx, {minAttr, maxAttr});
}

std::optional<mlir::DictionaryAttr> vpux::IE::getFusedClamp(const Logger& log, mlir::DictionaryAttr producerClamp,
                                                            IE::PostOpAttr consumerPostOp,
                                                            mlir::DictionaryAttr consumerClamp,
                                                            IE::LayerWithPostOpInterface fusedOp, mlir::Type type) {
    auto fusedClamp = producerClamp;
    if (consumerPostOp != nullptr && fusedClamp != nullptr) {
        fusedClamp = IE::remapClampThroughActivation(fusedClamp, consumerPostOp);
        if (fusedClamp == nullptr) {
            log.nest().trace("Producer clamp cannot be moved past the consumer post-op '{0}'", consumerPostOp);
            return std::nullopt;
        }
    }

    if (consumerClamp != nullptr) {
        auto min = consumerClamp.getAs<mlir::FloatAttr>("min").getValueAsDouble();
        auto max = consumerClamp.getAs<mlir::FloatAttr>("max").getValueAsDouble();
        if (fusedClamp != nullptr) {
            min = std::max(min, fusedClamp.getAs<mlir::FloatAttr>("min").getValueAsDouble());
            max = std::min(max, fusedClamp.getAs<mlir::FloatAttr>("max").getValueAsDouble());
        }

        fusedClamp = IE::createClampAttr(consumerClamp.getContext(), min, max);
    }

    // If producerClamp and consumerClamp are nullptr then there is no clamp to fuse
    if (fusedClamp == nullptr) {
        return nullptr;
    }

    const auto logCb = [&](const formatv_object_base& msg) {
        log.trace("{0}", msg.str());
    };
    auto min = fusedClamp.getAs<mlir::FloatAttr>("min").getValueAsDouble();
    auto max = fusedClamp.getAs<mlir::FloatAttr>("max").getValueAsDouble();
    if (!fusedOp.isSupportedClampProperties(min, max, type, logCb)) {
        log.nest().trace("Operation at '{0}' does not support the clamp range [{1}, {2}]", fusedOp.getLoc(), min, max);
        return std::nullopt;
    }

    return fusedClamp;
}

//
// LayoutInfoOpInterface
//

void vpux::IE::LayerLayoutInfo::setInput(size_t ind, const DimsOrder& info) {
    const auto prevInfo = getInput(ind);
    VPUX_THROW_UNLESS(info.numDims() == prevInfo.numDims(), "New order '{0}' doesn't match original rank '{1}'", info,
                      prevInfo.numDims());

    LayerDataInfo<DimsOrder>::setInput(ind, info);
}

void vpux::IE::LayerLayoutInfo::setOutput(size_t ind, const DimsOrder& info) {
    const auto prevInfo = getOutput(ind);
    VPUX_THROW_UNLESS(info.numDims() == prevInfo.numDims(), "New order '{0}' doesn't match original rank '{1}'", info,
                      prevInfo.numDims());

    LayerDataInfo<DimsOrder>::setOutput(ind, info);
}

mlir::LogicalResult vpux::IE::verifyLayout(mlir::Operation*) {
    // Tracking number [E#84955]
    return mlir::success();
}

IE::LayerLayoutInfo vpux::IE::getLayoutInfo(mlir::Operation* op) {
    SmallVector<DimsOrder> inputOrders;
    inputOrders.reserve(op->getNumOperands());
    for (const auto& val : op->getOperands()) {
        inputOrders.push_back(DimsOrder::fromValue(val));
    }

    SmallVector<DimsOrder> outputOrders;
    outputOrders.reserve(op->getNumResults());
    for (const auto& val : op->getResults()) {
        outputOrders.push_back(DimsOrder::fromValue(val));
    }

    return IE::LayerLayoutInfo(std::move(inputOrders), std::move(outputOrders));
}

void vpux::IE::fillDefaultLayoutInfo(IE::LayerLayoutInfo& info) {
    for (auto i : irange(info.getNumInputs())) {
        info.setInput(i, DimsOrder::fromNumDims(info.getInput(i).numDims()));
    }

    for (auto i : irange(info.getNumOutputs())) {
        info.setOutput(i, DimsOrder::fromNumDims(info.getOutput(i).numDims()));
    }
}

void vpux::IE::fillDefaultLayoutInfo(LayerLayoutInfo& info, FuncRef<bool(size_t)> inputFilter,
                                     FuncRef<bool(size_t)> outputFilter) {
    for (auto i : irange(info.getNumInputs()) | filtered(inputFilter)) {
        info.setInput(i, DimsOrder::fromNumDims(info.getInput(i).numDims()));
    }

    for (auto i : irange(info.getNumOutputs()) | filtered(outputFilter)) {
        info.setOutput(i, DimsOrder::fromNumDims(info.getOutput(i).numDims()));
    }
}

//
// EltwiseOp
//

mlir::LogicalResult vpux::IE::verifyEltwiseOp(mlir::Operation* op) {
    if (!mlir::isa<IE::LayerOpInterface>(op)) {
        return errorAt(op, "EltwiseOp trait is applied to non layer operation");
    }

    if (op->getNumResults() != 1) {
        return errorAt(op, "Operation with multiple results can't be EltwiseOp");
    }

    if (op->hasAttr("auto_broadcast")) {
        auto autoBroadcast = mlir::dyn_cast<vpux::IE::AutoBroadcastTypeAttr>(op->getAttr("auto_broadcast"));
        if (autoBroadcast == nullptr) {
            return errorAt(op, "Auto broadcast attribute cannot be cast");
        }
        auto broadcast = autoBroadcast.getValue();

        SmallVector<ArrayRef<int64_t>> inputShapes;
        for (auto operand : op->getOperands()) {
            const auto shape = mlir::cast<vpux::NDTypeInterface>(operand.getType()).getShape().raw();
            inputShapes.push_back(shape);
        }

        const auto outputShape = IE::broadcastEltwiseShape(inputShapes, broadcast, op->getLoc());

        if (mlir::failed(outputShape)) {
            return errorAt(op, "Eltwise inputs cannot be broadcast");
        }
    }

    return mlir::success();
}

vpux::IE::LayerDataInfo<mlir::Type> vpux::IE::getElemTypeInfo(mlir::Operation* op) {
    SmallVector<mlir::Type> inputTypes;
    inputTypes.reserve(op->getNumOperands());
    for (const auto& val : op->getOperands()) {
        inputTypes.push_back(mlir::cast<vpux::NDTypeInterface>(val.getType()).getElementType());
    }

    SmallVector<mlir::Type> outputTypes;
    outputTypes.reserve(op->getNumResults());
    for (const auto& val : op->getResults()) {
        outputTypes.push_back(mlir::cast<vpux::NDTypeInterface>(val.getType()).getElementType());
    }

    return vpux::IE::LayerDataInfo<mlir::Type>(std::move(inputTypes), std::move(outputTypes));
}

//
// isPureViewLike
//

bool vpux::IE::isPureViewOp(mlir::Operation* op) {
    return mlir::isa<IE::ViewLikeOpInterface, mlir::ViewLikeOpInterface>(op);
}

//
// Generated
//

#include <vpux/compiler/dialect/IE/ops_interfaces.cpp.inc>
