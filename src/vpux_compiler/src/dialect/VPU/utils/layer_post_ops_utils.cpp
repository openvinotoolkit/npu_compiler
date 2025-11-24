//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/layer_post_ops_utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux {
namespace VPU {

bool checkForQuantization(mlir::Operation* op, mlir::Operation* postOp) {
    auto isFakeQuantizeOpInput = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(op->getOperand(0).getDefiningOp());
    auto isFakeQuantizeOpOutput = true;
    for (auto user : postOp->getUsers()) {
        if (!mlir::isa<IE::FakeQuantizeOp>(user)) {
            isFakeQuantizeOpOutput = false;
            break;
        }
    }

    // since FusePostOps is called also after LowPrecisionPipeline
    const auto operandType = mlir::cast<vpux::NDTypeInterface>(postOp->getOperand(0).getType());
    const auto isQuantizedElemType = mlir::isa<mlir::quant::QuantizedType>(operandType.getElementType());

    return (isFakeQuantizeOpOutput && isFakeQuantizeOpInput) || isQuantizedElemType;
};

bool hasPerChannelQuantizedOutput(mlir::Operation* op) {
    for (auto user : op->getUsers()) {
        auto fq = mlir::dyn_cast<IE::FakeQuantizeOp>(user);
        if (fq == nullptr) {
            continue;
        }

        auto inLow = fq.getInputLow().getDefiningOp<Const::DeclareOp>();
        auto inHigh = fq.getInputHigh().getDefiningOp<Const::DeclareOp>();
        auto outLow = fq.getOutputLow().getDefiningOp<Const::DeclareOp>();
        auto outHigh = fq.getOutputHigh().getDefiningOp<Const::DeclareOp>();
        VPUX_THROW_WHEN(inLow == nullptr || inHigh == nullptr || outLow == nullptr || outHigh == nullptr,
                        "Got FakeQuantize with non-constant parameters, loc: {0}", fq->getLoc());

        if (!inLow.getContentAttr().isSplat() || !inHigh.getContentAttr().isSplat() ||
            !outLow.getContentAttr().isSplat() || !outHigh.getContentAttr().isSplat()) {
            return true;
        }
    }

    return false;
};

bool isSupportedHWClampOp(mlir::Operation* mainOp, mlir::Operation* clampOp, const LogCb& logCb) {
    if (auto clamp = mlir::dyn_cast<IE::ClampOp>(clampOp)) {
        const auto minVal = clamp.getMinAttr().getValueAsDouble();
        const auto isQuantized = vpux::VPU::checkForQuantization(mainOp, clampOp);
        if (!isDoubleEqual(minVal, 0.0) && !isQuantized) {
            logCb(llvm::formatv("Float {0} at `{1}` doesn't support non-zero clamp min", clampOp->getName(),
                                clampOp->getLoc()));
            return false;
        }
        // Disable MaxPool fused with Clamp since it is not fully supported by firmware.
        // Tracking Number: E#-145636
        if (mlir::isa<IE::MaxPoolOp>(mainOp)) {
            const auto maxVal = clamp.getMaxAttr().getValueAsDouble();
            const auto maxValueFP16 = checked_cast<double>(std::numeric_limits<vpux::type::float16>::max());
            // Given upper bound as fp16 max value, keep fusing Clamp into MaxPool to pass CI
            // Tracking Number: E#-146652
            if ((!isDoubleEqual(maxVal, maxValueFP16))) {
                logCb(llvm::formatv("{0} at `{1}` cannot be fused into MaxPool due to lack of firmware support",
                                    clampOp->getName(), clampOp->getLoc()));
                return false;
            }
        }
        return true;
    }
    logCb(llvm::formatv("{0} at `{1}` is not clamp op", clampOp->getName(), clampOp->getLoc()));
    return false;
}

mlir::DictionaryAttr mergeClampAttrs(mlir::DictionaryAttr currentClampAttr, IE::ClampOp clampOp) {
    SmallVector<mlir::NamedAttribute> newClampAttr;

    auto maxId = mlir::StringAttr::get(clampOp.getContext(), "max");
    auto minId = mlir::StringAttr::get(clampOp.getContext(), "min");

    const auto minClampOp = clampOp.getMinAttr().getValueAsDouble();
    const auto maxClampOp = clampOp.getMaxAttr().getValueAsDouble();

    double currentClampMin = 0, currentClampMax = 0;

    if (currentClampAttr.contains(maxId)) {
        currentClampMax = mlir::dyn_cast<mlir::FloatAttr>(currentClampAttr.get(maxId)).getValueAsDouble();
    }

    if (currentClampAttr.contains(minId)) {
        currentClampMin = mlir::dyn_cast<mlir::FloatAttr>(currentClampAttr.get(minId)).getValueAsDouble();
    }

    const auto newMin = std::max(currentClampMin, minClampOp);
    const auto newMax = std::min(currentClampMax, maxClampOp);
    const auto newMinAttr = getFPAttr(clampOp.getContext(), newMin);
    const auto newMaxAttr = getFPAttr(clampOp.getContext(), newMax);

    newClampAttr.emplace_back(maxId, newMaxAttr);
    newClampAttr.emplace_back(minId, newMinAttr);

    return mlir::DictionaryAttr::get(clampOp.getContext(), newClampAttr);
}

void setHWClampOp(mlir::Operation* mainOp, mlir::Operation* activationOp) {
    auto maybeClampOp = mlir::dyn_cast<IE::ClampOp>(activationOp);
    VPUX_THROW_WHEN(maybeClampOp == nullptr, "Not ClampOp provided at {0}", activationOp->getLoc());

    auto hasClampAttr = mainOp->hasAttr(vpux::OperationAttrName::CLAMP);
    mlir::DictionaryAttr clampOpInfo;

    if (hasClampAttr) {
        auto mainClampAttr = mlir::dyn_cast<mlir::DictionaryAttr>(mainOp->getAttr(vpux::OperationAttrName::CLAMP));
        VPUX_THROW_UNLESS(mainClampAttr, "The clamp attribute is expected to be a DictionaryAttr at {0}",
                          mainOp->getLoc());
        clampOpInfo = mergeClampAttrs(mainClampAttr, maybeClampOp);
    } else {
        clampOpInfo = activationOp->getAttrDictionary();
    }
    mainOp->setAttr(vpux::OperationAttrName::CLAMP, clampOpInfo);
}

}  // namespace VPU
}  // namespace vpux
