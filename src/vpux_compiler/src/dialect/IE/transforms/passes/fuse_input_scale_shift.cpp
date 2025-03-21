//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/algo.hpp"

#include <mlir/IR/IRMapping.h>

#include <algorithm>

namespace vpux::IE {
#define GEN_PASS_DECL_FUSEINPUTSCALESHIFT
#define GEN_PASS_DEF_FUSEINPUTSCALESHIFT
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

double calculateScale(double low, double high, int levels) {
    VPUX_THROW_UNLESS(low != high, "Low and high values must be different");
    VPUX_THROW_UNLESS(levels <= 256, "Levels must be less or equal to 256");

    return static_cast<double>((high - low) / static_cast<float>(levels - 1));
}

IE::FakeQuantizeOp createNewFqOp(mlir::OpBuilder& builder, IE::FakeQuantizeOp inputFqOp, mlir::Value newInput,
                                 ArrayRef<float> inLow, ArrayRef<float> inHigh, ArrayRef<float> outLow,
                                 ArrayRef<float> outHigh) {
    auto inLowConst = inputFqOp.getInputLow().getDefiningOp<Const::DeclareOp>();
    auto inHighConst = inputFqOp.getInputHigh().getDefiningOp<Const::DeclareOp>();
    auto outLowConst = inputFqOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
    auto outHighConst = inputFqOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();

    const auto newInLowConst = Const::createFloatConst(
            builder, inLowConst->getLoc(), mlir::cast<mlir::RankedTensorType>(inLowConst.getOutput().getType()), inLow);
    const auto newInHighConst =
            Const::createFloatConst(builder, inHighConst->getLoc(),
                                    mlir::cast<mlir::RankedTensorType>(inHighConst.getOutput().getType()), inHigh);
    const auto newOutLowConst =
            Const::createFloatConst(builder, outLowConst->getLoc(),
                                    mlir::cast<mlir::RankedTensorType>(outLowConst.getOutput().getType()), outLow);
    const auto newOutHighConst =
            Const::createFloatConst(builder, outHighConst->getLoc(),
                                    mlir::cast<mlir::RankedTensorType>(outHighConst.getOutput().getType()), outHigh);

    auto newInputFqOp = builder.clone(*inputFqOp);
    // We can't use IRMapping here since in/out low/high operands can be the same mlir::Value
    newInputFqOp->getOpOperand(0).set(newInput);
    newInputFqOp->getOpOperand(1).set(newInLowConst);
    newInputFqOp->getOpOperand(2).set(newInHighConst);
    newInputFqOp->getOpOperand(3).set(newOutLowConst);
    newInputFqOp->getOpOperand(4).set(newOutHighConst);

    return mlir::cast<IE::FakeQuantizeOp>(newInputFqOp);
}

// Looking for:
//       [input]        [Weights]
//          |              |
//      (Multiply)?        |
//          |              |
//        (Add)          (FQ2)
//          |              |
//        (FQ1)            |
//          |              |
//        (conv) --------- |
//          |
//        (Add) -------- [Bias]
//          |
//       [output]
//
// Disclamer: following is a rough description of the idea behind this transformation
// ConvInput =  [In] * scales + shifts
// ConvOutput = [ConvInput] * weights + biases
// =>
// ConvOutput = ([In] * scales + shifts) * weights + biases
// =>
// newWeighs = [In] * scales * weights
// newBias = shifts * weights + biases
//
// So the result is:
//       [input]      [new Weights]
//          |              |
//       (new FQ1)      (new FQ2)
//          |              |
//        (conv) --------- |
//          |
//        (Add) -------- [new Bias]
//          |
//       [output]

struct InputScaleShiftPattern {
    std::optional<IE::MultiplyOp> scaleOp;
    IE::AddOp shiftOp;
    IE::FakeQuantizeOp inputFqOp;
    SmallVector<IE::FakeQuantizeOp> weightsFqOps;
    SmallVector<IE::ConvolutionOp> convOps;
    SmallVector<IE::AddOp> addOps;

public:
    static std::optional<InputScaleShiftPattern> init(mlir::BlockArgument arg);

private:
    template <typename NextOpType, typename CurrOpType>
    static mlir::LogicalResult initNextOp(NextOpType& nextOp, CurrOpType currOp);
    static mlir::Operation* skipPreProcOps(mlir::Operation* currOp);
};

std::optional<InputScaleShiftPattern> InputScaleShiftPattern::init(mlir::BlockArgument arg) {
    if (arg.use_empty()) {
        return std::nullopt;
    }

    InputScaleShiftPattern pattern;
    auto firstUserOp = skipPreProcOps(*arg.getUsers().begin());
    if (mlir::isa_and_nonnull<IE::MultiplyOp>(firstUserOp)) {
        auto maybeValidScaleOp = mlir::cast<IE::MultiplyOp>(firstUserOp);
        auto scaleConst = maybeValidScaleOp.getInput2().getDefiningOp<Const::DeclareOp>();
        if (scaleConst == nullptr) {
            return std::nullopt;
        }
        pattern.scaleOp = maybeValidScaleOp;
    }

    auto maybeShiftOp = pattern.scaleOp.has_value() ? *(pattern.scaleOp->getResult().getUsers().begin()) : firstUserOp;
    if (!mlir::isa_and_nonnull<IE::AddOp>(maybeShiftOp)) {
        return std::nullopt;
    }
    pattern.shiftOp = mlir::cast<IE::AddOp>(maybeShiftOp);
    auto shiftConst = pattern.shiftOp.getInput2().getDefiningOp<Const::DeclareOp>();
    if (shiftConst == nullptr) {
        return std::nullopt;
    }

    if (mlir::failed(initNextOp(pattern.inputFqOp, pattern.shiftOp))) {
        return std::nullopt;
    }

    for (auto user : pattern.inputFqOp.getResult().getUsers()) {
        auto nextConvOp = mlir::dyn_cast_or_null<IE::ConvolutionOp>(user);
        if (nextConvOp == nullptr) {
            return std::nullopt;
        }
        pattern.convOps.push_back(nextConvOp);

        // transformation affects the chain Conv->Add
        // a direct descendant will get an unequal result: modified Conv != old Conv
        if (!nextConvOp->hasOneUse()) {
            return std::nullopt;
        }

        IE::AddOp nextAddOp;
        if (mlir::failed(initNextOp(nextAddOp, nextConvOp))) {
            return std::nullopt;
        }
        pattern.addOps.push_back(nextAddOp);

        auto maybeWeightsFqOp = nextConvOp.getFilter().getDefiningOp<IE::FakeQuantizeOp>();
        if (maybeWeightsFqOp == nullptr) {
            return std::nullopt;
        }

        pattern.weightsFqOps.push_back(maybeWeightsFqOp);
        auto weightsConst = maybeWeightsFqOp.getInput().getDefiningOp<Const::DeclareOp>();
        if (weightsConst == nullptr) {
            return std::nullopt;
        }

        const auto weightsType = mlir::cast<NDTypeInterface>(weightsConst.getOutput().getType());
        const auto weightsShape = weightsType.getShape();
        if (weightsShape.size() != 4) {
            return std::nullopt;
        }
    }

    return pattern;
}

template <typename NextOpType, typename CurrOpType>
mlir::LogicalResult InputScaleShiftPattern::initNextOp(NextOpType& nextOp, CurrOpType currOp) {
    auto maybeNextOp = *(currOp.getResult().getUsers().begin());
    nextOp = mlir::dyn_cast_or_null<NextOpType>(maybeNextOp);
    return mlir::failure(nextOp == nullptr);
}

mlir::Operation* InputScaleShiftPattern::skipPreProcOps(mlir::Operation* currOp) {
    while (mlir::isa<IE::ConvertOp, IE::TransposeOp>(currOp)) {
        // ignore branching intentionally
        // there is no need to handle this case now
        if (!currOp->getResult(0).hasOneUse()) {
            return nullptr;
        }

        currOp = *currOp->getResult(0).getUsers().begin();
    }
    return currOp;
}

void rewritePattern(const InputScaleShiftPattern& pattern) {
    auto maybeScaleOp = pattern.scaleOp;
    auto shiftOp = pattern.shiftOp;
    auto inputFqOp = pattern.inputFqOp;

    auto ctx = shiftOp.getContext();
    mlir::OpBuilder builder(ctx);
    builder.setInsertionPointAfterValue(shiftOp.getInput1());
    for (size_t idx = 0; idx < pattern.convOps.size(); ++idx) {
        auto convOp = pattern.convOps[idx];
        auto addOp = pattern.addOps[idx];
        auto weightsFqOp = pattern.weightsFqOps[idx];

        auto weightsConst = weightsFqOp.getInput().getDefiningOp<Const::DeclareOp>();
        const auto weightsType = mlir::cast<NDTypeInterface>(weightsConst.getOutput().getType());
        const auto weightsShape = weightsType.getShape();

        // clang-format off
        // we use FQ like scaleshift (because FQ with input low/high not equal to output low/high works like scaleshift)
        // from scaleshift res = input*scale + shift
        // from FQ res = round((input - input_low) / (input_high - input_low) * (levels-1)) / (levels-1) * (output_high - output_low) + output_low
        // we know that for u8 input_low=0 and input_high = 255 so after simplification
        // from FQ res = (x / 255) / (output_high - output_low) + output_low
        // from that (x / 255) / (output_high - output_low) + output_low = input*scale + shift
        // from that output_low = shift
        //           output_high = 255*scale + shift
        // clang-format on

        const size_t OC = weightsShape[Dims4D::Filter::OC];
        const size_t IC = weightsShape[Dims4D::Filter::IC];
        const size_t H = weightsShape[Dims4D::Filter::KY];
        const size_t W = weightsShape[Dims4D::Filter::KX];
        const size_t HW = H * W;
        const size_t IHW = IC * HW;

        const auto shiftConst = shiftOp.getInput2().getDefiningOp<Const::DeclareOp>();
        auto shiftData = IE::getConst(shiftConst);

        SmallVector<float> scaleData{1.0};
        if (maybeScaleOp.has_value()) {
            auto scaleConst = maybeScaleOp->getInput2().getDefiningOp<Const::DeclareOp>();
            scaleData = IE::getConst(scaleConst);
        }

        const auto validateAndBroadcast = [&](SmallVector<float>& data) -> mlir::LogicalResult {
            if (data.size() == IC) {
                return mlir::success();
            }

            if (data.size() == 1) {
                broadcast(data, IC);
                return mlir::success();
            }

            return mlir::failure();
        };

        if (mlir::failed(validateAndBroadcast(scaleData)) || mlir::failed(validateAndBroadcast(shiftData))) {
            return;
        }

        auto weightsInLowConst = weightsFqOp.getInputLow().getDefiningOp<Const::DeclareOp>();
        auto weightsInLowData = IE::getConst(weightsInLowConst);
        auto weightsInHighConst = weightsFqOp.getInputHigh().getDefiningOp<Const::DeclareOp>();
        auto weightsInHighData = IE::getConst(weightsInHighConst);
        auto weightsOutLowConst = weightsFqOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
        auto weightsOutLowData = IE::getConst(weightsOutLowConst);
        auto weightsOutHighConst = weightsFqOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();
        auto weightsOutHighData = IE::getConst(weightsOutHighConst);
        const auto validateSize = [&](SmallVector<float>& data) -> mlir::LogicalResult {
            if (data.size() == OC) {
                return mlir::success();
            }

            if (data.size() == 1) {
                return mlir::success();
            }

            return mlir::failure();
        };

        // constant should be either splat or per-(output)channel
        if (mlir::failed(validateSize(weightsInLowData)) || mlir::failed(validateSize(weightsInHighData)) ||
            mlir::failed(validateSize(weightsOutLowData)) || mlir::failed(validateSize(weightsOutHighData))) {
            return;
        }

        // try to fuse main part in input FQ to keep accuracy in padding (ZP works like pad value here)
        double avgShiftData =
                std::accumulate(shiftData.begin(), shiftData.end(), 0.0) / weightsShape[Dims4D::Filter::IC];
        double avgScaleData =
                std::accumulate(scaleData.begin(), scaleData.end(), 0.0) / weightsShape[Dims4D::Filter::IC];

        if (avgScaleData < 0) {
            return;
        }

        double inputMin = 0 * avgScaleData + avgShiftData;
        double inputMax = 255 * avgScaleData + avgShiftData;

        inputMin = std::min(inputMin, 0.);
        inputMax = std::max(inputMax, 0.);

        const auto maybeInputFQLevels = inputFqOp.getLevels();
        const auto maybeWeightsFQLevels = weightsFqOp.getLevels();

        if (!maybeInputFQLevels.has_value() || !maybeWeightsFQLevels.has_value()) {
            return;
        }

        const auto inputFQLevels = maybeInputFQLevels.value();
        const auto weightsFQLevels = maybeWeightsFQLevels.value();

        auto inputZP = checked_cast<double>(calculateZeroPoint(
                inputMin, inputMax, inputFQLevels, mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Unsigned)));
        double inputScale = calculateScale(inputMin, inputMax, inputFQLevels);
        inputMin = (0 - inputZP) * inputScale;
        inputMax = (255 - inputZP) * inputScale;
        if (inputScale < std::numeric_limits<double>::epsilon()) {
            return;
        }

        mlir::Value newInput = shiftOp.getInput1();
        if (maybeScaleOp.has_value()) {
            newInput = maybeScaleOp->getInput1();
        }

        auto newInputFqOp = createNewFqOp(builder, inputFqOp, newInput, /*inLow=*/ArrayRef(0.0F),
                                          /*inHigh=*/ArrayRef(checked_cast<float>(inputFQLevels - 1)),
                                          /*outLow=*/ArrayRef(static_cast<float>(inputMin)),
                                          /*outHigh=*/ArrayRef(static_cast<float>(inputMax)));
        auto newInputFqLoc = takeOpLoc(inputFqOp, llvm::formatv("new_input_fq_{0}", idx).str());
        newInputFqOp->setLoc(newInputFqLoc);
        convOp->getOpOperand(0).set(newInputFqOp->getResult(0));

        auto biasConst = addOp.getInput2().getDefiningOp<Const::DeclareOp>();
        auto biasData = IE::getConst(biasConst);
        auto weightsData = IE::getConst(weightsConst);

        SmallVector<float> newWeightsFqOutLow(weightsShape[Dims4D::Filter::OC]);
        SmallVector<float> newWeightsFqOutHigh(weightsShape[Dims4D::Filter::OC]);

        double sumOfZeroPoints = 0;

        // TODO: #-151978 these computations should be replaced with constant folding transformations
        for (size_t oc = 0; oc < OC; ++oc) {
            double weightsFqInLow = weightsInLowData[std::min(weightsInLowData.size() - 1, oc)];
            double weightsFqInHigh = weightsInHighData[std::min(weightsInHighData.size() - 1, oc)];
            double weightsFqOutLow = weightsOutLowData[std::min(weightsOutLowData.size() - 1, oc)];
            double weightsFqOutHigh = weightsOutHighData[std::min(weightsOutHighData.size() - 1, oc)];

            double weightsFqInRange = weightsFqInHigh - weightsFqInLow;
            double weightsFqOutRange = weightsFqOutHigh - weightsFqOutLow;
            double scaleshiftBiasAcc = 0;
            double weightsMin = -0.000061035156;  // fp16 closest to zero values
            double weightsMax = 0.000061035156;   // used to avoid inf scales in future calculations

            for (size_t ic = 0; ic < IC; ++ic) {
                for (size_t h = 0; h < H; ++h) {
                    for (size_t w = 0; w < W; ++w) {
                        const size_t idx = oc * IHW + ic * HW + h * W + w;
                        double storedWeight = weightsData[idx];
                        // dequantize weights using FQ formula
                        double realWeight = (storedWeight - weightsFqInLow) * weightsFqOutRange / weightsFqInRange +
                                            weightsFqOutLow;
                        // update weights to scaleshift scale per-channel difference
                        double rescaledWeight = realWeight * scaleData[ic] / inputScale;
                        // update biases to scaleshift shift per-channel difference
                        double biasModification = realWeight * (shiftData[ic] + inputZP * scaleData[ic]);
                        weightsData[idx] = rescaledWeight;
                        scaleshiftBiasAcc += biasModification;
                        // update min/max for weights FQ
                        if (weightsMax < rescaledWeight) {
                            weightsMax = rescaledWeight;
                        }
                        if (weightsMin > rescaledWeight) {
                            weightsMin = rescaledWeight;
                        }
                    }
                }
            }

            newWeightsFqOutLow[oc] = static_cast<float>(weightsMin);
            newWeightsFqOutHigh[oc] = static_cast<float>(weightsMax);
            biasData[oc] += scaleshiftBiasAcc;
            sumOfZeroPoints += -(weightsFQLevels - 1.0) * weightsMin / (weightsMax - weightsMin);
        }

        const auto newBiasConst = Const::createFloatConst(
                builder, biasConst->getLoc(), mlir::cast<mlir::RankedTensorType>(biasConst.getOutput().getType()),
                ArrayRef(biasData));
        addOp->getOpOperand(1).set(newBiasConst);

        // In original ngraph pass there was a flag: is_different_scales
        // that was used to decide if we need to update FQ levels and weights data,
        // but it seems like this a bug since functional test is failed for original implementation
        // when scales are spalt or absent, so the flag has now been removed.
        // TODO: #151977 probably original implementation should be restored

        float newWeightsFqInLow = 0;
        float newWeightsFqInHigh = weightsFQLevels - 1.0F;

        auto avgZeroPoints = std::round(sumOfZeroPoints / OC);
        for (size_t oc = 0; oc < OC; oc++) {
            double ol = newWeightsFqOutLow[oc];
            double oh = newWeightsFqOutHigh[oc];

            double zpl = oh * avgZeroPoints / (avgZeroPoints - (weightsFQLevels - 1.0));
            double zph = ol - ol * (weightsFQLevels - 1.0) / avgZeroPoints;

            ol = std::min(ol, zpl);
            oh = std::max(oh, zph);
            double scale = calculateScale(ol, oh, weightsFQLevels);
            newWeightsFqOutLow[oc] = static_cast<float>(ol);
            newWeightsFqOutHigh[oc] = static_cast<float>(oh);

            for (size_t ic = 0; ic < IC; ++ic) {
                for (size_t h = 0; h < H; ++h) {
                    for (size_t w = 0; w < W; ++w) {
                        const size_t idx = oc * IHW + ic * HW + h * W + w;
                        double q_weight = std::round((weightsData[idx] - ol) / scale);
                        weightsData[idx] = std::clamp(q_weight, 0., static_cast<double>(weightsFQLevels - 1));
                    }
                }
            }
        }

        const auto newWeightsConst = Const::createFloatConst(
                builder, weightsConst->getLoc(), mlir::cast<mlir::RankedTensorType>(weightsConst.getOutput().getType()),
                ArrayRef(weightsData));

        auto newWeightsFqOp =
                createNewFqOp(builder, weightsFqOp, newWeightsConst, /*inLow=*/ArrayRef(newWeightsFqInLow),
                              /*inHigh=*/ArrayRef(newWeightsFqInHigh), /*outLow=*/ArrayRef(newWeightsFqOutLow),
                              /*outHigh=*/ArrayRef(newWeightsFqOutHigh));

        auto newWeightsFqLoc = takeOpLoc(inputFqOp, llvm::formatv("new_weights_fq_{0}", idx).str());
        newWeightsFqOp->setLoc(newWeightsFqLoc);
        convOp->getOpOperand(1).set(newWeightsFqOp.getOutput());
    }
}

}  // namespace

//
// FuseInputScaleShiftPass
//

class FuseInputScaleShiftPass final : public IE::impl::FuseInputScaleShiftBase<FuseInputScaleShiftPass> {
public:
    explicit FuseInputScaleShiftPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void FuseInputScaleShiftPass::safeRunOnFunc() {
    auto func = getOperation();

    for (const auto& funcArg : func.getArguments()) {
        auto pattern = InputScaleShiftPattern::init(funcArg);
        if (pattern.has_value()) {
            _log.debug("Found pattern to fuse");
            rewritePattern(pattern.value());
        }
    }
}

//
// createFuseInputScaleShiftPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseInputScaleShiftPass(Logger log) {
    return std::make_unique<FuseInputScaleShiftPass>(log);
}
