//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <llvm/ADT/TypeSwitch.h>

namespace vpux::IE {
#define GEN_PASS_DECL_LOGOPOPTIMIZATIONS
#define GEN_PASS_DEF_LOGOPOPTIMIZATIONS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

void checkSEPInterpolate(IE::InterpolateOp op, const Logger& log) {
    if (op.getAttr() == nullptr) {
        return;
    }

    auto attr = op.getAttr();

    // Antialias is not supported
    if (attr.getAntialias() != nullptr && attr.getAntialias().getValue() == true) {
        return;
    }

    // Only integer scales are supported
    SmallVector<double> scales;
    auto shapeCalcModeAttr = attr.getShapeCalcMode();
    auto scalesAttr = op.getScalesAttr();
    if (shapeCalcModeAttr != nullptr && shapeCalcModeAttr.getValue() == IE::InterpolateCalcMode::SCALES &&
        scalesAttr.has_value()) {
        scales = parseFPArrayAttr<double>(scalesAttr.value());
    } else {
        auto inputShape = op.getInput().getType().cast<vpux::NDTypeInterface>().getShape();
        auto outputShape = op.getOutput().getType().cast<vpux::NDTypeInterface>().getShape();
        scales = {static_cast<double>(outputShape[Dims4D::Act::H]) / static_cast<double>(inputShape[Dims4D::Act::H]),
                  static_cast<double>(outputShape[Dims4D::Act::W]) / static_cast<double>(inputShape[Dims4D::Act::W])};
    }
    const auto nonIntegerScales = llvm::any_of(scales, [](const double scale) {
        return std::floor(scale) != scale;
    });
    if (nonIntegerScales) {
        return;
    }

    auto hasNonZeroPads = [&](mlir::ArrayAttr padsAttr) -> bool {
        if (padsAttr == nullptr) {
            return false;
        }
        auto pads = parseIntArrayAttr<int64_t>(padsAttr);
        return llvm::any_of(pads, [](int64_t pad) {
            return pad != 0;
        });
    };
    const auto hasPadding = hasNonZeroPads(attr.getPadsBegin()) || hasNonZeroPads(attr.getPadsEnd());

    // Limited support for different modes
    const auto mode = attr.getMode().getValue();
    const auto coordMode = attr.getCoordMode().getValue();
    if (mode == IE::InterpolateMode::NEAREST) {
        log.info("Interpolate at '{0}' can be optimized using SEP", op->getLoc());
        if (!hasPadding && coordMode != IE::InterpolateCoordMode::ALIGN_CORNERS) {
            log.nest().info("Case is already supported");
        }
    } else if (mode == IE::InterpolateMode::LINEAR || mode == IE::InterpolateMode::LINEAR_ONNX) {
        if (coordMode == IE::InterpolateCoordMode::ASYMMETRIC) {
            const auto scalesOutsideRange = llvm::any_of(scales, [](const double scale) {
                return !(scale >= 1.0 && scale <= 11.0);
            });
            if (scalesOutsideRange) {
                return;
            }
            log.info("Interpolate at '{0}' can be optimized using SEP", op->getLoc());
            log.nest().info("Case is already supported");
        } else if (coordMode == IE::InterpolateCoordMode::PYTORCH_HALF_PIXEL) {
            log.info("Interpolate at '{0}' might potentially be optimized using SEP", op->getLoc());
            log.nest().info("Case might have more constraints, as the proposal needs further analysis");
        }
    }
}

void checkSEPTransposedConv(IE::TransposedConvolutionOp op, const Logger& log) {
    log.info("TransposedConvolution at '{0}' can be optimized using SEP", op->getLoc());
}

void checkSEPDilatedConv(IE::ConvolutionOp op, const Logger& log) {
    if (op.getDilations() == nullptr) {
        return;
    }
    const auto dilations = parseIntArrayAttr<int64_t>(op.getDilations());
    const auto hasDilations = llvm::any_of(dilations, [](const int64_t d) {
        return d != 1;
    });
    if (!hasDilations) {
        return;
    }

    log.info("Dilated convolution at '{0}' can be optimized using SEP", op->getLoc());
}

void checkSEPPad(IE::PadOp op, const Logger& log) {
    // Tensor has to be at least 3D so that it has spatial dimensions for padding
    auto inputType = op.getInput().getType().cast<vpux::NDTypeInterface>();
    if (inputType.getRank() < 3) {
        return;
    }

    const auto extractPads = [](mlir::Value padsOperand, mlir::ArrayAttr padsAttr) -> SmallVector<int64_t> {
        if (padsOperand != nullptr) {
            auto constOp = padsOperand.getDefiningOp<Const::DeclareOp>();
            if (constOp == nullptr) {
                return {};
            }
            const auto content = constOp.getContent();
            return to_small_vector(content.getValues<int64_t>());
        } else if (padsAttr != nullptr) {
            return parseIntArrayAttr<int64_t>(padsAttr);
        }
        return {};
    };

    const auto padsBegin = extractPads(op.getPadsBegin(), op.getPadsBeginAttrAttr());
    const auto padsEnd = extractPads(op.getPadsEnd(), op.getPadsEndAttrAttr());
    if (padsBegin.empty() || padsEnd.empty()) {
        return;
    }

    // Only spatial dimensions can be padded with SEP
    for (size_t i = 0; i < padsBegin.size(); ++i) {
        if (padsBegin[i] > 0 && i < padsBegin.size() - 2) {
            return;
        }
    }
    for (size_t i = 0; i < padsEnd.size(); ++i) {
        if (padsEnd[i] > 0 && i < padsEnd.size() - 2) {
            return;
        }
    }

    // In case the data is padded with a zero constant, the sparsity map can be used to achieve this padding. For
    // other values, it would be necessary to bring them to CMX with a DMA which can affect the performance.
    std::optional<double> constantPadValue = std::nullopt;
    if (op.getMode() == IE::PadMode::CONSTANT) {
        if (op.getPadValue() != nullptr) {
            auto padValueConst = op.getPadValue().getDefiningOp<Const::DeclareOp>();
            if (padValueConst == nullptr || !padValueConst.getContentAttr().isSplat()) {
                return;
            }
            const auto padValueContent = padValueConst.getContent();
            constantPadValue = padValueContent.getSplatValue<double>();
        } else if (op.getPadValueAttrAttr() != nullptr) {
            const auto padValueAttr = op.getPadValueAttr();
            if (!padValueAttr.has_value()) {
                return;
            }
            constantPadValue = padValueAttr.value().convertToDouble();
        }
    }

    log.info("Pad operation at '{0}' can be optimized using SEP", op->getLoc());
    if (constantPadValue.has_value() && constantPadValue.value()) {
        const auto padValue = constantPadValue.value();
        if (!vpux::isDoubleEqual(padValue, 0.0)) {
            log.nest().info("Would require an extra small DMA to bring the constant '{0}' to CMX, which could hurt "
                            "performance",
                            padValue);
        }
    }
}

void checkSEPTile(IE::TileOp op, const Logger& log) {
    SmallVector<int64_t> repeats;
    if (op.getRepeats() != nullptr) {
        auto repeatsConst = op.getRepeats().getDefiningOp<Const::DeclareOp>();
        if (repeatsConst == nullptr) {
            return;
        }
        const auto repeatsContent = repeatsConst.getContent();
        repeats = to_small_vector(repeatsContent.getValues<int64_t>());
    } else if (op.getRepeatsValuesAttr() != nullptr) {
        repeats = parseIntArrayAttr<int64_t>(op.getRepeatsValuesAttr());
    }

    if (repeats.empty()) {
        return;
    }

    // Align the number of dimensions in the repeats vector and input shape by padding left the smaller container
    // with 1
    auto inputShape = to_small_vector(op.getInput().getType().cast<vpux::NDTypeInterface>().getShape());
    if (repeats.size() < inputShape.size()) {
        const auto numElems = inputShape.size() - repeats.size();
        repeats.insert(repeats.begin(), numElems, 1);
    } else if (repeats.size() > inputShape.size()) {
        const auto numElems = repeats.size() - inputShape.size();
        inputShape.insert(inputShape.begin(), numElems, 1);
    }

    // Only 3D and 4D Tile operations can be mapped to hardware using the SEP feature
    // For other instances, it might be necessary to split the operation into multiple smaller ones in order to map
    // to hardware, which could negatively impact the performance
    if (inputShape.size() < 3) {
        return;
    }

    for (auto repeatDim : repeats | indexed) {
        if (repeatDim.value() <= 1) {
            continue;
        }
        // Number of channels must be a multiple of 16 for the SEP feature to support the operation
        if (repeatDim.index() == repeats.size() - 3) {
            if (inputShape[repeatDim.index()] % 16 != 0) {
                return;
            }
            continue;
        }
        // Spatial dimensions can be repeated any number of times
        if (repeatDim.index() < repeats.size() - 2) {
            return;
        }
    }

    log.info("Tile operation at '{0}' can be optimized using SEP", op->getLoc());
}

//
// LogOpOptimizationsPass
//

class LogOpOptimizationsPass final : public IE::impl::LogOpOptimizationsBase<LogOpOptimizationsPass> {
public:
    explicit LogOpOptimizationsPass() {
    }

private:
    void safeRunOnFunc() final;
};

void LogOpOptimizationsPass::safeRunOnFunc() {
    Logger log("optimizations", LogLevel::Info);

    auto func = getOperation();
    func.walk([&](mlir::Operation* op) {
        llvm::TypeSwitch<mlir::Operation*, void>(op)
                .Case<IE::InterpolateOp>([&](IE::InterpolateOp interpOp) {
                    checkSEPInterpolate(interpOp, log);
                })
                .Case<IE::TransposedConvolutionOp>([&](IE::TransposedConvolutionOp transposedConvOp) {
                    checkSEPTransposedConv(transposedConvOp, log);
                })
                .Case<IE::ConvolutionOp>([&](IE::ConvolutionOp convOp) {
                    checkSEPDilatedConv(convOp, log);
                })
                .Case<IE::PadOp>([&](IE::PadOp padOp) {
                    checkSEPPad(padOp, log);
                })
                .Case<IE::TileOp>([&](IE::TileOp tileOp) {
                    checkSEPTile(tileOp, log);
                });
    });
}

}  // namespace

//
// createLogOpOptimizationsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createLogOpOptimizationsPass() {
    return std::make_unique<LogOpOptimizationsPass>();
}
