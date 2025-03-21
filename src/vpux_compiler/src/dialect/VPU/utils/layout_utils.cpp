//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/layout_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/const/utils/affine_reshape.hpp"

using namespace vpux;
using namespace VPU;

constexpr vpux::StringLiteral keep_dims = "keep_dims";
constexpr vpux::StringLiteral dim_mapping = "dim_mapping";
constexpr vpux::StringLiteral axes_value = "axes_value";

//
// verifyOpLayout
//

mlir::LogicalResult vpux::VPU::verifyOpLayout(mlir::Operation* op) {
    if (auto layoutInterface = mlir::dyn_cast<IE::LayoutInfoOpInterface>(op)) {
        return layoutInterface.verifyLayout();
    }
    return mlir::success();
}

//
// inferReduceLayoutInfo
//

void vpux::VPU::inferReduceLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info) {
    const bool keepDims = op->hasAttr(keep_dims) ? op->getAttr(keep_dims) != nullptr : false;
    llvm::SmallVector<int64_t> axesVec;
    if (op->getNumOperands() > 1) {
        axesVec = parseIntArrayAttr<int64_t>(vpux::IE::getIntArrayAttrValue(op->getOperand(1)));
    } else if (op->hasAttr(axes_value)) {
        if (auto axesValue = op->getAttr(axes_value).dyn_cast_or_null<mlir::ArrayAttr>()) {
            axesVec = parseIntArrayAttr<int64_t>(axesValue);
        }
    }

    const auto mainOrder = info.getInput(0);

    if (!keepDims) {
        info.setOutput(0, IE::calculateReducedOutputLayout(mainOrder, axesVec));
    } else {
        info.setOutput(0, mainOrder);
    }
}

mlir::LogicalResult vpux::VPU::verifyReduceLayoutInfo(mlir::Operation* op) {
    auto layer = mlir::dyn_cast<VPU::LayerOpInterface>(op);
    VPUX_THROW_UNLESS(layer != nullptr, "Operation {0} does not implement VPU::LayerOpInterface", op->getName());

    const auto input = layer.getInputs()[0];
    const auto output = layer.getOutputs()[0];

    const auto inOrder = DimsOrder::fromValue(input);
    const auto outOrder = DimsOrder::fromValue(output);

    const bool keepDims = op->hasAttr(keep_dims) ? op->getAttr(keep_dims) != nullptr : false;
    if (!keepDims) {
        llvm::SmallVector<int64_t> axesVec;
        if (op->getNumOperands() > 1) {
            axesVec = parseIntArrayAttr<int64_t>(vpux::IE::getIntArrayAttrValue(op->getOperand(1)));
        } else if (op->hasAttr(axes_value)) {
            if (auto axesValue = op->getAttr(axes_value).dyn_cast_or_null<mlir::ArrayAttr>()) {
                axesVec = parseIntArrayAttr<int64_t>(axesValue);
            }
        }

        const auto expectedOutOrder = IE::calculateReducedOutputLayout(inOrder, axesVec);

        if (expectedOutOrder != outOrder) {
            return errorAt(op->getLoc(), "Operation output order is not as expected. outL={0}, expectedOutL={1}",
                           outOrder, expectedOutOrder);
        }
    } else {
        if (inOrder != outOrder) {
            return errorAt(op->getLoc(), "Operation must have the same input and output order. inL={0}, outL={1}",
                           inOrder, outOrder);
        }
    }

    return mlir::success();
}

//
// verifySameInOutDimsOrder
//

mlir::LogicalResult vpux::VPU::verifySameInOutDimsOrder(mlir::Operation* op) {
    auto layer = mlir::dyn_cast<VPU::LayerOpInterface>(op);
    VPUX_THROW_UNLESS(layer != nullptr, "Operation {0} does not implement VPU::LayerOpInterface", op->getName());

    const auto input = layer.getInputs()[0];
    const auto output = layer.getOutputs()[0];

    const auto inOrder = DimsOrder::fromValue(input);
    const auto outOrder = DimsOrder::fromValue(output);

    if (inOrder != outOrder) {
        return errorAt(op->getLoc(), "Operation must have the same input and output order. inL={0}, outL={1}", inOrder,
                       outOrder);
    }

    return mlir::success();
}

//
// verifyDefaultDimsOrder
//

mlir::LogicalResult vpux::VPU::verifyDefaultDimsOrder(mlir::Operation* op) {
    auto layer = mlir::dyn_cast<VPU::LayerOpInterface>(op);
    VPUX_THROW_UNLESS(layer != nullptr, "Operation {0} does not implement VPU::LayerOpInterface", op->getName());

    for (const auto& val : layer->getOpOperands()) {
        const auto inOrder = DimsOrder::fromValue(val.get());
        auto expectedInputOrder = DimsOrder::fromNumDims(inOrder.numDims());

        if (inOrder != expectedInputOrder) {
            return errorAt(op->getLoc(),
                           "Operation must have the same input and output order. inL={0}, expectedInL={1}", inOrder,
                           expectedInputOrder);
        }
    }

    for (const auto& val : layer->getResults()) {
        const auto outOrder = DimsOrder::fromValue(val);
        auto expectedOutputOrder = DimsOrder::fromNumDims(outOrder.numDims());

        if (outOrder != expectedOutputOrder) {
            return errorAt(op->getLoc(),
                           "Operation must have the same input and output order. outL={0}, expectedOutL={1}", outOrder,
                           expectedOutputOrder);
        }
    }

    return mlir::success();
}

void vpux::VPU::inferLayoutInfoDefaultDimsOrder(IE::LayerLayoutInfo& info) {
    IE::fillDefaultLayoutInfo(info);
}

//
// verifySameInOutDefaultDimsOrder
//

mlir::LogicalResult vpux::VPU::verifySameInOutDefaultDimsOrder(mlir::Operation* op) {
    if (VPU::verifySameInOutDimsOrder(op).failed()) {
        return mlir::failure();
    }

    if (verifyDefaultDimsOrder(op).failed()) {
        return mlir::failure();
    }

    return mlir::success();
}

void vpux::VPU::inferLayoutInfoSameInOutDefaultDimsOrder(IE::LayerLayoutInfo& info) {
    IE::fillDefaultLayoutInfo(info);
    info.setOutput(0, info.getInput(0));
}

//
// SameAnyDimsOrder
//

mlir::LogicalResult vpux::VPU::verifySameAnyDimsOrder(mlir::Operation* op) {
    auto layer = mlir::dyn_cast<VPU::LayerOpInterface>(op);
    if (layer == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement Layer interface", op->getName());
    }

    auto inputs = layer.getInputs();

    const auto firstInput = inputs.front();
    const auto mainOrder = DimsOrder::fromValue(firstInput);

    for (const auto& val : layer->getOpOperands()) {
        const auto order = DimsOrder::fromValue(val.get());

        if (order != mainOrder) {
            return errorAt(op, "Operation's input/output layout mismatch");
        }
    }

    for (const auto& val : layer->getResults()) {
        const auto order = DimsOrder::fromValue(val);

        if (order != mainOrder) {
            return errorAt(op, "Operation's input/output layout mismatch");
        }
    }

    return mlir::success();
}

void vpux::VPU::inferLayoutInfoSameAnyDimsOrder(IE::LayerLayoutInfo& info) {
    const auto inOrder = info.getInput(0);
    info.fill(inOrder);
}

//
// SameInOutSpecificDimsOrder
//

mlir::LogicalResult vpux::VPU::verifySameInOutSpecificDimsOrder(mlir::Operation* op,
                                                                ArrayRef<DimsOrder> supportedLayouts) {
    if (VPU::verifySameInOutDimsOrder(op).failed()) {
        return mlir::failure();
    }

    auto layerOp = mlir::dyn_cast<VPU::LayerOpInterface>(op);
    VPUX_THROW_UNLESS(layerOp != nullptr, "Operation {0} does not implement VPU::LayerOpInterface", op->getName());

    const auto input = layerOp.getInputs()[0];
    const auto inOrder = DimsOrder::fromValue(input);

    const auto isSupported = std::count(supportedLayouts.begin(), supportedLayouts.end(), inOrder);
    if (!isSupported) {
        return errorAt(op->getLoc(), "Operation does not support {0} layout", inOrder);
    }

    return mlir::success();
}

void vpux::VPU::inferLayoutInfoSameInOutSpecificDimsOrder(IE::LayerLayoutInfo& info,
                                                          ArrayRef<DimsOrder> supportedLayouts) {
    const auto filter = [](size_t ind) {
        return ind != 0;
    };
    IE::fillDefaultLayoutInfo(info, filter, filter);

    const auto mainOrder = info.getInput(0);

    if (llvm::is_contained(supportedLayouts, mainOrder)) {
        info.setOutput(0, mainOrder);
        return;
    }

    const auto supportedOrderIt = llvm::find_if(supportedLayouts, [mainOrder](DimsOrder order) {
        return order.numDims() == mainOrder.numDims();
    });

    VPUX_THROW_UNLESS(supportedOrderIt != supportedLayouts.end(),
                      "Layouts supported by the operation '{0}' do not match the rank '{1}' of the input shape",
                      supportedLayouts, mainOrder.numDims());

    const auto supportedOrder = *supportedOrderIt;
    info.setInput(0, supportedOrder);
    info.setOutput(0, supportedOrder);
}

//
// SameMultipleInOutSpecificDimsOrder
//

void vpux::VPU::inferSameMultipleInOutSpecificDimsOrder(IE::LayerLayoutInfo& info,
                                                        llvm::ArrayRef<vpux::DimsOrder> supportedLayouts) {
    const auto mainOrder = info.getInput(0);

    // Sets input and output layouts to either the supportedOrder or default order based on rank
    auto setLayouts = [&](const vpux::DimsOrder& order) {
        const auto numDims = order.numDims();
        for (auto i : irange(info.getNumInputs())) {
            const auto inputNumDims = info.getInput(i).numDims();
            const auto layoutOrder = (inputNumDims == numDims) ? order : DimsOrder::fromNumDims(inputNumDims);
            info.setInput(i, layoutOrder);
        }

        for (auto i : irange(info.getNumOutputs())) {
            const auto outputNumDims = info.getOutput(i).numDims();
            const auto layoutOrder = (outputNumDims == numDims) ? order : DimsOrder::fromNumDims(outputNumDims);
            info.setOutput(i, layoutOrder);
        }
    };

    if (llvm::is_contained(supportedLayouts, mainOrder)) {
        setLayouts(mainOrder);
        return;
    }

    const auto supportedOrderIt = llvm::find_if(supportedLayouts, [mainOrder](DimsOrder order) {
        return order.numDims() == mainOrder.numDims();
    });

    VPUX_THROW_UNLESS(supportedOrderIt != supportedLayouts.end(),
                      "Layouts supported by the operation '{0}' do not match the rank '{1}' of the input shape",
                      supportedLayouts, mainOrder.numDims());

    const auto supportedOrder = *supportedOrderIt;
    setLayouts(supportedOrder);
}

mlir::LogicalResult vpux::VPU::verifySameMultipleInOutSpecificDimsOrder(mlir::Operation* op,
                                                                        ArrayRef<DimsOrder> supportedLayouts) {
    if (VPU::verifySameInOutDimsOrder(op).failed()) {
        return mlir::failure();
    }

    auto layerOp = mlir::dyn_cast<VPU::LayerOpInterface>(op);
    VPUX_THROW_UNLESS(layerOp != nullptr, "Operation {0} does not implement VPU::LayerOpInterface", op->getName());

    const auto mainInput = layerOp.getInputs()[0];
    const auto mainOrderNumDims = DimsOrder::fromValue(mainInput).numDims();

    for (const auto& val : layerOp->getOpOperands()) {
        const auto inOrder = DimsOrder::fromValue(val.get());
        const auto inNumDims = inOrder.numDims();
        const auto defaultDimsOrder = DimsOrder::fromNumDims(inNumDims);

        // Inputs with ranks different than the main input should have the default layout
        if (inNumDims != mainOrderNumDims && inOrder != defaultDimsOrder) {
            return errorAt(op->getLoc(), "Operation does not support {0} layout", inOrder);
        }

        const auto isSupported = std::count(supportedLayouts.begin(), supportedLayouts.end(), inOrder);
        if (!isSupported) {
            return errorAt(op->getLoc(), "Operation does not support {0} layout", inOrder);
        }
    }

    for (const auto& val : layerOp->getResults()) {
        const auto outOrder = DimsOrder::fromValue(val);
        const auto outNumDims = outOrder.numDims();
        const auto defaultDimsOrder = DimsOrder::fromNumDims(outNumDims);

        if (outNumDims != mainOrderNumDims && outOrder != defaultDimsOrder) {
            return errorAt(op->getLoc(), "Operation does not support {0} layout", outOrder);
        }

        const auto isSupported = std::count(supportedLayouts.begin(), supportedLayouts.end(), outOrder);
        if (!isSupported) {
            return errorAt(op->getLoc(), "Operation does not support {0} layout", outOrder);
        }
    }

    return mlir::success();
}

//
// inferAffineReshapeLayoutInfo
//

void vpux::VPU::inferAffineReshapeLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info) {
    const auto dimMapping = op->hasAttr(dim_mapping) ? op->getAttr(dim_mapping) : nullptr;
    const auto dimMappingAttr = dimMapping.dyn_cast_or_null<mlir::ArrayAttr>();
    const auto inOrder = info.getInput(0);
    const auto inPermutation = inOrder.toPermutation();
    const auto outPermutation = Const::inferAffineReshapeOutputLayout(inPermutation, dimMappingAttr);
    if (!outPermutation.has_value()) {
        IE::fillDefaultLayoutInfo(info);
        return;
    }

    info.setInput(0, inOrder);
    info.setOutput(0, outPermutation.value());
}

mlir::LogicalResult vpux::VPU::verifyAffineReshapeLayoutInfo(mlir::Operation* op) {
    auto layer = mlir::dyn_cast<VPU::LayerOpInterface>(op);
    VPUX_THROW_UNLESS(layer != nullptr, "Operation {0} does not implement VPU::LayerOpInterface", op->getName());

    const auto input = layer.getInputs()[0];
    const auto output = layer.getOutputs()[0];

    const auto inOrder = DimsOrder::fromValue(input);
    const auto outOrder = DimsOrder::fromValue(output);

    const auto dimMapping = op->hasAttr(dim_mapping) ? op->getAttr(dim_mapping) : nullptr;
    const auto dimMappingAttr = dimMapping.dyn_cast_or_null<mlir::ArrayAttr>();

    const auto inPermutation = inOrder.toPermutation();
    const auto outPermutation = Const::inferAffineReshapeOutputLayout(inPermutation, dimMappingAttr);

    if (!outPermutation.has_value()) {
        return verifyDefaultDimsOrder(op);
    }

    if (outPermutation.value() != outOrder) {
        return errorAt(op->getLoc(), "Operation output order is not as expected. outL={0}, expectedOutL={1}", outOrder,
                       outPermutation.value());
    }

    return mlir::success();
}

//
// inferInterpolateLayoutInfo
//

void vpux::VPU::inferInterpolateLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info) {
    auto inputShape = op->getOperand(0).getType().cast<vpux::NDTypeInterface>().getShape().raw();
    VPUX_THROW_UNLESS(inputShape.size() == 4, "Interpolate input shape expected to have 4 dimensions, but has {0}",
                      inputShape.size());

    // Select NCHW layout due to performance reasons
    // [Track number: E#25302]
    auto channels = inputShape[Dims4D::Act::C.ind()];
    const auto antialias = mlir::cast<IE::InterpolateOp>(op).getAttr().getAntialias().getValue();
    if (channels == 1 || antialias) {
        VPU::inferLayoutInfoSameInOutSpecificDimsOrder(info, {DimsOrder::NCHW, DimsOrder::CHW});
    } else {
        VPU::inferLayoutInfoSameInOutSpecificDimsOrder(
                info, {DimsOrder::NCHW, DimsOrder::NHWC, DimsOrder::CHW, DimsOrder::HWC});
    }
}

mlir::LogicalResult vpux::VPU::verifyInterpolateLayoutInfo(mlir::Operation* op) {
    auto inputShape = op->getOperand(0).getType().cast<vpux::NDTypeInterface>().getShape().raw();
    VPUX_THROW_UNLESS(inputShape.size() == 4, "Interpolate input shape expected to have 4 dimensions, but has {0}",
                      inputShape.size());

    // Select NCHW layout due to performance reasons
    // [Track number: E#25302]
    auto channels = inputShape[Dims4D::Act::C.ind()];
    const auto antialias = mlir::cast<VPU::InterpolateOp>(op).getAttr().getAntialias().getValue();
    if (channels == 1 || antialias) {
        return VPU::verifySameInOutSpecificDimsOrder(op, {DimsOrder::NCHW, DimsOrder::CHW});
    } else {
        return VPU::verifySameInOutSpecificDimsOrder(
                op, {DimsOrder::NCHW, DimsOrder::NHWC, DimsOrder::CHW, DimsOrder::HWC});
    }
}

//
// inferRegionYoloLayoutInfo
//

void vpux::VPU::inferRegionYoloLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info) {
    auto regionYoloOp = mlir::dyn_cast<IE::RegionYoloOp>(op);
    VPUX_THROW_UNLESS(regionYoloOp != nullptr, "Operation '{0}' is not a RegionYolo", op->getName());

    if (regionYoloOp.getDoSoftmax()) {
        IE::fillDefaultLayoutInfo(info);
    } else {
        VPU::inferLayoutInfoSameInOutSpecificDimsOrder(info, {DimsOrder::NCHW});
    }
}

mlir::LogicalResult vpux::VPU::verifyRegionYoloLayoutInfo(mlir::Operation* op) {
    auto regionYoloOp = mlir::dyn_cast<VPU::RegionYoloOp>(op);
    VPUX_THROW_UNLESS(regionYoloOp != nullptr, "Operation '{0}' is not a RegionYolo", op->getName());

    if (regionYoloOp.getDoSoftmax()) {
        return verifyDefaultDimsOrder(op);
    }

    return VPU::verifySameInOutSpecificDimsOrder(op, {DimsOrder::NCHW});
}

//
// inferLSTMSequenceLayoutInfo
//

void vpux::VPU::inferLSTMSequenceLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info) {
    auto lstmSequenceOp = mlir::dyn_cast<IE::LSTMSequenceOp>(op);
    VPUX_THROW_UNLESS(lstmSequenceOp, "Expected an IE::LSTMSequenceOp, got {0}", op);

    const size_t firstOptionalInputInd = 3;
    const auto inputFilter = [=](size_t ind) {
        return ind < firstOptionalInputInd;
    };
    const auto outputFilter = [](size_t) {
        return true;
    };
    IE::fillDefaultLayoutInfo(info, inputFilter, outputFilter);

    size_t ind = firstOptionalInputInd;
    if (lstmSequenceOp.getWeights()) {
        info.setInput(ind++, DimsOrder::NWHC);  // weights
    }
    info.setInput(ind++, DimsOrder::NWHC);  // recurrenceWeights
    if (lstmSequenceOp.getBiases()) {
        info.setInput(ind++, DimsOrder::NCWH);  // biases
    }
}

mlir::LogicalResult vpux::VPU::verifyLSTMSequenceLayoutInfo(mlir::Operation* op) {
    auto lstmSequenceOp = mlir::dyn_cast<VPU::LSTMSequenceOp>(op);
    VPUX_THROW_UNLESS(lstmSequenceOp, "Expected a VPU::LSTMSequenceOp, got {0}", op);

    if (DimsOrder::fromValue(lstmSequenceOp.getInputData()) != DimsOrder::NCHW ||
        DimsOrder::fromValue(lstmSequenceOp.getInitialHiddenState()) != DimsOrder::NCHW ||
        DimsOrder::fromValue(lstmSequenceOp.getInitialCellState()) != DimsOrder::NCHW ||
        DimsOrder::fromValue(lstmSequenceOp.getReccurenceWeights()) != DimsOrder::NWHC ||
        DimsOrder::fromValue(lstmSequenceOp.getSyncBuffer()) != DimsOrder::NCHW) {
        return mlir::failure();
    }

    return mlir::success();
}

void vpux::VPU::inferDequantizeLayoutInfo(mlir::Operation* origOp, IE::LayerLayoutInfo& info) {
    const auto inType = origOp->getOperand(0).getType().cast<vpux::NDTypeInterface>().getElementType();

    const auto qType = inType.cast<mlir::quant::QuantizedType>();

    if (qType.isa<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto numDims = info.getInput(0).numDims();
        if (numDims == 3) {
            info.fill(DimsOrder::HWC);
        } else if (numDims == 4) {
            info.fill(DimsOrder::NHWC);
        } else {
            VPUX_THROW("Unsupported rank '{0}'", numDims);
        }
    } else {
        VPU::inferLayoutInfoSameInOutSpecificDimsOrder(
                info, {DimsOrder::CHW, DimsOrder::HWC, DimsOrder::NCHW, DimsOrder::NHWC});
    }
}

mlir::LogicalResult vpux::VPU::verifyDequantizeLayoutInfo(mlir::Operation* op) {
    const auto inType = op->getOperand(0).getType().cast<vpux::NDTypeInterface>();

    const auto qType = inType.getElementType().cast<mlir::quant::QuantizedType>();

    if (qType.isa<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto numDims = inType.getShape().raw().size();
        if (numDims == 3) {
            return VPU::verifySameInOutSpecificDimsOrder(op, {DimsOrder::HWC});
        } else if (numDims == 4) {
            return VPU::verifySameInOutSpecificDimsOrder(op, {DimsOrder::NHWC});
        } else {
            VPUX_THROW("Unsupported rank '{0}'", numDims);
        }
    } else {
        return VPU::verifySameInOutSpecificDimsOrder(
                op, {DimsOrder::CHW, DimsOrder::HWC, DimsOrder::NCHW, DimsOrder::NHWC});
    }
}

void vpux::VPU::inferQuantizeLayoutInfo(mlir::Operation* origOp, IE::LayerLayoutInfo& info) {
    const auto outType = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getElementType();

    const auto qType = outType.cast<mlir::quant::QuantizedType>();

    if (qType.isa<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto numDims = info.getInput(0).numDims();
        if (numDims == 3) {
            info.fill(DimsOrder::HWC);
        } else if (numDims == 4) {
            info.fill(DimsOrder::NHWC);
        } else {
            VPUX_THROW("Unsupported rank '{0}'", numDims);
        }
    } else {
        VPU::inferLayoutInfoSameInOutSpecificDimsOrder(
                info, {DimsOrder::CHW, DimsOrder::HWC, DimsOrder::NCHW, DimsOrder::NHWC});
    }
}

mlir::LogicalResult vpux::VPU::verifyQuantizeLayoutInfo(mlir::Operation* op) {
    const auto outType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();

    const auto qType = outType.getElementType().cast<mlir::quant::QuantizedType>();

    if (qType.isa<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto numDims = outType.getShape().raw().size();
        if (numDims == 3) {
            return VPU::verifySameInOutSpecificDimsOrder(op, {DimsOrder::HWC});
        } else if (numDims == 4) {
            return VPU::verifySameInOutSpecificDimsOrder(op, {DimsOrder::NHWC});
        } else {
            VPUX_THROW("Unsupported rank '{0}'", numDims);
        }
    } else {
        return VPU::verifySameInOutSpecificDimsOrder(
                op, {DimsOrder::CHW, DimsOrder::HWC, DimsOrder::NCHW, DimsOrder::NHWC});
    }
}

//
// inferSqueezeOutputLayout
//

DimsOrder vpux::VPU::inferSqueezeOutputLayout(const DimArr& inPerm, const SmallVector<int64_t>& axesVec,
                                              ArrayRef<int64_t> inShape) {
    SmallVector<vpux::Dim> perm;
    SmallVector<int64_t> axes = axesVec;

    // If axes attr is empty, find all dims equal to 1
    if (axes.empty()) {
        for (auto inInd : irange(inShape.size())) {
            if (inShape[inInd] == 1) {
                axes.push_back(inInd);
            }
        }
    }

    // Iterate over input dims in the given order and push back corresponding output dims.
    for (const auto& p : inPerm) {
        // Skip over squeezed dim
        if (llvm::find(axes, p.ind()) != axes.end())
            continue;

        auto dim = p.ind();
        // Decrement input dim index by the number of squeezed axes smaller than itself
        for (const auto& squeezeAxis : axes) {
            if (p.ind() > squeezeAxis) {
                dim--;
            }
        }

        perm.push_back(vpux::Dim(dim));
    }

    return DimsOrder::fromPermutation(ArrayRef(perm));
}

//
// inferUnsqueezeOutputLayout
//

DimsOrder vpux::VPU::inferUnsqueezeOutputLayout(const DimArr& inPerm, const SmallVector<int64_t>& axesVec,
                                                ArrayRef<int64_t> inShape) {
    SmallVector<vpux::Dim> perm;
    SmallVector<int64_t> axes = axesVec;

    // If axes attr is empty, find all dims not equal to 1
    if (axes.empty()) {
        for (auto inInd : irange(inShape.size())) {
            if (inShape[inInd] != 1) {
                axes.push_back(inInd);
            }
        }
    }

    // Iterate over input dims in the given order and push back corresponding output dims.
    for (const auto& p : inPerm) {
        auto dim = p.ind();
        for (const auto& unsqueezedAxis : axes) {
            if (dim > unsqueezedAxis) {
                dim++;
            } else if (dim == unsqueezedAxis) {
                perm.push_back(vpux::Dim(dim));
                dim++;
            }
        }

        perm.push_back(vpux::Dim(dim));
    }

    // If unsqueezed 1s are at the end, push their corresponding axes in the perm vec
    const auto sz = static_cast<int64_t>(perm.size());
    for (const auto& unsqueezedAxis : axes) {
        if (unsqueezedAxis >= sz) {
            perm.push_back(vpux::Dim(unsqueezedAxis));
        }
    }

    return DimsOrder::fromPermutation(ArrayRef(perm));
}

void vpux::VPU::inferSqueezeUnsqueezeLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info) {
    const auto axesValue = op->hasAttr(axes_value) ? op->getAttr(axes_value) : nullptr;
    const auto axesValueAttr = axesValue.dyn_cast_or_null<mlir::ArrayAttr>();
    const auto axes =
            axesValueAttr != nullptr ? parseIntArrayAttr<int64_t>(axesValueAttr) : mlir::SmallVector<int64_t>{};
    const auto inOrder = info.getInput(0);
    const auto inShape = op->getOperand(0).getType().cast<NDTypeInterface>().getShape().raw();
    const auto inPermutation = inOrder.toPermutation();

    info.setInput(0, inOrder);
    if (mlir::isa<IE::SqueezeOp>(op)) {
        info.setOutput(0, inferSqueezeOutputLayout(inPermutation, axes, inShape));
    } else {
        info.setOutput(0, inferUnsqueezeOutputLayout(inPermutation, axes, inShape));
    }
}

mlir::LogicalResult vpux::VPU::verifyNCEConvolutionLayoutInfo(mlir::Operation* op) {
    auto layer = mlir::dyn_cast<VPU::LayerOpInterface>(op);
    if (layer == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement Layer interface", op->getName());
    }

    const auto input = layer.getInputs()[0];
    const auto filter = layer.getInputs()[1];
    const auto output = layer.getOutputs()[0];

    const auto inOrder = DimsOrder::fromValue(input);
    const auto filterOrder = DimsOrder::fromValue(filter);
    const auto outOrder = DimsOrder::fromValue(output);

    if (inOrder != DimsOrder::NHWC) {
        return errorAt(op->getLoc(), "Operation input order is not as expected. inL={0}, expectedInL=NHWC", inOrder);
    }
    if (outOrder != DimsOrder::NHWC) {
        return errorAt(op->getLoc(), "Operation output order is not as expected. outL={0}, expectedOutL=NHWC",
                       outOrder);
    }
    if (filterOrder != DimsOrder::OYXI) {
        return errorAt(op->getLoc(), "Operation filter order is not as expected. filterL={0}, expectedFilterL=OYXI",
                       filterOrder);
    }
    return mlir::success();
}

mlir::LogicalResult vpux::VPU::verifyTopKLayoutInfo(mlir::Operation* op) {
    auto layer = mlir::dyn_cast<VPU::LayerOpInterface>(op);
    if (layer == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement Layer interface", op->getName());
    }

    const auto input = layer.getInputs()[0];
    const auto outputFirst = layer.getOutputs()[0];
    const auto outputSecond = layer.getOutputs()[1];

    const auto inOrder = DimsOrder::fromValue(input);
    const auto firstOutOrder = DimsOrder::fromValue(outputFirst);
    const auto secondOutOrder = DimsOrder::fromValue(outputSecond);

    if (firstOutOrder != inOrder) {
        return errorAt(op->getLoc(), "Operation output order is not as expected. outL={0}, expectedOutL={1}",
                       firstOutOrder, inOrder);
    }
    if (secondOutOrder != inOrder) {
        return errorAt(op->getLoc(), "Operation output order is not as expected. outL={0}, expectedOutL={1}",
                       secondOutOrder, inOrder);
    }
    return mlir::success();
}

mlir::LogicalResult vpux::VPU::verifyScatterNDUpdateLayoutInfo(mlir::Operation* op) {
    auto layer = mlir::dyn_cast<VPU::LayerOpInterface>(op);
    if (layer == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement Layer interface", op->getName());
    }

    const auto inputFirst = layer.getInputs()[0];
    const auto inputSecond = layer.getInputs()[2];
    const auto output = layer.getOutputs()[0];

    const auto firstInOrder = DimsOrder::fromValue(inputFirst);
    const auto secondInOrder = DimsOrder::fromValue(inputSecond);
    const auto outOrder = DimsOrder::fromValue(output);

    if (secondInOrder != firstInOrder) {
        return errorAt(op->getLoc(), "Operation output order is not as expected. in2L={0}, expectedIn2L={1}",
                       secondInOrder, firstInOrder);
    }
    if (outOrder != firstInOrder) {
        return errorAt(op->getLoc(), "Operation output order is not as expected. outL={0}, expectedOutL={1}", outOrder,
                       firstInOrder);
    }
    return mlir::success();
}

mlir::LogicalResult vpux::VPU::verifySWGroupConvolutionLayoutInfo(mlir::Operation* op) {
    auto layer = mlir::dyn_cast<VPU::LayerOpInterface>(op);
    if (layer == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement Layer interface", op->getName());
    }

    const auto filter = layer.getInputs()[1];

    const auto filterOrder = DimsOrder::fromValue(filter);

    if (filterOrder != DimsOrder::OIYX) {
        return errorAt(op->getLoc(), "Operation filter order is not as expected. filterL={0}, expectedFilterL=OIYX",
                       filterOrder);
    }

    return VPU::verifySameInOutSpecificDimsOrder(op,
                                                 {DimsOrder::NCHW, DimsOrder::NHWC, DimsOrder::CHW, DimsOrder::HWC});
}
mlir::LogicalResult vpux::VPU::verifyNCEPermuteLayoutInfo(mlir::Operation* op) {
    auto layer = mlir::dyn_cast<VPU::LayerOpInterface>(op);
    if (layer == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement Layer interface", op->getName());
    }

    const auto input = layer.getInputs()[0];
    const auto output = layer.getOutputs()[0];

    const auto inOrder = DimsOrder::fromValue(input);
    const auto outOrder = DimsOrder::fromValue(output);

    if (inOrder != DimsOrder::NCHW) {
        return errorAt(op->getLoc(), "Operation input order is not as expected. inL={0}, expectedInL=NCHW", inOrder);
    }
    if (outOrder != DimsOrder::NHWC) {
        return errorAt(op->getLoc(), "Operation output order is not as expected. outL={0}, expectedOutL=NHWC",
                       outOrder);
    }
    return mlir::success();
}

//
// inferRollLayoutInfo
//

void vpux::VPU::inferRollLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info) {
    VPU::inferLayoutInfoSameInOutSpecificDimsOrder(
            info, {DimsOrder::NCHW, DimsOrder::NHWC, DimsOrder::CHW, DimsOrder::NC, DimsOrder::C});
}

mlir::LogicalResult vpux::VPU::verifyRollLayoutInfo(mlir::Operation* op) {
    return VPU::verifySameInOutSpecificDimsOrder(
            op, {DimsOrder::NCHW, DimsOrder::NHWC, DimsOrder::CHW, DimsOrder::NC, DimsOrder::C});
}
