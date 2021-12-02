//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/VPUIP/blob_reader.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/IE/float16.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

namespace {

std::pair<VPUIP::BlobWriter::Vector<uint16_t>, VPUIP::BlobWriter::Vector<uint16_t>> serializeScalesAndZeroPoints(
        mlir::Value input, mlir::Value output, VPUIP::BlobWriter& writer) {
    const auto inType = input.getType().cast<mlir::MemRefType>().getElementType();
    const auto outType = output.getType().cast<mlir::MemRefType>().getElementType();

    const auto qType = inType.isa<mlir::quant::QuantizedType>() ? inType.cast<mlir::quant::QuantizedType>()
                                                                : outType.cast<mlir::quant::QuantizedType>();

    const auto getRawFP16 = [](auto val) {
        const auto valFP16 = float16(val);
        return valFP16.to_bits();
    };

    const auto getVecFP16 = [&](auto range) {
        return writer.createVector(range | transformed(getRawFP16));
    };

    SmallVector<double> scales;
    SmallVector<int64_t> zeroPoints;
    if (qType.isa<mlir::quant::UniformQuantizedType>()) {
        auto quantParams = qType.cast<mlir::quant::UniformQuantizedType>();
        scales = {quantParams.getScale()};
        zeroPoints = {quantParams.getZeroPoint()};
    } else if (qType.isa<mlir::quant::UniformQuantizedPerAxisType>()) {
        auto quantParams = qType.cast<mlir::quant::UniformQuantizedPerAxisType>();
        scales = {quantParams.getScales().begin(), quantParams.getScales().end()};
        zeroPoints = {quantParams.getZeroPoints().begin(), quantParams.getZeroPoints().end()};
    } else {
        VPUX_THROW("Unsupported quantized type {0}", qType);
    }

    return {getVecFP16(scales), getVecFP16(zeroPoints)};
}

}  // namespace

mlir::LogicalResult vpux::VPUIP::verifyOp(QuantCastUPAOp op) {
    const auto inType = op.input().getType().cast<mlir::MemRefType>().getElementType();
    const auto outType = op.output().getType().cast<mlir::MemRefType>().getElementType();
    const auto arch = VPU::getArch(op.getOperation()->getParentOfType<mlir::ModuleOp>());

    if (arch != VPU::ArchKind::MTL && (inType.isBF16() || outType.isBF16())) {
        return errorAt(op, "BF16 is only supported by MTL");
    }

    if (!((inType.isF16() && outType.isa<mlir::quant::QuantizedType>()) ||
          (inType.isa<mlir::quant::QuantizedType>() && outType.isF16()))) {
        return errorAt(op, "Unsupported quantize/dequantize conversion '{0}' -> '{1}'", inType, outType);
    }

    const auto qType = inType.isa<mlir::quant::QuantizedType>() ? inType.cast<mlir::quant::QuantizedType>()
                                                                : outType.cast<mlir::quant::QuantizedType>();

    if (!qType.getStorageType().isInteger(CHAR_BIT)) {
        return errorAt(op, "Unsupported quantized storage type '{0}'", qType.getStorageType());
    }

    if (const auto perAxis = qType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        if (perAxis.getQuantizedDimension() != 1) {
            return errorAt(op, "Only per-channel quantization is supported");
        }

        // TODO: support per-channel zero point
        const auto zeroPoints = perAxis.getZeroPoints();
        if (zeroPoints.empty()) {
            return errorAt(op, "Missing zero points");
        }

        const auto firstVal = zeroPoints[0];
        for (auto val : zeroPoints.drop_front()) {
            if (val != firstVal) {
                return errorAt(op, "Only splat zero points are supported");
            }
        }
    } else if (!qType.isa<mlir::quant::UniformQuantizedType>()) {
        return errorAt(op, "Unsupported quantized type '{0}'", qType);
    }

    return mlir::success();
}

void vpux::VPUIP::QuantCastUPAOp::inferLayoutInfo(mlir::Operation* origOp, IE::LayerLayoutInfo& info) {
    const auto inType = origOp->getOperand(0).getType().cast<mlir::ShapedType>().getElementType();
    const auto outType = origOp->getResult(0).getType().cast<mlir::ShapedType>().getElementType();

    const auto qType = inType.isa<mlir::quant::QuantizedType>() ? inType.cast<mlir::quant::QuantizedType>()
                                                                : outType.cast<mlir::quant::QuantizedType>();

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
        info.fill(info.getInput(0));
    }
}

void vpux::VPUIP::QuantCastUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                        mlir::Value output) {
    build(builder, state, input, output, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::QuantCastUPAOp::serialize(BlobWriter& writer) {
    auto scalesAndZeroPoints = serializeScalesAndZeroPoints(input(), output(), writer);

    MVCNN::QuantizeParamsBuilder builder(writer);
    builder.add_scale(scalesAndZeroPoints.first);
    builder.add_zero(scalesAndZeroPoints.second);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_QuantizeParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseQuantCast(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                         ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask*) {
    VPUX_THROW_UNLESS(inputs.size() == 1, "UPAQuantCast supports only 1 input, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPAQuantCast supports only 1 output, got {0}", outputs.size());
    return builder.create<VPUIP::QuantCastUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
}
