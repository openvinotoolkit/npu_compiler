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

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::ConvertOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ConvertOpAdaptor cvt(operands, attrs);
    if (mlir::failed(cvt.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = cvt.input().getType().cast<mlir::RankedTensorType>();
    const auto dstElemType = cvt.dstElemType().getValue();

    inferredReturnShapes.emplace_back(inType.getShape(), dstElemType);
    return mlir::success();
}

bool vpux::IE::ConvertOp::areCastCompatible(mlir::TypeRange inputs, mlir::TypeRange outputs) {
    if (inputs.size() != 1 || outputs.size() != 1) {
        return false;
    }

    const auto input = inputs.front().dyn_cast<mlir::RankedTensorType>();
    const auto output = outputs.front().dyn_cast<mlir::RankedTensorType>();

    if (!input || !output || input.getShape() != output.getShape()) {
        return false;
    }

    return true;
}

namespace {

#include <vpux/compiler/dialect/IE/rewriters/generated/convert.hpp.inc>

}  // namespace

void vpux::IE::ConvertOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext*) {
    populateWithGenerated(patterns);
}

mlir::OpFoldResult vpux::IE::ConvertOp::fold(ArrayRef<mlir::Attribute> operands) {
    VPUX_THROW_UNLESS(operands.size() == 1, "Wrong number of operands : {0}", operands.size());

    if (const auto attr = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        return attr.convertElemType(dstElemType());
    }

    return nullptr;
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::IE::ConvertOp::serialize(EMU::BlobWriter& writer) {

    MVCNN::ConvertParamsBuilder builder(writer);
    builder.add_scale(checked_cast<float>(1.0));
    builder.add_bias(checked_cast<float>(0.0));
    builder.add_from_detection_output(false);
    builder.add_have_batch(false);
    builder.add_batch_id(0);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ConvertParams});
}
