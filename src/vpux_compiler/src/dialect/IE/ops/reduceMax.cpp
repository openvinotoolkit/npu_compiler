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

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ReduceMaxOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ReduceMaxOpAdaptor reduceMax(operands, attrs);
    if (mlir::failed(reduceMax.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = reduceMax.input().getType().cast<mlir::ShapedType>();
    const auto inShape = reduceMax.input().getType().cast<mlir::ShapedType>().getShape();
    const auto keep_dims = reduceMax.keep_dims().getValue();
    auto axes = IE::constInputToData(loc, reduceMax.axes()).getValue();
    SmallVector<int64_t> outShape;

    std::sort(axes.begin(), axes.end());
    bool isAllUnique = std::unique(axes.begin(), axes.end()) == axes.end();
    if (!isAllUnique) {
        return errorAt(loc, "Axes values should be unique");
    }

    // Add to outShape the values with indices not found in axes_set.
    for (size_t i = 0; i < inShape.size(); i++) {
        if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
            outShape.push_back(inShape[i]);
        } else if (keep_dims) {
            outShape.push_back(1);
        }
    }

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::IE::ReduceMaxOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::ReduceParamsBuilder builder(writer);

    EMU::BlobWriter::String type;
    type = writer.createString("max");

    builder.add_keep_dims(checked_cast<bool>(keep_dims()));
    builder.add_operation(type);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ReduceParams});
}
