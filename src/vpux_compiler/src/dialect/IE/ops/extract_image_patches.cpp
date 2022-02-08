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

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
//#include "vpux/compiler/core/attributes/shape.hpp"
//#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

//TODO

mlir::LogicalResult vpux::IE::ExtractImagePatchesOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ExtractImagePatchesOpAdaptor extractImagePatches(operands, attrs);
    if (mlir::failed(extractImagePatches.verify(loc))) {
        return mlir::failure();
    }

    const auto paddingType = extractImagePatches.auto_pad().getValue();
    const auto inType = extractImagePatches.data().getType().cast<mlir::ShapedType>();
    const auto inShape = inType.getShape();

    if (inShape.size() != 4) {
        return errorAt(loc, "Dimension of the input data should be 4. Got {0} D tensor",
                        inShape.size());
    }

    const auto sizes = parseIntArrayAttr<int32_t>(extractImagePatches.sizes());

    if (sizes.size() != 2) {
        return errorAt(loc, "Dimension of sizes attributes is expected to be equal to 2. Got {0}", sizes.size());
    }

    if (sizes[0] <= 0 || sizes[1] <= 0) {
        return errorAt(loc, "Sizes attributes sizeRows and sizeCols should be positive.");
    }

    const auto strides = parseIntArrayAttr<int32_t>(extractImagePatches.strides());
    
    if (strides.size() != 2) {
        return errorAt(loc, "Dimension of strides attributes is expected to be equal to 2. Got {0}", strides.size());
    }

    if (strides[0] <= 0 || strides[1] <= 0) {
        return errorAt(loc, "strides attributes stridesRows and stridesCols should be positive.");
    }

    const auto rates = parseIntArrayAttr<int32_t>(extractImagePatches.rates());
    
    if (rates.size() != 2) {
        return errorAt(loc, "Dimension of rates attributes is expected to be equal to 2. Got {0}", rates.size());
    }

    if (rates[0] <= 0 || rates[1] <= 0) {
        return errorAt(loc, "rates attributes ratesRows and ratesCols should be positive.");
    }

    SmallVector<int64_t> output_shape;

    if (paddingType == IE::PadType::SAME_UPPER || paddingType == IE::PadType::SAME_LOWER) {
//      SMTH - the input is padded by zeros to match the output size. In case of odd padding value an extra padding is added at the end (at the beginning).
//      output_shape = ;
    } else if (paddingType == IE::PadType::VALID) {
//      SMTH -  do not use padding
//      output_shape = ;
    }
//     data: the 4-D tensor of type T with shape [batch, depth, in_rows, in_cols].
//     output: 4-D tensor with shape [batch, size[0] * size[1] * depth, out_rows, out_cols]

    output_shape.push_back(inType.getShape()[0]); // ?? - batch
//     output_shape.push_back(inType.getShape()[1] * ... * ... ); // ?? - size[0] * size[1] * depth
//     output_shape.push_back(..); // ?? -  out_rows
//     output_shape.push_back(..); // ?? - out_cols
//     Note out_rows and out_cols are the dimensions of the output patches.

    inferredReturnShapes.emplace_back(output_shape, inType.getElementType());
    return mlir::success();
}
