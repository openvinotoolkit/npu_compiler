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

using namespace vpux;

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
        return errorAt(loc, "Dimension of the input data should be 4. Got {0} D tensor", inShape.size());
    }

    const auto sizes = parseIntArrayAttr<int64_t>(extractImagePatches.sizes());

    if (sizes.size() != 2) {
        return errorAt(loc, "Dimension of sizes attributes is expected to be equal to 2. Got {0}", sizes.size());
    }

    if (sizes[0] <= 0 || sizes[1] <= 0) {
        return errorAt(loc, "Sizes attributes sizeRows and sizeCols should be positive.");
    }

    const auto strides = parseIntArrayAttr<int64_t>(extractImagePatches.strides());

    if (strides.size() != 2) {
        return errorAt(loc, "Dimension of strides attributes is expected to be equal to 2. Got {0}", strides.size());
    }

    if (strides[0] <= 0 || strides[1] <= 0) {
        return errorAt(loc, "strides attributes stridesRows and stridesCols should be positive.");
    }

    const auto rates = parseIntArrayAttr<int64_t>(extractImagePatches.rates());

    if (rates.size() != 2) {
        return errorAt(loc, "Dimension of rates attributes is expected to be equal to 2. Got {0}", rates.size());
    }

    if (rates[0] <= 0 || rates[1] <= 0) {
        return errorAt(loc, "rates attributes ratesRows and ratesCols should be positive.");
    }

    SmallVector<int64_t> output_shape;

    const auto  input_shape = inShape.begin();

    int64_t input_rows = input_shape[2];
    int64_t input_cols = input_shape[3];

    int64_t out_rows(0);
    int64_t out_cols(0);

    if (input_rows == 0 || input_cols == 0) {

        output_shape.push_back(inShape[0]); 
        output_shape.push_back(inShape[1] * sizes[0] * sizes[1]); 
        output_shape.push_back(inShape[2]); 
        output_shape.push_back(inShape[3]); 

        } else {

            if (paddingType == IE::PadType::SAME_UPPER || paddingType == IE::PadType::SAME_LOWER) {
                out_rows = 1 + ( input_rows - 1 ) / strides[0];
                out_cols = 1 + ( input_cols - 1 ) / strides[1];
            } else {
                    if (paddingType == IE::PadType::VALID) {
                        out_rows = ( input_rows - rates[0] * ( sizes[0] - 1) - 1 ) / strides[0] + 1 ;
                        out_cols = ( input_cols - rates[1] * ( sizes[1] - 1) - 1 ) / strides[1] + 1 ;
                    }
                }

            if ( out_rows < 0 )
                out_rows = 0 ;

            if ( out_cols < 0 )
                out_cols = 0 ;

            output_shape.push_back(inShape[0]); 
            output_shape.push_back(inShape[1] * sizes[0] * sizes[1]); 
            output_shape.push_back(out_rows); 
            output_shape.push_back(out_cols); 

        }

    inferredReturnShapes.emplace_back(output_shape, inType.getElementType());
    return mlir::success();
}
