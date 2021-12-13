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
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"

#include <ngraph/coordinate.hpp>
#include <ngraph/op/max_pool.hpp>
#include <ngraph/util.hpp>
#include <ngraph/validation_util.hpp>

using namespace vpux;

mlir::LogicalResult vpux::IE::MaxPoolOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::MaxPoolOpAdaptor maxPool(operands, attrs);
    if (mlir::failed(maxPool.verify(loc))) {
        return mlir::failure();
    }

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(maxPool.pads_end());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(maxPool.pads_begin());
    const auto windowShape = parseIntArrayAttr<int64_t>(maxPool.kernel_size());
    const auto windowStrides = parseIntArrayAttr<int64_t>(maxPool.strides());
    const auto roundingType = maxPool.rounding_type().getValue();

    const auto inType = maxPool.input().getType().cast<mlir::ShapedType>().getElementType();
    const auto inShape = maxPool.input().getType().cast<mlir::ShapedType>().getShape();

    const auto outputShape = ngraph::infer_batched_pooling_forward(
            nullptr, ngraph::Shape(inShape.begin(), inShape.end()),
            ngraph::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),
            ngraph::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),
            ngraph::Shape(windowShape.begin(), windowShape.end()),
            ngraph::Strides(windowStrides.begin(), windowStrides.end()), true,
            roundingType == vpux::IE::RoundingType::CEIL);

    const auto shapeI64 = to_small_vector(outputShape.get_shape() | transformed([](size_t val) {
                                              return checked_cast<int64_t>(val);
                                          }));
    inferredReturnShapes.emplace_back(shapeI64, inType);

    return mlir::success();
}

mlir::Value vpux::IE::MaxPoolOp::reifyTile(const TileInfo& outputTile, mlir::OpBuilder& builder) {
    const auto origInputShape = getShape(input());

    const auto tileConf =
            backInferPoolTile(outputTile, origInputShape, kernel_size(), strides(), pads_begin(), pads_end());

    const std::array<int64_t, 2> padsBegin = {tileConf.pads.top, tileConf.pads.left};
    const std::array<int64_t, 2> padsEnd = {tileConf.pads.bottom, tileConf.pads.right};

    const auto inputTileVal = IE::makeTile(builder, getLoc(), input(), tileConf.inputTile, "input");

    const auto tileName = llvm::formatv("output tile {0}", outputTile.offsets).str();
    const auto tileLoc = appendLoc(getLoc(), tileName);

    const auto tiledResType = getDenseTileType(getType(), outputTile.offsets, outputTile.shape);

    auto tiledOp = builder.create<IE::MaxPoolOp>(tileLoc, tiledResType, inputTileVal, kernel_sizeAttr(), stridesAttr(),
                                                 getIntArrayAttr(builder, padsBegin), getIntArrayAttr(builder, padsEnd),
                                                 rounding_typeAttr(), post_opAttr());

    return tiledOp.output();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::IE::MaxPoolOp::serialize(EMU::BlobWriter& writer) {
    const auto kernel = VPUIP::createOrder3(kernel_sizeAttr());
    const auto strides = VPUIP::createOrder3(stridesAttr());
    const auto padsBegin = VPUIP::createOrder3(pads_beginAttr());
    const auto padsEnd = VPUIP::createOrder3(pads_endAttr());

    EMU::BlobWriter::String type;
    type = writer.createString("max");

    MVCNN::PoolingParamsBuilder builder(writer);
    builder.add_pool_method(type);
    builder.add_kernel(&kernel);
    builder.add_strides(&strides);
    builder.add_pads_begin(&padsBegin);
    builder.add_pads_end(&padsEnd);
    builder.add_exclude_pad(false);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PoolingParams});
}