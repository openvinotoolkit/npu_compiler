//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>

using namespace vpux;

//
// build
//

void vpux::VPU::GroupSparseTensorOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value data,
                                           bool isWeights, VPU::SparsityCompressionAttr sparsityCompression) {
    build(builder, state, data, nullptr, nullptr, isWeights, sparsityCompression);
}

void vpux::VPU::GroupSparseTensorOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value data,
                                           mlir::Value sparsityMap, bool isWeights,
                                           VPU::SparsityCompressionAttr sparsityCompression) {
    build(builder, state, data, sparsityMap, nullptr, isWeights, sparsityCompression);
}

void vpux::VPU::GroupSparseTensorOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value data,
                                           mlir::Value sparsityMap, mlir::Value storageElementTable, bool isWeights,
                                           VPU::SparsityCompressionAttr sparsityCompression) {
    const auto isWeightsAttr = isWeights ? mlir::UnitAttr::get(builder.getContext()) : nullptr;
    build(builder, state, data, sparsityMap, storageElementTable, isWeightsAttr, sparsityCompression, nullptr);
}

void vpux::VPU::GroupSparseTensorOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value data,
                                           mlir::Value sparsityMap, mlir::Value storageElementTable,
                                           VPU::SEAttr seAttr) {
    build(builder, state, data, sparsityMap, storageElementTable, nullptr, nullptr, seAttr);
}

//
// inferReturnTypes
//

mlir::LogicalResult vpux::VPU::GroupSparseTensorOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*ranges*/,
        SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::GroupSparseTensorOpAdaptor groupOp(operands, attrs, prop);
    if (mlir::failed(groupOp.verify(loc))) {
        return mlir::failure();
    }

    const auto dataType = groupOp.getData().getType();
    const auto sparsityMapType = groupOp.getSparsityMap() != nullptr ? groupOp.getSparsityMap().getType() : nullptr;
    const auto storageElementTableType =
            groupOp.getStorageElementTable() != nullptr ? groupOp.getStorageElementTable().getType() : nullptr;

    inferredReturnTypes.push_back(
            VPU::SparseTensorType::get(dataType, sparsityMapType, storageElementTableType, groupOp.getIsWeightsAttr(),
                                       groupOp.getSparsityCompressionAttr(), groupOp.getSeAttrAttr()));

    return mlir::success();
}

//
// MoveViewLikeOps
//

/*
 * Patterns such as the following:
 *      Data   Const SM   SETable
 *        \       |       /
 *        GroupSparseTensor
 *       /              \
 *     Slice           Slice
 *
 * get transformed into:
 *      Data    Const SM*   SETable     Const Data  Const SM* SETable
 *        |        |         |             |           |       |
 *      Slice      |       Slice         Slice         |      Slice
 *        \        |        /               \          |      /
 *         GroupSparseTensor                 GroupSparseTensor
 *
 * This can allow the Slice canonicalizer convert the operation into a constant transformation.
 * The sparsity map for weights is attached directly as a transformation since the original constant's type has the
 * shape flattened for each output channel (i.e. OCx1x1xSIZExi1), making it incompatible with the attributes of the
 * Slice operation. Therefore, it is applied as a transformation to the dense constant before the
 * transformation that generates the sparsity map.
 * StorageElementTableOp has its own canonicalizer and SliceOp will be fused into it.
 */

namespace {

class MoveViewLikeOps final : public mlir::OpRewritePattern<VPU::GroupSparseTensorOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPU::GroupSparseTensorOp origOp, mlir::PatternRewriter& rewriter) const final {
        if (origOp.getData() == nullptr) {
            return mlir::failure();
        }
        VPU::StorageElementTableOp seTableOp = nullptr;
        if (origOp.getStorageElementTable() != nullptr) {
            seTableOp = origOp.getStorageElementTable().getDefiningOp<VPU::StorageElementTableOp>();
            if (seTableOp == nullptr) {
                return mlir::failure();
            }
        }

        bool changed = false;
        for (auto userOp : llvm::make_early_inc_range(origOp.getOutput().getUsers())) {
            changed |= llvm::TypeSwitch<mlir::Operation*, bool>(userOp)
                               .Case<VPU::SliceOp>([&](VPU::SliceOp sliceUserOp) {
                                   return tryMoveSliceUser(sliceUserOp, origOp, seTableOp, rewriter);
                               })
                               .Case<VPU::ExpandOp>([&](VPU::ExpandOp expandUserOp) {
                                   return tryMoveExpandUser(expandUserOp, origOp, rewriter);
                               })
                               .Case<VPU::ReshapeOp>([&](VPU::ReshapeOp reshapeUserOp) {
                                   return tryMoveReshapeUser(reshapeUserOp, origOp, rewriter);
                               })
                               .Case<VPU::LayoutCastOp>([&](VPU::LayoutCastOp layoutCastUserOp) {
                                   return tryMoveLayoutCastUser(layoutCastUserOp, origOp, rewriter);
                               })
                               .Default([](mlir::Operation*) {
                                   return false;
                               });
        }
        return changed ? mlir::success() : mlir::failure();
    }

    bool tryMoveSliceUser(VPU::SliceOp sliceUserOp, VPU::GroupSparseTensorOp origOp,
                          VPU::StorageElementTableOp seTableOp, mlir::PatternRewriter& rewriter) const {
        const auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceUserOp.getStaticOffsets());
        const auto sliceSizes = parseIntArrayAttr<int64_t>(sliceUserOp.getStaticSizes());

        // In case the sparse type represents weights, the sparsity map has the weight set (ICxKHxKW) flattened
        // and aligned to 128 bits. As such, if the slice happens on any of these dimensions, it is difficult to
        // adapt the slice parameters to the flattened shape when the slice is offsetted, so these cases are skipped
        if (origOp.getIsWeights()) {
            const auto offsetNotOnOC = std::any_of(sliceOffsets.begin() + 1, sliceOffsets.end(), [](int64_t offset) {
                return offset != 0;
            });
            if (offsetNotOnOC) {
                return false;
            }
        }

        auto seAttr = origOp.getSeAttr().value_or(nullptr);
        auto sparsityCompressionAttr = origOp.getSparsityCompression().value_or(nullptr);
        if (sparsityCompressionAttr != nullptr) {
            sparsityCompressionAttr =
                    VPU::tileSparsityCompression(sparsityCompressionAttr, ShapeRef(sliceOffsets), ShapeRef(sliceSizes));
        }

        // In case the parent operation is a constant, fold the view-like operation directly into the constant
        // manually. This is necessary because the last transformation in the constant will be either Sparsify
        // or GetSparsityMap, which can change the type of the constant (e.g. GetSparsityMap will convert
        // 16x16x1x1xf16 into 16x1x1x128xui8, to align the data). As these sparsity-related transformations must
        // remain at the end of the list of transformations, the new transformation that is introduced for this
        // view-like operation will end up being inserted before the sparsity-related transformation, so it
        // must have its parameters associated to the type of the constant before the sparsity-related
        // transformation
        const auto tryFoldIntoConstant = [&](mlir::Value value, ShapeRef offsets, ShapeRef sizes) -> mlir::Value {
            auto constOp = value.getDefiningOp<Const::DeclareOp>();
            if (constOp == nullptr) {
                return nullptr;
            }
            auto newContentAttr = constOp.transformContentAttr().subview(offsets, sizes).get();
            auto newConstOp =
                    rewriter.create<Const::DeclareOp>(constOp.getLoc(), newContentAttr.getType(), newContentAttr);
            return newConstOp.getOutput();
        };

        const auto rewriteData = [&](mlir::Value origData, ShapeRef offsets, ShapeRef sizes) -> mlir::Value {
            if (auto constResult = tryFoldIntoConstant(origData, offsets, sizes)) {
                return constResult;
            }
            auto sliceOp = rewriter.create<VPU::SliceOp>(origOp->getLoc(), origData,
                                                         getIntArrayAttr(origOp.getContext(), offsets),
                                                         getIntArrayAttr(origOp.getContext(), sizes));
            return sliceOp.getResult();
        };

        const auto rewriteSparsityMap = [&](mlir::Value origSparsityMap, mlir::Value newData, ShapeRef offsets,
                                            ShapeRef sizes) -> mlir::Value {
            if (origSparsityMap == nullptr) {
                return nullptr;
            }
            if (auto constResult = tryFoldIntoConstant(origSparsityMap, offsets, sizes)) {
                return constResult;
            }
            // Note: in case the sparse type is a weight, the slice is expected to be over a non-flattened dimension
            // (see the check above), so it is safe to reuse the original offsets
            SmallVector<int64_t> smOffsets(offsets.raw());
            SmallVector<int64_t> smSizes(sizes.raw());
            if (origOp.getIsWeights()) {
                auto newDataShape = mlir::cast<NDTypeInterface>(newData.getType()).getShape();
                auto newSMShape = VPU::NCESparsity::inferWeightsSparsityMapShape(newDataShape);
                smSizes = newSMShape.raw();
            }
            auto sliceOp = rewriter.create<VPU::SliceOp>(origOp->getLoc(), origOp.getSparsityMap(),
                                                         getIntArrayAttr(origOp.getContext(), smOffsets),
                                                         getIntArrayAttr(origOp.getContext(), smSizes));
            return sliceOp.getResult();
        };

        // Data
        auto newDataOffsets = Shape(sliceOffsets);
        auto newDataSizes = Shape(sliceSizes);
        if (seAttr != nullptr) {
            seAttr = seAttr.extractTile(ShapeRef(sliceOffsets), ShapeRef(sliceSizes),
                                        mlir::cast<NDTypeInterface>(origOp.getData().getType()).getShape(),
                                        newDataOffsets, newDataSizes);
        }
        auto newDataValue = rewriteData(origOp.getData(), newDataOffsets, newDataSizes);

        // SM
        auto newSparsityMapValue =
                rewriteSparsityMap(origOp.getSparsityMap(), newDataValue, ShapeRef(sliceOffsets), ShapeRef(sliceSizes));

        // SETable
        mlir::Value newSETableValue = nullptr;
        if (seTableOp != nullptr) {
            const auto seDepth = getShape(seTableOp.getOutput())[Dims4D::Act::C];
            const auto [seTableOffsets, seTableSizes] = VPU::getUpdatedSliceOffsetsAndShapesForSETable(
                    seDepth, seTableOp.getSeSize(), sliceOffsets, sliceSizes);

            newSETableValue = rewriter.createOrFold<VPU::SliceOp>(sliceUserOp.getLoc(), origOp.getStorageElementTable(),
                                                                  seTableOffsets, seTableSizes);
        }
        rewriter.replaceOpWithNewOp<VPU::GroupSparseTensorOp>(sliceUserOp, newDataValue, newSparsityMapValue,
                                                              newSETableValue, origOp.getIsWeightsAttr(),
                                                              sparsityCompressionAttr, seAttr);
        return true;
    }

    bool tryMoveExpandUser(VPU::ExpandOp expandUserOp, VPU::GroupSparseTensorOp origOp,
                           mlir::PatternRewriter& rewriter) const {
        if (!origOp.getIsWeights()) {
            return false;
        }

        // In case the parent operation is a constant, fold the view-like operation directly into the constant
        // manually. This is necessary because the last transformation in the constant will be either Sparsify
        // or GetSparsityMap, which can change the type of the constant (e.g. GetSparsityMap will convert
        // 16x16x1x1xf16 into 16x1x1x128xui8, to align the data). As these sparsity-related transformations must
        // remain at the end of the list of transformations, the new transformation that is introduced for this
        // view-like operation will end up being inserted before the sparsity-related transformation, so it
        // must have its parameters associated to the type of the constant before the sparsity-related
        // transformation
        auto tryFoldIntoConstant = [&](mlir::Value value, ArrayRef<int64_t> padsBegin,
                                       ArrayRef<int64_t> padsEnd) -> mlir::Value {
            auto constOp = value.getDefiningOp<Const::DeclareOp>();
            if (constOp == nullptr) {
                return nullptr;
            }
            auto newContentAttr =
                    constOp.transformContentAttr().padWithZero(ShapeRef(padsBegin), ShapeRef(padsEnd)).get();
            auto newConstOp =
                    rewriter.create<Const::DeclareOp>(constOp.getLoc(), newContentAttr.getType(), newContentAttr);
            return newConstOp.getOutput();
        };

        const auto rewriteData = [&](mlir::Value origData, ArrayRef<int64_t> padsBegin,
                                     ArrayRef<int64_t> padsEnd) -> mlir::Value {
            if (auto constResult = tryFoldIntoConstant(origData, padsBegin, padsEnd)) {
                return constResult;
            }
            auto expandOp = rewriter.create<VPU::ExpandOp>(origOp->getLoc(), origData,
                                                           getIntArrayAttr(origOp->getContext(), padsBegin),
                                                           getIntArrayAttr(origOp->getContext(), padsEnd));
            return expandOp.getOutput();
        };

        const auto rewriteSparsityMap = [&](mlir::Value origSparsityMap, mlir::Value newData,
                                            ArrayRef<int64_t> padsBegin, ArrayRef<int64_t> padsEnd) -> mlir::Value {
            if (origSparsityMap == nullptr) {
                return nullptr;
            }
            if (auto constResult = tryFoldIntoConstant(origSparsityMap, padsBegin, padsEnd)) {
                return constResult;
            }
            auto newDataShape = mlir::cast<NDTypeInterface>(newData.getType()).getShape();
            auto newSMShape = VPU::NCESparsity::inferWeightsSparsityMapShape(newDataShape);
            auto origSMShape = mlir::cast<NDTypeInterface>(origSparsityMap.getType()).getShape();
            SmallVector<int64_t> smPadsBeing(origSMShape.size());
            SmallVector<int64_t> smPadsEnd(origSMShape.size());
            for (size_t i = 0; i < origSMShape.size(); ++i) {
                auto diff = newSMShape[Dim(i)] - origSMShape[Dim(i)];
                if (diff > 0) {
                    smPadsEnd[i] = static_cast<int64_t>(diff);
                }
            }
            auto expandOp = rewriter.create<VPU::ExpandOp>(origOp->getLoc(), origOp.getSparsityMap(),
                                                           getIntArrayAttr(origOp->getContext(), smPadsBeing),
                                                           getIntArrayAttr(origOp->getContext(), smPadsEnd));
            return expandOp.getOutput();
        };

        const auto padsBegin = parseIntArrayAttr<int64_t>(expandUserOp.getPadsBegin());
        const auto padsEnd = parseIntArrayAttr<int64_t>(expandUserOp.getPadsEnd());

        auto newData = rewriteData(origOp.getData(), padsBegin, padsEnd);
        auto newSparsityMap = rewriteSparsityMap(origOp.getSparsityMap(), newData, padsBegin, padsEnd);

        rewriter.replaceOpWithNewOp<VPU::GroupSparseTensorOp>(
                expandUserOp, newData, newSparsityMap, origOp.getStorageElementTable(), origOp.getIsWeightsAttr(),
                origOp.getSparsityCompressionAttr(), origOp.getSeAttrAttr());
        return true;
    }

    bool tryMoveReshapeUser(VPU::ReshapeOp reshapeUserOp, VPU::GroupSparseTensorOp origOp,
                            mlir::PatternRewriter& rewriter) const {
        if (!origOp.getIsWeights()) {
            return false;
        }

        // In case the parent operation is a constant, fold the view-like operation directly into the constant
        // manually. This is necessary because the last transformation in the constant will be either Sparsify
        // or GetSparsityMap, which can change the type of the constant (e.g. GetSparsityMap will convert
        // 16x16x1x1xf16 into 16x1x1x128xui8, to align the data). As these sparsity-related transformations must
        // remain at the end of the list of transformations, the new transformation that is introduced for this
        // view-like operation will end up being inserted before the sparsity-related transformation, so it
        // must have its parameters associated to the type of the constant before the sparsity-related
        // transformation
        const auto tryFoldIntoConstant = [&](mlir::Value value, ShapeRef shape) -> mlir::Value {
            auto constOp = value.getDefiningOp<Const::DeclareOp>();
            if (constOp == nullptr) {
                return nullptr;
            }
            auto newContentAttr = constOp.transformContentAttr().reshape(shape).get();
            auto newConstOp =
                    rewriter.create<Const::DeclareOp>(constOp.getLoc(), newContentAttr.getType(), newContentAttr);
            return newConstOp.getOutput();
        };

        const auto rewriteData = [&](mlir::Value origData, ShapeRef shape) -> mlir::Value {
            if (auto constResult = tryFoldIntoConstant(origData, shape)) {
                return constResult;
            }
            auto reshapeOp = rewriter.create<VPU::ReshapeOp>(origOp->getLoc(), origData, /*shape=*/nullptr,
                                                             /*specialZero=*/false,
                                                             getIntArrayAttr(origOp->getContext(), shape));
            return reshapeOp.getOutput();
        };

        const auto rewriteSparsityMap = [&](mlir::Value origSparsityMap, mlir::Value newData,
                                            ShapeRef shape) -> mlir::Value {
            if (origSparsityMap == nullptr) {
                return nullptr;
            }
            if (auto constResult = tryFoldIntoConstant(origSparsityMap, shape)) {
                return constResult;
            }
            auto newDataShape = mlir::cast<NDTypeInterface>(newData.getType()).getShape();
            auto newSMShape = VPU::NCESparsity::inferWeightsSparsityMapShape(newDataShape);
            auto reshapeOp = rewriter.create<VPU::ReshapeOp>(origOp->getLoc(), origSparsityMap, /*shape=*/nullptr,
                                                             /*specialZero=*/false,
                                                             getIntArrayAttr(origOp->getContext(), newSMShape));
            return reshapeOp.getOutput();
        };

        auto shape = mlir::cast<NDTypeInterface>(reshapeUserOp.getOutput().getType()).getShape();

        auto newData = rewriteData(origOp.getData(), shape);
        auto newSparsityMap = rewriteSparsityMap(origOp.getSparsityMap(), newData, shape);

        rewriter.replaceOpWithNewOp<VPU::GroupSparseTensorOp>(
                reshapeUserOp, newData, newSparsityMap, origOp.getStorageElementTable(), origOp.getIsWeightsAttr(),
                origOp.getSparsityCompressionAttr(), origOp.getSeAttrAttr());
        return true;
    }

    bool tryMoveLayoutCastUser(VPU::LayoutCastOp layoutCastUserOp, VPU::GroupSparseTensorOp origOp,
                               mlir::PatternRewriter& rewriter) const {
        if (!origOp.getIsWeights()) {
            return false;
        }
        auto dataLayoutCastOp = rewriter.create<VPU::LayoutCastOp>(origOp->getLoc(), origOp.getData(),
                                                                   layoutCastUserOp.getDstOrderAttr());
        // Note: the sparsity map does not have its layout changed, as the GetSparsityMap transformation resets its
        // layout to the default
        rewriter.replaceOpWithNewOp<VPU::GroupSparseTensorOp>(
                layoutCastUserOp, dataLayoutCastOp.getOutput(), origOp.getSparsityMap(),
                origOp.getStorageElementTable(), origOp.getIsWeightsAttr(), origOp.getSparsityCompressionAttr(),
                origOp.getSeAttrAttr());
        return true;
    }
};

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::VPU::GroupSparseTensorOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results,
                                                                 mlir::MLIRContext* ctx) {
    results.add<MoveViewLikeOps>(ctx);
}

//
// TilingViewLikeOpInterface
//

InputTiling vpux::VPU::GroupSparseTensorOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    VPU::StorageElementTableOp seTableOp = nullptr;
    if (auto seTable = getStorageElementTable()) {
        if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(seTable)) {
            if (auto blockParent = this->getOperation()->getParentOp()) {
                VPUX_THROW_WHEN(blockParent->getNumOperands() < blockArg.getArgNumber(),
                                "Number of block operands {0} doesn't match with argument number {1}",
                                blockParent->getNumOperands(), blockArg.getArgNumber());
                seTableOp =
                        blockParent->getOperand(blockArg.getArgNumber()).getDefiningOp<VPU::StorageElementTableOp>();
            }
        } else {
            seTableOp = getStorageElementTable().getDefiningOp<VPU::StorageElementTableOp>();
        }
    }

    auto inputTile = TileInfo(outputTile.shape, outputTile.offsets, outputTile.axis);
    if (auto seAttr = getSeAttr().value_or(nullptr)) {
        seAttr.extractTile(outputTile.offsets, outputTile.shape, getBoundedShape(getData()), inputTile.offsets,
                           inputTile.shape);
    }

    SmallVector<TileInfo> inputTiles = {inputTile};
    auto sparsityMap = getSparsityMap();

    if (sparsityMap != nullptr) {
        inputTiles.push_back(outputTile);
    }

    if (seTableOp != nullptr) {
        const auto seDepth = getShape(seTableOp.getOutput())[Dims4D::Act::C];
        const auto [seTableOffsets, seTableSizes] = VPU::getUpdatedSliceOffsetsAndShapesForSETable(
                seDepth, seTableOp.getSeSize(), outputTile.offsets.raw(), outputTile.shape.raw());

        const Shape axis(seTableOffsets.size(), 1);
        inputTiles.push_back(TileInfo(ShapeRef(seTableSizes), ShapeRef(seTableOffsets), axis));
    }
    return InputTiling(inputTiles);
}

void vpux::VPU::GroupSparseTensorOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& outputTile,
                                                 ShapeRef outputShape) {
    VPUX_THROW_WHEN(inputTiling.tiles.empty(), "There is no tiling for {0}", getLoc());
    if (auto seAttr = getSeAttr().value_or(nullptr)) {
        auto inputTile = inputTiling.tiles.front();
        const auto inputShape = seAttr.backInferInputShape(outputShape);
        seAttr = seAttr.extractTile(outputTile.offsets, outputTile.shape, inputShape, inputTile.offsets,
                                    inputTile.shape);
        setSeAttrAttr(seAttr);

        auto dataType = mlir::cast<vpux::NDTypeInterface>(getData().getType());
        const auto sparsityMapType = getSparsityMap() != nullptr ? getSparsityMap().getType() : nullptr;
        const auto storageElementTableType =
                getStorageElementTable() != nullptr ? getStorageElementTable().getType() : nullptr;

        auto newType = VPU::SparseTensorType::get(dataType, sparsityMapType, storageElementTableType,
                                                  getIsWeightsAttr(), getSparsityCompressionAttr(), seAttr);

        getOutput().setType(newType);
    }
}
