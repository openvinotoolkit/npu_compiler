//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/VPU/impl/convert_tile_with_eltwise_to_nce_pool_strategy.hpp"

#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/common_rewriters/convert_tile_with_eltwise_to_nce_pool.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/core/IR/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <llvm/ADT/bit.h>

using namespace vpux;

namespace vpux::VPU::arch50xx {

namespace {

mlir::Value createLegacyWeightsTable(mlir::Location loc, mlir::Value scale, mlir::Value bias, int64_t weightsTableC,
                                     mlir::PatternRewriter& rewriter) {
    auto ctx = rewriter.getContext();
    auto fp32ElemType = mlir::Float32Type::get(ctx);

    const auto pointerShape = Shape({weightsTableC, 1, 1, 2});
    const auto pointerType = mlir::RankedTensorType::get(pointerShape.raw(), fp32ElemType);
    const auto noSparsityPtr = llvm::bit_cast<float>(VPU::NCESparsity::SPARSITY_PTR_WHEN_NO_SPARSITY);
    SmallVector<float> pointerValues;
    pointerValues.reserve(checked_cast<size_t>(weightsTableC * 2));
    for (int64_t ind = 0; ind < weightsTableC; ++ind) {
        pointerValues.push_back(0.0f);
        pointerValues.push_back(noSparsityPtr);
    }
    auto pointerTable = Const::createConst<float>(rewriter, appendLoc(loc, "weight_table_pointers"), pointerType,
                                                  ArrayRef(pointerValues),
                                                  [](Const::ContentSetup& contentSetup) -> Const::ContentSetup {
                                                      return contentSetup.reorder(DimsOrder::NCHW);
                                                  });

    const auto weightsTableOutType =
            mlir::cast<NDTypeInterface>(mlir::RankedTensorType::get({weightsTableC, 1, 1, 4}, fp32ElemType))
                    .changeDimsOrder(DimsOrder::NCHW);
    const SmallVector<SmallVector<int64_t>> staticOffsets = {{0, 0, 0, 0}, {0, 0, 0, 2}, {0, 0, 0, 3}};
    auto weightsTable =
            VPU::ConcatOp::create(rewriter, appendLoc(loc, "weight_table_concat"), weightsTableOutType,
                                  mlir::ValueRange{pointerTable, scale, bias}, getIntArrayOfArray(ctx, staticOffsets))
                    .getOutput();

    const auto si32TableType = mlir::cast<NDTypeInterface>(weightsTable.getType()).changeElemType(getSInt32Type(ctx));
    return Core::ReinterpretCastOp::create(rewriter, appendLoc(loc, "weight_table"), si32TableType, weightsTable)
            .getOutput();
}

class ConvertTileWithEltwiseToNCEPool final : public VPU::ConvertTileWithEltwiseToNCEPoolRewriter {
public:
    ConvertTileWithEltwiseToNCEPool(mlir::MLIRContext* ctx, Logger log)
            : VPU::ConvertTileWithEltwiseToNCEPoolRewriter(ctx, log) {
    }

private:
    VPU::TileWithEltwiseWeightTables createWeightTableOperands(mlir::Location loc, mlir::Value scale, mlir::Value bias,
                                                               int64_t weightsTableC,
                                                               mlir::PatternRewriter& rewriter) const final {
        auto weightsTable = createLegacyWeightsTable(loc, scale, bias, weightsTableC, rewriter);
        return {weightsTable, /*scale=*/nullptr, /*bias=*/nullptr};
    }
};

}  // namespace

void ConvertTileWithEltwiseToNCEPoolStrategy::addPatterns(SmallVector<mlir::RewritePatternSet>& patterns,
                                                          Logger& log) const {
    mlir::RewritePatternSet patternSet(_ctx);
    patternSet.add<ConvertTileWithEltwiseToNCEPool>(_ctx, log);
    patterns.push_back(std::move(patternSet));
}

}  // namespace vpux::VPU::arch50xx
