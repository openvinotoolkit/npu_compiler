//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>

namespace vpux {
namespace IE {

template <typename ConcreteOp>
mlir::LogicalResult findQuantizeOrQuantizedNCE(ConcreteOp origOp, mlir::PatternRewriter& rewriter,
                                               mlir::Value eltwiseInput,
                                               SmallVector<mlir::Operation*>& eltwiseToQuantizeOps, Logger log);
template <typename ConcreteOp>
mlir::LogicalResult removeQuantOrFusedQuant(ConcreteOp origOp, mlir::PatternRewriter& rewriter,
                                            ArrayRef<mlir::Operation*> eltwiseToQuantizeOps,
                                            mlir::Operation* quantOrQuantizedNCE, mlir::Type elementType, Logger log);

//
// QuantizeWithTwoInputsNCEEltwiseOpGeneric
//

//
// Case 1:
//      [input 1]       [input 2]
//          |               |
//      (quantize)      (quantize)
//          |               |
//         u8 -(EltwiseOp)- u8
//
// Case 2:
//             [input 1]
//                 |
//             (quantize)
//          |               |
//         u8 -(EltwiseOp)- u8
//
// Case 3:
//      [input 1]      [u8_Conv_u8]
//          |               |
//      (quantize)
//          |               |
//         u8 -(EltwiseOp)- u8
//

template <typename ConcreteOp>
class QuantizeWithTwoInputsNCEEltwiseOpGeneric final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    QuantizeWithTwoInputsNCEEltwiseOpGeneric(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

class QuantizeWithAvgPool final : public mlir::OpRewritePattern<IE::AvgPoolOp> {
public:
    QuantizeWithAvgPool(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AvgPoolOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AvgPoolOp avgPoolOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <typename ConcreteOp>
class QuantizeWithNCEOp final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    QuantizeWithNCEOp(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};
}  // namespace IE
}  // namespace vpux
