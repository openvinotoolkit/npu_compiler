//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/utils/interpolate_utils.hpp"

#include <mlir/IR/PatternMatch.h>

namespace vpux::IE {

bool isLegalInterpolateOp(IE::InterpolateOp op, bool interpolateAsSEOp, LogCb logCb);

//
// MapBilinearInterpolateOnDPUBaseRewriter
//

class MapBilinearInterpolateOnDPUBaseRewriter : public mlir::OpRewritePattern<IE::InterpolateOp> {
public:
    MapBilinearInterpolateOnDPUBaseRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::InterpolateOp>(ctx), _log(log) {
        setDebugName("MapBilinearInterpolateOnDPURewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::Value createIdentityPooling(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value input,
                                      vpux::NDTypeInterface outType) const;
    mlir::Value scaleOnAxis(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value input,
                            vpux::NDTypeInterface outType, int64_t inputSize, int64_t outputSize, vpux::Dim axis,
                            IE::MapCoordFuncT mapCoord) const;

    Logger _log;
};

}  // namespace vpux::IE
