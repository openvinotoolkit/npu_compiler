//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes/map_bilinear_interpolate_on_DPU.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE::arch37xx {
#define GEN_PASS_DECL_MAPBILINEARINTERPOLATEONDPUPASS
#define GEN_PASS_DEF_MAPBILINEARINTERPOLATEONDPUPASS
#include "vpux/compiler/NPU37XX/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE::arch37xx

using namespace vpux;

namespace {

//
// MapBilinearInterpolateOnDPUPass
//

class MapBilinearInterpolateOnDPUPass final :
        public IE::arch37xx::impl::MapBilinearInterpolateOnDPUPassBase<MapBilinearInterpolateOnDPUPass> {
public:
    explicit MapBilinearInterpolateOnDPUPass(const bool interpolateAsSEOp, Logger log)
            : _interpolateAsSEOp(interpolateAsSEOp) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

public:
    class MapBilinearInterpolateOnDPURewriter;

private:
    void safeRunOnFunc() final;

private:
    bool _interpolateAsSEOp;
};

class MapBilinearInterpolateOnDPUPass::MapBilinearInterpolateOnDPURewriter final :
        public vpux::IE::MapBilinearInterpolateOnDPUBaseRewriter {
public:
    MapBilinearInterpolateOnDPURewriter(mlir::MLIRContext* ctx, Logger log)
            : vpux::IE::MapBilinearInterpolateOnDPUBaseRewriter(ctx, log) {
        setDebugName("MapBilinearInterpolateOnDPURewriterVPUX37XX");
    }
};

mlir::LogicalResult MapBilinearInterpolateOnDPUPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // When this parameter has a value, it probably comes from LIT test.
    // Override the default
    if (interpolateAsSEOp.hasValue()) {
        _interpolateAsSEOp = interpolateAsSEOp.getValue();
    }

    return mlir::success();
}

void MapBilinearInterpolateOnDPUPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::InterpolateOp>([&](IE::InterpolateOp op) {
        return isLegalInterpolateOp(op, _interpolateAsSEOp, logCb);
    });

    target.addLegalOp<IE::ExpandOp>();
    target.addLegalOp<IE::AvgPoolOp>();
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::GroupConvolutionOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<MapBilinearInterpolateOnDPURewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMapBilinearInterpolateOnDPUPass
//

std::unique_ptr<mlir::Pass> vpux::IE::arch37xx::createMapBilinearInterpolateOnDPUPass(const bool interpolateAsSEOp,
                                                                                      Logger log) {
    return std::make_unique<MapBilinearInterpolateOnDPUPass>(interpolateAsSEOp, log);
}
