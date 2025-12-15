//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/feasible_memory_scheduler_spilling.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"
#include "vpux/utils/profiling/common.hpp"

#include <functional>
#include <map>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_UPDATESWKERNELPARAMS
#define GEN_PASS_DEF_UPDATESWKERNELPARAMS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

enum SoftmaxOptType : int64_t {
    SoftmaxModeDynamic = 0,
    SoftmaxModeInner = 1,
    SoftmaxModeInnerHardcoded = 2,
    SoftmaxModeOuter = 3
};

uint64_t packAsI32intoU64(int64_t val1, int64_t val2) {
    static constexpr uint64_t bitWidth = sizeof(uint32_t) * CHAR_BIT;
    auto v1 = checked_cast<uint32_t>(val1);
    auto v2 = checked_cast<uint32_t>(val2);
    uint64_t patch = (static_cast<uint64_t>(v2) << bitWidth) | (static_cast<uint64_t>(v1));
    return patch;
}

static SmallVector<uint64_t> generateVpuipSoftmaxAttr(mlir::Value input, mlir::Value output, int64_t axis,
                                                      int64_t padSize, bool hasDynamicShape) {
    SmallVector<uint64_t> storage;
    const auto inShape = getShape(input);
    const auto inOrder = DimsOrder::fromValue(input);
    const auto inMemShape = inOrder.toMemoryOrder(inShape);
    const auto inMemStrides = getMemStrides(input);

    const auto outShape = getShape(output);
    const auto outOrder = DimsOrder::fromValue(output);
    const auto outMemShape = outOrder.toMemoryOrder(outShape);
    const auto outMemStrides = getMemStrides(output);

    int64_t ndims = inShape.size();
    SoftmaxOptType mode = SoftmaxModeDynamic;

    if (hasDynamicShape) {
        storage.push_back(packAsI32intoU64(axis, padSize));
        storage.push_back(packAsI32intoU64(mode, ndims));
        return storage;
    }

    SmallVector<int64_t> inDims;
    SmallVector<int64_t> inStrides;
    SmallVector<int64_t> outStrides;

    for (auto& dim : inMemShape | reversed) {
        inDims.push_back(dim);
    }

    auto bitTypeSize = mlir::cast<vpux::NDTypeInterface>(input.getType()).getElemTypeSize();
    for (auto& stride : inMemStrides | reversed) {
        inStrides.push_back(stride.count() / bitTypeSize.count());
    }
    for (auto& stride : outMemStrides | reversed) {
        outStrides.push_back(stride.count() / bitTypeSize.count());
    }

    inDims.resize(MAX_NUM_DIMS, 0);
    inStrides.resize(MAX_NUM_DIMS, 0);
    outStrides.resize(MAX_NUM_DIMS, 0);

    // excluding dim == 1 from dims
    for (int64_t i = ndims - 1; i >= 0; i--) {
        if (ndims <= 1) {
            break;
        }
        if ((inDims[i] == 1) && (axis != i)) {
            for (int64_t j = i; j < ndims - 1; j++) {
                inDims[j] = inDims[j + 1];
                inStrides[j] = inStrides[j + 1];
                outStrides[j] = outStrides[j + 1];
            }
            axis = (axis > i) ? axis - 1 : axis;
            ndims--;
        }
    }

    // fuse dims if stride allow
    for (int64_t i = ndims - 2; i > axis && i >= 0; i--) {
        if ((inStrides[i + 1] == (inStrides[i] * inDims[i])) && (outStrides[i + 1] == (outStrides[i] * inDims[i]))) {
            inDims[i] = inDims[i + 1] * inDims[i];  // fuse dims
            for (int64_t j = i + 1; j < ndims - 1; j++) {
                inDims[j] = inDims[j + 1];
                inStrides[j] = inStrides[j + 1];
                outStrides[j] = outStrides[j + 1];
            }
            ndims--;
        }
    }

    //    Inner 1 dimension added to make the algorithm work in 'calculateSoftMaxOuter' way
    //    in the case when inner stride  more than size of one tensor element
    if (axis == 0 && ((inStrides[0]) > 1 || (outStrides[0] > 1))) {
        for (int64_t i = ndims; i >= 1; i--) {
            inDims[i] = inDims[i - 1];
            inStrides[i] = inStrides[i - 1];
            outStrides[i] = outStrides[i - 1];
        }
        inDims[0] = 1;
        ndims++;
        axis = 1;
    }
    // works only with ndims >= 3 to simplicity and for speed increase if stride requested
    if (ndims < 3) {
        for (int64_t i = ndims; i < 3 && i >= 0; i++) {
            inStrides[i] = inStrides[ndims - 1] * inDims[ndims - 1];
            outStrides[i] = outStrides[ndims - 1] * inDims[ndims - 1];
            inDims[i] = 1;
        }
        ndims = 3;
    }

    if (0 == axis) {
        if (((inDims[axis] == 2) || (inDims[axis] == 4)) && (padSize == 0) && (inStrides[2] == outStrides[2]) &&
            ((inDims[0]) == (inStrides[1]))) {
            mode = SoftmaxModeInnerHardcoded;
        } else {
            mode = SoftmaxModeInner;
        }
    } else {
        mode = SoftmaxModeOuter;
    }

    storage.push_back(packAsI32intoU64(axis, padSize));
    storage.push_back(packAsI32intoU64(mode, ndims));

    for (int64_t i = 0; i < ndims; i++) {
        storage.push_back(packAsI32intoU64(inDims[i], inDims[i]));
        storage.push_back(packAsI32intoU64(inStrides[i], outStrides[i]));
    }
    return storage;
}

//
// UpdateSwKernelParamsRewriter
//

class UpdateSwKernelParamsRewriter final : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    UpdateSwKernelParamsRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _log(log) {
        setDebugName("UpdateSwKernelParamsRewriter");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp swkernelOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult UpdateSwKernelParamsRewriter::matchAndRewrite(VPUIP::SwKernelOp swKernelOp,
                                                                  mlir::PatternRewriter& /*rewriter*/) const {
    _log.trace("Identified SwKernel operation '{0}' at location '{1}'", swKernelOp, swKernelOp->getLoc());
    auto module = swKernelOp->getParentOfType<mlir::ModuleOp>();
    auto kernelFunc = module.lookupSymbol<mlir::func::FuncOp>(swKernelOp.getKernelFunctionAttr());
    if (kernelFunc == nullptr) {
        return mlir::failure();
    }

    if (kernelFunc->getAttrOfType<mlir::StringAttr>("VPU.kernel_entry") == nullptr) {
        return mlir::failure();
    }

    auto kernelEntryName = getSwKernelEntryName(swKernelOp);

    // Only SoftMax and LogSoftMax need params updated as of now
    if (kernelEntryName != "softmax" && kernelEntryName != "log_softmax") {
        return mlir::failure();
    }

    _log.trace("Try apply params update '{0}' at '{1}'", kernelEntryName, swKernelOp->getLoc());
    for (auto&& kernelRun : swKernelOp.getBody().getOps<VPUIP::SwKernelRun>()) {
        if (!kernelRun.getAttrs().has_value()) {
            return mlir::failure();
        }

        const mlir::ArrayAttr arrayAttrs = kernelRun.getAttrs().value();
        const auto& attrs = arrayAttrs.getValue();
        if (auto arr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(attrs[0])) {
            _log.trace("Attrs was already updated '{0}' at '{1}'", arr, swKernelOp->getLoc());
            return mlir::failure();
        }

        const auto axis = mlir::dyn_cast<mlir::IntegerAttr>(attrs[0]).getInt();
        const auto padSize = mlir::dyn_cast<mlir::IntegerAttr>(attrs[1]).getInt();
        const auto hasDynamicShape = VPUIP::hasBoundedBuffers(swKernelOp);

        const auto params = generateVpuipSoftmaxAttr(kernelRun.getOperand(0), swKernelOp.getResult(0), axis, padSize,
                                                     hasDynamicShape);
        const auto paramsAttr = getIntArrayAttr(kernelRun->getContext(), params);

        const auto newAttrs = SmallVector<mlir::Attribute>{paramsAttr};
        kernelRun.setAttrsAttr(mlir::ArrayAttr::get(kernelRun->getContext(), newAttrs));
    }
    _log.trace("Attrs is updated'{0}' at '{1}'", swKernelOp, swKernelOp->getLoc());

    return mlir::success();
}

//
// UpdateSwKernelParamsPass
//

class UpdateSwKernelParamsPass final : public VPUIP::impl::UpdateSwKernelParamsBase<UpdateSwKernelParamsPass> {
public:
    explicit UpdateSwKernelParamsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UpdateSwKernelParamsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<UpdateSwKernelParamsRewriter>(&ctx, _log);

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createUpdateSwKernelParamsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUpdateSwKernelParamsPass(Logger log) {
    return std::make_unique<UpdateSwKernelParamsPass>(log);
}
