//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/bounded_buffer.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"

#include <intel_npu/prefix.hpp>

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LogicalResult.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_UNGROUPBOUNDEDBUFFERSASFUNCARGS
#define GEN_PASS_DEF_UNGROUPBOUNDEDBUFFERSASFUNCARGS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// UngroupBoundedBuffersAsFuncArgs
//

class UngroupBoundedBuffersAsFuncArgs final :
        public VPUIP::impl::UngroupBoundedBuffersAsFuncArgsBase<UngroupBoundedBuffersAsFuncArgs> {
public:
    explicit UngroupBoundedBuffersAsFuncArgs(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void UngroupBoundedBuffersAsFuncArgs::safeRunOnModule() {
    auto module = getOperation();

    IE::CNNNetworkOp netInfo;
    mlir::func::FuncOp func;
    IE::CNNNetworkOp::getFromModule(module, netInfo, func);

    // TODO: multiple functions support TBD
    // Track: E#118218

    const auto isDynamicOperand = [&](auto type) {
        return type.template isa<VPUIP::BoundedBufferType>();
    };
    const auto hasDynamicInputs = llvm::any_of(func.getFunctionType().getInputs(), isDynamicOperand);
    const auto hasDynamicOutputs = llvm::any_of(func.getFunctionType().getResults(), isDynamicOperand);
    if (!hasDynamicInputs && !hasDynamicOutputs) {
        return;
    }

    auto ctx = module.getContext();
    OpBuilderLogger builderLog(_log.nest());
    auto& entryBlock = func.front();
    auto builder = mlir::OpBuilder::atBlockBegin(&entryBlock, &builderLog);

    _log.trace("Old block inputs: {0}", entryBlock.getArgumentTypes());

    _log = _log.nest(4);

    struct BoundedBuffer {
        mlir::MemRefType dataType;
        mlir::MemRefType dynamicShapeType;
    };
    auto unpackBoundedBuffer = [](VPUIP::BoundedBufferType type) -> BoundedBuffer {
        // TODO: support for other buffer types will be added separately
        // Track E#118173
        return {type.getData().cast<mlir::MemRefType>(), type.getDynamicShape().cast<mlir::MemRefType>()};
    };

    const auto addDataInfo = [&](auto boundedType, auto& infoBlock, auto dataInfo, int index, int dataBufferCount) {
        const auto& [boundedMemRef, dynamicShapeMemRef] = unpackBoundedBuffer(boundedType);
        _log.trace("Memory to store dynamic tensor: {0}, {1}", boundedMemRef, dynamicShapeMemRef);

        const auto typeAttr = mlir::TypeAttr::get(
                mlir::RankedTensorType::get(dynamicShapeMemRef.getShape(), dynamicShapeMemRef.getElementType()));
        const auto nameAttr =
                mlir::StringAttr::get(ctx, std::string(intel_npu::SHAPE_TENSOR_PREFIX) + dataInfo[index].getName());

        auto insertionPointAfter = std::next(infoBlock.begin(), dataBufferCount);
        auto infoBuilder = mlir::OpBuilder(&infoBlock, insertionPointAfter, builder.getListener());
        infoBuilder.create<IE::DataInfoOp>(takeOpLoc(func, StringLiteral("{0}"), func.getName()), nameAttr, typeAttr,
                                           /*OptionalAttr originalShape*/ nullptr,
                                           /*OptionalAttr friendlyName*/ nullptr,
                                           /*OptionalAttr inputName*/ nullptr,
                                           /*OptionalAttr tensorNames*/ nullptr,
                                           /*profilingSectionsCount=*/0);
        _log.trace("Added new DataInfo '{0}' with type {1}", nameAttr, typeAttr);
    };
    const auto originalInputSize = func.getFunctionType().getInputs().size();
    for (const auto& index : irange(originalInputSize)) {
        const auto input = func.getFunctionType().getInputs()[index];
        if (const auto boundedType = input.dyn_cast_or_null<VPUIP::BoundedBufferType>()) {
            _log.trace("Found dynamic input {0}", input);

            addDataInfo(boundedType, netInfo.getInputsInfo().front(), netInfo.getInputsDataInfo(), index,
                        /*current dataBufferCount*/ netInfo.getInputsDataInfo().size());

            const auto& [boundedMemRef, dynamicShapeMemRef] = unpackBoundedBuffer(boundedType);
            auto arg0 = entryBlock.insertArgument(index + 1, boundedMemRef, func.getLoc());
            auto arg1 = entryBlock.insertArgument(entryBlock.getNumArguments(), dynamicShapeMemRef, func.getLoc());

            auto alloc0 = builder.create<mlir::memref::AllocOp>(func.getLoc(), boundedMemRef.cast<mlir::MemRefType>());
            auto alloc1 =
                    builder.create<mlir::memref::AllocOp>(func.getLoc(), dynamicShapeMemRef.cast<mlir::MemRefType>());

            auto copy0 = builder.create<VPUIP::CopyOp>(func.getLoc(), arg0, alloc0);
            auto copy1 = builder.create<VPUIP::CopyOp>(func.getLoc(), arg1, alloc1);

            auto groupOp = builder.create<VPUIP::GroupBoundedBufferOp>(func.getLoc(), copy0, copy1);

            _log.trace("Wrapped newly added network arguments into {0}", groupOp);

            entryBlock.getArgument(index).replaceAllUsesWith(groupOp->getResult(0));
            entryBlock.eraseArgument(index);

            _log.trace("All uses of an old tensor replaced with value {0}", groupOp->getResult(0));
        }
    }

    _log = _log.unnest(4);
    _log.trace("New block inputs: {0}", entryBlock.getArgumentTypes());

    auto returnOps = func.getOps<mlir::func::ReturnOp>();
    if (const auto returnOpsCount = std::distance(returnOps.begin(), returnOps.end()); returnOpsCount != 1) {
        VPUX_THROW("Expected to find one ReturnOp, but got {0}", returnOpsCount);
    }

    auto returnOp = *returnOps.begin();
    builder.setInsertionPoint(returnOp);

    _log.trace("Old function outputs: {0}", returnOp->getOperandTypes());
    _log = _log.nest(4);
    const auto originalOutputSize = func.getFunctionType().getResults().size();
    for (const auto& index : irange(originalOutputSize)) {
        const auto output = func.getFunctionType().getResults()[index];
        if (const auto boundedType = output.dyn_cast_or_null<VPUIP::BoundedBufferType>()) {
            _log.trace("Found dynamic output {0}", output);

            addDataInfo(boundedType, netInfo.getOutputsInfo().front(), netInfo.getOutputsDataInfo(), index,
                        /*current dataBufferCount*/ netInfo.getOutputsDataInfo().size());

            const auto operand = returnOp.getOperand(index);
            auto ungroupOp = builder.create<VPUIP::UngroupBoundedBufferOp>(func.getLoc(), operand);
            _log.trace("Wrapped newly added network results into {0}", ungroupOp);

            returnOp->eraseOperand(index);
            returnOp->insertOperands(index, ungroupOp.getData());
            returnOp->insertOperands(returnOp->getOpOperands().size(), ungroupOp.getDynamicShape());
        }
    }
    _log = _log.unnest(4);
    _log.trace("New function outputs: {0}", returnOp->getOperandTypes());
    const auto newOutFuncTypes =
            mlir::FunctionType::get(func.getContext(), entryBlock.getArgumentTypes(), returnOp->getOperandTypes());
    func.setType(newOutFuncTypes);
}

}  // namespace

//
// UngroupBoundedBuffersAsFuncArgs
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUngroupBoundedBuffersAsFuncArgsPass(Logger log) {
    return std::make_unique<UngroupBoundedBuffersAsFuncArgs>(log);
}
