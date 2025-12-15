//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/weights_separation.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/Hashing.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>

#include <cstdint>

namespace vpux::VPU {
#define GEN_PASS_DECL_CONCATINITINPUTS
#define GEN_PASS_DEF_CONCATINITINPUTS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

// Returns a unique name for concatenated init inputs.
std::string getUniqueConcatenatedNameOfInitInputs(ArrayRef<StringRef> names) {
    if (names.size() == 1) {
        // Note: preserve the original name for a single argument
        return names.front().str();
    }

    llvm::hash_code hashCode = llvm::hash_combine(names);
    return formatv("{0}hash_{1}_concat", Const::IMPORTED_WEIGHT_PREFIX, hashCode);
}

class ConcatInitInputs final : public VPU::impl::ConcatInitInputsBase<ConcatInitInputs> {
public:
    enum class Mode { Unspecified, GenerateMain, GenerateInit, GenerateAll };

    explicit ConcatInitInputs(const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;

    size_t concatenateFunctionInputs(mlir::func::FuncOp funcOp);
    void updateNetworkInfo(net::NetworkInfoOp netInfo, mlir::func::FuncOp funcOp, size_t newInputsOffset);
};

size_t ConcatInitInputs::concatenateFunctionInputs(mlir::func::FuncOp funcOp) {
    SmallVector<size_t> indices(funcOp.getNumArguments(), 0);
    std::iota(indices.begin(), indices.end(), 0);

    VPU::obfuscateInputs(_log, appendLoc(funcOp.getLoc(), "obfuscated_inputs"), funcOp, indices,
                         [](mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input, ArrayRef<int64_t> offsets,
                            ArrayRef<int64_t> sizes) {
                             return builder.create<IE::SliceOp>(loc, input, offsets, sizes);
                         });
    return 0;
}

void ConcatInitInputs::updateNetworkInfo(net::NetworkInfoOp netInfo, mlir::func::FuncOp funcOp,
                                         size_t newInputsOffset) {
    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(&getContext(), &builderLog);

    auto& inputsRegion = netInfo.getInputsInfo();
    const auto namesToBeMerged = to_small_vector(inputsRegion.front().getOps<net::DataInfoOp>() |
                                                 transformed([](net::DataInfoOp op) -> StringRef {
                                                     return op.getName();
                                                 }));

    // update input types
    // Note: preserve original, non-constant inputs information
    net::eraseSectionEntries(inputsRegion, newInputsOffset);
    builder.setInsertionPointToEnd(&inputsRegion.front());

    builder.create<net::DataInfoOp>(appendLoc(netInfo.getLoc(), "concat_in"),
                                    getUniqueConcatenatedNameOfInitInputs(namesToBeMerged),
                                    funcOp.getFunctionType().getInput(newInputsOffset));
}

void ConcatInitInputs::safeRunOnModule() {
    auto moduleOp = getOperation();

    net::NetworkInfoOp netInfo;
    mlir::func::FuncOp entryPointFunc;
    net::NetworkInfoOp::getFromModule(moduleOp, netInfo, entryPointFunc);

    const auto offset = concatenateFunctionInputs(entryPointFunc);
    updateNetworkInfo(netInfo, entryPointFunc, offset);
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createConcatInitInputsPass(const Logger& log) {
    return std::make_unique<ConcatInitInputs>(log);
}
