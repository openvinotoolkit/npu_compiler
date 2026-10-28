//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/IR/dialect.hpp"
#include "vpux/compiler/dialect/ELF/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUASM/dialect.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/constraints.hpp"
namespace vpux::ELF {
#define GEN_PASS_DECL_SETENTRYPOINT
#define GEN_PASS_DEF_SETENTRYPOINT
#include "vpux/compiler/dialect/ELF/passes.hpp.inc"
}  // namespace vpux::ELF

using namespace vpux;
using MappedInferenceFormat = config::NPUConstraints::MappedInferenceFormat;

namespace {
class SetEntryPointPass : public ELF::impl::SetEntryPointBase<SetEntryPointPass> {
public:
    SetEntryPointPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

template <typename InferenceOpType>
mlir::SymbolRefAttr getMPI(ELF::MainOp mainOp) {
    for (auto dataSection : mainOp.getOps<ELF::DataSectionOp>()) {
        auto inferenceOps = dataSection.getBlock()->getOps<InferenceOpType>();
        if (inferenceOps.empty()) {
            continue;
        }
        const auto mpiCount = std::distance(inferenceOps.begin(), inferenceOps.end());
        VPUX_THROW_UNLESS(mpiCount == 1, "Expected single {0}, found {1}", InferenceOpType::getOperationName(),
                          mpiCount);

        auto inferenceOp = *inferenceOps.begin();
        auto mpiRef = mlir::FlatSymbolRefAttr::get(
                mlir::cast<mlir::SymbolOpInterface>(inferenceOp.getOperation()).getNameAttr());

        return mlir::SymbolRefAttr::get(dataSection.getNameAttr(), {mpiRef});
    }
    VPUX_THROW("Could not find {0}", InferenceOpType::getOperationName());
}

void SetEntryPointPass::safeRunOnFunc() {
    auto netFunc = getOperation();

    auto mainOps = to_small_vector(netFunc.getOps<ELF::MainOp>());
    VPUX_THROW_UNLESS(mainOps.size() == 1, "Expected exactly one ELF mainOp. Got {0}", mainOps.size());
    auto elfMain = mainOps[0];
    auto getSymTab = [](ELF::MainOp main) -> ELF::CreateSymbolTableSectionOp {
        for (auto symTab : main.getOps<ELF::CreateSymbolTableSectionOp>()) {
            if (symTab.getName() == "symtab") {
                return symTab;
            }
        }
        VPUX_THROW("Coult not find default symtab");
        return nullptr;
    };

    auto ctx = netFunc->getContext();
    bool useDirectMmi =
            (config::getNPUConstraints(ctx).mappedInferenceFormat == MappedInferenceFormat::ManagedMappedInference);

    auto mpiRef = useDirectMmi ? getMPI<VPUASM::ManagedMappedInferenceOp>(elfMain)
                               : getMPI<VPUASM::MappedInferenceOp>(elfMain);

    ELF::CreateSymbolTableSectionOp symTab = getSymTab(elfMain);

    auto builder = mlir::OpBuilder::atBlockEnd(symTab.getBlock());

    ELF::SymbolOp::create(builder, elfMain.getLoc(),        // location
                          "entry",                          // sym_name
                          mpiRef,                           // reference
                          ELF::SymbolType::VPU_STT_ENTRY);  // type
}
}  // namespace

//
// createSetEntryPointPass
//

std::unique_ptr<mlir::Pass> ELF::createSetEntryPointPass(Logger log) {
    return std::make_unique<SetEntryPointPass>(log);
}
