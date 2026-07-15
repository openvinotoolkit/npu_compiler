//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/conversion/passes/VPUMI40XX2VPUASM/symbolization_type_converter.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/utils/symbolization.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

template <typename OperationType>
class VPUASMSymbolizationPattern : public SymbolizationPattern<OperationType> {
public:
    using Base = VPUASMSymbolizationPattern<OperationType>;
    using SymbolMapper = typename SymbolizationPattern<OperationType>::SymbolMapper;
    using SectionMapper = typename SymbolizationPattern<OperationType>::SectionMapper;
    using OpAdaptor = typename SymbolizationPattern<OperationType>::OpAdaptor;

    VPUASMSymbolizationPattern(mlir::func::FuncOp netFunc, SymbolizationTypeConverter& typeConverter,
                               SymbolMapper& mapper, SectionMapper& sectionMap, mlir::MLIRContext* ctx, Logger log)
            : SymbolizationPattern<OperationType>(netFunc, typeConverter, mapper, sectionMap, ctx), _log(log) {
    }

protected:
    mlir::ArrayAttr vectorizeBarriers(mlir::Operation::operand_range&& barrierRange) const {
        mlir::MLIRContext* ctx = this->getContext();
        llvm::SmallVector<mlir::Attribute> barrierVec(barrierRange.size());

        auto u16Attr = [&ctx](uint16_t value) -> mlir::IntegerAttr {
            auto u16Type = mlir::IntegerType::get(ctx, 16, mlir::IntegerType::Unsigned);
            return mlir::IntegerAttr::get(u16Type, value);
        };

        for (auto barrier : llvm::enumerate(barrierRange)) {
            auto barrierVal = barrier.value();
            auto barrierIdx = barrier.index();
            // hard-cast since it should a by-default-expected relationship
            auto barrierOp = mlir::cast<VPUMI40XX::ConfigureBarrierOp>(barrierVal.getDefiningOp());

            barrierVec[barrierIdx] = u16Attr(barrierOp.getId());
        }

        return mlir::ArrayAttr::get(ctx, barrierVec);
    };

    Logger _log;
};

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
