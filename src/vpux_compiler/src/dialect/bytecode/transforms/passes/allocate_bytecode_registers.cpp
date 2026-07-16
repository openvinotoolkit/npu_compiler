//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/IR/attributes.hpp"
#include "vpux/compiler/dialect/bytecode/IR/dialect.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/register.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/section.hpp"
#include "vpux/compiler/dialect/bytecode/transforms/passes.hpp"
#include "vpux/utils/core/dense_map.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Analysis/Liveness.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <set>
#include <vpux/utils/core/range.hpp>

namespace vpux {
#define GEN_PASS_DECL_ALLOCATEBYTECODEREGISTERS
#define GEN_PASS_DEF_ALLOCATEBYTECODEREGISTERS
#include "vpux/compiler/dialect/bytecode/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

namespace {

struct LiveInterval {
    bytecode::VirtualGeneralRegisterOp op;
    size_t start;
    size_t end;
    int16_t regNum = -1;
};

// Resolve the number of parameters (P) for a function by looking up the referenced
// FunctionTypeAttr in the type section. The `bytecode.func` verifier already guarantees
// that the reference resolves to a FunctionTypeAttr, so this lookup is expected to succeed.
int64_t resolveNumParams(bytecode::FuncOp funcOp) {
    const auto refName = funcOp.getFunctionTypeRef();
    auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();
    VPUX_THROW_WHEN(moduleOp == nullptr,
                    "bytecode.func @{0} is not nested inside a ModuleOp; cannot resolve function type @{1}",
                    funcOp.getSymName(), refName);

    for (auto typeSection : moduleOp.getOps<bytecode::TypeSectionOp>()) {
        for (auto typeOp : typeSection.getContent().getOps<bytecode::TypeOp>()) {
            if (typeOp.getSymName() != refName) {
                continue;
            }
            auto funcTypeAttr = mlir::dyn_cast<bytecode::FunctionTypeAttr>(typeOp.getValue());
            VPUX_THROW_WHEN(funcTypeAttr == nullptr,
                            "bytecode.func @{0} references type @{1}, which is not a function type",
                            funcOp.getSymName(), refName);
            return static_cast<int64_t>(funcTypeAttr.getArguments().size());
        }
    }

    VPUX_THROW("bytecode.func @{0} references type @{1}, which is not present in the type section", funcOp.getSymName(),
               refName);
}

struct BlockOpPositions {
    size_t firstOpPosition;
    size_t onePastLastOpPosition;
};

int16_t linearScanAllocate(mlir::Region& body, SmallVector<LiveInterval>& intervals) {
    DenseMap<mlir::Operation*, size_t> opIndex;
    DenseMap<mlir::Block*, BlockOpPositions> blockPositions;
    size_t i = 0;
    for (auto& block : body) {
        const size_t firstOp = i;
        for (auto& op : block) {
            opIndex[&op] = i++;
        }
        blockPositions[&block] = {firstOp, i};
    }

    SmallVector<bytecode::VirtualGeneralRegisterOp> virtualOps;
    for (auto& block : body) {
        for (auto vgrOp : block.getOps<bytecode::VirtualGeneralRegisterOp>()) {
            virtualOps.push_back(vgrOp);
        }
    }

    mlir::Liveness liveness(body.getParentOp());
    for (auto virtualOp : virtualOps) {
        if (virtualOp->use_empty()) {
            virtualOp.erase();
            continue;
        }

        size_t minStart = std::numeric_limits<size_t>::max();
        size_t maxEnd = 0;
        bool crossBlock = false;

        for (auto& block : body) {
            const auto* info = liveness.getLiveness(&block);
            VPUX_THROW_WHEN(info == nullptr, "Liveness information missing for block in bytecode.func body");
            if (info->isLiveIn(virtualOp.getResult()) || info->isLiveOut(virtualOp.getResult())) {
                const auto posIt = blockPositions.find(&block);
                VPUX_THROW_UNLESS(posIt != blockPositions.end(), "Block not found in blockPositions map");
                minStart = std::min(minStart, posIt->second.firstOpPosition);
                maxEnd = std::max(maxEnd, posIt->second.onePastLastOpPosition);
                crossBlock = true;
            }
        }

        if (!crossBlock) {
            for (auto& use : virtualOp.getResult().getUses()) {
                const auto idxIt = opIndex.find(use.getOwner());
                VPUX_THROW_UNLESS(idxIt != opIndex.end(), "Use owner not found in opIndex map");
                const auto idx = idxIt->second;
                minStart = std::min(minStart, idx);
                maxEnd = std::max(maxEnd, idx);
            }
            ++maxEnd;
        }
        intervals.push_back({virtualOp, minStart, maxEnd, -1});
    }

    llvm::sort(intervals, [](const LiveInterval& a, const LiveInterval& b) {
        return a.start < b.start;
    });

    // Linear scan with half-open interval expiry (end <= current.start).
    std::set<int16_t> freePool;
    SmallVector<size_t> active;  // sorted by interval end
    int16_t nextReg = 0;
    for (size_t k = 0; k < intervals.size(); ++k) {
        auto& current = intervals[k];

        auto it = active.begin();
        while (it != active.end() && intervals[*it].end <= current.start) {
            freePool.insert(intervals[*it].regNum);
            it = active.erase(it);
        }

        if (!freePool.empty()) {
            current.regNum = *freePool.begin();
            freePool.erase(freePool.begin());
        } else {
            VPUX_THROW_WHEN(nextReg == std::numeric_limits<int16_t>::max(),
                            "Ran out of bytecode general registers during allocation");
            current.regNum = nextReg++;
        }

        auto pos = std::lower_bound(active.begin(), active.end(), k, [&](size_t a, size_t b) {
            return intervals[a].end < intervals[b].end;
        });
        active.insert(pos, k);
    }

    return nextReg;
}

void rewriteGeneralRegisters(mlir::OpBuilder& builder, MutableArrayRef<LiveInterval> intervals) {
    auto registerType = bytecode::RegisterType::get(builder.getContext());
    for (auto& iv : intervals) {
        builder.setInsertionPoint(iv.op);
        auto concrete = builder.create<bytecode::GeneralRegisterOp>(iv.op.getLoc(), registerType, iv.regNum);
        iv.op.getResult().replaceAllUsesWith(concrete.getResult());
        iv.op.erase();
    }
}

void rewriteParameterRegisters(mlir::Block& block, int16_t numGeneralRegisters, int64_t numParams) {
    SmallVector<bytecode::VirtualParameterRegisterOp> virtualParams;
    for (auto op : block.getOps<bytecode::VirtualParameterRegisterOp>()) {
        virtualParams.push_back(op);
    }
    if (virtualParams.empty() && numParams == 0) {
        return;
    }

    auto* ctx = block.getParentOp()->getContext();
    auto registerType = bytecode::RegisterType::get(ctx);

    DenseMap<int16_t, bytecode::GeneralRegisterOp> paramRegs;
    for (auto virtualParam : virtualParams) {
        const auto regNum =
                static_cast<int16_t>(static_cast<int64_t>(numGeneralRegisters) + virtualParam.getParamIndex());
        auto& slot = paramRegs[regNum];
        if (slot == nullptr) {
            mlir::OpBuilder paramBuilder(virtualParam);
            slot = paramBuilder.create<bytecode::GeneralRegisterOp>(virtualParam.getLoc(), registerType, regNum);
        }
        virtualParam.getResult().replaceAllUsesWith(slot.getResult());
        virtualParam.erase();
    }

    // Pin the highest parameter slot so that `num_general_registers`, derived at serialization
    // time as `max(regNum) + 1`, reflects the full calling-convention frame size `G + P` even
    // when trailing parameters are unreferenced by the function body.
    if (numParams > 0) {
        const auto lastParamRegNum = static_cast<int16_t>(static_cast<int64_t>(numGeneralRegisters) + numParams - 1);
        if (paramRegs.find(lastParamRegNum) == paramRegs.end()) {
            auto* terminator = block.getTerminator();
            mlir::OpBuilder paramBuilder(terminator);
            paramBuilder.create<bytecode::GeneralRegisterOp>(terminator->getLoc(), registerType, lastParamRegNum);
        }
    }
}

void validateCallFrameLayout(bytecode::FuncOp funcOp, mlir::Block& block, int16_t numGeneralRegisters,
                             int64_t numParams) {
    constexpr auto MAX_ADDRESSABLE_REGISTER_COUNT = static_cast<int64_t>(std::numeric_limits<int16_t>::max()) + 1;

    const auto totalRegisters = static_cast<int64_t>(numGeneralRegisters) + numParams;
    VPUX_THROW_WHEN(totalRegisters > MAX_ADDRESSABLE_REGISTER_COUNT,
                    "Bytecode call frame would exceed the addressable register space (G={0}, P={1})",
                    numGeneralRegisters, numParams);

    for (auto virtualParam : block.getOps<bytecode::VirtualParameterRegisterOp>()) {
        const auto paramIndex = static_cast<int64_t>(virtualParam.getParamIndex());
        VPUX_THROW_WHEN(paramIndex < 0 || paramIndex >= numParams,
                        "bytecode.func @{0} uses parameter register index {1}, but the signature only defines {2} "
                        "parameter(s)",
                        funcOp.getSymName(), paramIndex, numParams);
    }
}

void allocate(bytecode::FuncOp funcOp) {
    auto& body = funcOp.getBody();
    auto& entryBlock = body.front();

    SmallVector<LiveInterval> intervals;
    const auto numGeneralRegisters = linearScanAllocate(body, intervals);
    const auto numParams = resolveNumParams(funcOp);
    validateCallFrameLayout(funcOp, entryBlock, numGeneralRegisters, numParams);

    mlir::OpBuilder builder(funcOp);
    rewriteGeneralRegisters(builder, intervals);
    rewriteParameterRegisters(entryBlock, numGeneralRegisters, numParams);
}

}  // namespace

namespace vpux {

class AllocateBytecodeRegistersPass final : public impl::AllocateBytecodeRegistersBase<AllocateBytecodeRegistersPass> {
public:
    explicit AllocateBytecodeRegistersPass(const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final {
        getOperation().walk([](bytecode::FuncOp funcOp) {
            allocate(funcOp);
        });
    }
};

}  // namespace vpux

std::unique_ptr<mlir::Pass> vpux::bytecode::createAllocateBytecodeRegistersPass(const Logger& log) {
    return std::make_unique<AllocateBytecodeRegistersPass>(log);
}
