//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"

namespace {
const size_t bits64 = sizeof(uint64_t) * CHAR_BIT;
const size_t bits128 = 2 * bits64;
}  // namespace

namespace vpux {
namespace VPUMI40XX {

uint64_t computeMaskHi(mlir::ArrayAttr barriers) {
    auto barriersVector = parseIntArrayAttr<uint8_t>(barriers);
    uint64_t mask = 0;
    for (auto barrier : barriersVector) {
        VPUX_THROW_WHEN(barrier >= bits128, "Barrier physical ID out of range: got {0}", barrier);
        if (barrier >= bits64) {
            mask |= static_cast<uint64_t>(1) << (barrier - bits64);
        }
    }
    return mask;
}

uint64_t computeMaskLo(mlir::ArrayAttr barriers) {
    auto barriersVector = parseIntArrayAttr<uint8_t>(barriers);
    uint64_t mask = 0;
    for (auto barrier : barriersVector) {
        if (barrier < bits64) {
            mask |= static_cast<uint64_t>(1) << barrier;
        }
    }
    return mask;
}

bool isConfigureBarrierOpType(const mlir::Operation::operand_range& barriers) {
    return std::all_of(barriers.begin(), barriers.end(), [](mlir::Value barrier) {
        return barrier.getDefiningOp<VPUMI40XX::ConfigureBarrierOp>() != nullptr;
    });
}

// Update indexes in list of operations
size_t reindexList(VPURegMapped::TaskOpInterface head) {
    if (!head)
        return 0;

    auto ctx = head.getOperation()->getContext();
    const uint32_t listIdx = head.getIndexType().getListIdx();
    const uint32_t tileIdx = head.getIndexType().getTileIdx();
    uint32_t taskIdx = 0;

    do {
        head.getOperation()->getResult(0).setType(VPURegMapped::IndexType::get(ctx, tileIdx, listIdx, taskIdx));
        taskIdx++;

        auto headId = llvm::find_if(head.getOperation()->getResult(0).getUsers(), [&head](mlir::Operation* op) {
            if (auto prev = mlir::dyn_cast<VPURegMapped::TaskOpInterface>(op)) {
                if (prev.getPreviousTask() == head)
                    return true;
            }
            return false;
        });

        head = headId != head.getOperation()->getResult(0).getUsers().end()
                       ? mlir::cast<VPURegMapped::TaskOpInterface>(*headId)
                       : nullptr;

    } while (head);

    return taskIdx;
}

MappedInferenceOp getMPI(mlir::func::FuncOp mainFunc) {
    auto mpiOps = to_small_vector(mainFunc.getOps<MappedInferenceOp>());
    VPUX_THROW_WHEN(mpiOps.size() != 1, "IR needs to have exactly one MPI OP. Got {0}", mpiOps.size());
    return mpiOps[0];
}

VPURegMapped::ExecutionGroupOp getNextGroup(VPURegMapped::ExecutionGroupOp op) {
    auto users = op.getEndIndexes()[0].getUsers();
    auto nexOpIt = llvm::find_if(users, [&op](mlir::Operation* user) {
        auto nextUser = mlir::dyn_cast<VPURegMapped::ExecutionGroupOp>(user);
        return nextUser && llvm::all_of(nextUser.getPreviousTaskIdx(), [&op](mlir::Value operand) {
                   return operand.getDefiningOp() == op.getOperation();
               });
    });

    op = nexOpIt != users.end() ? mlir::cast<VPURegMapped::ExecutionGroupOp>(*nexOpIt) : nullptr;
    return op;
}

void printIndex(llvm::raw_ostream& os, VPURegMapped::IndexType index, llvm::StringRef head, llvm::StringRef middle,
                llvm::StringRef end) {
    os << head << "Index: " << middle << index.getTileIdx() << ":" << index.getListIdx() << ":" << index.getValue()
       << end;
}

bool checkBarrierProductionRelationship(mlir::Operation* barr, VPUMI40XX::ExecutableTaskOpInterface exec) {
    auto barrOp = mlir::dyn_cast_or_null<VPUMI40XX::ConfigureBarrierOp>(barr);
    if (barrOp) {
        auto it = llvm::find_if(exec.updateBarriers(), [&barrOp](mlir::Value val) {
            return val == barrOp.getResult();
        });
        return it != exec.updateBarriers().end();
    }

    return false;
}

size_t reindexEnqueueList(VPURegMapped::EnqueueOp head) {
    if (!head)
        return 0;

    auto ctx = head.getOperation()->getContext();
    const uint32_t listIdx = head.getType().getListIdx();
    const uint32_t tileIdx = head.getType().getTileIdx();
    uint32_t taskIdx = 0;

    do {
        head.getOperation()->getResult(0).setType(VPURegMapped::IndexType::get(ctx, tileIdx, listIdx, taskIdx));
        taskIdx++;

        auto headId = llvm::find_if(head.getOperation()->getResult(0).getUsers(), [&head](mlir::Operation* op) {
            if (auto next = mlir::dyn_cast<VPURegMapped::EnqueueOp>(op)) {
                if (next.getPreviousTaskIdx() == head)
                    return true;
            }
            return false;
        });

        head = headId != head.getOperation()->getResult(0).getUsers().end()
                       ? mlir::cast<VPURegMapped::EnqueueOp>((*headId))
                       : nullptr;

    } while (head);

    return taskIdx;
}

uint32_t generateTileMask(mlir::ArrayRef<uint32_t> usedTileIndexes) {
    // this offset is actually maybe generation-specific, so shouldn't be
    // exposed to VPUMI40XX dialect that is generic for NPU40XX gens
    // E#146739
    constexpr auto CMX_TILE_SELECT_OFFSET = uint32_t{21};
    auto tileMask = uint32_t{0};
    for (auto tileIndex : usedTileIndexes) {
        tileMask |= 1 << (tileIndex + CMX_TILE_SELECT_OFFSET);
    }
    return tileMask;
}

}  // namespace VPUMI40XX
}  // namespace vpux
