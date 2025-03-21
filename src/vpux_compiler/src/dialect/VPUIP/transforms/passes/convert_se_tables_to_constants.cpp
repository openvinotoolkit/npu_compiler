//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"

#include <mlir/IR/BuiltinAttributes.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_CONVERTSETABLESTOCONSTANTS
#define GEN_PASS_DEF_CONVERTSETABLESTOCONSTANTS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

// Receives the start offsets of the data for the cluster and the strides of the input data.
// Uses this information to compute the byte offset of the start of the cluster.
//
// For example, let's take an input data with the shape [1, 16, 20, 20], layout NHWC, split equally across two clusters
// over the height dimension, where we want to find the start offset of the second cluster:
//                                 N   C    H   W
//   clusterOffsets:          [    0,  0,  10,  0]
//   dataStrides (in bytes):  [12800,  2, 640, 32]
// The start offset of the cluster is 10*640 = 6400.
int32_t convertClusterOffsetToByteOffset(ShapeRef clusterOffsets, StridesRef dataStrides) {
    VPUX_THROW_UNLESS(clusterOffsets.size() == dataStrides.size(),
                      "Expected cluster offsets size {0} to be equal to the strides size {1}", clusterOffsets.size(),
                      dataStrides.size());

    int32_t byteOffset = 0;
    for (auto offset : clusterOffsets | indexed) {
        auto stride = dataStrides[Dim(offset.index())].to<Byte>().count();
        byteOffset += checked_cast<int32_t>(offset.value() * stride);
    }
    return byteOffset;
}

namespace {
//
// ConvertSETablesToConstantsPass
//

class ConvertSETablesToConstantsPass final :
        public VPUIP::impl::ConvertSETablesToConstantsBase<ConvertSETablesToConstantsPass> {
public:
    explicit ConvertSETablesToConstantsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertSETablesToConstantsPass::safeRunOnFunc() {
    auto func = getOperation();

    func.walk([&](VPUIP::NCEClusterTaskOp nceOp) {
        if (nceOp.getInputStorageElementTable() == nullptr) {
            return;
        }

        _log.trace("Found NCE operation with an input SE table at '{0}'", nceOp->getLoc());

        VPUX_THROW_WHEN(nceOp.getTaskType() == VPUIP::NCETaskType::ELTWISE,
                        "Eltwise operations with input storage element tables are not yet supported");

        auto inputOperand = nceOp.getInput();
        auto seTableOperand = nceOp.getInputStorageElementTable();
        if (auto parentTilingOp = nceOp->getParentOfType<VPUIP::NCEClusterTilingOp>()) {
            auto inputArg = nceOp.getInput().dyn_cast<mlir::BlockArgument>();
            VPUX_THROW_WHEN(inputArg == nullptr, "Input is not a block argument");
            inputOperand = parentTilingOp.getOperand(inputArg.getArgNumber());

            auto seTableBlockArg = nceOp.getInputStorageElementTable().dyn_cast<mlir::BlockArgument>();
            VPUX_THROW_WHEN(seTableBlockArg == nullptr, "Input storage element table is not a block argument");
            seTableOperand = parentTilingOp.getOperand(seTableBlockArg.getArgNumber());
        }

        auto seTable = VPUIP::findSETableOp(seTableOperand);
        VPUX_THROW_WHEN(seTable == nullptr, "Unable to find the storage element table");
        if (mlir::isa<Const::DeclareOp>(seTable)) {
            _log.nest().trace("Storage element table was already converted to a constant");
            return;
        }

        auto seTableOp = mlir::cast<VPUIP::StorageElementTableOp>(seTable);
        auto seAttr = seTableOp.getSeAttr();
        if (!seAttr.has_value()) {
            _log.nest().trace("Storage element table has no SE attribute. Skipping");
            return;
        }

        auto inputType = inputOperand.getType().cast<vpux::NDTypeInterface>();

        const auto dataShape = Shape(parseIntArrayAttr<int64_t>(seTableOp.getDataShape()));
        const auto dataStrides = inputType.getStrides();
        const auto elemByteSize = getElemTypeSize(seTableOp.getDataElemType()).to<Byte>();
        const auto seSize = seTableOp.getSeSize();
        auto seOffsets = seAttr.value().computeSEOffsets(dataShape, dataStrides, elemByteSize, seSize);

        std::vector<int32_t> basePtrs(seOffsets.size(), 0);
        if (seTableOp.getBasePtrs().has_value()) {
            basePtrs = to_std_vector(seTableOp.getBasePtrs().value().getValues<int32_t>());
            VPUX_THROW_UNLESS(seOffsets.size() == basePtrs.size(), "Expected {0} base pointers, got {1}",
                              seOffsets.size(), basePtrs.size());
        }

        if (auto distType = inputType.dyn_cast<VPUIP::DistributedBufferType>()) {
            auto perClusterOffsets = distType.getPerClusterMemoryShapeOffsets();

            // Reset offsets when the base pointer value changes
            // This can happen for SOH strategy, when the offset refers to a new cluster
            std::optional<int32_t> prevBasePtr = std::nullopt;
            int32_t baseOffset = 0;
            for (auto p : zip(seOffsets, basePtrs) | indexed) {
                const auto offset = std::get<0>(p.value());
                const auto basePtr = std::get<1>(p.value());
                if (!prevBasePtr.has_value()) {
                    prevBasePtr = basePtr;
                }
                if (basePtr != prevBasePtr.value()) {
                    VPUX_THROW_UNLESS(
                            basePtr > prevBasePtr.value(),
                            "Expected incremental base pointer values, got previous value {0}, current value {1}",
                            prevBasePtr.value(), basePtr);
                    prevBasePtr = basePtr;
                    VPUX_THROW_UNLESS(checked_cast<size_t>(basePtr) < perClusterOffsets.size(),
                                      "Unexpected base pointer value {0}, while {1} clusters are used", basePtr,
                                      perClusterOffsets.size());
                    baseOffset = convertClusterOffsetToByteOffset(perClusterOffsets[basePtr], dataStrides);
                }
                seOffsets[p.index()] = offset - baseOffset;
            }
        }

        // SE pointers have the following format:
        //   31-29 28                            9 8         0
        //   -------------------------------------------------
        //   | xx |           DATA_PTR            | BASE_PTR |
        //   -------------------------------------------------
        // The DATA_PTRs are expected to be aligned to 16 bytes, so the last 4 bits are always zero.
        // That is why they are not stored in the table. Instead, the hardware will right pad the offset
        // with 4 bits on execution.
        // For example, offset 48 has the following binary value:
        //    0000 0000 0000 0011 0000
        // The DATA_PTR will store:
        //    0000 0000 0000 0000 0011
        std::vector<int32_t> sePtrs(seOffsets.size(), 0);
        for (const auto& p : zip(seOffsets, basePtrs) | indexed) {
            const auto offset = std::get<0>(p.value());
            const auto basePtr = std::get<1>(p.value());
            sePtrs[p.index()] = (offset >> 4) << VPU::NCESparsity::BASE_PTR_SIZE | basePtr;
        }

        mlir::OpBuilder builder(seTableOp);
        auto seTableType = seTableOp.getOutput().getType().cast<vpux::NDTypeInterface>();
        auto contentType = mlir::RankedTensorType::get(seTableType.getShape().raw(), seTableType.getElementType())
                                   .cast<vpux::NDTypeInterface>()
                                   .changeDimsOrder(seTableType.getDimsOrder());
        auto seTableContent = Const::createConstContent(mlir::cast<mlir::ShapedType>(contentType), ArrayRef(sePtrs));
        auto constantOp = builder.create<Const::DeclareOp>(seTableOp.getLoc(), seTableType,
                                                           Const::ContentAttr::get(seTableContent));

        seTableOp.replaceAllUsesWith(constantOp.getOutput());
        seTableOp.erase();
    });
}

}  // namespace

//
// createConvertSETablesToConstantsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertSETablesToConstantsPass(Logger log) {
    return std::make_unique<ConvertSETablesToConstantsPass>(log);
}
