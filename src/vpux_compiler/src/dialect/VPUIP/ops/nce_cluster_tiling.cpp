//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/asm.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// RegionBranchOpInterface
//

mlir::OperandRange vpux::VPUIP::NCEClusterTilingOp::getSuccessorEntryOperands(unsigned index) {
    VPUX_THROW_UNLESS(index == 0, "Invalid region index: {0}", index);
    return getOperands();
}

void vpux::VPUIP::NCEClusterTilingOp::getSuccessorRegions(Optional<unsigned> index, ArrayRef<mlir::Attribute>,
                                                          SmallVectorImpl<mlir::RegionSuccessor>& regions) {
    if (index.hasValue()) {
        VPUX_THROW_UNLESS(*index == 0, "Invalid region index: {0}", *index);
        regions.push_back(mlir::RegionSuccessor(results()));
        return;
    }

    regions.push_back(mlir::RegionSuccessor(&body(), body().getArguments()));
}

//
// Inner info
//

mlir::Operation* vpux::VPUIP::NCEClusterTilingOp::getInnerTaskOp() {
    for (auto& op : body().front().getOperations()) {
        if (VPUIP::isPureViewOp(&op)) {
            continue;
        }
        return &op;
    }
    return nullptr;
}

mlir::MutableArrayRef<mlir::BlockArgument> vpux::VPUIP::NCEClusterTilingOp::getInnerInputs() {
    return body().getArguments().take_front(getInputs().size());
}

mlir::MutableArrayRef<mlir::BlockArgument> vpux::VPUIP::NCEClusterTilingOp::getInnerOutputs() {
    return body().getArguments().slice(getInputs().size(), getOutputs().size());
}

//
// Executor info
//

IndexedSymbolAttr vpux::VPUIP::NCEClusterTilingOp::getExecutor() {
    // For NCEClusterTiling retrieve executer of inner operation
    auto op = mlir::dyn_cast<VPUIP::AsyncLayerOpInterface>(getInnerTaskOp());
    VPUX_THROW_UNLESS(op != nullptr, "Inner operation does not support interface for executors");
    return op.getExecutor();
}

//
// print/parse
//

void vpux::VPUIP::print(mlir::OpAsmPrinter& p, VPUIP::NCEClusterTilingOp op) {
    // (%operand as %blockArg: <type>, ...)

    auto& body = op.body();
    VPUX_THROW_UNLESS(!body.empty(), "Cannot serialize operation with empty body.");

    auto* entry = &op.body().front();
    VPUX_THROW_UNLESS(op.getNumOperands() == entry->getNumArguments(),
                      "Mismatch between the number of operands({0}) and body arguments({1}).", op.getNumOperands(),
                      entry->getNumArguments());

    unsigned opIdx = 0;

    printGroupOfOperands(p, entry, "inputs", op.inputs(), opIdx);
    printGroupOfOperands(p, entry, "outputs", op.output_buffs(), opIdx);

    p.printOptionalAttrDictWithKeyword(op->getAttrs(), {"operand_segment_sizes"});
    p.printOptionalArrowTypeList(op.getResultTypes());
    p << " ";
    p.printRegion(op.body(), /*printEntryBlockArgs=*/false);
}

mlir::ParseResult vpux::VPUIP::parseNCEClusterTilingOp(mlir::OpAsmParser& parser, mlir::OperationState& result) {
    // Parse operands (%operand as %blockArg : <type>).
    SmallVector<mlir::OpAsmParser::OperandType> blockArgs;
    SmallVector<mlir::Type> blockTypes;

    int32_t inCount = 0;
    if (parseGroupOfOperands(parser, result, blockArgs, blockTypes, "inputs", inCount).failed()) {
        return mlir::failure();
    }

    int32_t outCount = 0;
    if (parseGroupOfOperands(parser, result, blockArgs, blockTypes, "outputs", outCount).failed()) {
        return mlir::failure();
    }

    // Add derived `operand_segment_sizes` attribute based on parsed operands.
    auto operandSegmentSizes = mlir::DenseIntElementsAttr::get(
            mlir::VectorType::get({2}, parser.getBuilder().getI32Type()), {inCount, outCount});
    result.addAttribute("operand_segment_sizes", operandSegmentSizes);

    // Parse operation attributes.
    mlir::NamedAttrList attrs;
    if (parser.parseOptionalAttrDictWithKeyword(attrs)) {
        return mlir::failure();
    }
    result.addAttributes(attrs);

    // Parse operation results.
    SmallVector<mlir::Type> resultTypes;
    if (parser.parseOptionalArrowTypeList(resultTypes)) {
        return mlir::failure();
    }
    result.addTypes(resultTypes);

    // Parse region.
    auto* body = result.addRegion();
    if (parser.parseRegion(*body, blockArgs, blockTypes, /*argLocations=*/{}, /*enableNameShadowing=*/false)) {
        return mlir::failure();
    }

    return mlir::success();
}

//
// build
//

void vpux::VPUIP::NCEClusterTilingOp::build(mlir::OpBuilder& builder, mlir::OperationState& result,
                                            mlir::TypeRange resultTypes, mlir::ValueRange operands,
                                            BodyBuilderFn bodyBuilder) {
    result.addOperands(operands);
    result.addTypes(resultTypes);

    int32_t inCount = static_cast<int32_t>(operands.size()) - static_cast<int32_t>(resultTypes.size());
    int32_t outCount = static_cast<int32_t>(resultTypes.size());

    auto operandSegmentSizes =
            mlir::DenseIntElementsAttr::get(mlir::VectorType::get({2}, builder.getI32Type()), {inCount, outCount});
    result.addAttribute("operand_segment_sizes", operandSegmentSizes);

    // Add a body region with block arguments
    auto* bodyRegion = result.addRegion();
    auto& bodyBlock = bodyRegion->emplaceBlock();
    for (auto operand : operands) {
        auto type = operand.getType();
        if (auto distributedType = type.dyn_cast<DistributedBufferType>()) {
            type = distributedType.getCompactType();
        } else if (auto sparseType = type.dyn_cast<SparseBufferType>()) {
            if (auto distDataType = sparseType.getData().dyn_cast<DistributedBufferType>()) {
                mlir::MemRefType dataType = distDataType.getCompactType();
                mlir::MemRefType smType = nullptr;
                if (sparseType.getSparsityMap() != nullptr &&
                    sparseType.getSparsityMap().isa<DistributedBufferType>()) {
                    smType = sparseType.getSparsityMap().cast<DistributedBufferType>().getCompactType();
                }
                mlir::MemRefType seType = nullptr;
                if (sparseType.getStorageElementTable() != nullptr &&
                    sparseType.getStorageElementTable().isa<DistributedBufferType>()) {
                    seType = sparseType.getStorageElementTable().cast<DistributedBufferType>().getCompactType();
                }
                type = SparseBufferType::get(dataType, smType, seType, sparseType.getIsWeights(),
                                             sparseType.getCompressionScheme());
            }
        }

        bodyBlock.addArgument(type, operand.getLoc());
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);

    VPUX_THROW_UNLESS(bodyBuilder, "Got empty body builder.");
    bodyBuilder(builder, result.location, bodyBlock.getArguments());
}

//
// verifyOp
//

mlir::LogicalResult vpux::VPUIP::verifyOp(vpux::VPUIP::NCEClusterTilingOp op) {
    auto& body = op.body();
    if (!body.hasOneBlock()) {
        return errorAt(op->getLoc(), "Operation must have only one block.");
    }

    auto numOperands = op->getNumOperands();
    if (numOperands == 0) {
        return errorAt(op->getLoc(), "Operation must have at least one operand.");
    }

    auto bodyNumArgs = body.getNumArguments();
    if (numOperands != bodyNumArgs) {
        return errorAt(op->getLoc(), "Mismatch between the number of operands({0}) and body arguments({1}).",
                       numOperands, bodyNumArgs);
    }

    if (op->getNumResults() == 0) {
        return errorAt(op->getLoc(), "Operation must have at least one result.");
    }

    auto isNceClusterTaskType = mlir::isa<VPUIP::NCEClusterTaskOp>(op.getInnerTaskOp());
    const auto checkShape = [&](mlir::ValueRange operands) {
        for (auto operand : operands) {
            auto operandType = operand.getType();
            if (auto ndType = operand.getType().dyn_cast<vpux::NDTypeInterface>()) {
                auto rank = ndType.getRank();
                if (rank != 4 && rank != 1 && isNceClusterTaskType) {
                    return errorAt(op->getLoc(), "Only 4D/1D tensors are supported. Got {0}", rank);
                }
                continue;
            }

            VPUX_THROW("Unsupported type of operand: {0}", operandType);
        }

        return mlir::success();
    };

    auto isOperandsValid = checkShape(op->getOperands());
    if (isOperandsValid.failed()) {
        return mlir::failure();
    }

    auto isArgsValid = checkShape(op.body().getArguments());
    if (isArgsValid.failed()) {
        return mlir::failure();
    }

    return mlir::success();
}
