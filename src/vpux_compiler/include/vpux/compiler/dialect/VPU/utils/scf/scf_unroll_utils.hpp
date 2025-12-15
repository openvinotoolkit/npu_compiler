//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU40XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Affine/Utils.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/Interfaces/TilingInterface.h>

namespace vpux::VPU {

struct TileDimensionInfo {
    vpux::Dim dimension;
    int64_t numBlocks;
    bool isUnrolled;
    mlir::scf::ForOp forOp;
    int64_t id;  // identifier to sort the loop from outermost to innermost
};

struct UnrollConfig {
    llvm::SmallVector<int64_t> unrollFactors;
    llvm::SmallVector<int64_t> accessOrder;
    size_t totalBlocks;
};

constexpr uint32_t TWO_BIT_MASK = 0x3;
constexpr uint32_t UNROLL_N_START_BLK_SHIFT = 12;
constexpr uint32_t UNROLL_N_END_BLK_SHIFT = 14;
constexpr uint32_t UNROLL_C_START_BLK_SHIFT = 8;
constexpr uint32_t UNROLL_C_END_BLK_SHIFT = 10;
constexpr uint32_t UNROLL_H_START_BLK_SHIFT = 4;
constexpr uint32_t UNROLL_H_END_BLK_SHIFT = 6;
constexpr uint32_t UNROLL_W_START_BLK_SHIFT = 0;
constexpr uint32_t UNROLL_W_END_BLK_SHIFT = 2;

constexpr uint32_t UNROLL_N_SHIFT = 12;
constexpr uint32_t UNROLL_C_SHIFT = 8;
constexpr uint32_t UNROLL_H_SHIFT = 4;

constexpr uint32_t N_SHIFT = 6;
constexpr uint32_t C_SHIFT = 4;
constexpr uint32_t H_SHIFT = 2;

struct BlockEncodingUtils {
    int64_t encodeValue(vpux::Dim dim, int64_t newValue, int64_t startPos, int64_t endPos) {
        auto encodedValue = newValue;
        startPos = startPos & TWO_BIT_MASK;
        endPos = endPos & TWO_BIT_MASK;

        if (dim == vpux::Dims4D::Act::N) {
            encodedValue |= (startPos << UNROLL_N_START_BLK_SHIFT);
            encodedValue |= (endPos << UNROLL_N_END_BLK_SHIFT);
        } else if (dim == vpux::Dims4D::Act::C) {
            encodedValue |= (startPos << UNROLL_C_START_BLK_SHIFT);
            encodedValue |= (endPos << UNROLL_C_END_BLK_SHIFT);
        } else if (dim == vpux::Dims4D::Act::H) {
            encodedValue |= (startPos << UNROLL_H_START_BLK_SHIFT);
            encodedValue |= (endPos << UNROLL_H_END_BLK_SHIFT);
        } else if (dim == vpux::Dims4D::Act::W) {
            encodedValue |= (startPos << UNROLL_W_START_BLK_SHIFT);
            encodedValue |= (endPos << UNROLL_W_END_BLK_SHIFT);
        } else {
            assert(false && "Unsupported dimension");
        }

        return encodedValue;
    }

    void insertBlockId(vpux::Dim dim, int64_t& value, int64_t blockId) {
        if (dim == vpux::Dims4D::Act::N) {
            value |= (blockId << N_SHIFT);
        } else if (dim == vpux::Dims4D::Act::C) {
            value |= (blockId << C_SHIFT);
        } else if (dim == vpux::Dims4D::Act::H) {
            value |= (blockId << H_SHIFT);
        } else if (dim == vpux::Dims4D::Act::W) {
            value |= blockId;
        } else {
            assert(false && "Unsupported dimension");
        }
    }

    int64_t getBlockId(vpux::Dim dim, int64_t value) {
        if (dim == vpux::Dims4D::Act::N) {
            return (value >> N_SHIFT) & TWO_BIT_MASK;
        } else if (dim == vpux::Dims4D::Act::C) {
            return (value >> C_SHIFT) & TWO_BIT_MASK;
        } else if (dim == vpux::Dims4D::Act::H) {
            return (value >> H_SHIFT) & TWO_BIT_MASK;
        } else if (dim == vpux::Dims4D::Act::W) {
            return value & TWO_BIT_MASK;
        } else {
            assert(false && "Unsupported dimension");
        }
    }

    std::pair<int64_t, int64_t> decodeStartAndEndBlkIds(vpux::Dim dim, int64_t value) {
        int64_t decodedValue = 0x0;
        if (dim == vpux::Dims4D::Act::N) {
            decodedValue = (value >> UNROLL_N_SHIFT);
        } else if (dim == vpux::Dims4D::Act::C) {
            decodedValue = (value >> UNROLL_C_SHIFT);
        } else if (dim == vpux::Dims4D::Act::H) {
            decodedValue = (value >> UNROLL_H_SHIFT);
        } else if (dim == vpux::Dims4D::Act::W) {
            decodedValue = value;
        } else {
            assert(false && "Unsupported dimension");
        }

        auto startBlk = decodedValue & TWO_BIT_MASK;
        auto endBlk = (decodedValue >> 2) & TWO_BIT_MASK;
        return std::pair<int64_t, int64_t>(startBlk, endBlk);
    }
};

mlir::LogicalResult mergeUnrollOperationsInBlock(mlir::Block* block, const UnrollConfig& config, int64_t numOriginalOps,
                                                 bool cleanupAfterMerge = true);
mlir::LogicalResult mergeUnrolledOperations(mlir::scf::ForOp forOp, SmallVector<TileDimensionInfo>& tileDimInfoVec);

mlir::scf::ForOp fuseSiblingForLoops(mlir::scf::ForOp target, mlir::scf::ForOp source, mlir::RewriterBase& rewriter,
                                     bool residualLoops = false);

}  // namespace vpux::VPU
