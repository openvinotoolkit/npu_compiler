//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include <mlir/IR/Operation.h>
#include <mlir/Support/LogicalResult.h>

namespace vpux {
namespace VPUIP {

class NCEInvariant final {
public:
    static constexpr int64_t WEIGHT_TABLE_NUM_ELEMENTS_PER_OC = 4;

public:
    static mlir::LogicalResult verifyPipeliningCMX(VPU::ConvolutionOp origOp, const vpux::OutputTiling& tiling,
                                                   Logger log = Logger::global());
    static mlir::LogicalResult verifyPipeliningCMX(VPU::MaxPoolOp origOp, const vpux::OutputTiling& tiling,
                                                   Logger log = Logger::global());
    static mlir::LogicalResult verifyPipeliningCMX(VPU::GroupConvolutionOp origOp, const vpux::OutputTiling& tiling,
                                                   Logger log = Logger::global());

    static mlir::LogicalResult verifyPipeliningCMX(VPU::NCEConvolutionOp origOp, const vpux::OutputTiling& tiling,
                                                   Logger log = Logger::global());
    static mlir::LogicalResult verifyPipeliningCMX(VPU::NCEInterpolateOp origOp, const vpux::OutputTiling& tiling,
                                                   Logger log = Logger::global());
    static mlir::LogicalResult verifyPipeliningCMX(VPU::NCECompressConvolutionOp origOp,
                                                   const vpux::OutputTiling& tiling, Logger log = Logger::global());
    static mlir::LogicalResult verifyPipeliningCMX(VPU::NCEMaxPoolOp origOp, const vpux::OutputTiling& tiling,
                                                   Logger log = Logger::global());
    static mlir::LogicalResult verifyPipeliningCMX(VPU::NCEAveragePoolOp origOp, const vpux::OutputTiling& tiling,
                                                   Logger log = Logger::global());
    static mlir::LogicalResult verifyPipeliningCMX(VPU::NCEDepthConvolutionOp origOp, const vpux::OutputTiling& tiling,
                                                   Logger log = Logger::global());
    static mlir::LogicalResult verifyPipeliningCMX(VPU::NCEMatMulOp origOp, const vpux::OutputTiling& tiling,
                                                   Logger log = Logger::global());

    static mlir::LogicalResult verifyPipeliningCMX(VPU::AddOp origOp, const vpux::OutputTiling& tiling,
                                                   Logger log = Logger::global());
    static mlir::LogicalResult verifyPipeliningCMX(VPU::MultiplyOp origOp, const vpux::OutputTiling& tiling,
                                                   Logger log = Logger::global());
    static mlir::LogicalResult verifyPipeliningCMX(VPU::SubtractOp origOp, const vpux::OutputTiling& tiling,
                                                   Logger log = Logger::global());
    static mlir::LogicalResult verifyPipeliningCMX(VPU::NCEEltwiseOp origOp, const vpux::OutputTiling& tiling,
                                                   Logger log = Logger::global());
    static mlir::LogicalResult verifyEltwisePipeliningCMX(mlir::Operation* op, const vpux::OutputTiling& tiling,
                                                          Logger log = Logger::global());
    static Byte getRequiredCMXSizeForLastTile(mlir::Operation* op, Logger Log = Logger::global());
    static mlir::LogicalResult verifyPrefetchCMX(mlir::Operation* op, const vpux::OutputTiling& tiling,
                                                 mlir::Operation* parentOp, const vpux::OutputTiling& parentTiling,
                                                 Logger log);

public:
    static mlir::LogicalResult verifyChannels(IE::ConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(VPU::NCEConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(VPU::NCECompressConvolutionOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyChannels(IE::MaxPoolOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(VPU::NCEMaxPoolOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyPoolChannels(mlir::Operation* op, vpux::NDTypeInterface inputType,
                                                  Logger log = Logger::global());

    static mlir::LogicalResult verifyChannels(IE::AvgPoolOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(VPU::NCEAveragePoolOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyChannels(IE::AddOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(IE::ReduceMeanOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(IE::ReduceSumOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(IE::MultiplyOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(IE::SubtractOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(VPU::NCEEltwiseOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyEltwiseChannels(mlir::Operation* op, vpux::NDTypeInterface firstInputType,
                                                     vpux::NDTypeInterface secondInputType,
                                                     Logger log = Logger::global());
    static mlir::LogicalResult verifyReduceChannels(mlir::Operation* origOp, vpux::NDTypeInterface inputType,
                                                    Logger log = Logger::global());

    static mlir::LogicalResult verifyChannels(IE::GroupConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(VPU::NCEDepthConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(VPU::NCEPermuteOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyChannels(IE::InterpolateOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(VPU::NCEInterpolateOp origOp, Logger log = Logger::global());

    static mlir::LogicalResult verifyChannels(IE::TransposedConvolutionOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(IE::PadOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(IE::MatMulOp origOp, Logger log = Logger::global());
    static mlir::LogicalResult verifyChannels(VPU::NCEMatMulOp origOp, Logger log = Logger::global());

public:
    static mlir::LogicalResult isSupported(mlir::Operation* origOp, Logger log = Logger::global());
};

}  // namespace VPUIP
}  // namespace vpux
