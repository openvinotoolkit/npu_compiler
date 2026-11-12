//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU60XX/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/nce_op_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/mpe_engine_utils.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"

namespace vpux::IE::arch60xx {

enum class DcimOpKind { Convolution, MatMul, FullyConnected };

struct DcimMappingInfo final {
    DcimOpKind opKind = DcimOpKind::Convolution;

    vpux::NDTypeInterface inputType = nullptr;
    vpux::NDTypeInterface filterType = nullptr;
    vpux::NDTypeInterface outputType = nullptr;

    // Used by MatMul only. For Convolution, channels are taken from the logical NCHW C dimension.
    bool transposeA = false;
    // True when the operand is produced by a FakeQuantize, Dequantize or DynamicDequantize
    bool inputHasQuantizeProducer = false;
    bool filterHasQuantizeProducer = false;
    bool inputChannelsPadded = false;

    IE::FakeQuantizeOp inputFakeQuantizeOp = nullptr;

    int64_t inputFqLevels = 0;
    bool inputFqIsPerTensor = false;
    // True when the filter is produced by a DynamicDequantize whose scale is per-tensor (all dims are 1).
    bool filterDynDeqScaleIsPerTensor = true;

    // True when the filter operand resolves to a genuine BlockArgument
    bool filterIsWeightsAsInput = false;
    mlir::quant::QuantizedType filterWaiQuantType = nullptr;

    bool dcimEnabled = false;
    bool powerOptimized = false;
};

// Populates the *Fq* fields above from inputFakeQuantizeOp. Call after the FakeQuantize op has been set on the info.
void setDcimFakeQuantizeFacts(DcimMappingInfo& info);

// Activation-side producer: FakeQuantize is still a live representation for activations (pre SplitFakeQuant), so
// the matched FakeQuantizeOp (if any) is returned for later inspection via setDcimFakeQuantizeFacts. Dequantize is
// also recognized, since SplitFakeQuant turns FakeQuantize into Quantize+Dequantize on activations.
void setDcimInputProducerInfo(mlir::Value operand, vpux::NDTypeInterface& operandType, bool& hasQuantizeProducer,
                              IE::FakeQuantizeOp& fakeQuantizeOp);

// Weights-side producer: weights are only ever imported as DynamicDequantize, so
// this reports whether the DynamicDequantize scale is per-tensor instead of returning a FakeQuantizeOp.
void setDcimFilterProducerInfo(mlir::Value operand, vpux::NDTypeInterface& operandType, bool& hasQuantizeProducer,
                               bool& dynamicDequantizeScaleIsPerTensor);

// Populates filterIsWeightsAsInput / filterWaiQuantType on info from nceOp filter operand.
void setDcimFilterWaiInfo(mlir::Operation* nceOp, DcimMappingInfo& info);

mlir::Type getDcimIntStorageType(mlir::Type elemType);

DcimMappingInfo getDcimMappingInfo(mlir::Operation* rootOp);
DcimMappingInfo getProspectiveConvDcimMappingInfo(vpux::NDTypeInterface inputType, vpux::NDTypeInterface filterType,
                                                  vpux::NDTypeInterface outputType, bool dcimEnabled,
                                                  bool powerOptimized, bool inputHasQuantizeProducer = false,
                                                  bool filterHasQuantizeProducer = false);

// Walks up the use-def chain through pure view-like ops (Reshape, AffineReshape, Transpose, ...)
// to locate a producing FakeQuantize. Returns the FakeQuantize if found, otherwise returns the
// originally-supplied defining op.
mlir::Operation* findProducingFakeQuantizeThroughViewLikes(mlir::Operation* definingOp);
mlir::Operation* findProducingDynamicDequantizeThroughViewLikes(mlir::Operation* definingOp);
int32_t getInputChannelCount(const DcimMappingInfo& info);
int32_t getOutputChannelCount(const DcimMappingInfo& info);
int32_t getInputHeight(const DcimMappingInfo& info);
int32_t getInputWidth(const DcimMappingInfo& info);

bool isMappableToDCIM(mlir::Operation* op, bool checkChannelAlignment);
bool isMappedToDCIM(mlir::Operation* op);

// Shared tail of isMappableToDCIM: given a DCIM-eligible op and its mapping info, checks that the input and output
// channels, once padded to the op's channel alignment, are multiples of 32
bool dcimChannelAlignmentSatisfied(mlir::Operation* op, const DcimMappingInfo& info);

bool passesSWHeuristicForDCIM(const DcimMappingInfo& info, bool checkChannelAlignment);
bool respectHWConstraintsForDCIM(const DcimMappingInfo& info);

// Backward-compatible op wrappers used by the existing MPE engine selection code.
bool passesSWHeuristicForDCIM(mlir::Operation* rootOp, bool checkChannelAlignment);
bool respectHWConstraintsForDCIM(mlir::Operation* rootOp);
mlir::Attribute makeMPEEngineModeAttr(mlir::Operation* op, bool useDCIM, mlir::Attribute weightZp,
                                      mlir::Attribute activationZp);

// Shared getMPEEngine body for Convolution models. dcimEligible is the arch-specific eligibility result; DCIM is
// selected only when the convolution is not better served by NCECompressConvolution.
mlir::Attribute getConvMPEEngineMode(mlir::Operation* op, bool dcimEligible, mlir::Attribute weightZp,
                                     mlir::Attribute activationZp);

bool isEligibleForDCIM(mlir::Operation* op, bool checkChannelAlignment);

template <typename OpType>
class MPEEngineInfoOpModel : public IE::MPEEngineInfoOpInterface::ExternalModel<MPEEngineInfoOpModel<OpType>, OpType> {
public:
    mlir::Attribute getMPEEngine(mlir::Operation* op) const {
        return getMPEEngineWithZP(op, /*weightZp=*/nullptr, /*activationZp=*/nullptr);
    }

    mlir::Attribute getMPEEngineWithZP(mlir::Operation* op, mlir::Attribute weightZp,
                                       mlir::Attribute activationZp) const {
        return VPU::MPEEngine60XXAttr::get(
                op->getContext(), VPU::MPEEngine60XXModeAttr::get(op->getContext(), VPU::MPEEngine60XXMode::SCL),
                weightZp, activationZp);
    }
};

class MPEEngineInfoConvOpModel : public IE::MPEEngineInfoOpInterface::FallbackModel<MPEEngineInfoConvOpModel> {
public:
    bool supportsSpecializedMPE(mlir::Operation*) const {
        return true;
    }

    mlir::Attribute getMPEEngine(mlir::Operation* op) const {
        return getMPEEngineWithZP(op, /*weightZp=*/nullptr, /*activationZp=*/nullptr);
    }

    mlir::Attribute getMPEEngineWithZP(mlir::Operation* op, mlir::Attribute weightZp,
                                       mlir::Attribute activationZp) const {
        return getConvMPEEngineMode(op, isEligibleForDCIM(op, /*checkChannelAlignment=*/true), weightZp, activationZp);
    }
};

class MPEEngineInfoMatMulOpModel : public IE::MPEEngineInfoOpInterface::FallbackModel<MPEEngineInfoMatMulOpModel> {
public:
    bool supportsSpecializedMPE(mlir::Operation*) const {
        return true;
    }

    mlir::Attribute getMPEEngine(mlir::Operation* op) const {
        return getMPEEngineWithZP(op, /*weightZp=*/nullptr, /*activationZp=*/nullptr);
    }

    mlir::Attribute getMPEEngineWithZP(mlir::Operation* op, mlir::Attribute weightZp,
                                       mlir::Attribute activationZp) const {
        return makeMPEEngineModeAttr(op, isEligibleForDCIM(op, /*checkChannelAlignment=*/true), weightZp, activationZp);
    }
};

class MPEEngineInfoFullyConnectedOpModel :
        public IE::MPEEngineInfoOpInterface::FallbackModel<MPEEngineInfoFullyConnectedOpModel> {
public:
    bool supportsSpecializedMPE(mlir::Operation*) const {
        return true;
    }

    mlir::Attribute getMPEEngine(mlir::Operation* op) const {
        return getMPEEngineWithZP(op, /*weightZp=*/nullptr, /*activationZp=*/nullptr);
    }

    mlir::Attribute getMPEEngineWithZP(mlir::Operation* op, mlir::Attribute weightZp,
                                       mlir::Attribute activationZp) const {
        const bool useDCIM = isEligibleForDCIM(op, /*checkChannelAlignment=*/true);
        return makeMPEEngineModeAttr(op, useDCIM, weightZp, activationZp);
    }
};

}  // namespace vpux::IE::arch60xx
