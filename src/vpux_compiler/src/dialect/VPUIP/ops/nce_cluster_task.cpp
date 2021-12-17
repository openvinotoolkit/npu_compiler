//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// NCEClusterTaskOp::build
//

void vpux::VPUIP::NCEClusterTaskOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                          mlir::Value weights, mlir::Value weight_table, mlir::Value activation_window,
                                          mlir::Value parent_input, mlir::Value parent_output, mlir::Value output_buff,
                                          vpux::VPUIP::NCETaskType task_type, mlir::ArrayAttr kernel_size,
                                          mlir::ArrayAttr kernel_strides, mlir::ArrayAttr kernel_padding,
                                          mlir::IntegerAttr activation_window_channel_length,
                                          mlir::UnitAttr is_continued) {
    build(builder, state, output_buff.getType(), input, weights, weight_table, activation_window, parent_input,
          parent_output, output_buff, nullptr, vpux::VPUIP::NCETaskTypeAttr::get(builder.getContext(), task_type),
          kernel_size, kernel_strides, kernel_padding, activation_window_channel_length, is_continued);

    for (auto& region : state.regions) {
        region->emplaceBlock();
    }
}

void vpux::VPUIP::NCEClusterTaskOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type output,
                                          mlir::Value input, mlir::Value weights, mlir::Value weight_table,
                                          mlir::Value activation_window, mlir::Value parent_input,
                                          mlir::Value parent_output, mlir::Value output_buff,
                                          vpux::VPUIP::NCETaskType task_type, mlir::ArrayAttr kernel_size,
                                          mlir::ArrayAttr kernel_strides, mlir::ArrayAttr kernel_padding,
                                          mlir::IntegerAttr activation_window_channel_length,
                                          mlir::UnitAttr is_continued) {
    build(builder, state, output, input, weights, weight_table, activation_window, parent_input, parent_output,
          output_buff, nullptr, vpux::VPUIP::NCETaskTypeAttr::get(builder.getContext(), task_type), kernel_size,
          kernel_strides, kernel_padding, activation_window_channel_length, is_continued);

    for (auto& region : state.regions) {
        region->emplaceBlock();
    }
}

//
// NCEClusterTaskOp::addDPUTask
//

VPUIP::DPUTaskOp vpux::VPUIP::NCEClusterTaskOp::addDPUTask(mlir::OpBuilder& builder, mlir::ArrayAttr start,
                                                           mlir::ArrayAttr end, VPUIP::PaddingAttr pad,
                                                           VPUIP::MPEMode mpeMode) {
    if (variants().empty()) {
        variants().emplaceBlock();
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&variants().front());

    return builder.create<VPUIP::DPUTaskOp>(getLoc(), start, end, pad, mpeMode);
}

//
// NCEClusterTaskOp::getNumVariants
//

int64_t vpux::VPUIP::NCEClusterTaskOp::getNumVariants() {
    return variants().getBlocks().front().getOperations().size();
}

//
// NCEClusterTaskOp::inferLayoutInfo
//

void vpux::VPUIP::NCEClusterTaskOp::inferLayoutInfo(mlir::Operation* origOp, IE::LayerLayoutInfo& info) {
    llvm::TypeSwitch<mlir::Operation*, void>(origOp)
            .Case<IE::ConvolutionOp>([&](IE::ConvolutionOp) {
                info.setInput(0, DimsOrder::NHWC);
                info.setInput(1, DimsOrder::OYXI);
                info.setOutput(0, DimsOrder::NHWC);
            })
            .Case<IE::GroupConvolutionOp>([&](IE::GroupConvolutionOp) {
                info.setInput(0, DimsOrder::NHWC);
                info.setInput(1, DimsOrder::OYXI);
                info.setOutput(0, DimsOrder::NHWC);
            })
            .Case<IE::MaxPoolOp>([&](IE::MaxPoolOp) {
                info.fill(DimsOrder::NHWC);
            })
            .Case<IE::AddOp>([&](IE::AddOp) {
                info.fill(DimsOrder::NHWC);
            })
            .Case<IE::MultiplyOp>([&](IE::MultiplyOp) {
                info.fill(DimsOrder::NHWC);
            })
            .Case<IE::SubtractOp>([&](IE::SubtractOp) {
                info.fill(DimsOrder::NHWC);
            })
            .Case<IE::AndOp>([&](IE::AndOp) {
                info.fill(DimsOrder::NHWC);
            })
            .Default([](mlir::Operation* unknownOp) {
                VPUX_THROW("Operation '{0}' is not supported by the DPU", unknownOp->getName());
            });
}

//
// NCEClusterTaskOp::sparsitySupport
//

vpux::VPU::SparsitySupport vpux::VPUIP::NCEClusterTaskOp::sparsitySupport() {
    const auto sparseInputs =
            (task_type() == vpux::VPUIP::NCETaskType::CONV) || (task_type() == vpux::VPUIP::NCETaskType::ELTWISE)
                    ? vpux::VPU::SparsitySupport::SPARSE_INPUTS
                    : vpux::VPU::SparsitySupport::NONE;
    return sparseInputs | vpux::VPU::SparsitySupport::SPARSE_OUTPUTS;
}

//
// NCEClusterTaskOp::inferReturnTypes
//

mlir::LogicalResult vpux::VPUIP::NCEClusterTaskOp::inferReturnTypes(
        mlir::MLIRContext*, llvm::Optional<mlir::Location>, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange ranges, llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    VPUIP::NCEClusterTaskOpAdaptor adaptor(operands, attrs, ranges);
    inferredReturnTypes.push_back(adaptor.output_buff().getType());
    if (adaptor.profiling_data() != nullptr) {
        inferredReturnTypes.push_back(adaptor.profiling_data().getType());
    }
    return mlir::success();
}

//
// verifyOp
//

namespace {

mlir::LogicalResult verifyNCEConv(VPUIP::NCEClusterTaskOp op, VPU::ArchKind arch) {
    VPUX_THROW_UNLESS(op.task_type() == VPUIP::NCETaskType::CONV, "Expected task type '{0}', but got '{1}'",
                      VPUIP::NCETaskType::CONV, op.task_type());

    if (op.weights() == nullptr) {
        return errorAt(op, "weights is required for NCETaskType : '{0}'", op.task_type());
    }
    if (op.weight_table() == nullptr) {
        return errorAt(op, "weight_table is required for NCETaskType : '{0}'", op.task_type());
    }

    if (op.kernel_sizeAttr() == nullptr) {
        return errorAt(op, "kernel_size is required for NCETaskType : '{0}'", op.task_type());
    }
    if (op.kernel_stridesAttr() == nullptr) {
        return errorAt(op, "kernel_strides is required for NCETaskType : '{0}'", op.task_type());
    }
    if (op.kernel_paddingAttr() == nullptr) {
        return errorAt(op, "kernel_padding is required for NCETaskType : '{0}'", op.task_type());
    }

    const auto kernelSize = parseIntArrayAttr<int64_t>(op.kernel_sizeAttr());
    const auto KY = kernelSize[0];
    const auto KX = kernelSize[1];

    const auto kernelStrides = parseIntArrayAttr<int64_t>(op.kernel_stridesAttr());
    const auto SY = kernelStrides[0];
    const auto SX = kernelStrides[1];

    const auto kernelPadding = parseIntArrayAttr<int64_t>(op.kernel_paddingAttr());
    const auto padLeft = kernelPadding[0];
    const auto padRight = kernelPadding[1];
    const auto padTop = kernelPadding[2];
    const auto padBottom = kernelPadding[3];

    if (mlir::failed(VPUIP::NCEInvariant::verifyKernel(op->getLoc(), KY, KX, SY, SX, padTop, padBottom, padLeft,
                                                       padRight, arch))) {
        return mlir::failure();
    }

    const auto weightsShape = getShape(op.weights());
    const auto OC = weightsShape[Dims4D::Filter::OC];

    const auto weightTableShape = getShape(op.weight_table());
    const auto weightTableNumElements = weightTableShape.totalSize();

    if (OC * VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC != weightTableNumElements) {
        return errorAt(op, "Weight table must have '{0}' elements, got '{1}'",
                       OC * VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC, weightTableNumElements);
    }

    if (verifySameInOutSpecificDimsOrder(op, {DimsOrder::NHWC}).failed()) {
        return mlir::failure();
    }

    const auto weightsLayout = DimsOrder::fromValue(op.weights());
    if (weightsLayout != DimsOrder::NHWC) {
        return errorAt(op, "weights layout must be NHWC, got {0}", weightsLayout);
    }

    return mlir::success();
}

mlir::LogicalResult verifyNCEPool(VPUIP::NCEClusterTaskOp op, VPU::ArchKind arch) {
    VPUX_THROW_UNLESS(op.task_type() == VPUIP::NCETaskType::AVEPOOL || op.task_type() == VPUIP::NCETaskType::MAXPOOL,
                      "Expected task type '{0}' or '{1}', but got '{2}'", VPUIP::NCETaskType::AVEPOOL,
                      VPUIP::NCETaskType::MAXPOOL, op.task_type());

    // MTL hw doesn't require weights table and activation window for max/average pool ops
    if (arch != VPU::ArchKind::MTL) {
        if (op.weight_table() == nullptr) {
            return errorAt(op, "weight_table is required for NCETaskType : '{0}'", op.task_type());
        }
        if (op.activation_window() == nullptr) {
            return errorAt(op, "activation_window is required for NCETaskType : '{0}'", op.task_type());
        }
        if (op.activation_window_channel_lengthAttr() == nullptr) {
            return errorAt(op, "activation_window_channel_length is required for NCETaskType : '{0}'", op.task_type());
        }
    }

    if (op.kernel_sizeAttr() == nullptr) {
        return errorAt(op, "kernel_size is required for NCETaskType : '{0}'", op.task_type());
    }
    if (op.kernel_stridesAttr() == nullptr) {
        return errorAt(op, "kernel_strides is required for NCETaskType : '{0}'", op.task_type());
    }
    if (op.kernel_paddingAttr() == nullptr) {
        return errorAt(op, "kernel_padding is required for NCETaskType : '{0}'", op.task_type());
    }

    const auto kernelSize = parseIntArrayAttr<int64_t>(op.kernel_sizeAttr());
    const auto KY = kernelSize[0];
    const auto KX = kernelSize[1];

    const auto kernelStrides = parseIntArrayAttr<int64_t>(op.kernel_stridesAttr());
    const auto SY = kernelStrides[0];
    const auto SX = kernelStrides[1];

    const auto kernelPadding = parseIntArrayAttr<int64_t>(op.kernel_paddingAttr());
    const auto padLeft = kernelPadding[0];
    const auto padRight = kernelPadding[1];
    const auto padTop = kernelPadding[2];
    const auto padBottom = kernelPadding[3];

    if (mlir::failed(VPUIP::NCEInvariant::verifyKernel(op->getLoc(), KY, KX, SY, SX, padTop, padBottom, padLeft,
                                                       padRight, arch))) {
        return mlir::failure();
    }

    if (verifySameInOutSpecificDimsOrder(op, {DimsOrder::NHWC}).failed()) {
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult verifyNCEEltwise(VPUIP::NCEClusterTaskOp op, VPU::ArchKind) {
    VPUX_THROW_UNLESS(op.task_type() == VPUIP::NCETaskType::ELTWISE, "Expected task type '{0}', but got '{1}'",
                      VPUIP::NCETaskType::ELTWISE, op.task_type());

    if (op.weight_table() != nullptr) {
        return errorAt(op, "weight_table should be empty for NCETaskType : '{0}'", op.task_type());
    }
    if (op.activation_window() != nullptr) {
        return errorAt(op, "activation_window should be empty for NCETaskType : '{0}'", op.task_type());
    }

    if (op.kernel_sizeAttr() != nullptr) {
        return errorAt(op, "kernel_size should be empty for NCETaskType : '{0}'", op.task_type());
    }
    if (op.kernel_stridesAttr() != nullptr) {
        return errorAt(op, "kernel_strides should be empty for NCETaskType : '{0}'", op.task_type());
    }
    if (op.kernel_paddingAttr() != nullptr) {
        return errorAt(op, "kernel_padding should be empty for NCETaskType : '{0}'", op.task_type());
    }

    return mlir::success();
}

mlir::LogicalResult verifyNCEDWConv(VPUIP::NCEClusterTaskOp op, VPU::ArchKind arch) {
    VPUX_THROW_UNLESS(op.task_type() == VPUIP::NCETaskType::DWCONV, "Expected task type '{0}', but got '{1}'",
                      VPUIP::NCETaskType::CONV, op.task_type());

    if (op.weights() == nullptr) {
        return errorAt(op, "weights is required for NCETaskType : '{0}'", op.task_type());
    }
    if (op.weight_table() == nullptr) {
        return errorAt(op, "weight_table is required for NCETaskType : '{0}'", op.task_type());
    }
    if (op.activation_window() == nullptr) {
        return errorAt(op, "activation_window is required for NCETaskType : '{0}'", op.task_type());
    }

    if (op.kernel_sizeAttr() == nullptr) {
        return errorAt(op, "kernel_size is required for NCETaskType : '{0}'", op.task_type());
    }
    if (op.kernel_stridesAttr() == nullptr) {
        return errorAt(op, "kernel_strides is required for NCETaskType : '{0}'", op.task_type());
    }
    if (op.kernel_paddingAttr() == nullptr) {
        return errorAt(op, "kernel_padding is required for NCETaskType : '{0}'", op.task_type());
    }

    const auto kernelSize = parseIntArrayAttr<int64_t>(op.kernel_sizeAttr());
    const auto KY = kernelSize[0];
    const auto KX = kernelSize[1];

    const auto kernelStrides = parseIntArrayAttr<int64_t>(op.kernel_stridesAttr());
    const auto SY = kernelStrides[0];
    const auto SX = kernelStrides[1];

    const auto kernelPadding = parseIntArrayAttr<int64_t>(op.kernel_paddingAttr());
    const auto padLeft = kernelPadding[0];
    const auto padRight = kernelPadding[1];
    const auto padTop = kernelPadding[2];
    const auto padBottom = kernelPadding[3];

    if (mlir::failed(VPUIP::NCEInvariant::verifyKernel(op->getLoc(), KY, KX, SY, SX, padTop, padBottom, padLeft,
                                                       padRight, arch))) {
        return mlir::failure();
    }

    const auto weightsShape = getShape(op.weights());
    const auto OC = weightsShape[Dims4D::Filter::OC];

    const auto weightTableShape = getShape(op.weight_table());
    const auto weightTableNumElements = weightTableShape.totalSize();

    if (OC * VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC != weightTableNumElements) {
        return errorAt(op, "Weight table must have '{0}' elements, got '{1}'",
                       OC * VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC, weightTableNumElements);
    }

    if (verifySameInOutSpecificDimsOrder(op, {DimsOrder::NHWC}).failed()) {
        return mlir::failure();
    }

    const auto weightsLayout = DimsOrder::fromValue(op.weights());
    if (weightsLayout != DimsOrder::NHWC) {
        return errorAt(op, "weights layout must be NHWC, got {0}", weightsLayout);
    }

    return mlir::success();
}

}  // namespace

mlir::LogicalResult vpux::VPUIP::verifyOp(VPUIP::DPUTaskOp op) {
    static const size_t NUM_WORKLOAD_DIMS = 3;

    if (op.start().size() != NUM_WORKLOAD_DIMS) {
        return errorAt(op, "start coords should {0}-D, but got {1}-D", NUM_WORKLOAD_DIMS, op.start().size());
    }
    if (op.end().size() != NUM_WORKLOAD_DIMS) {
        return errorAt(op, "end coords should {0}-D, but got {1}-D", NUM_WORKLOAD_DIMS, op.end().size());
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::verifyOp(VPUIP::NCEClusterTaskOp op) {
    const auto arch = VPU::getArch(op.getOperation()->getParentOfType<mlir::ModuleOp>());

    for (const auto& operand : op.getOpOperands()) {
        const auto val = operand.get();
        const auto type = val.getType().cast<mlir::MemRefType>().getElementType();

        if (arch != VPU::ArchKind::MTL && type.isBF16()) {
            return errorAt(op, "BF16 is only supported by MTL");
        }
    }

    if (op.task_type() == VPUIP::NCETaskType::CONV) {
        if (mlir::failed(verifyNCEConv(op, arch))) {
            return mlir::failure();
        }
    } else if (op.task_type() == VPUIP::NCETaskType::MAXPOOL || op.task_type() == VPUIP::NCETaskType::AVEPOOL) {
        if (mlir::failed(verifyNCEPool(op, arch))) {
            return mlir::failure();
        }
    } else if (op.task_type() == VPUIP::NCETaskType::ELTWISE) {
        if (mlir::failed(verifyNCEEltwise(op, arch))) {
            return mlir::failure();
        }
    } else if (op.task_type() == VPUIP::NCETaskType::DWCONV) {
        if (mlir::failed(verifyNCEDWConv(op, arch))) {
            return mlir::failure();
        }
    } else {
        return errorAt(op, "NCE Task Type '{0}' in not supported", op.task_type());
    }

    size_t numDPUTasks = 0;
    for (auto& dpuOp : op.variants().getOps()) {
        if (!mlir::isa<VPUIP::DPUTaskOp>(dpuOp)) {
            return errorAt(op, "Got unsupported Operation '{0}' in 'variants' region", dpuOp.getName());
        }

        ++numDPUTasks;
    }

    static const size_t MAX_NUM_DPUS_PER_CLUSTER = 5;
    static const size_t MIN_NUM_DPUS_PER_CLUSTER = 1;

    if (numDPUTasks > MAX_NUM_DPUS_PER_CLUSTER || numDPUTasks < MIN_NUM_DPUS_PER_CLUSTER) {
        return errorAt(op, "There should be a total of {0}-{1} DPU Tasks per NCEClusterTask, but got {2}",
                       MIN_NUM_DPUS_PER_CLUSTER, MAX_NUM_DPUS_PER_CLUSTER, numDPUTasks);
    }

    for (auto& ppeOp : op.ppe().getOps()) {
        if (!mlir::isa<VPUIP::PPETaskOp>(ppeOp)) {
            return errorAt(op, "Got unsupported Operation '{0}' in 'PPE' region", ppeOp.getName());
        }
    }

    for (const auto& operand : op.getOpOperands()) {
        const auto val = operand.get();
        const auto type = val.getType().cast<mlir::MemRefType>();

        const auto mem = VPU::getMemoryKind(type);
        if (mem != VPU::MemoryKind::CMX_NN && mem != VPU::MemoryKind::Register) {
            return errorAt(op, "Can't operate with '{0}' MemoryKind. Only '{1}' or '{2} MemoryKind is allowed", mem,
                           VPU::MemoryKind::CMX_NN, VPU::MemoryKind::Register);
        }

        const auto strideReqs = StrideReqs().add(DimStrideReq::compact(MemDim(type.getRank() - 1)));

        if (!strideReqs.checkStrides(val)) {
            return errorAt(op, "Value '{0}' strides do not match requirements '{1}'", val, strideReqs);
        }
    }

    return mlir::success();
}

//
// NCEClusterTaskOp::serialize
//

namespace {

MVCNN::MPE_Mode getMPEMode(VPUIP::MPEMode mpeMode) {
    switch (mpeMode) {
    case VPUIP::MPEMode::VECTOR:
        return MVCNN::MPE_Mode_VECTOR;
    case VPUIP::MPEMode::MATRIX:
        return MVCNN::MPE_Mode_MATRIX;
    case VPUIP::MPEMode::VECTOR_FP16:
        return MVCNN::MPE_Mode_VECTOR_FP16;
    case VPUIP::MPEMode::CUBOID_16x16:
        return MVCNN::MPE_Mode_CUBOID_16x16;
    case VPUIP::MPEMode::CUBOID_8x16:
        return MVCNN::MPE_Mode_CUBOID_8x16;
    case VPUIP::MPEMode::NOP:
        return MVCNN::MPE_Mode_NOP;
    default:
        VPUX_THROW("Unsupported MPE mode type: '{0}'", mpeMode);
    }
}

MVCNN::DPULayerType getDPULayerType(VPUIP::NCETaskType taskType) {
    switch (taskType) {
    case VPUIP::NCETaskType::CONV:
        return MVCNN::DPULayerType_CONV;
    case VPUIP::NCETaskType::DWCONV:
        return MVCNN::DPULayerType_DWCONV;
    case VPUIP::NCETaskType::MAXPOOL:
        return MVCNN::DPULayerType_MAXPOOL;
    case VPUIP::NCETaskType::AVEPOOL:
        return MVCNN::DPULayerType_AVEPOOL;
    case VPUIP::NCETaskType::FCL:
        return MVCNN::DPULayerType_FCL;
    case VPUIP::NCETaskType::ELTWISE:
        return MVCNN::DPULayerType_ELTWISE;
    case VPUIP::NCETaskType::IDENTITY:
        return MVCNN::DPULayerType_IDENTITY;
    case VPUIP::NCETaskType::CMCONV:
        return MVCNN::DPULayerType_CMCONV;
    default:
        VPUX_THROW("Unsupported DPU Layer type: '{0}'", taskType);
    }
}

MVCNN::PPELayerType getPPELayerType(VPUIP::PPELayerType ppeType) {
    switch (ppeType) {
    case VPUIP::PPELayerType::STORE:
        return MVCNN::PPELayerType_STORE;
    case VPUIP::PPELayerType::LOAD:
        return MVCNN::PPELayerType_LOAD;
    case VPUIP::PPELayerType::CLEAR:
        return MVCNN::PPELayerType_CLEAR;
    case VPUIP::PPELayerType::NOOP:
        return MVCNN::PPELayerType_NOOP;
    case VPUIP::PPELayerType::HALT:
        return MVCNN::PPELayerType_HALT;
    case VPUIP::PPELayerType::ADD:
        return MVCNN::PPELayerType_ADD;
    case VPUIP::PPELayerType::SUB:
        return MVCNN::PPELayerType_SUB;
    case VPUIP::PPELayerType::MULT:
        return MVCNN::PPELayerType_MULT;
    case VPUIP::PPELayerType::MAXIMUM:
        return MVCNN::PPELayerType_MAXIMUM;
    case VPUIP::PPELayerType::MINIMUM:
        return MVCNN::PPELayerType_MINIMUM;
    case VPUIP::PPELayerType::AND:
        return MVCNN::PPELayerType_AND;
    case VPUIP::PPELayerType::OR:
        return MVCNN::PPELayerType_OR;
    case VPUIP::PPELayerType::XOR:
        return MVCNN::PPELayerType_XOR;
    case VPUIP::PPELayerType::LRELU:
        return MVCNN::PPELayerType_LRELU;
    case VPUIP::PPELayerType::LRELUX:
        return MVCNN::PPELayerType_LRELUX;
    case VPUIP::PPELayerType::LPRELU:
        return MVCNN::PPELayerType_LPRELU;
    case VPUIP::PPELayerType::CEIL:
        return MVCNN::PPELayerType_CEIL;
    case VPUIP::PPELayerType::FLOOR:
        return MVCNN::PPELayerType_FLOOR;
    case VPUIP::PPELayerType::EXP:
        return MVCNN::PPELayerType_EXP;
    case VPUIP::PPELayerType::SIGMOID:
        return MVCNN::PPELayerType_SIGMOID;
    case VPUIP::PPELayerType::TANH:
        return MVCNN::PPELayerType_TANH;
    case VPUIP::PPELayerType::SQRT:
        return MVCNN::PPELayerType_SQRT;
    case VPUIP::PPELayerType::RSQRT:
        return MVCNN::PPELayerType_RSQRT;
    case VPUIP::PPELayerType::FLEXARB:
        return MVCNN::PPELayerType_FLEXARB;
    case VPUIP::PPELayerType::NOT:
        return MVCNN::PPELayerType_NOT;
    case VPUIP::PPELayerType::ABS:
        return MVCNN::PPELayerType_ABS;
    case VPUIP::PPELayerType::NEG:
        return MVCNN::PPELayerType_NEG;
    default:
        VPUX_THROW("Unsupported PPE Layer type: '{0}'", ppeType);
    }
}

VPUIP::MPEMode getMPEFrequentModeFromDPUTasks(mlir::Region& dpuTaskOps) {
    std::unordered_map<VPUIP::MPEMode, size_t> umap;
    for (auto dpuTaskOp : dpuTaskOps.getOps<VPUIP::DPUTaskOp>()) {
        ++umap[dpuTaskOp.mpe_mode()];
        if (umap.size() > 1) {
            VPUX_THROW("Non-uniform DPU task MPE modes is not supported yet.");
        }
    }
    return umap.begin()->first;
}

// This is a helper routine to build new TensorReference out of NCE task output with provided
// quantization scale parameters
vpux::VPUIP::BlobWriter::TensorReference getTensorReferenceWithUpdatedQuantParams(VPUIP::NCEClusterTaskOp nceTask,
                                                                                  VPUIP::BlobWriter& writer,
                                                                                  ArrayRef<uint16_t> ppeQuantMult,
                                                                                  ArrayRef<uint8_t> ppeQuantShift,
                                                                                  int8_t ppeQuantPostShift) {
    // Get also ZP from output
    SmallVector<uint8_t> quantZeroPoints;

    auto outputElementType = nceTask.output().getType().cast<mlir::ShapedType>().getElementType();
    if (const auto uniformQuantType = outputElementType.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        quantZeroPoints.push_back(checked_cast<uint8_t>(uniformQuantType.getZeroPoint()));
    } else if (const auto uniformQuantPerAxisType =
                       outputElementType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        auto zp_output = uniformQuantPerAxisType.getZeroPoints();
        quantZeroPoints.resize(zp_output.size());
        std::transform(zp_output.begin(), zp_output.end(), quantZeroPoints.begin(), [](int64_t a) {
            return checked_cast<uint8_t>(a);
        });
    } else {
        quantZeroPoints.push_back(0);
    }

    VPUX_THROW_UNLESS(ppeQuantShift.size() == quantZeroPoints.size(),
                      "Mismatch of size between quant shift/mult vector and quant ZP:  {0} != {1}",
                      ppeQuantShift.size(), quantZeroPoints.size());

    // Find corresponding DeclareBufferOp to get all the data needed to build new TensorReference
    auto bufferOp = nceTask.output_buff().getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(bufferOp != nullptr, "Unable to find parent DeclareBufferOp to build new TensorReference");

    return writer.createTensorRef("output_tensor_scale_updated", nceTask.getType(0).cast<mlir::ShapedType>(),
                                  bufferOp.section(), bufferOp.sectionIndex().getValueOr(0), bufferOp.byteOffset(),
                                  ppeQuantMult, ppeQuantShift, ppeQuantPostShift, quantZeroPoints);
}

}  // namespace

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::NCEClusterTaskOp::serialize(VPUIP::BlobWriter& writer) {
    SmallVector<flatbuffers::Offset<MVCNN::NCEVariantFields>> variantList;
    for (auto dpuTaskOp : variants().getOps<VPUIP::DPUTaskOp>()) {
        const auto start = parseIntArrayAttr<int64_t>(dpuTaskOp.start());
        const auto end = parseIntArrayAttr<int64_t>(dpuTaskOp.end());
        const auto pad = dpuTaskOp.pad();
        const auto profilingData =
                (profiling_data() != nullptr && !variantList.size()) ? writer.getTensorRef(profiling_data()) : 0;

        const auto variant = MVCNN::CreateNCEVariantFields(writer,
                                                           0,                                            // Barriers
                                                           getMPEMode(dpuTaskOp.mpe_mode()),             // MPE mode
                                                           static_cast<int16_t>(pad.left().getInt()),    // padLeft
                                                           static_cast<int16_t>(pad.right().getInt()),   // padRight
                                                           static_cast<int16_t>(pad.top().getInt()),     // padTop
                                                           static_cast<int16_t>(pad.bottom().getInt()),  // padBottom
                                                           static_cast<int16_t>(start[0]),  // workload_start_X
                                                           static_cast<int16_t>(start[1]),  // workload_start_Y
                                                           static_cast<int16_t>(start[2]),  // workload_start_Z
                                                           static_cast<int16_t>(end[0]),    // workload_end_X
                                                           static_cast<int16_t>(end[1]),    // workload_end_Y
                                                           static_cast<int16_t>(end[2]),    // workload_end_Z
                                                           0,                               // flex_map_column
                                                           0,                               // flex_map_array
                                                           0,                               // flex_inner
                                                           0,                               // flex_outer
                                                           0,                               // flex_outer_order
                                                           profilingData                    // profiling_data
        );
        variantList.push_back(variant);
    }
    const auto variant = writer.createVector(variantList);

    SmallVector<uint8_t> ppeList;
    int32_t clampLow = std::numeric_limits<int32_t>::min();
    int32_t clampHigh = std::numeric_limits<int32_t>::max();
    int32_t LreluMult = 1;
    uint32_t LreluShift = 0;
    ::llvm::Optional<SmallVector<uint16_t>> ppeQuantMult;
    ::llvm::Optional<SmallVector<uint8_t>> ppeQuantShift;
    ::llvm::Optional<int8_t> ppeQuantPostShift;

    for (auto ppeOp : ppe().getOps<VPUIP::PPETaskOp>()) {
        const auto type = getPPELayerType(ppeOp.ppe_layer_type());
        if (type != MVCNN::PPELayerType_NOOP) {
            ppeList.push_back(type);
        }
        if (ppeOp.clamp_low().hasValue()) {
            clampLow = checked_cast<int32_t>(ppeOp.clamp_low().getValue());
        }
        if (ppeOp.clamp_high().hasValue()) {
            clampHigh = checked_cast<int32_t>(ppeOp.clamp_high().getValue());
        }
        if (ppeOp.lrelu_mult().hasValue()) {
            LreluMult = checked_cast<int32_t>(ppeOp.lrelu_mult().getValue());
        }
        if (ppeOp.lrelu_shift().hasValue()) {
            LreluShift = checked_cast<uint32_t>(ppeOp.lrelu_shift().getValue());
        }
        if (ppeOp.quant_mult().hasValue()) {
            ppeQuantMult = parseIntArrayAttr<uint16_t>(ppeOp.quant_mult().getValue());
        }
        if (ppeOp.quant_shift().hasValue()) {
            ppeQuantShift = parseIntArrayAttr<uint8_t>(ppeOp.quant_shift().getValue());
        }
        if (ppeOp.quant_post_shift().hasValue()) {
            ppeQuantPostShift = checked_cast<int8_t>(ppeOp.quant_post_shift().getValue());
        }
    }
    VPUX_THROW_UNLESS(ppeList.size() <= 1, "Cannot set more than one PPE task");

    auto ppeLayerTypes = writer.createVector(ppeList);
    // TODO: Lrelu_Mult, Lrelu_Shift
    auto ppeFixedFunction =
            MVCNN::CreatePPEFixedFunction(writer, ppeLayerTypes, clampLow, clampHigh, LreluMult, LreluShift);
    // TODO: scale_data, rounding, instruction_list_data
    auto ppeTask = MVCNN::CreatePPETask(writer, 0, ppeFixedFunction);

    int16_t kernelSizeH = 1, kernelSizeW = 1;
    int16_t kernelStridesH = 1, kernelStridesW = 1;
    int16_t kernelPadL = 0, kernelPadR = 0, kernelPadT = 0, kernelPadB = 0;
    flatbuffers::Offset<flatbuffers::Vector<int8_t>> enabled_optimizations = 0;
    int32_t odu_offset = 0;
    int32_t out_channel_offset = 0;
    bool is_segmented = false;
    bool is_continued = false;

    if (kernel_sizeAttr() != nullptr) {
        const auto kernelSize = parseIntArrayAttr<int64_t>(kernel_sizeAttr());
        kernelSizeH = checked_cast<int16_t>(kernelSize[0]);
        kernelSizeW = checked_cast<int16_t>(kernelSize[1]);
    }

    if (kernel_stridesAttr() != nullptr) {
        const auto kernelStrides = parseIntArrayAttr<int64_t>(kernel_stridesAttr());
        kernelStridesH = checked_cast<int16_t>(kernelStrides[0]);
        kernelStridesW = checked_cast<int16_t>(kernelStrides[1]);
    }

    if (kernel_paddingAttr() != nullptr) {
        const auto kernelPadding = parseIntArrayAttr<int64_t>(kernel_paddingAttr());
        kernelPadL = checked_cast<int16_t>(kernelPadding[0]);
        kernelPadR = checked_cast<int16_t>(kernelPadding[1]);
        kernelPadT = checked_cast<int16_t>(kernelPadding[2]);
        kernelPadB = checked_cast<int16_t>(kernelPadding[3]);
    }

    if (is_continuedAttr() != nullptr) {
        is_continued = true;
    }

    const auto inputData = writer.getTensorRef(input());
    const auto weightsData = weights() != nullptr ? writer.getTensorRef(weights()) : 0;
    const auto weightsTable = weight_table() != nullptr ? writer.getTensorRef(weight_table()) : 0;
    const auto activationWindow = activation_window() != nullptr ? writer.getTensorRef(activation_window()) : 0;
    const auto activationWindowChannelLength = checked_cast<int32_t>(activation_window_channel_length().getValueOr(0));

    auto outputData = writer.getTensorRef(output());

    // If quant scale (mult, shift) settings were provided as part of PPE block then use it to build new
    // output TensorReference. This is required for Eltwise operation which doesn't have weights table
    // and PPE quantization settings (Mult, Shift) need to be provided for NN runtime in output tensor descriptor
    const auto isQuantizationProvided =
            ppeQuantMult.hasValue() && ppeQuantShift.hasValue() && ppeQuantPostShift.hasValue();
    const auto isQuantizationNotProvided =
            !ppeQuantMult.hasValue() && !ppeQuantShift.hasValue() && !ppeQuantPostShift.hasValue();
    VPUX_THROW_WHEN(!isQuantizationProvided && !isQuantizationNotProvided, "Missing quantization scale settings.");

    if (isQuantizationProvided) {
        outputData = getTensorReferenceWithUpdatedQuantParams(*this, writer, ppeQuantMult.getValue(),
                                                              ppeQuantShift.getValue(), ppeQuantPostShift.getValue());
    }

    const auto parentInputTensor = writer.getTensorRef(parent_input());
    const auto parentOutputTensor = writer.getTensorRef(parent_output());

    const auto invariantMPEMode = getMPEFrequentModeFromDPUTasks(variants());

    const auto invariant =
            MVCNN::CreateNCEInvariantFields(writer,
                                            getDPULayerType(task_type()),   // dpu_task_type
                                            ppeTask,                        // ppe_task
                                            getMPEMode(invariantMPEMode),   // mpe_frequent_mode
                                            kernelSizeH,                    // kernelH
                                            kernelSizeW,                    // kernelW
                                            kernelStridesH,                 // kernel_strideH
                                            kernelStridesW,                 // kernel_strideW
                                            kernelPadL,                     // kernel_padLeft
                                            kernelPadR,                     // kernel_padRight
                                            kernelPadT,                     // kernel_padTop
                                            kernelPadB,                     // kernel_padBottom
                                            parentInputTensor,              // parent_input_tensor
                                            parentOutputTensor,             // parent_output_tensor
                                            0,                              // parent_weights_tensor
                                            inputData,                      // input_data
                                            outputData,                     // output_data
                                            weightsData,                    // weights_data
                                            weightsTable,                   // weights_table
                                            activationWindow,               // activation_window
                                            activationWindowChannelLength,  // activation_window_channel_length
                                            enabled_optimizations,          // enabled_optimizations
                                            odu_offset,                     // odu_offset
                                            out_channel_offset,             // out_channel_offset
                                            is_segmented,                   // is_segmented
                                            is_continued                    // is_continued
            );

    MVCNN::NCE2TaskBuilder builder(writer);
    builder.add_variant(variant);
    builder.add_invariant(invariant);

    return {builder.Finish().Union(), MVCNN::SpecificTask_NCE2Task};
}
