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

#include "vpux/compiler/dialect/VPUIP/utils.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPURT/attributes.hpp"

using namespace vpux;

//
// Run-time info
//

double vpux::VPUIP::getMemoryDerateFactor(IERT::MemoryResourceOp mem) {
    VPUX_THROW_UNLESS(mem.kindAttr() != nullptr, "Got empty memory resource kind");
    VPUX_THROW_UNLESS(mem.kindAttr().isa<VPU::MemoryKindAttr>(), "Unsupported memory resource kind '{0}'", mem.kind());

    auto attr = mem->getAttr(VPU::getMemoryDerateAttrName());
    VPUX_THROW_UNLESS(attr != nullptr, "Memory resource '{0}' has no '{1}' attribute", mem.kind(),
                      VPU::getMemoryDerateAttrName());
    VPUX_THROW_UNLESS(attr.isa<mlir::FloatAttr>(), "Memory resource '{0}' has wrong '{1}' attribute : '{2}'",
                      mem.kind(), VPU::getMemoryDerateAttrName(), attr);

    return attr.cast<mlir::FloatAttr>().getValueAsDouble();
}

uint32_t vpux::VPUIP::getMemoryBandwidth(IERT::MemoryResourceOp mem) {
    VPUX_THROW_UNLESS(mem.kindAttr() != nullptr, "Got empty memory resource kind");
    VPUX_THROW_UNLESS(mem.kindAttr().isa<VPU::MemoryKindAttr>(), "Unsupported memory resource kind '{0}'", mem.kind());

    auto attr = mem->getAttr(VPU::getMemoryBandwidthAttrName());
    VPUX_THROW_UNLESS(attr != nullptr, "Memory resource '{0}' has no '{1}' attribute", mem.kind(),
                      VPU::getMemoryBandwidthAttrName());
    VPUX_THROW_UNLESS(attr.isa<mlir::IntegerAttr>(), "Memory resource '{0}' has wrong '{1}' attribute : '{2}'",
                      mem.kind(), VPU::getMemoryBandwidthAttrName(), attr);

    return checked_cast<uint32_t>(attr.cast<mlir::IntegerAttr>().getInt());
}

double vpux::VPUIP::getProcessorFrequency(IERT::ExecutorResourceOp res) {
    VPUX_THROW_UNLESS(res.kindAttr() != nullptr, "Got empty executor resource kind");

    auto attr = res->getAttr(VPU::getProcessorFrequencyAttrName());
    VPUX_THROW_UNLESS(attr != nullptr, "Executor resource '{0}' has no '{1}' attribute", res.kind(),
                      VPU::getProcessorFrequencyAttrName());
    VPUX_THROW_UNLESS(attr.isa<mlir::FloatAttr>(), "Executor resource '{0}' has wrong '{1}' attribute : '{2}'",
                      res.kind(), VPU::getProcessorFrequencyAttrName(), attr);

    return attr.cast<mlir::FloatAttr>().getValueAsDouble();
}

//
// DW Convolution utility
//

namespace {

mlir::Value getAlignedConstWeights(mlir::OpBuilder& builder, mlir::Location loc, Const::DeclareOp weightsConst,
                                   Shape flatWeightShape, int64_t alignment) {
    auto weightsContentAttr = weightsConst.contentAttr();
    auto nchwWeightsContentAttr = weightsContentAttr.reorder(DimsOrder::NCHW);

    auto flatWeightsContentAttr = nchwWeightsContentAttr.reshape(flatWeightShape);
    auto alignedWeightsContentAttr = flatWeightsContentAttr.padWithZero({0, 0, 0, 0}, {0, alignment, 0, 0});
    auto nhwcWeightsContentAttr = alignedWeightsContentAttr.reorder(DimsOrder::NHWC);

    const auto OC = flatWeightShape[Dims4D::Filter::OC];
    const auto flatWeightChannelsCount = flatWeightShape[Dims4D::Filter::IC];
    const auto alignedWeightShape = SmallVector<int64_t>{OC, flatWeightChannelsCount + alignment, 1, 1};
    const auto origFilterType = weightsConst.output().getType().cast<mlir::ShapedType>();
    const auto outAllocType = mlir::MemRefType::get(alignedWeightShape, origFilterType.getElementType());
    const auto outAllocTypeNHWC = changeDimsOrder(outAllocType, DimsOrder::NHWC);
    auto alignedWeightsOp = builder.create<Const::DeclareOp>(loc, outAllocTypeNHWC, nhwcWeightsContentAttr);

    return alignedWeightsOp.output();
}

mlir::Value getAlignedNonConstWeights(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value origFilter,
                                      Shape flatWeightShape, int64_t alignment) {
    auto ctx = builder.getContext();
    // Step 1: Flatten input to OCxICx1x1, where IC = filters * KY * KX.
    const auto origFilterType = origFilter.getType().cast<mlir::ShapedType>();
    const auto flatWeightType =
            changeDimsOrder(changeShape(origFilterType, flatWeightShape), DimsOrder::fromValue(origFilter));
    auto flatWeightsOp = builder.create<IERT::GenericReshapeOp>(loc, flatWeightType, origFilter);

    // Step 2: Permute flat input to NCHW.
    auto flatWeightTypeNCHWType = changeDimsOrder(flatWeightType, DimsOrder::NCHW);
    const auto nchwAttr = mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(ctx));
    const auto flatWeightsDimsAttr =
            mlir::AffineMapAttr::get(DimsOrder::fromValue(flatWeightsOp.output()).toAffineMap(ctx));
    auto flatWeightsNCHW = builder.create<IERT::PermuteCastOp>(loc, flatWeightTypeNCHWType, flatWeightsOp.output(),
                                                               nchwAttr, flatWeightsDimsAttr);

    // Step 3: Create padding for flat NCHW input. IC must be a multiple of 16.
    const auto OC = flatWeightShape[Dims4D::Filter::OC];
    const auto flatWeightChannelsCount = flatWeightShape[Dims4D::Filter::IC];
    const auto alignedWeightShape = SmallVector<int64_t>{OC, flatWeightChannelsCount + alignment, 1, 1};
    const auto outAllocType = changeDimsOrder(
            mlir::MemRefType::get(alignedWeightShape, origFilterType.getElementType()), DimsOrder::NCHW);

    const auto padShape = SmallVector<int64_t>{OC, alignment, 1, 1};
    const auto padValues = std::vector<ngraph::float16>(OC * alignment, 0.f);
    const auto padType =
            changeDimsOrder(mlir::RankedTensorType::get(padShape, origFilterType.getElementType()), DimsOrder::NCHW);
    const auto padAttr = mlir::DenseElementsAttr::get(padType, makeArrayRef(padValues));
    const auto padContentAttr = Const::ContentAttr::get(padAttr);

    const auto padAllocType = mlir::MemRefType::get(padShape, origFilterType.getElementType());
    const auto padAllocTypeNHWC = changeDimsOrder(padAllocType, DimsOrder::NCHW);
    auto paddedTensor = builder.create<Const::DeclareOp>(loc, padAllocTypeNHWC, padContentAttr);

    // Step 4: Concatenate flat NCHW input with padding.
    auto subViewAlloc = builder.create<mlir::memref::AllocOp>(loc, outAllocType);

    const SmallVector<int64_t> filterOffsets = {0, 0, 0, 0};
    const auto filterOffsetsAttr = getIntArrayAttr(ctx, filterOffsets);
    const auto flatWeightShapeAttr = getIntArrayAttr(ctx, flatWeightShape);

    const SmallVector<int64_t> paddingOffsets = {0, flatWeightChannelsCount, 0, 0};
    const auto paddingOffsetsAttr = getIntArrayAttr(ctx, paddingOffsets);
    const auto padShapeAttr = getIntArrayAttr(ctx, padShape);

    auto subViewFilter = builder.create<IERT::SubViewOp>(loc, subViewAlloc, filterOffsetsAttr, flatWeightShapeAttr);
    auto subViewPadding = builder.create<IERT::SubViewOp>(loc, subViewAlloc, paddingOffsetsAttr, padShapeAttr);

    auto subViewFilterCopy = builder.create<IERT::CopyOp>(loc, flatWeightsNCHW.result(), subViewFilter);
    auto subViewPaddingCopy = builder.create<IERT::CopyOp>(loc, paddedTensor.output(), subViewPadding);

    auto concatViewOp = builder.create<IERT::ConcatViewOp>(
            loc, SmallVector<mlir::Value>{subViewFilterCopy.output(), subViewPaddingCopy.output()}, subViewAlloc);

    // Step 5: Permute the result to NHWC.
    auto outNHWCType = changeDimsOrder(outAllocType, DimsOrder::NHWC);
    const auto nhwcAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(ctx));

    auto outOpNCHW = builder.create<IERT::PermuteCastOp>(loc, outNHWCType, concatViewOp.output(), nhwcAttr, nchwAttr);

    return outOpNCHW.result();
}

}  // namespace

mlir::Value vpux::VPUIP::alignDepthWiseWeightsTensor(mlir::OpBuilder& builder, mlir::Location loc,
                                                     mlir::Value origFilter) {
    const auto filterShape = getShape(origFilter);
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto filtersPerInChan = filterShape[Dims4D::Filter::IC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto origFilterType = origFilter.getType().cast<mlir::ShapedType>();
    const auto depthwiseConvAlignment = VPUIP::NCEInvariant::getChannelAlignment(origFilterType.getElementType());
    const int64_t remainder = (filtersPerInChan * KY * KX) % depthwiseConvAlignment;
    VPUX_THROW_UNLESS(remainder >= 0, "Channel alignment cannot be negative: {0}", remainder);
    if (remainder == 0) {
        // nothing to align
        return origFilter;
    }

    const int64_t alignment = depthwiseConvAlignment - remainder;
    const auto flatWeightChannelsCount = filtersPerInChan * KY * KX;
    const auto flatWeightShape = Shape{OC, flatWeightChannelsCount, 1, 1};
    mlir::Value alignedFilter;
    if (auto weightsConst = origFilter.getDefiningOp<Const::DeclareOp>()) {
        alignedFilter = getAlignedConstWeights(builder, loc, weightsConst, flatWeightShape, alignment);
    } else {
        alignedFilter = getAlignedNonConstWeights(builder, loc, origFilter, flatWeightShape, alignment);
    }
    return alignedFilter;
}

MVCNN::TargetDevice vpux::VPUIP::mapTargetDevice(VPU::ArchKind kind) {
    switch (kind) {
    case VPU::ArchKind::KMB:
        return MVCNN::TargetDevice::TargetDevice_KMB;
    case VPU::ArchKind::TBH:
        return MVCNN::TargetDevice::TargetDevice_TBH;
    case VPU::ArchKind::MTL:
        return MVCNN::TargetDevice::TargetDevice_MTL;
    case VPU::ArchKind::LNL:
        return MVCNN::TargetDevice::TargetDevice_LNL;
    default:
        VPUX_THROW("Unsupported architecture '{0}'", kind);
    }
}

MVCNN::TargetDeviceRevision vpux::VPUIP::mapTargetDeviceRevision(VPU::ArchKind kind) {
    switch (kind) {
    case VPU::ArchKind::KMB:
        return MVCNN::TargetDeviceRevision::TargetDeviceRevision_B0;
    default:
        return MVCNN::TargetDeviceRevision::TargetDeviceRevision_NONE;
    }
}

MVCNN::DType vpux::VPUIP::createDType(mlir::Type type) {
    if (type.isF64()) {
        return MVCNN::DType_FP64;
    } else if (type.isF32()) {
        return MVCNN::DType_FP32;
    } else if (type.isF16()) {
        return MVCNN::DType_FP16;
    } else if (type.isBF16()) {
        return MVCNN::DType_BFP16;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int64_t))) {
        return MVCNN::DType_I64;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int32_t))) {
        return MVCNN::DType_I32;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int16_t))) {
        return MVCNN::DType_I16;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int8_t))) {
        return MVCNN::DType_I8;
    } else if (type.isSignedInteger(4)) {
        return MVCNN::DType_I4;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint64_t))) {
        return MVCNN::DType_U64;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint32_t))) {
        return MVCNN::DType_U32;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint16_t))) {
        return MVCNN::DType_U16;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint8_t))) {
        return MVCNN::DType_U8;
    } else if (type.isInteger(4)) {
        return MVCNN::DType_U4;
    } else if (type.isInteger(2)) {
        return MVCNN::DType_I2;
    } else if (type.isInteger(1)) {
        return MVCNN::DType_BIN;
    } else if (type.isa<mlir::quant::QuantizedType>()) {
        return createDType(type.cast<mlir::quant::QuantizedType>().getStorageType());
    } else {
        VPUX_THROW("Unsupported element type {0}", type);
    }
}

MVCNN::MemoryLocation vpux::VPUIP::createMemoryLocation(VPURT::BufferSection section) {
    switch (section) {
    case VPURT::BufferSection::NetworkInput:
        return MVCNN::MemoryLocation_ProgrammableInput;
    case VPURT::BufferSection::NetworkOutput:
        return MVCNN::MemoryLocation_ProgrammableOutput;
    case VPURT::BufferSection::ProfilingOutput:
        return MVCNN::MemoryLocation_ProfilingOutput;
    case VPURT::BufferSection::Constant:
        return MVCNN::MemoryLocation_GraphFile;
    case VPURT::BufferSection::SW_KernelText:
        return MVCNN::MemoryLocation_GFEmbeddedKernel;
    case VPURT::BufferSection::DDR:
        return MVCNN::MemoryLocation_VPU_DDR_Heap;
    case VPURT::BufferSection::CSRAM:
        return MVCNN::MemoryLocation_VPU_CSRAM;
    case VPURT::BufferSection::CMX_UPA:
        return MVCNN::MemoryLocation_VPU_CMX_UPA;
    case VPURT::BufferSection::CMX_NN:
        return MVCNN::MemoryLocation_VPU_CMX_NN;
    case VPURT::BufferSection::Register:
        return MVCNN::MemoryLocation_AbsoluteAddr;
    case VPURT::BufferSection::MAC_Accumulators:
        return MVCNN::MemoryLocation_MAC_Accumulators;
    default:
        VPUX_THROW("Unsupported BufferSection {0}", section);
    }
}

MVCNN::order3 vpux::VPUIP::createOrder3(mlir::ArrayAttr attr) {
    auto vec = parseIntArrayAttr<int64_t>(attr);
    std::reverse(vec.begin(), vec.end());

    VPUX_THROW_UNLESS(vec.size() <= 3, "Got wrong order array : {0}", vec);

    uint8_t x = 0, y = 0, z = 0;
    if (vec.size() >= 1) {
        x = checked_cast<uint8_t>(vec[0]);
    }
    if (vec.size() >= 2) {
        y = checked_cast<uint8_t>(vec[1]);
    }
    if (vec.size() >= 3) {
        z = checked_cast<uint8_t>(vec[2]);
    }

    return MVCNN::order3(x, y, z);
}
