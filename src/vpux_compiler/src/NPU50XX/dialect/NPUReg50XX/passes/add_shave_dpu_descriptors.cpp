//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/descriptors.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/platform_resources.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <npu_40xx_nnrt.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>

namespace vpux::NPUReg50XX {
#define GEN_PASS_DECL_ADDSHAVEDPUDESCRIPTORS
#define GEN_PASS_DEF_ADDSHAVEDPUDESCRIPTORS
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/passes.hpp.inc"
}  // namespace vpux::NPUReg50XX

using namespace vpux;

namespace {

using DpuInvariantDescriptor = vpux::NPUReg50XX::Descriptors::DpuInvariantRegister;
using DpuVariantDescriptor = vpux::NPUReg50XX::Descriptors::DpuVariantRegister;

// DpuParamsBuffers is preloaded at the same address on every tile, so it never overlaps the
// compiler managed workspace:
//
//   +0x000  uint32_t                 weightTable[16 * 4];   256 B
//   +0x100  VpuDPUInvariantRegisters invariant;             288 B
//   +0x220  VpuDPUVariantRegisters   variant[2];            192 B each, one per SHAVE
//
// The block lives inside the CMX metadata region, at a fixed offset from its start. The
// metadata region's own base is not a fixed address - it moves depending on which other
// reservations (stack frames, ...) are placed ahead of it - so it is read from
// config::getCMXMetadataReservedMemory() at pass run-time rather than assumed to be a constant.
constexpr int64_t DPU_PARAMS_METADATA_SUB_OFFSET = 0xEA00;
constexpr size_t VARIANT_COUNT = 2;
constexpr size_t WEIGHT_TABLE_SIZE_BYTES = 16 * 4 * sizeof(uint32_t);
constexpr size_t INVARIANT_REGS_SIZE_BYTES = sizeof(npu40xx::nn_public::VpuDPUInvariantRegisters);
constexpr size_t VARIANT_REGS_SIZE_BYTES = sizeof(npu40xx::nn_public::VpuDPUVariantRegisters);
constexpr size_t DPU_PARAMS_SIZE_BYTES =
        WEIGHT_TABLE_SIZE_BYTES + INVARIANT_REGS_SIZE_BYTES + VARIANT_COUNT * VARIANT_REGS_SIZE_BYTES;

// Byte offsets of the block members, relative to the block's own (run-time computed) CMX
// address. The descriptors reference their own block, so the dummy workload only ever touches
// these 928 bytes. All 32 byte aligned, as the pointer fields require.
constexpr uint32_t WEIGHT_TABLE_REL_OFFSET = 0x000;  // input/output/weights
constexpr uint32_t INVARIANT_REL_OFFSET = 0x100;     // invar_ptr target
// variant[i].halo_region, where the HW profiling stats of that variant are kept
constexpr std::array<uint32_t, VARIANT_COUNT> VARIANT_STATS_REL_OFFSET = {0x260, 0x320};

// The DPU encodes CMX addresses in 16 byte units (odu_ac_base) or 32 byte units (invar_ptr,
// hwp_wload_id). No masking is needed, CMX is 1.5 MB so every address fits the 21 bit field.
constexpr uint32_t DPU_CMX_ADDR_SHIFT = 4;
constexpr uint32_t DPU_PTR_SHIFT = 5;

// cmx_slice*_low_addr / cmx_slice_size describe the DPU CMX window itself rather than a buffer, so
// they carry no relocation - none of the NPUReg40XX/NPUReg50XX DPUInvariantOp::getRelocs entries
// target them. Derived from the compiler wide CMX constants, matching what the real invariant
// lowering (vpuipdpu2npureg50xx::DPUInvariantRewriter) writes.
constexpr uint32_t CMX_SLICE_LOW_ADDR = CMX_BASE_ADDR >> DPU_CMX_ADDR_SHIFT;
constexpr uint32_t CMX_SLICE_SIZE = static_cast<uint32_t>(CMX_SIZE_1_5MB.count()) >> DPU_CMX_ADDR_SHIFT;
static_assert(CMX_SLICE_LOW_ADDR == 0x4000000 && CMX_SLICE_SIZE == 0x18000,
              "CMX window must match DPUInvariantRewriter");

// The dummy workload is a single 1x1 spatial position over the minimal channel granularity
constexpr uint16_t DUMMY_SIZE_X = 1;
constexpr uint16_t DUMMY_SIZE_Y = 1;
constexpr uint16_t DUMMY_SIZE_Z = 16;
constexpr uint32_t DUMMY_WEIGHT_SIZE = 16;
constexpr uint32_t DUMMY_WEIGHT_NUM = 16;

// odu_cfg.dtype is a 3 bit type encoding, not a bit count. Mirrors ODUDataBitWidth::ODU_DTYPE_16BIT
constexpr uint8_t DUMMY_ODU_DTYPE_16BIT = 4;
constexpr uint32_t BASE_OFFSET_A = 0x200;
constexpr uint8_t ODU_NTHW_16 = 3;
constexpr uint8_t IDU_NTHW_NTK_16 = 3;
constexpr uint8_t PPE_FP_CONVERT_FP16 = 1;
// SINT fields, so the 0x7FFFFFFF / 0x80000000 register values are the signed extremes
constexpr int32_t PPE_INT_SCALE_HCLAMP = std::numeric_limits<int32_t>::max();
constexpr int32_t PPE_INT_SCALE_LCLAMP = std::numeric_limits<int32_t>::min();
constexpr uint8_t HWP_STAT_MODE_IDU_ODU = 3;
constexpr uint16_t HWP_UNIQUE_ID = 0xA55A;

DpuInvariantDescriptor buildDummyInvariant(uint32_t dpuParamsCmxOffset) {
    DpuInvariantDescriptor descriptor;
    const uint32_t weightTableCmxAddr = dpuParamsCmxOffset + WEIGHT_TABLE_REL_OFFSET;

    // CMX window the DPU sees
    descriptor.write<NPUReg50XX::Fields::cmx_slice0_low_addr>(CMX_SLICE_LOW_ADDR);
    descriptor.write<NPUReg50XX::Fields::cmx_slice1_low_addr>(CMX_SLICE_LOW_ADDR);
    descriptor.write<NPUReg50XX::Fields::cmx_slice2_low_addr>(CMX_SLICE_LOW_ADDR);
    descriptor.write<NPUReg50XX::Fields::cmx_slice3_low_addr>(CMX_SLICE_LOW_ADDR);
    descriptor.write<NPUReg50XX::Fields::cmx_slice_size>(CMX_SLICE_SIZE);

    // IDU: dense 1x1 kernel over the minimal FP16 input. amode / wmode stay 0, which already is
    // VpuInputTensorDType::FP16
    descriptor.write<NPUReg50XX::Fields::addr_format_sel>(1);
    descriptor.write<NPUReg50XX::Fields::kernel_y>(1);
    descriptor.write<NPUReg50XX::Fields::kernel_x>(1);
    descriptor.write<NPUReg50XX::Fields::act_dense>(1);
    descriptor.write<NPUReg50XX::Fields::wt_dense>(1);
    descriptor.write<NPUReg50XX::Fields::dynamic_bw_en>(1);
    descriptor.write<NPUReg50XX::Fields::rst_ctxt>(1);
    descriptor.write<NPUReg50XX::Fields::zm_input>(1);
    descriptor.write<NPUReg50XX::Fields::base_offset_a>(BASE_OFFSET_A);
    descriptor.write<NPUReg50XX::Fields::tensor_size_x>(DUMMY_SIZE_X);
    descriptor.write<NPUReg50XX::Fields::tensor_size_y>(DUMMY_SIZE_Y);
    descriptor.write<NPUReg50XX::Fields::tensor_size_z>(DUMMY_SIZE_Z);

    // Input and weights both point at the weight table of this very block, so the dummy workload
    // only ever touches its own 256 reserved bytes
    descriptor.write<NPUReg50XX::Fields::act_offset0>(weightTableCmxAddr);
    descriptor.write<NPUReg50XX::Fields::act_offset1>(weightTableCmxAddr);
    descriptor.write<NPUReg50XX::Fields::act_offset2>(weightTableCmxAddr);
    descriptor.write<NPUReg50XX::Fields::act_offset3>(weightTableCmxAddr);
    descriptor.write<NPUReg50XX::Fields::wt_offset>(weightTableCmxAddr);

    // ODU: te_dim_* holds the last valid index, hence the -1
    descriptor.write<NPUReg50XX::Fields::dtype>(DUMMY_ODU_DTYPE_16BIT);
    descriptor.write<NPUReg50XX::Fields::nthw>(ODU_NTHW_16);
    descriptor.write<NPUReg50XX::Fields::te_dim_x>(DUMMY_SIZE_X - 1);
    descriptor.write<NPUReg50XX::Fields::te_dim_y>(DUMMY_SIZE_Y - 1);
    descriptor.write<NPUReg50XX::Fields::te_dim_z>(DUMMY_WEIGHT_NUM - 1);
    // ac_base is 16 byte granular, unlike act_offset / wt_offset above
    descriptor.write<NPUReg50XX::Fields::ac_base>(weightTableCmxAddr >> DPU_CMX_ADDR_SHIFT);

    // The ODU write stays disabled, so the dummy workload never stores its result
    descriptor.write<NPUReg50XX::Fields::write_ac>(0);

    // PPE
    descriptor.write<NPUReg50XX::Fields::ppe_fp16_clamp>(1);
    descriptor.write<NPUReg50XX::Fields::ppe_fp_convert>(PPE_FP_CONVERT_FP16);
    descriptor.write<NPUReg50XX::Fields::ppe_scale_mult>(1);
    descriptor.write<NPUReg50XX::Fields::ppe_scale_override>(1);
    descriptor.write<NPUReg50XX::Fields::ppe_prelu_mult>(1);
    descriptor.write<NPUReg50XX::Fields::ppe_int_scale_hclamp>(PPE_INT_SCALE_HCLAMP);
    descriptor.write<NPUReg50XX::Fields::ppe_int_scale_lclamp>(PPE_INT_SCALE_LCLAMP);

    // HW profiling, the per variant stats land in the halo region addressed by hwp_wload_id
    descriptor.write<NPUReg50XX::Fields::hwp_en>(1);
    descriptor.write<NPUReg50XX::Fields::hwp_stat_mode>(HWP_STAT_MODE_IDU_ODU);
    descriptor.write<NPUReg50XX::Fields::unique_ID>(HWP_UNIQUE_ID);
    descriptor.write<NPUReg50XX::Fields::hwp_cmx_mem_addr>(CMX_BASE_ADDR >> DPU_PTR_SHIFT);

    descriptor.write<NPUReg50XX::Fields::nvar_tag>(1);

    // Note: variant_count_ / cluster_invariant_ are deliberately not set - they live in
    // VpuDPUInvariant past the end of VpuDPUInvariantRegisters, so they are not part of the block

    return descriptor;
}

DpuVariantDescriptor buildDummyVariant(uint32_t dpuParamsCmxOffset, size_t variantIdx) {
    DpuVariantDescriptor descriptor;
    const uint32_t weightTableCmxAddr = dpuParamsCmxOffset + WEIGHT_TABLE_REL_OFFSET;
    const uint32_t invariantCmxAddr = dpuParamsCmxOffset + INVARIANT_REL_OFFSET;

    descriptor.write<NPUReg50XX::Fields::var_tag>(1);
    // Points back at the invariant of this same block
    descriptor.write<NPUReg50XX::Fields::invar_ptr>(invariantCmxAddr >> DPU_PTR_SHIFT);

    descriptor.write<NPUReg50XX::Fields::nthw_ntk>(IDU_NTHW_NTK_16);
    descriptor.write<NPUReg50XX::Fields::dense_se>(1);
    descriptor.write<NPUReg50XX::Fields::odu_stat_en>(1);
    descriptor.write<NPUReg50XX::Fields::idu_stat_en>(1);
    //  Note: wt_swizzle_sel needs to be on by default to match NNRT GF behaviour
    descriptor.write<NPUReg50XX::Fields::wt_swizzle_sel>(1);

    descriptor.write<NPUReg50XX::Fields::invar_lptr_force>(1);
    descriptor.write<NPUReg50XX::Fields::workload_start_odu>(1);
    descriptor.write<NPUReg50XX::Fields::workload_start_idu>(1);
    descriptor.write<NPUReg50XX::Fields::workload_idu_auto_upd_0>(1);

    // Weights come from the weight table of this block, same as the invariant input
    descriptor.write<NPUReg50XX::Fields::weight_start>(weightTableCmxAddr);
    descriptor.write<NPUReg50XX::Fields::weight_size>(DUMMY_WEIGHT_SIZE);
    descriptor.write<NPUReg50XX::Fields::weight_num>(DUMMY_WEIGHT_NUM);

    // The whole (minimal) output tensor is produced by this variant
    descriptor.write<NPUReg50XX::Fields::workload_size_x>(DUMMY_SIZE_X);
    descriptor.write<NPUReg50XX::Fields::workload_size_y>(DUMMY_SIZE_Y);
    descriptor.write<NPUReg50XX::Fields::workload_size_z>(DUMMY_SIZE_Z);
    descriptor.write<NPUReg50XX::Fields::te_end_x>(DUMMY_SIZE_X - 1);
    descriptor.write<NPUReg50XX::Fields::te_end_y>(DUMMY_SIZE_Y - 1);
    descriptor.write<NPUReg50XX::Fields::te_end_z>(DUMMY_WEIGHT_NUM - 1);

    // Stats of this variant go to its own halo region, which stays zeroed in the block
    const uint32_t variantStatsCmxAddr = dpuParamsCmxOffset + VARIANT_STATS_REL_OFFSET.at(variantIdx);
    descriptor.write<NPUReg50XX::Fields::hwp_wload_id>(variantStatsCmxAddr >> DPU_PTR_SHIFT);

    return descriptor;
}

// Builds the contents of
//
//     typedef struct {
//         uint32_t                 weightTable[16 * 4];
//         VpuDPUInvariantRegisters invariant;
//         VpuDPUVariantRegisters   variant[2];
//     } DpuParamsBuffers;
//
// Note the structure embeds the plain register blocks, while a descriptor covers the whole
// VpuDPUInvariant / VpuDPUVariant task. Since the register block is the first member of both
// tasks, taking the front of the descriptor storage yields exactly the needed bytes.
SmallVector<uint32_t> buildDpuParamsPayload(uint32_t dpuParamsCmxOffset) {
    SmallVector<uint8_t> payload;
    payload.reserve(DPU_PARAMS_SIZE_BYTES);

    // The dummy workload reads no weights, the weight table is left zeroed
    payload.resize(WEIGHT_TABLE_SIZE_BYTES, 0);

    const auto invariant = buildDummyInvariant(dpuParamsCmxOffset);
    VPUX_THROW_UNLESS(invariant.size() >= INVARIANT_REGS_SIZE_BYTES,
                      "DpuInvariantRegister descriptor of size {0} is smaller than the {1} bytes register block",
                      invariant.size(), INVARIANT_REGS_SIZE_BYTES);
    llvm::append_range(payload, invariant.getStorage().take_front(INVARIANT_REGS_SIZE_BYTES));

    for (size_t variantIdx = 0; variantIdx < VARIANT_COUNT; ++variantIdx) {
        const auto variant = buildDummyVariant(dpuParamsCmxOffset, variantIdx);
        VPUX_THROW_UNLESS(variant.size() >= VARIANT_REGS_SIZE_BYTES,
                          "DpuVariantRegister descriptor of size {0} is smaller than the {1} bytes register block",
                          variant.size(), VARIANT_REGS_SIZE_BYTES);
        llvm::append_range(payload, variant.getStorage().take_front(VARIANT_REGS_SIZE_BYTES));
    }

    VPUX_THROW_UNLESS(payload.size() == DPU_PARAMS_SIZE_BYTES, "Built {0} bytes of DPU parameters, expected {1}",
                      payload.size(), DPU_PARAMS_SIZE_BYTES);

    SmallVector<uint32_t> words(DPU_PARAMS_SIZE_BYTES / sizeof(uint32_t));
    std::memcpy(words.data(), payload.data(), DPU_PARAMS_SIZE_BYTES);

    return words;
}

// SHAVE kernels which drive the DPU expect the DPU parameters block to be already present in CMX.
// This is the VPUMI40XX counterpart of VPURT::EnqueueBarrierHandler::findShvTasksWithDpu - at this
// point in the pipeline VPURT::TaskOp is gone, but DeclareKernelEntryOp still carries the very same
// "VPU.kernel_entry" string that VPUIP::isSwKernelUseDpu matches against.
bool hasDpuShaveTasks(mlir::func::FuncOp netFunc) {
    for (auto rangeOp : netFunc.getOps<VPUMI40XX::ActKernelRangeOp>()) {
        auto kernelEntryIndex = rangeOp.getKernelEntryIndex();
        if (kernelEntryIndex == nullptr) {
            continue;
        }

        auto entryOp = mlir::dyn_cast_or_null<VPUMI40XX::DeclareKernelEntryOp>(kernelEntryIndex.getDefiningOp());
        if (entryOp == nullptr) {
            continue;
        }

        if (llvm::is_contained(VPUIP::SW_KERNELS_USE_DPU, entryOp.getKernelPath())) {
            return true;
        }
    }

    return false;
}

class AddShaveDpuDescriptors : public NPUReg50XX::impl::AddShaveDpuDescriptorsBase<AddShaveDpuDescriptors> {
public:
    explicit AddShaveDpuDescriptors(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    Const::DeclareOp createDpuParamsConstant(mlir::OpBuilder& builder, mlir::Operation* insertionPoint,
                                             ArrayRef<uint32_t> payload);
    VPURT::DeclareBufferOp createDpuParamsBufferOp(mlir::OpBuilder& builder, mlir::Operation* insertionPoint,
                                                   mlir::Type constType, int64_t tile, int64_t dpuParamsCmxOffset);
    VPUMI40XX::NNDMAOp createDpuParamsDmaOp(mlir::OpBuilder& builder, mlir::Operation* insertionPoint,
                                            mlir::Value input, mlir::Value output, mlir::Value previousTask);

private:
    void safeRunOnFunc() final;
};

Const::DeclareOp AddShaveDpuDescriptors::createDpuParamsConstant(mlir::OpBuilder& builder,
                                                                 mlir::Operation* insertionPoint,
                                                                 ArrayRef<uint32_t> payload) {
    const auto elemType = getUInt32Type(builder.getContext());
    const Shape valShape = {static_cast<int64_t>(payload.size())};
    const auto dataStorageType = mlir::RankedTensorType::get(valShape.raw(), elemType);
    const auto dataAttr = mlir::DenseElementsAttr::get(dataStorageType, payload);

    auto memType = vpux::getMemRefType(ShapeRef(dataStorageType.getShape()), dataStorageType.getElementType());
    builder.setInsertionPoint(insertionPoint);

    return Const::DeclareOp::create(builder, builder.getUnknownLoc(), memType, Const::ContentAttr::get(dataAttr));
}

VPURT::DeclareBufferOp AddShaveDpuDescriptors::createDpuParamsBufferOp(mlir::OpBuilder& builder,
                                                                       mlir::Operation* insertionPoint,
                                                                       mlir::Type constType, int64_t tile,
                                                                       int64_t dpuParamsCmxOffset) {
    const auto memKindAttr = IndexedSymbolAttr::get(builder.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN), tile);
    const auto type = mlir::cast<vpux::NDTypeInterface>(constType);
    const auto newType = type.changeMemSpace(memKindAttr);
    auto memType = vpux::getMemRefType(newType.getShape(), newType.getElementType(),
                                       DimsOrder::fromNumDims(newType.getShape().size()), newType.getMemSpace());

    builder.setInsertionPoint(insertionPoint);

    return VPURT::DeclareBufferOp::create(builder, builder.getUnknownLoc(), memType, VPURT::BufferSection::CMX_NN, tile,
                                          dpuParamsCmxOffset);
}

VPUMI40XX::NNDMAOp AddShaveDpuDescriptors::createDpuParamsDmaOp(mlir::OpBuilder& builder,
                                                                mlir::Operation* insertionPoint, mlir::Value input,
                                                                mlir::Value output, mlir::Value previousTask) {
    auto ctx = builder.getContext();

    // Plain contiguous transfer of the whole block
    auto zeroAttr = vpux::getIntAttr(ctx, 0);
    auto sizeAttr = vpux::getIntAttr(ctx, DPU_PARAMS_SIZE_BYTES);
    auto dmaDescriptorAttr = VPUIP::DMADescriptorAttr::get(ctx, /*numPlane*/ zeroAttr, /*len*/ sizeAttr,
                                                           /*srcWidth*/ sizeAttr, /*srcStride*/ zeroAttr,
                                                           /*srcPlaneStride*/ zeroAttr, /*dstWidth*/ sizeAttr,
                                                           /*dstStride*/ zeroAttr, /*dstPlaneStride*/ zeroAttr);

    // Any index will do, reindexList below assigns the final one
    auto indexType = VPURegMapped::IndexType::get(ctx, 0, 0, 0);

    auto wlmPageAttr = vpux::getIntAttr(ctx, -1);

    builder.setInsertionPoint(insertionPoint);

    return VPUMI40XX::NNDMAOp::create(builder, builder.getUnknownLoc(), indexType, /*taskLocation*/ nullptr, input,
                                      output, previousTask, /*waitBarriers*/ mlir::ValueRange(),
                                      /*updateBarriers*/ mlir::ValueRange(),
                                      /*startAfter*/ 0,
                                      /*cleanAfter*/ 0, false, false, false, 0, VPUIP::DMAAccMode::DISABLE, nullptr,
                                      nullptr,
                                      /*transactionAttr*/ nullptr, dmaDescriptorAttr, nullptr, nullptr, false, nullptr,
                                      /*enqueueBarrier*/ nullptr, wlmPageAttr);
}

void AddShaveDpuDescriptors::safeRunOnFunc() {
    auto netFunc = getOperation();

    if (!hasDpuShaveTasks(netFunc)) {
        _log.trace("No SHAVE task uses the DPU, skip creating the DPU parameters buffers");
        return;
    }

    auto parentModule = netFunc.getOperation()->getParentOfType<mlir::ModuleOp>();
    const auto tilesCount = config::getTileExecutor(parentModule).getCount();

    // The metadata region's own base offset is not fixed - it depends on which other bottom
    // anchored reservations (e.g. stack frames) were registered ahead of it - so it is queried
    // rather than assumed. See CMXMetadataReserveMemPass, which registers this resource early in
    // the pipeline, long before this pass runs.
    auto metadataMem = config::getCMXMetadataReservedMemory(parentModule);
    VPUX_THROW_UNLESS(metadataMem != nullptr, "Missing reserved CMX memory for metadata");
    auto metadataMemOffset = metadataMem.getOffset();
    VPUX_THROW_UNLESS(metadataMemOffset.has_value(), "No address allocated for metadata in CMX");
    const auto metadataMemSize = static_cast<int64_t>(metadataMem.getByteSize());
    VPUX_THROW_UNLESS(DPU_PARAMS_METADATA_SUB_OFFSET + static_cast<int64_t>(DPU_PARAMS_SIZE_BYTES) <= metadataMemSize,
                      "DPU parameters block of {0} bytes at metadata sub-offset {1:x} does not fit inside the {2} "
                      "byte CMX metadata region",
                      DPU_PARAMS_SIZE_BYTES, DPU_PARAMS_METADATA_SUB_OFFSET, metadataMemSize);
    const int64_t dpuParamsCmxOffset = metadataMemOffset.value() + DPU_PARAMS_METADATA_SUB_OFFSET;

    auto mpi = VPUMI40XX::getMPI(netFunc);
    auto builder = mlir::OpBuilder(mpi.getOperation());

    auto bufferOps = netFunc.getOps<VPURT::DeclareBufferOp>();
    auto bufferInsertionPoint = !bufferOps.empty() ? *bufferOps.begin() : &netFunc.getBody().front().front();

    auto declOps = netFunc.getOps<Const::DeclareOp>();
    auto cstInsertionPoint = !declOps.empty() ? *declOps.begin() : &netFunc.getBody().front().front();

    auto dmaTypeOps = netFunc.getOps<VPURegMapped::DMATypeOpInterface>();
    mlir::Operation* dmaInsertionPoint = !dmaTypeOps.empty() ? *dmaTypeOps.begin() : &netFunc.getBody().front().front();

    _log.trace("Creating DPU parameters buffers of {0} bytes at CMX offset {1:x} for {2} tile(s)",
               DPU_PARAMS_SIZE_BYTES, dpuParamsCmxOffset, tilesCount);

    // The very same content is written to every tile, so a single constant is enough
    auto constOp = createDpuParamsConstant(builder, cstInsertionPoint,
                                           buildDpuParamsPayload(static_cast<uint32_t>(dpuParamsCmxOffset)));

    VPUMI40XX::NNDMAOp firstDmaOp;
    VPUMI40XX::NNDMAOp previousDmaOp;
    for (int64_t tile = 0; tile < tilesCount; ++tile) {
        auto bufferOp = createDpuParamsBufferOp(builder, bufferInsertionPoint, constOp.getOutput().getType(), tile,
                                                dpuParamsCmxOffset);
        auto dmaOp = createDpuParamsDmaOp(builder, dmaInsertionPoint, constOp.getOutput(), bufferOp.getBuffer(),
                                          previousDmaOp != nullptr ? previousDmaOp.getResult() : nullptr);

        if (firstDmaOp == nullptr) {
            firstDmaOp = dmaOp;
        }
        previousDmaOp = dmaOp;
    }

    // Chain the previously first DMA after the new ones so that they are executed at bootstrap
    if (auto dmaTypeOp = llvm::dyn_cast<VPURegMapped::DMATypeOpInterface>(dmaInsertionPoint)) {
        dmaTypeOp.setPreviousTaskForOp(previousDmaOp);
    }

    // Reindex from the new head. Passing mpi.getListHead() instead would start the walk at the old
    // head, leaving the DMAs created above with their placeholder index
    VPUMI40XX::reindexList<VPUMI40XX::NNDMAOp>(mpi, firstDmaOp, 0, 0);
}

}  // namespace

//
// createAddShaveDpuDescriptors
//

std::unique_ptr<mlir::Pass> vpux::NPUReg50XX::createAddShaveDpuDescriptors(Logger log) {
    return std::make_unique<AddShaveDpuDescriptors>(log);
}
