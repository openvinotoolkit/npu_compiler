//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/ELFNPU37XX/utils.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/utils/core/mem_size.hpp"

#include "vpux/compiler/dialect/VPUMI37XX/utils.hpp"

#include <npu_37xx_nnrt.hpp>

using namespace vpux;
using namespace npu37xx;

//
// NNDMAOp
//

// For further development, please refer to the ticket E#36225.

void VPUMI37XX::NNDMAOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, mlir::Type index,
                               mlir::Value input, mlir::ValueRange output_buffs, mlir::Value previousDMAIdx,
                               mlir::ValueRange waitBarriers, mlir::ValueRange updateBarriers, uint64_t start_after,
                               uint64_t clean_after, bool is_out_of_order, bool is_critical, int64_t port,
                               vpux::VPUIP::DMAAccMode acceleration_mode, VPUIP::DMADescriptorAttr dma_descriptor) {
    build(odsBuilder, odsState, index, nullptr, input, output_buffs, previousDMAIdx, waitBarriers, updateBarriers,
          start_after, clean_after, is_out_of_order, is_critical, port, acceleration_mode, dma_descriptor);
}

namespace {

void decode_storage_order(ShapeRef dims, StridesRef strides, unsigned char* order) {
    const size_t S = dims.size();

    for (unsigned int i = 0; i < S; ++i) {
        order[i] = i;
    }

    std::sort(&order[0], &order[0] + S, [&](int lhs, int rhs) {
        return std::make_tuple(strides[Dim(lhs)], dims[Dim(lhs)], lhs) <
               std::make_tuple(strides[Dim(rhs)], dims[Dim(rhs)], rhs);
    });
}

class SimplifiedTensorLayout {
public:
    explicit SimplifiedTensorLayout(mlir::Value value) {
        VPUX_THROW_UNLESS(value, "Encountered nullptr value");

        auto ndType = mlir::cast<vpux::NDTypeInterface>(value.getType());
        const auto sizes = ndType.getShape();
        const auto strides = ndType.getStrides();
        auto dims = static_cast<unsigned int>(sizes.size());

        std::vector<unsigned char> order(dims, 0);
        decode_storage_order(sizes, strides, order.data());

        unsigned int line_stride_in_bits = 0;
        unsigned int plane_stride_in_bits = 0;
        unsigned int* rt_dims[SimplifiedTensorLayout::STRIDING_LEVELS] = {&line_length_, &plane_length_};
        unsigned int* rt_strides[SimplifiedTensorLayout::STRIDING_LEVELS] = {&line_stride_in_bits,
                                                                             &plane_stride_in_bits};

        auto bit_strides = [&](Dim i) -> unsigned int {
            return static_cast<unsigned int>(strides[i].count());
        };

        unsigned int previous_size = 1;
        unsigned int previous_stride = static_cast<unsigned int>(vpux::getElemTypeSize(ndType).count());
        unsigned int total_length_in_bits = previous_stride;

        // In case of plane stride dimension at (dims-1)
        plane_dim_ = dims - 1;
        for (unsigned int dim = 0, level = 0; dim < dims; ++dim) {
            const unsigned int crt_size = checked_cast<unsigned int>(sizes[Dim(order[dim])]);
            unsigned int crt_stride = bit_strides(Dim(order[dim]));
            total_length_in_bits *= crt_size;

            if (previous_size * previous_stride < crt_stride) {
                if (sizes[Dim(order[dim])] == 1) {
                    if (dim + 1 == dims) {
                        continue;
                    }

                    crt_stride = bit_strides(Dim(order[dim + 1]));
                }

                VPUX_THROW_UNLESS(level < SimplifiedTensorLayout::STRIDING_LEVELS, "Max striding levels exceeded");

                *rt_strides[level] = crt_stride;
                *rt_dims[level] = (previous_size * previous_stride) / (level ? *rt_strides[level - 1] : CHAR_BIT);
                if (level == (SimplifiedTensorLayout::STRIDING_LEVELS - 1)) {
                    plane_dim_ = dim;
                }
                ++level;
            }

            previous_size = crt_size;
            previous_stride = crt_stride;
        }

        line_stride_ = line_stride_in_bits / CHAR_BIT;
        plane_stride_ = plane_stride_in_bits / CHAR_BIT;
        total_length_ = total_length_in_bits / CHAR_BIT;
    }

    unsigned int line_stride() const {
        return line_stride_;
    }
    unsigned int line_length() const {
        return line_length_;
    }
    unsigned int plane_dimension() const {
        return plane_dim_;
    }
    unsigned int plane_stride() const {
        return plane_stride_;
    }
    unsigned int plane_length() const {
        return plane_length_;
    }
    unsigned int plane_count() const {
        return plane_length_ ? (total_length_ / plane_length_ / (line_length_ ? line_length_ : 1)) : 1;
    }
    unsigned int total_length() const {
        return total_length_;
    }

private:
    static constexpr auto STRIDING_LEVELS = 2;
    unsigned int line_stride_ = 0;
    unsigned int line_length_ = 0;
    unsigned int plane_stride_ = 0;
    unsigned int plane_length_ = 0;
    unsigned int plane_dim_ = 0;
    unsigned int total_length_ = 0;
};

}  // namespace

void vpux::VPUMI37XX::NNDMAOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    nn_public::VpuDMATask dmaTask;

    // safe init to zero the structure
    memset(reinterpret_cast<void*>(&dmaTask), 0, sizeof(dmaTask));

    const auto hasDescriptor = getDmaDescriptor().has_value();

    dmaTask.barriers_sched_.start_after_ = checked_cast<uint32_t>(getStartAfter());
    dmaTask.barriers_sched_.clean_after_ = checked_cast<uint32_t>(getCleanAfter());

    auto& descriptor = dmaTask.transaction_;
    descriptor.cfg_link.cfg_bits.burst_length = 255;  // set burst lenght to max value
    descriptor.cfg_link.cfg_bits.barrier_en = 1;

    // In case of multicasting (multiple outputs) we will mask the destination with the multicast mask;

    if (getOutputBuffs().size() > 1) {
        descriptor.dst = 0xC00000;
    }

    if (auto nextDMAIndexValue = getNextDMAIdx()) {
        descriptor.link_address =
                checked_cast<uint64_t>(mlir::cast<VPURegMapped::IndexType>(nextDMAIndexValue.getType()).getValue());
    }

    descriptor.cfg_link.cfg_bits.critical = 1;
    descriptor.cfg_link.cfg_bits.order_forced = !getIsOutOfOrder();
    descriptor.cfg_link.cfg_bits.skip_nr = 63;

    auto src_layout = SimplifiedTensorLayout(getInput());
    auto dst_layout = SimplifiedTensorLayout(getOutputBuffs()[0]);

    uint32_t src_width = src_layout.line_length();
    uint32_t dst_width = dst_layout.line_length();
    uint32_t src_stride = src_layout.line_stride();
    uint32_t dst_stride = dst_layout.line_stride();
    uint32_t num_planes = src_layout.plane_count();
    uint32_t src_plane_stride = src_layout.plane_stride();
    uint32_t dst_plane_stride = dst_layout.plane_stride();
    uint32_t size = src_layout.total_length();

    const auto getPlaneStride = [&](mlir::Value value, unsigned int dim) -> uint32_t {
        const auto ndType = mlir::cast<vpux::NDTypeInterface>(value.getType());
        const auto memStride = ndType.getMemStrides();
        const auto reversedDim = ndType.getRank() - 1 - dim;
        return checked_cast<uint32_t>(Byte(memStride[MemDim(reversedDim)]).count());
    };

    if (!hasDescriptor && getAccelerationMode() == vpux::VPUIP::DMAAccMode::DISABLE) {
        if (!!src_plane_stride ^ !!dst_plane_stride) {
            // For the case that src-stride-level=2 and dst-stride-level=1 or vice verse, we can't simply
            // calculate plane stride with 'size/num_planes' because src or dst is with stride.
            // Use src/dst plane stride dimension to search for des/src plane stride in its memory stride shape,
            // so the plane dimension is valid only when the correlative plane stride is not zero.
            if (src_plane_stride) {
                num_planes = std::max(1u, src_layout.plane_count());
                dst_plane_stride = getPlaneStride(getOutputBuffs()[0], src_layout.plane_dimension());
            } else {
                num_planes = std::max(1u, dst_layout.plane_count());
                src_plane_stride = getPlaneStride(getInput(), dst_layout.plane_dimension());
            }
        }

        VPUX_THROW_UNLESS(num_planes > 0, "Encountered num planes = {0}", num_planes);

        // Plane size calculated here because num_planes might change above and there would be a need to re-adjust
        size = size / num_planes;

        if (src_width == src_stride) {
            src_width = size;
            src_stride = size;
        }
        if (dst_width == dst_stride) {
            dst_width = size;
            dst_stride = size;
        }
    }

    if (hasDescriptor) {
        const auto dmaDescriptor = getDmaDescriptor().value();
        descriptor.length = checked_cast<uint32_t>(dmaDescriptor.getLen().getInt());
        descriptor.attr2d.src_width = checked_cast<uint32_t>(dmaDescriptor.getSrcWidth().getInt());
        descriptor.attr2d.dst_width = checked_cast<uint32_t>(dmaDescriptor.getDstWidth().getInt());
        descriptor.attr2d.src_stride = checked_cast<int32_t>(dmaDescriptor.getSrcStride().getInt());
        descriptor.attr2d.dst_stride = checked_cast<int32_t>(dmaDescriptor.getDstStride().getInt());
        descriptor.src_plane_stride = checked_cast<int32_t>(dmaDescriptor.getSrcPlaneStride().getInt());
        descriptor.dst_plane_stride = checked_cast<int32_t>(dmaDescriptor.getDstPlaneStride().getInt());
        descriptor.num_planes = checked_cast<uint32_t>(dmaDescriptor.getNumPlanes().getInt());
    } else {
        descriptor.length = size;
        descriptor.attr2d.src_width = src_width;
        descriptor.attr2d.dst_width = dst_width;
        descriptor.attr2d.src_stride = checked_cast<int32_t>(src_stride);
        descriptor.attr2d.dst_stride = checked_cast<int32_t>(dst_stride);
        descriptor.src_plane_stride = checked_cast<int32_t>(src_plane_stride);
        descriptor.dst_plane_stride = checked_cast<int32_t>(dst_plane_stride);
        descriptor.num_planes = num_planes;
    }

    --descriptor.num_planes;
    if (!descriptor.attr2d.src_width && !descriptor.attr2d.dst_width && !descriptor.attr2d.src_stride &&
        !descriptor.attr2d.dst_stride) {
        descriptor.num_planes = descriptor.src_plane_stride = descriptor.dst_plane_stride = 0;
        descriptor.cfg_link.cfg_bits.type = 0;
    } else if (!descriptor.num_planes) {
        descriptor.src_plane_stride = descriptor.dst_plane_stride = 0;
        descriptor.cfg_link.cfg_bits.type = 1;
    } else {
        descriptor.cfg_link.cfg_bits.type = 1;
    }

    switch (getAccelerationMode()) {
    case vpux::VPUIP::DMAAccMode::DECOMPRESSION:
        descriptor.cfg_link.cfg_bits.dec_en = 1;
        VPUX_THROW_UNLESS(descriptor.num_planes == 0,
                          "For DMA compression to be possible, the computed num_planes for the transaction needs to be "
                          "0, got {0}",
                          checked_cast<uint8_t>(descriptor.num_planes));

        // Ensure plane strides are set to 0 and set transaction type to 1D
        descriptor.src_plane_stride = descriptor.dst_plane_stride = 0;
        descriptor.cfg_link.cfg_bits.type = 0;
        break;
    case vpux::VPUIP::DMAAccMode::DISABLE:
        break;
    default:
        VPUX_THROW("{0} acceleration mode is not supported by DMA for NPU37XX arch", getAccelerationMode());
    }

    auto& barrierConsMask =
            descriptor.cfg_link.cfg_bits.type ? descriptor.barriers.cons_mask : descriptor.barriers1d.cons_mask;
    auto& barrierProdMask =
            descriptor.cfg_link.cfg_bits.type ? descriptor.barriers.prod_mask : descriptor.barriers1d.prod_mask;

    barrierConsMask = VPUMI37XX::computeMask(getWaitBarriers());
    barrierProdMask = VPUMI37XX::computeMask(getUpdateBarriers());

    uint8_t* ptrCharTmp = reinterpret_cast<uint8_t*>(&dmaTask);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::VPUMI37XX::NNDMAOp::getBinarySize() {
    return sizeof(nn_public::VpuDMATask);
}

size_t vpux::VPUMI37XX::NNDMAOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuDMATask);
}

size_t vpux::VPUMI37XX::NNDMAOp::getOffsetOfWithinOperation(mlir::Value val) {
    if (val == getInput()) {
        return offsetof(nn_public::VpuDMATask, transaction_) + offsetof(vpu_dma_descriptor_t, src);
    } else if (val == getOutputBuffs()[0]) {
        return offsetof(nn_public::VpuDMATask, transaction_) + offsetof(vpu_dma_descriptor_t, dst);
    } else if (val == getNextDMAIdx()) {
        return offsetof(nn_public::VpuDMATask, transaction_);
    }

    VPUX_THROW("Provided Value is not linked to the DMA Op or getOffset does not support it");
}

vpux::VPURT::BufferSection vpux::VPUMI37XX::NNDMAOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::NNDMAOp::getAccessingProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_EXECINSTR | ELFNPU37XX::SectionFlagsAttr::VPU_SHF_PROC_DMA);
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::NNDMAOp::getUserProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::VPU_SHF_PROC_DMA);
}
