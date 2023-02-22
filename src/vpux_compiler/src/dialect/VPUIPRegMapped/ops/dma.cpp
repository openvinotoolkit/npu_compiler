//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/ELF/utils.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/nn_public/vpu_nnrt_api.h"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/utils/core/mem_size.hpp"

#include "vpux/compiler/dialect/VPUIPRegMapped/utils.hpp"

using namespace vpux;

//
// NNDMAOp
//

// For further development, please refer to the ticket E#36225.

namespace {

void decode_storage_order(ShapeRef dims, StridesRef strides, unsigned char* order) {
    const unsigned int S = dims.size();

    for (unsigned int i = 0; i < S; ++i)
        order[i] = i;

    std::sort(&order[0], &order[0] + S, [&](int lhs, int rhs) {
        return std::make_tuple(strides[Dim(lhs)], dims[Dim(lhs)], lhs) <
               std::make_tuple(strides[Dim(rhs)], dims[Dim(rhs)], rhs);
    });
}

class SimplifiedTensorLayout {
public:
    explicit SimplifiedTensorLayout(mlir::Value value) {
        VPUX_THROW_UNLESS(value, "Encountered nullptr value");

        auto ndType = value.getType().cast<vpux::NDTypeInterface>();
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

        for (unsigned int dim = 0, level = 0; dim < dims; ++dim) {
            const unsigned int crt_size = sizes[Dim(order[dim])];
            unsigned int crt_stride = bit_strides(Dim(order[dim]));
            total_length_in_bits *= crt_size;

            if (previous_size * previous_stride < crt_stride) {
                if (sizes[Dim(order[dim])] == 1) {
                    if (dim + 1 == dims)
                        continue;

                    crt_stride = bit_strides(Dim(order[dim + 1]));
                }

                VPUX_THROW_UNLESS(level < SimplifiedTensorLayout::STRIDING_LEVELS, "Max striding levels exceeded");

                *rt_strides[level] = crt_stride;
                *rt_dims[level] = (previous_size * previous_stride) / (level ? *rt_strides[level - 1] : CHAR_BIT);
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
    unsigned int total_length_ = 0;
};

}  // namespace

void vpux::VPUIPRegMapped::NNDMAOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    nn_public::VpuDMATask dmaTask;

    // safe init to zero the structure
    memset(reinterpret_cast<void*>(&dmaTask), 0, sizeof(dmaTask));

    const auto hasDescriptor = dma_descriptor().hasValue();

    auto inputType = input().getType().cast<mlir::MemRefType>();

    dmaTask.barriers_sched_.start_after_ = checked_cast<uint16_t>(start_after());
    dmaTask.barriers_sched_.clean_after_ = checked_cast<uint16_t>(clean_after());

    auto& descriptor = dmaTask.transaction_;
    descriptor.cfg_link.cfg_bits.burst_length = 16;
    descriptor.cfg_link.cfg_bits.barrier_en = 1;

    // In case of multicasting (multiple outputs) we will mask the destination with the multicast mask;

    if (output_buffs().size() > 1)
        descriptor.dst = 0xC00000;

    // If this DMA Op is not used by any other Op this is the last Op in DMA chain and link_address shall be zero

    vpux::VPUIPRegMapped::NNDMAOp dmaUser = nullptr;

    for (auto user : getResult().getUsers()) {
        auto newDmaUser = mlir::dyn_cast<vpux::VPUIPRegMapped::NNDMAOp>(user);

        if (newDmaUser) {
            VPUX_THROW_UNLESS(!dmaUser, "VPUIPRegMapped::NNDMAOp '{0}' at loc '{1}' has more than one DMA user",
                              getOperation()->getName(), getLoc());

            dmaUser = newDmaUser;
        }
    }

    if (dmaUser) {
        auto dmaOpIndex = dmaUser.getResult().getType().cast<VPUIPRegMapped::IndexType>();
        descriptor.link_address = static_cast<uint64_t>(dmaOpIndex.getValue());
    } else {
        descriptor.link_address = 0;
    }

    descriptor.cfg_link.cfg_bits.critical = 1;
    descriptor.cfg_link.cfg_bits.order_forced = !is_out_of_order();
    descriptor.cfg_link.cfg_bits.skip_nr = 63;

    auto src_layout = SimplifiedTensorLayout(input());
    auto dst_layout = SimplifiedTensorLayout(output_buffs()[0]);

    uint32_t src_width = src_layout.line_length();
    uint32_t dst_width = dst_layout.line_length();
    uint32_t src_stride = src_layout.line_stride();
    uint32_t dst_stride = dst_layout.line_stride();
    uint32_t num_planes = src_layout.plane_count();
    uint32_t src_plane_stride = src_layout.plane_stride();
    uint32_t dst_plane_stride = dst_layout.plane_stride();
    uint32_t size = src_layout.total_length();

    if (!hasDescriptor && !compression()) {
        if (!!src_plane_stride ^ !!dst_plane_stride) {
            if (src_plane_stride)
                num_planes = std::max(1u, src_layout.plane_count()), dst_plane_stride = size / num_planes;
            else
                num_planes = std::max(1u, dst_layout.plane_count()), src_plane_stride = size / num_planes;
        }

        VPUX_THROW_UNLESS(num_planes > 0, "Encountered num planes = {0}", num_planes);

        if (src_width == src_stride)
            src_width = src_stride = size / num_planes;

        if (dst_width == dst_stride)
            dst_width = dst_stride = size / num_planes;
    }

    if (hasDescriptor) {
        const auto dmaDescriptor = dma_descriptor().getValue();
        descriptor.length = checked_cast<uint32_t>(dmaDescriptor.len().getInt());
        descriptor.attr2d.src_width = checked_cast<uint32_t>(dmaDescriptor.srcWidth().getInt());
        descriptor.attr2d.dst_width = checked_cast<uint32_t>(dmaDescriptor.dstWidth().getInt());
        descriptor.attr2d.src_stride = checked_cast<int32_t>(dmaDescriptor.srcStride().getInt());
        descriptor.attr2d.dst_stride = checked_cast<int32_t>(dmaDescriptor.dstStride().getInt());
        descriptor.src_plane_stride = checked_cast<int32_t>(dmaDescriptor.srcPlaneStride().getInt());
        descriptor.dst_plane_stride = checked_cast<int32_t>(dmaDescriptor.dstPlaneStride().getInt());
        descriptor.num_planes = checked_cast<uint32_t>(dmaDescriptor.numPlanes().getInt());
    } else {
        const auto elemSize = vpux::getElemTypeSize(inputType);
        const auto mult = vpux::Bit(vpux::MemMultiplier<vpux::MemType::Byte, vpux::MemType::Bit>::value);
        auto totalSizeBits = inputType.getNumElements() * elemSize;
        if (totalSizeBits % mult) {
            vpux::Logger::global().info("SubByte shape not aligned to byte. Increasing copy size to byte alignment, "
                                        "assuming memory allocations are at byte level: ElemSize {0} Shape: {1}",
                                        elemSize, inputType.getShape());

            totalSizeBits += vpux::Bit(mult.count() - (totalSizeBits % mult));
        }
        descriptor.length = vpux::Byte(totalSizeBits).count();
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

    if (compression()) {
        descriptor.cfg_link.cfg_bits.dec_en = 1;
        VPUX_THROW_UNLESS(descriptor.num_planes == 0,
                          "For DMA compression to be possible, the computed num_planes for the transaction needs to be "
                          "0, got {0}",
                          checked_cast<uint8_t>(descriptor.num_planes));

        // Ensure plane strides are set to 0 and set transaction type to 1D
        descriptor.src_plane_stride = descriptor.dst_plane_stride = 0;
        descriptor.cfg_link.cfg_bits.type = 0;
    }

    auto& barrierConsMask =
            descriptor.cfg_link.cfg_bits.type ? descriptor.barriers.cons_mask : descriptor.barriers1d.cons_mask;
    auto& barrierProdMask =
            descriptor.cfg_link.cfg_bits.type ? descriptor.barriers.prod_mask : descriptor.barriers1d.prod_mask;

    barrierConsMask = VPUIPRegMapped::computeMask(waitBarriers());
    barrierProdMask = VPUIPRegMapped::computeMask(updateBarriers());

    uint8_t* ptrCharTmp = reinterpret_cast<uint8_t*>(&dmaTask);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::VPUIPRegMapped::NNDMAOp::getBinarySize() {
    return sizeof(nn_public::VpuDMATask);
}

size_t vpux::VPUIPRegMapped::NNDMAOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuDMATask);
}

mlir::FailureOr<uint64_t> vpux::VPUIPRegMapped::NNDMAOp::getOffsetOfWithinOperation(mlir::Value val) {
    if (val == input()) {
        return offsetof(nn_public::VpuDMATask, transaction_) + offsetof(vpu_dma_descriptor_t, src);
    } else if (val == output_buffs()[0]) {
        return offsetof(nn_public::VpuDMATask, transaction_) + offsetof(vpu_dma_descriptor_t, dst);
    } else if (val == previousDMAIdx()) {
        return offsetof(nn_public::VpuDMATask, transaction_);
    }

    return mlir::failure();
}

vpux::VPURT::BufferSection vpux::VPUIPRegMapped::NNDMAOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}

vpux::ELF::SectionFlagsAttr vpux::VPUIPRegMapped::NNDMAOp::getAccessingProcs() {
    return (ELF::SectionFlagsAttr::SHF_EXECINSTR | ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA);
}

vpux::ELF::SectionFlagsAttr vpux::VPUIPRegMapped::NNDMAOp::getUserProcs() {
    return (ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA);
}
