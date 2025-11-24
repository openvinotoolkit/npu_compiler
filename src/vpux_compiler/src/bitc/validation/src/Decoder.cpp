//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "Decoder.hpp"
#include <cassert>
#include <stdexcept>
#include "commons.hpp"

using namespace vpux::bitc;

class Decoder::Impl {
public:
    Impl(const std::vector<uint8_t>& bits, BitCompactorConfig config);
    Impl(const std::vector<uint8_t>&& bits, BitCompactorConfig config);
    void decode(std::vector<uint8_t>& out);
    void decode(std::vector<uint8_t>& out, std::vector<uint8_t>& bitmap, unsigned sparse_block_size);

private:
    typedef void (Decoder::Impl::*AlgorithmDecoder)();
    void verify_config();
    void unpack_sparse_data(std::vector<uint8_t>& src, std::vector<uint8_t>& dst, std::vector<uint8_t>& bitmap,
                            unsigned sparse_block_size);
    void fp16deprdct(unsigned char* dst);
    void decode_noprdct();
    void decode_signshftproc();
    void decode_signshftaddproc();
    void decode_addproc();
    void decode_binexpproc();
    void decode_lftshftproc();
    void decode_btexpproc();
    void decode_signshftaddblk();
    void init();
    void read_residual();
    void signed_residual();
    std::vector<AlgorithmDecoder> algorithm_decoder_;
    void eofr_hp(const BitcHeader& header);
    void cmprsd_hp(const BitcHeader& header);
    void uncmprsd_hp(const BitcHeader& header);
    void lastblk_hp(const BitcHeader& header);
    BitCompactorConfig config_;
    BitStream bit_stream_;
    uint32_t stream_bit_offset_{};
    int32_t bit_length_{};
    uint32_t block_size_{};
    DualEncoder dual_encoder_{};
    std::vector<uint8_t> block_residual_;
    uint8_t block_offset_{};
    std::vector<uint8_t> output_stream_;
    bool signed_residual_{false};
    bool prev_block_{false};
};

Decoder::Decoder(const std::vector<uint8_t>& bits, BitCompactorConfig config) {
    impl_ = std::make_unique<Impl>(bits, config);
}

Decoder::Decoder(const std::vector<uint8_t>&& bits, BitCompactorConfig config) {
    impl_ = std::make_unique<Impl>(bits, config);
}

void Decoder::decode(std::vector<uint8_t>& out) {
    impl_->decode(out);
}

void Decoder::decode(std::vector<uint8_t>& out, std::vector<uint8_t>& bitmap, unsigned sparse_block_size) {
    impl_->decode(out, bitmap, sparse_block_size);
}

Decoder::~Decoder() {
}

Decoder::Impl::Impl(const std::vector<uint8_t>& bits, BitCompactorConfig config): bit_stream_{bits}, config_{config} {
    init();
}

Decoder::Impl::Impl(const std::vector<uint8_t>&& bits, BitCompactorConfig config): bit_stream_{bits}, config_{config} {
    init();
}

void Decoder::Impl::verify_config() {
    if (config_.arch_type == ArchType::NPU27) {
        if (config_.mode_fp16_enable) {
            throw std::logic_error{"FP16 is not supported when using NPU27"};
        }
        if (!config_.weight_compress_enable) {
            throw std::logic_error{"NPU27 doesn't support activation compression"};
        }
    }
}

void Decoder::Impl::init() {
    verify_config();

    algorithm_decoder_.resize(static_cast<uint32_t>(DecoderAlgorithm::ALGO_COUNT));

    algorithm_decoder_[static_cast<uint32_t>(DecoderAlgorithm::NOPROC)] = &Decoder::Impl::decode_noprdct;
    algorithm_decoder_[static_cast<uint32_t>(DecoderAlgorithm::SIGNSHFTPROC)] = &Decoder::Impl::decode_signshftproc;
    algorithm_decoder_[static_cast<uint32_t>(DecoderAlgorithm::SIGNSHFTADDPROC)] =
            &Decoder::Impl::decode_signshftaddproc;
    algorithm_decoder_[static_cast<uint32_t>(DecoderAlgorithm::ADDPROC)] = &Decoder::Impl::decode_addproc;
    algorithm_decoder_[static_cast<uint32_t>(DecoderAlgorithm::SIGNSHFTADDBLK)] =
            &Decoder::Impl::decode_signshftaddblk;  // NPU4/5 only
    algorithm_decoder_[static_cast<uint32_t>(DecoderAlgorithm::BINEXPPROC)] = &Decoder::Impl::decode_binexpproc;
    algorithm_decoder_[static_cast<uint32_t>(DecoderAlgorithm::LFTSHFTPROC)] = &Decoder::Impl::decode_lftshftproc;
    algorithm_decoder_[static_cast<uint32_t>(DecoderAlgorithm::BTEXPPROC)] = &Decoder::Impl::decode_btexpproc;
    block_residual_.resize(BLOCK_SIZE);
}

// No Predict, just look at the maximum in the array
void Decoder::Impl::decode_noprdct() {
}

void Decoder::Impl::decode_signshftproc() {
    signed_residual_ = true;
}

void Decoder::Impl::decode_signshftaddproc() {
    signed_residual_ = true;
    decode_addproc();
}

void Decoder::Impl::signed_residual() {
    if (signed_residual_) {
        for (auto& residual : block_residual_) {
            if ((residual & 0x1) == 0x1) {
                residual = (~residual >> 1) | 0x80;
            } else {
                residual >>= 1;
            }
        }
    }
}

void Decoder::Impl::decode_addproc() {
    uint64_t bits{};
    bit_stream_.read(stream_bit_offset_, bits);
    block_offset_ = static_cast<uint8_t>(bits & 0xffull);
    stream_bit_offset_ += 8u;
}

void Decoder::Impl::decode_binexpproc() {
}

void Decoder::Impl::decode_lftshftproc() {
}

void Decoder::Impl::decode_btexpproc() {
}

void Decoder::Impl::decode_signshftaddblk() {
    signed_residual_ = true;
    prev_block_ = true;
}

void Decoder::Impl::eofr_hp(const BitcHeader& header) {
    block_size_ = 0u;
}

void Decoder::Impl::read_residual() {
    std::fill(std::begin(block_residual_), std::end(block_residual_), 0);
    auto residual_iter{std::begin(block_residual_)};

    uint64_t bits{};
    int32_t bit_count{64};
    const uint64_t symbol_mask{(1ull << bit_length_) - 1ull};
    bit_stream_.read(stream_bit_offset_, bits);
    stream_bit_offset_ += 64;
    auto symbol_bitmap{dual_encoder_.bitmap};

    for (uint32_t i{}; i < block_size_; ++i) {
        auto bitlen{bit_length_};
        auto mask{symbol_mask};

        if (symbol_bitmap & 0x1ull) {
            bitlen = 8;
            mask = 0xffull;
        }

        symbol_bitmap >>= 1;

        if (bitlen > bit_count) {
            stream_bit_offset_ -= bit_count;
            bit_count = 64;
            bit_stream_.read(stream_bit_offset_, bits);
            stream_bit_offset_ += bit_count;
        }

        *residual_iter++ = static_cast<uint8_t>(bits & mask);
        bits >>= bitlen;
        bit_count -= bitlen;
    }

    stream_bit_offset_ -= bit_count;
    if (config_.arch_type != ArchType::NPU27) {
        stream_bit_offset_ = BYTE_ALIGN(stream_bit_offset_);
    }
    signed_residual();

    if (prev_block_) {
        size_t prev_block_idx = output_stream_.size() - BLOCK_SIZE;
        for (unsigned int idx = 0; idx < BLOCK_SIZE; ++idx) {
            block_residual_[idx] += output_stream_[prev_block_idx + idx];
        }
    }

    for (auto& residual : block_residual_) {
        residual += block_offset_;
        output_stream_.push_back(residual);
    }
}

void Decoder::Impl::lastblk_hp(const BitcHeader& header) {
    const auto& last_block_header{*reinterpret_cast<const LastBlockHeader*>(&header)};
    block_size_ = static_cast<uint32_t>(last_block_header.block_length);
    signed_residual_ = false;
    prev_block_ = false;
    bit_length_ = 8;
    dual_encoder_.bitmap = 0ull;
    stream_bit_offset_ += 8u;
    read_residual();
}

void Decoder::Impl::uncmprsd_hp(const BitcHeader& header) {
    dual_encoder_.bitmap = 0ull;
    signed_residual_ = false;
    prev_block_ = false;
    bit_length_ = 8u;
    block_offset_ = 0u;
    block_size_ = BLOCK_SIZE;
    stream_bit_offset_ += 2u;
    // Byte alignment
    if (config_.arch_type != ArchType::NPU27) {
        stream_bit_offset_ += 6u;
    }
    read_residual();
}

void Decoder::Impl::cmprsd_hp(const BitcHeader& header) {
    const auto& cmp_header{*reinterpret_cast<const CompressionHeader*>(&header)};
    bit_length_ = static_cast<int32_t>(cmp_header.bit_length);
    block_size_ = BLOCK_SIZE;

    if (!bit_length_) {
        bit_length_ = 8;
    }

    stream_bit_offset_ += 10u;
    dual_encoder_.bitmap = 0ull;

    if (cmp_header.dual_encode) {
        stream_bit_offset_ += 10u;
        dual_encoder_.length = static_cast<uint32_t>(cmp_header.dual_encode_length);
        // Byte alignment
        if (config_.arch_type != ArchType::NPU27) {
            stream_bit_offset_ += 4;
        }
    }
    // Byte alignment
    else if (config_.arch_type != ArchType::NPU27) {
        stream_bit_offset_ += 6u;
    }

    signed_residual_ = false;
    prev_block_ = false;

    (this->*algorithm_decoder_[cmp_header.algo])();

    if (cmp_header.dual_encode) {
        bit_stream_.read(stream_bit_offset_, dual_encoder_.bitmap);
        stream_bit_offset_ += BLOCK_SIZE;
    }

    read_residual();
}

void Decoder::Impl::unpack_sparse_data(std::vector<uint8_t>& src, std::vector<uint8_t>& dst,
                                       std::vector<uint8_t>& bitmap, unsigned sparse_block_size) {
    unsigned int sparseBlockSize = sparse_block_size;
    unsigned int bitmapSize = bitmap.size();
    unsigned int srcIdx = 0;
    unsigned int typeBytes = (config_.mode_fp16_enable) ? sizeof(uint16_t) : sizeof(uint8_t);
    unsigned int btmBlockSize = sparseBlockSize / (8 * typeBytes);
    unsigned int btmBlockStart = 0;
    unsigned int dstBlockStart = 0;

    if ((sparseBlockSize != 0) && ((sparseBlockSize % 16) == 0)) {
        srcIdx = 0;
        while (srcIdx < src.size()) {
            // Copy current bitmap block
            if (((srcIdx + btmBlockSize) > src.size()) || ((btmBlockStart + btmBlockSize) > bitmapSize)) {
                throw std::logic_error{"UnpackSparseData: Bitmap handling failed"};
            }
            memcpy(&bitmap[btmBlockStart], &src[srcIdx], btmBlockSize);
            btmBlockStart += btmBlockSize;
            srcIdx += btmBlockSize;

            // Read sparsity header and retrieve the number of valid bytes
            unsigned int validBytes;
            if (src[srcIdx] & 0x01) {
                // 2-byte header, stores number of valid elements as {14'h<elemCount>,
                // 2'b01} Number of valid elements is in the range [128..8192]
                unsigned int header = static_cast<unsigned int>(src[srcIdx] + (src[srcIdx + 1] << 8));
                validBytes = (header >> 2) * typeBytes;
                srcIdx += 2;
            } else {
                // 1-byte header, stores number of valid elements as {7'b<elemCount>,
                // 1'b0} Number of valid elements is in the range [0..127]
                unsigned int header = static_cast<unsigned int>(src[srcIdx]);
                validBytes = (header >> 1) * typeBytes;
                srcIdx++;
            }

            // Copy current valid data block
            if ((srcIdx + validBytes) > src.size()) {
                throw std::logic_error{"UnpackSparseData: Data handling failed"};
            }
            std::fill(&dst[dstBlockStart], &dst[dstBlockStart + sparseBlockSize], 0x00);
            memcpy(&dst[dstBlockStart], &src[srcIdx], validBytes);
            dstBlockStart += sparseBlockSize;
            srcIdx += validBytes;
        }

        // Set buffer sizes
        dst.resize(dstBlockStart);
        bitmapSize = btmBlockStart;

        // Check resulting bitmap size against programmed size
        if (bitmap.size() != bitmapSize) {
            throw std::logic_error{"UnpackSparseData: Bitmap size mismatch."};
        }
    } else {
        throw std::logic_error{"UnpackSparseData: Invalid configuration"};
    }
}

void Decoder::Impl::decode(std::vector<uint8_t>& out, std::vector<uint8_t>& bitmap, unsigned sparse_block_size) {
    decode(out);
    std::vector<uint8_t> unpacked(2 * out.size());
    unpack_sparse_data(out, unpacked, bitmap, sparse_block_size);
    out = unpacked;
}

void Decoder::Impl::decode(std::vector<uint8_t>& out) {
    if (config_.bypass_compression) {
        out.resize(bit_stream_.source_stream_length());
        memcpy(out.data(), bit_stream_.get_byte_pointer(0), bit_stream_.source_stream_length());
        return;
    }

    uint64_t bits{};
    stream_bit_offset_ = 0u;
    output_stream_ = std::move(out);
    output_stream_.reserve(bit_stream_.stream_length() << 3);
    bool end_of_stream{false};
    uint32_t decompression_length{};

    while (!end_of_stream) {
        block_offset_ = 0;
        bit_stream_.read(stream_bit_offset_, bits);
        BitcHeader& header{*reinterpret_cast<BitcHeader*>(&bits)};

        switch (static_cast<CompressionType>(header.compression_type)) {
        case CompressionType::eofr:
            eofr_hp(header);
            end_of_stream = true;
            break;
        case CompressionType::cmprsd:
            cmprsd_hp(header);
            break;
        case CompressionType::lastblk:
            lastblk_hp(header);
            break;
        case CompressionType::uncmprsd:
            uncmprsd_hp(header);
            break;
        default:
            assert(false);
            break;
        }

        decompression_length += block_size_;
    }
    output_stream_.resize(decompression_length);

    if (config_.mode_fp16_enable) {
        unsigned char* dst = output_stream_.data();
        for (size_t i = 0; i <= (output_stream_.size() - (2 * BLOCK_SIZE)); i += (2 * BLOCK_SIZE)) {
            fp16deprdct((dst + i));
        }
    }

    out = std::move(output_stream_);
}

void Decoder::Impl::fp16deprdct(unsigned char* dst) {
    unsigned int twoXblkSize = 2 * BLOCK_SIZE;
    unsigned char lLSB, uLSB;
    unsigned char srcBackup[BLOCK_SIZE * 2];

    for (unsigned int i = 0; i < twoXblkSize; i++) {
        srcBackup[i] = *(dst + i);
    }
    for (unsigned int i = 0; i < twoXblkSize; i += 2) {
        lLSB = (*(srcBackup + (i / 2)) & 0x01);
        uLSB = (*(srcBackup + BLOCK_SIZE + (i / 2)) & 0x01);
        *(dst + i) = (*(srcBackup + (i / 2)) >> 1) | (uLSB << 7);
        *(dst + i + 1) = (*(srcBackup + BLOCK_SIZE + (i / 2)) >> 1) | (lLSB << 7);
    }
}
