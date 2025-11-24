//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/bitc/Encoder.hpp"

#include "commons.hpp"

#include <algorithm>
#include <cassert>
#include <memory>
#include <stdexcept>
using namespace vpux::bitc;

class Encoder::Impl {
public:
    Impl();
    void encode(const BitCompactorConfig& config, const std::vector<uint8_t>& in, std::vector<uint8_t>& out);

private:
    typedef void (Encoder::Impl::*AlgorithmEncoder)(AlgorithmParam& param);
    void verify_config(const BitCompactorConfig& config);
    void fp16enprdct(const unsigned char* src, const unsigned int srcLen, unsigned char* dst, const unsigned BLKSIZE);

    void binexpproc(AlgorithmParam& param);
    void btexpproc(AlgorithmParam& param);
    void muprdct(AlgorithmParam& param);
    void minprdct(AlgorithmParam& param);
    void minsprdct(AlgorithmParam& param);
    void medprdct(AlgorithmParam& param);
    void noprdct(AlgorithmParam& param);
    void nosprdct(AlgorithmParam& param);
    void dummyprdct(AlgorithmParam& param);
    void prevblkprdct(AlgorithmParam& param);
    void dual_encode(AlgorithmParam& param);
    void write_residual(const BitCompactorConfig& config, BitStream& stream, const AlgorithmParam& param);
    uint32_t to_unsigned(int8_t* p_data, uint8_t* p_data_out, const uint32_t& block_size);
    void init(const BitCompactorConfig& config, const std::vector<uint8_t>& in);
    void fp16_preprocess(uint32_t input_bytes);
    void pack_sparse_data(const BitCompactorConfig& config);
    void init_block_param(AlgorithmParam& block_param, uint8_t* p_input_block, uint32_t algo);
    void add_overhead_and_padding(const BitCompactorConfig& config, AlgorithmParam& block_param, uint32_t algo);
    bool is_algo_dual_encode_compatible(const BitCompactorConfig& config, uint32_t algo);
    bool is_algo_valid(uint32_t algo, int blk);
    uint64_t get_dual_encode_len(const BitCompactorConfig& config, AlgorithmParam& block_param);
    BestAlgorithm get_best_algorithm(const BitCompactorConfig& config, int blk,
                                     std::vector<AlgorithmParam>& block_params);
    CompressionHeader create_compression_header(const BitCompactorConfig& config, CompressionType type,
                                                AlgorithmParam& block_param, uint32_t dual_encode = 0);
    int write_dual_encode_header(const BitCompactorConfig& config, int blk, BitStream& blk_stream,
                                 std::vector<AlgorithmParam>& block_params, BestAlgorithm& best_algorithm);
    int write_compressed_header(const BitCompactorConfig& config, int blk, BitStream& blk_stream,
                                std::vector<AlgorithmParam>& block_params, BestAlgorithm& best_algorithm);
    void write_compressed_blk(const BitCompactorConfig& config, int blk, const bool using_dual_encoder,
                              std::vector<BitStream>& block_stream, std::vector<AlgorithmParam>& block_params,
                              BestAlgorithm& best_algorithm);
    void write_uncompressed_blk(const BitCompactorConfig& config, int blk, std::vector<BitStream>& block_stream,
                                std::vector<AlgorithmParam>& block_params);
    void write_last_blk(uint32_t input_blocks, std::vector<BitStream>& block_stream, unsigned last_block_elements);
    void write_to_output(std::vector<uint8_t>& out, std::vector<BitStream>& block_stream);
    bool is_compression_better_than_uncompressed(BestAlgorithm& best_algorithm);
    bool using_dual_encoder(BestAlgorithm& best_algorithm);

    BitStream bit_stream_in_;
    BitStream bit_stream_out_;
    uint32_t stream_bit_offset_in_{};
    uint32_t stream_bit_offset_out_{};
    std::vector<AlgorithmEncoder> algorithm_encoder_;
    std::vector<uint8_t> log2_lut_;
    std::vector<uint8_t> dual_log2_lut_;
    static const std::map<uint32_t, DecoderAlgorithm> encoder_to_decoder_mapping_;
    const std::array<uint32_t, static_cast<uint32_t>(EncoderAlgorithm::ALGO_COUNT)> algorithm_overhead_bits_{
            18u, 18u, 18u, 10u, 10u, 18u, 10u, 14u, 90u};
    uint32_t output_byte_alignment_{32u};

    static const uint8_t DUAL_CMPRS_PAD{4u};
    static const uint8_t CMPRS_PAD{6u};

    static const uint16_t dual_encoder_enable_{1u};

    uint32_t ALGORITHMS;
    uint32_t MAX_BLOCK_COMPRESSION_BITS;
};

Encoder::Encoder() {
    impl_ = std::make_unique<Impl>();
}

void Encoder::encode(const BitCompactorConfig& config, const std::vector<uint8_t>& in, std::vector<uint8_t>& out) {
    impl_->encode(config, in, out);
}

Encoder::~Encoder() {
}

Encoder::Impl::Impl() {
    init(BitCompactorConfig{}, std::vector<uint8_t>());
}

const std::map<uint32_t, DecoderAlgorithm> Encoder::Impl::encoder_to_decoder_mapping_{
        {0, DecoderAlgorithm::ADDPROC},         {1, DecoderAlgorithm::SIGNSHFTADDPROC},
        {2, DecoderAlgorithm::SIGNSHFTADDPROC}, {3, DecoderAlgorithm::NOPROC},
        {4, DecoderAlgorithm::SIGNSHFTPROC},    {5, DecoderAlgorithm::SIGNSHFTADDPROC},
        {6, DecoderAlgorithm::SIGNSHFTADDBLK},  // >= NPU4 only
        {7, DecoderAlgorithm::BINEXPPROC},      {8, DecoderAlgorithm::BTEXPPROC}};

void Encoder::Impl::verify_config(const BitCompactorConfig& config) {
    if (config.arch_type == ArchType::NPU27) {
        if (config.mode_fp16_enable) {
            throw std::logic_error{"FP16 is not supported when using NPU27"};
        }
        if (!config.weight_compress_enable) {
            throw std::logic_error{"NPU27 doesn't support activation compression"};
        }
    }
}

void Encoder::Impl::init(const BitCompactorConfig& config, const std::vector<uint8_t>& in) {
    bit_stream_in_ = BitStream{in};

    verify_config(config);

    algorithm_encoder_.resize(static_cast<uint32_t>(EncoderAlgorithm::ALGO_COUNT));
    algorithm_encoder_[static_cast<uint32_t>(EncoderAlgorithm::MINPRDCT)] = &Encoder::Impl::minprdct;

    if (config.weight_compress_enable) {
        algorithm_encoder_[static_cast<uint32_t>(EncoderAlgorithm::MINSPRDCT)] = &Encoder::Impl::minsprdct;
    } else {
        algorithm_encoder_[static_cast<uint32_t>(EncoderAlgorithm::MINSPRDCT)] = &Encoder::Impl::dummyprdct;
    }

    algorithm_encoder_[static_cast<uint32_t>(EncoderAlgorithm::MUPRDCT)] = &Encoder::Impl::muprdct;
    algorithm_encoder_[static_cast<uint32_t>(EncoderAlgorithm::MEDPRDCT)] = &Encoder::Impl::medprdct;
    algorithm_encoder_[static_cast<uint32_t>(EncoderAlgorithm::NOPRDCT)] = &Encoder::Impl::noprdct;
    algorithm_encoder_[static_cast<uint32_t>(EncoderAlgorithm::NOSPRDCT)] = &Encoder::Impl::nosprdct;
    algorithm_encoder_[static_cast<uint32_t>(EncoderAlgorithm::PREVBLKPRDCT)] =
            &Encoder::Impl::prevblkprdct;  // >= NPU4 only
    algorithm_encoder_[static_cast<uint32_t>(EncoderAlgorithm::BINCMPCT)] = &Encoder::Impl::binexpproc;
    algorithm_encoder_[static_cast<uint32_t>(EncoderAlgorithm::BTMAP)] = &Encoder::Impl::btexpproc;

    ALGORITHMS = static_cast<uint32_t>(algorithm_encoder_.size()) - (config.arch_type == ArchType::NPU27 ? 3u : 2u);
    MAX_BLOCK_COMPRESSION_BITS = (BLOCK_SIZE << 3) + (config.arch_type == ArchType::NPU27 ? 2u : 8u);

    log2_lut_.resize(256);
    dual_log2_lut_.resize(256);

    uint32_t bits{2u};
    uint32_t dual_bits{config.arch_type == ArchType::NPU27 ? 0u : 1u};

    uint32_t bit_mask{(1u << (bits + 1)) - 1u};
    uint32_t dual_bit_mask{(1u << (dual_bits + 1)) - 1u};

    for (uint32_t lbyte{}; lbyte < 256; ++lbyte) {
        if (lbyte > bit_mask) {
            bits++;
            bit_mask = (1u << (bits + 1)) - 1u;
        }

        log2_lut_[lbyte] = bits + 1;

        if (lbyte > dual_bit_mask) {
            dual_bits++;
            dual_bit_mask = (1u << (dual_bits + 1)) - 1u;
        }

        dual_log2_lut_[lbyte] = dual_bits + 1;
    }
}

void Encoder::Impl::fp16enprdct(const unsigned char* src, const unsigned int srcLen, unsigned char* dst,
                                const unsigned BLKSIZE) {
    unsigned int twoXblkSize = 2 * BLKSIZE;
    unsigned int idx = 0;
    unsigned char lMSB, uMSB;

    while (idx < srcLen) {
        if ((idx + twoXblkSize) <= srcLen) {
            for (unsigned int i = 0; i < twoXblkSize; i += 2) {
                lMSB = (*(src + idx + i) & 0x80) >> 7;
                uMSB = (*(src + idx + i + 1) & 0x80) >> 7;
                *(dst + idx + (i / 2)) = (*(src + idx + i) << 1) | uMSB;
                *(dst + idx + BLKSIZE + (i / 2)) = (*(src + idx + i + 1) << 1) | lMSB;
            }
            idx += twoXblkSize;
        } else {
            while (idx < srcLen) {
                *(dst + idx) = *(src + idx);
                idx++;
            }
        }
    }
}

// No Preprocess
// No Predict, just look at the maximum in the array
void Encoder::Impl::noprdct(AlgorithmParam& param) {
    param.bit_length = 1u;

    for (uint32_t i{}; i < param.block_size; ++i) {
        param.residual[i] = param.p_data[i];

        const auto bits{log2_lut_[param.residual[i]]};

        if (bits > param.bit_length) {
            param.bit_length = bits;
        }
    }

    param.encoding_bits = param.bit_length * param.block_size;
}

// No Sign Preprocess
void Encoder::Impl::nosprdct(AlgorithmParam& param) {
    param.bit_length = to_unsigned(reinterpret_cast<int8_t*>(param.p_data), param.residual, param.block_size);

    param.encoding_bits = param.bit_length * param.block_size;
}

void Encoder::Impl::medprdct(AlgorithmParam& param) {
    std::vector<uint8_t> data_sorted(param.block_size);

    std::memcpy(reinterpret_cast<void*>(data_sorted.data()), reinterpret_cast<const void*>(param.p_data),
                param.block_size);

    std::sort(std::begin(data_sorted), std::end(data_sorted));

    const auto med_indx{(param.block_size >> 1) - 1u};

    param.minimum = data_sorted[med_indx];

    auto p_data{reinterpret_cast<int8_t*>(param.p_data)};
    auto p_residual{reinterpret_cast<int8_t*>(param.residual)};
    auto& median{*reinterpret_cast<int8_t*>(&param.minimum)};

    for (uint32_t i{}; i < param.block_size; ++i) {
        p_residual[i] = p_data[i] - median;
    }

    param.bit_length = to_unsigned(p_residual, param.residual, param.block_size);
    param.encoding_bits = param.bit_length * param.block_size;
}

// Do Min Signed Predict algo on a buffer, return minimum number and the bitln
void Encoder::Impl::minsprdct(AlgorithmParam& param) {
    auto p_data{reinterpret_cast<int8_t*>(param.p_data)};
    auto p_residual{reinterpret_cast<int8_t*>(param.residual)};
    auto& minimum{*reinterpret_cast<int8_t*>(&param.minimum)};

    minimum = 127;

    for (uint32_t i{}; i < param.block_size; ++i) {
        if (p_data[i] < minimum) {
            minimum = p_data[i];
        }
    }

    for (uint32_t i{}; i < param.block_size; ++i) {
        p_residual[i] = p_data[i] - minimum;
    }

    param.bit_length = to_unsigned(p_residual, param.residual, param.block_size);
    param.encoding_bits = param.bit_length * param.block_size;
}

uint32_t Encoder::Impl::to_unsigned(int8_t* p_data, uint8_t* p_data_out, const uint32_t& block_size) {
    uint8_t bit_length{1};

    for (uint32_t i{}; i < block_size; ++i) {
        if (p_data[i] < 0) {
            p_data_out[i] = (static_cast<uint8_t>(~p_data[i] << 1) | 0x1);
        } else {
            p_data_out[i] = static_cast<uint8_t>(p_data[i] << 1);
        }

        const auto bits{log2_lut_[static_cast<uint8_t>(p_data_out[i])]};
        if (bits > bit_length) {
            bit_length = bits;
        }
    }

    return static_cast<uint32_t>(bit_length);
}

// Mean Preprocess
// Do Mean Signed Predict algo on a buffer, return minimum number and the bitln.
void Encoder::Impl::muprdct(AlgorithmParam& param) {
    auto p_data{reinterpret_cast<int8_t*>(param.p_data)};
    auto p_residual{reinterpret_cast<int8_t*>(param.residual)};
    int32_t mean{};

    for (uint32_t i{}; i < param.block_size; ++i) {
        mean += p_data[i];
    }

    double d_mean{static_cast<double>(mean) / static_cast<double>(param.block_size)};
    mean = static_cast<int32_t>(std::round(d_mean));

    auto& mu{*reinterpret_cast<int8_t*>(&param.minimum)};
    mu = static_cast<int8_t>(mean);

    for (uint32_t i{}; i < param.block_size; ++i) {
        p_residual[i] = p_data[i] - mu;
    }

    param.bit_length = to_unsigned(p_residual, param.residual, param.block_size);
    param.encoding_bits = param.bit_length * param.block_size;
}

// Min Preprocess
// Do Min Predict algo on a buffer, return minimum number and the bitln.
void Encoder::Impl::minprdct(AlgorithmParam& param) {
    param.minimum = 255;

    for (uint32_t i{}; i < param.block_size; ++i) {
        if (param.p_data[i] < param.minimum) {
            param.minimum = param.p_data[i];
        }
    }

    param.bit_length = 1u;

    for (uint32_t i{}; i < param.block_size; ++i) {
        param.residual[i] = param.p_data[i] - param.minimum;
        const auto residual_bits{log2_lut_[param.residual[i]]};

        if (residual_bits > param.bit_length) {
            param.bit_length = residual_bits;
        }
    }

    param.encoding_bits = param.block_size * param.bit_length;
}

void Encoder::Impl::dummyprdct(AlgorithmParam& param) {
    param.bit_length = 8u;

    std::memcpy(reinterpret_cast<void*>(param.residual), reinterpret_cast<const void*>(param.p_data), param.block_size);

    param.encoding_bits = param.block_size * param.bit_length;
}
void Encoder::Impl::prevblkprdct(AlgorithmParam& param) {
    const uint8_t* prev_blk = param.p_data - param.block_size;

    for (uint32_t i{}; i < param.block_size; ++i) {
        param.residual[i] = param.p_data[i] - prev_blk[i];
    }
    param.bit_length = to_unsigned(reinterpret_cast<int8_t*>(param.residual), param.residual, param.block_size);
    param.encoding_bits = param.bit_length * param.block_size;
}
void Encoder::Impl::binexpproc(AlgorithmParam& param) {
    dummyprdct(param);
}

void Encoder::Impl::btexpproc(AlgorithmParam& param) {
    dummyprdct(param);
}

void Encoder::Impl::dual_encode(AlgorithmParam& param) {
    std::vector<uint32_t> bits_histogram(9);
    std::vector<uint8_t> symbol_bits(param.block_size);

    for (uint32_t i{}; i < param.block_size; ++i) {
        symbol_bits[i] = dual_log2_lut_[param.residual[i]];
        bits_histogram[symbol_bits[i]]++;
    }

    // For each of the bins, calculate the compressed Size.
    // And find the bitln that results in the minimum compressed Size.
    uint32_t cumSumL{};
    uint32_t cumSumH{};
    std::vector<uint32_t> cSize(9);
    param.dual_encoding_bits = ~0u;

    for (int i = 1; i < 9; ++i) {
        cumSumL = 0u;
        for (int j = 1; j <= i; ++j) {
            cumSumL += bits_histogram[j];
        }
        cumSumH = 0;
        for (int j = i + 1; j < 9; ++j) {
            cumSumH += bits_histogram[j];
        }
        cSize[i] = cumSumL * i + cumSumH * 8;
        // Find the minimum compressed Size.
        if (i == 1) {
            param.dual_encoding_bits = cSize[i];
            param.dual_bit_length = i;
        } else {
            if (cSize[i] < param.dual_encoding_bits) {
                param.dual_encoding_bits = cSize[i];
                param.dual_bit_length = i;
            }
        }
    }

    param.dual_bitmap = 0ull;
    bool using_8bits{false};

    for (uint32_t i{}; i < param.block_size; ++i) {
        const auto use_8bits{symbol_bits[i] > param.dual_bit_length};

        param.dual_bitmap |= (static_cast<uint64_t>(use_8bits)) << i;

        using_8bits |= use_8bits;
    }

    if (!using_8bits) {
        param.dual_encoding_bits += (8u - param.dual_bit_length);
        param.dual_bitmap |= 0x1ull;
    }
}

void Encoder::Impl::fp16_preprocess(uint32_t input_bytes) {
    std::vector<uint8_t> processed_bits(input_bytes);
    fp16enprdct(bit_stream_in_.get_byte_pointer(0), input_bytes, processed_bits.data(), BLOCK_SIZE);
    bit_stream_in_ = BitStream{processed_bits};
}

void Encoder::Impl::init_block_param(AlgorithmParam& block_param, uint8_t* p_input_block, uint32_t algo) {
    block_param.p_data = p_input_block;
    block_param.block_size = BLOCK_SIZE;
    block_param.decoder = encoder_to_decoder_mapping_.at(algo);
    block_param.encoder_index = algo;
    block_param.dual_encoder_enable = dual_encoder_enable_;
}

CompressionHeader Encoder::Impl::create_compression_header(const BitCompactorConfig& config, CompressionType type,
                                                           AlgorithmParam& block_param, uint32_t dual_encode) {
    CompressionHeader header{};
    header.compression_type = static_cast<uint32_t>(type);
    header.algo = static_cast<uint32_t>(block_param.decoder);
    header.bit_length = block_param.bit_length;
    if (dual_encode == 0) {
        return header;
    }

    header.dual_encode = 1;
    header.dual_encode_length = get_dual_encode_len(config, block_param);

    return header;
}

void Encoder::Impl::add_overhead_and_padding(const BitCompactorConfig& config, AlgorithmParam& block_param,
                                             uint32_t algo) {
    block_param.encoding_bits += algorithm_overhead_bits_[algo];
    block_param.dual_encoding_bits += algorithm_overhead_bits_[algo];
    block_param.dual_encoding_bits += 74u;

    if (config.arch_type != ArchType::NPU27) {
        block_param.encoding_bits += CMPRS_PAD;
        block_param.dual_encoding_bits = BYTE_ALIGN(block_param.dual_encoding_bits + DUAL_CMPRS_PAD);
    }
}

bool Encoder::Impl::is_algo_dual_encode_compatible(const BitCompactorConfig& config, uint32_t algo) {
    return config.arch_type != ArchType::NPU27 || algo != static_cast<uint32_t>(EncoderAlgorithm::MINSPRDCT);
}

bool Encoder::Impl::is_algo_valid(uint32_t algo, int blk) {
    return !(algo == static_cast<uint32_t>(EncoderAlgorithm::PREVBLKPRDCT) && blk == 0);
}
BestAlgorithm Encoder::Impl::get_best_algorithm(const BitCompactorConfig& config, int blk,
                                                std::vector<AlgorithmParam>& block_params) {
    uint32_t min_encoder_bits{~0u};
    uint32_t min_algo{};

    uint32_t dual_min_encoder_bits{~0u};
    uint32_t dual_min_algo{};

    const auto input_offset{blk * BLOCK_SIZE};
    auto p_input_block{bit_stream_in_.get_byte_pointer(input_offset)};
    const auto algorithm_offset{blk * ALGORITHMS};

    for (uint32_t algo{}; algo < ALGORITHMS; ++algo) {
        if (!is_algo_valid(algo, blk)) {
            continue;
        }

        const auto offset{algorithm_offset + algo};
        auto& block_param{block_params[offset]};
        init_block_param(block_param, p_input_block, algo);

        (this->*algorithm_encoder_[algo])(block_param);

        if (is_algo_dual_encode_compatible(config, algo)) {
            dual_encode(block_param);
        } else {
            block_param.dual_encoding_bits = MAX_BLOCK_COMPRESSION_BITS;
        }

        add_overhead_and_padding(config, block_param, algo);

        if (block_param.encoding_bits < min_encoder_bits) {
            min_encoder_bits = block_param.encoding_bits;
            min_algo = algo;
        }

        if (block_param.dual_encoding_bits < dual_min_encoder_bits) {
            dual_min_encoder_bits = block_param.dual_encoding_bits;
            dual_min_algo = algo;
        }
    }

    return BestAlgorithm{min_encoder_bits, min_algo, dual_min_encoder_bits, dual_min_algo};
}

uint64_t Encoder::Impl::get_dual_encode_len(const BitCompactorConfig& config, AlgorithmParam& block_param) {
    uint64_t encode_len{};

    if (config.arch_type == ArchType::NPU27) {
        encode_len = block_param.dual_encoding_bits - algorithm_overhead_bits_[block_param.encoder_index] - 74u;
    } else {
        // Recalculating encode_len because of byte alignment
        for (unsigned i = 0; i < BLOCK_SIZE; ++i) {
            encode_len = encode_len + (((block_param.dual_bitmap >> i) & 1) ? 8 : block_param.bit_length);
        }
    }

    return encode_len;
}

int Encoder::Impl::write_dual_encode_header(const BitCompactorConfig& config, int blk, BitStream& blk_stream,
                                            std::vector<AlgorithmParam>& block_params, BestAlgorithm& best_algorithm) {
    const auto algorithm_offset{blk * ALGORITHMS};

    int block_index = algorithm_offset + best_algorithm.dual_min_algo;
    auto& min_block_params{block_params[block_index]};
    min_block_params.bit_length = min_block_params.dual_bit_length;

    blk_stream.allocate_bits(min_block_params.dual_encoding_bits);

    CompressionHeader header{create_compression_header(config, CompressionType::cmprsd, min_block_params, 1)};

    blk_stream.write(*reinterpret_cast<uint64_t*>(&header), 20u);

    // Byte alignment
    if (config.arch_type != ArchType::NPU27) {
        blk_stream.write(0ull, 4u);
    }
    return block_index;
}

int Encoder::Impl::write_compressed_header(const BitCompactorConfig& config, int blk, BitStream& blk_stream,
                                           std::vector<AlgorithmParam>& block_params, BestAlgorithm& best_algorithm) {
    const auto algorithm_offset{blk * ALGORITHMS};
    int block_index = algorithm_offset + best_algorithm.min_algo;
    auto& min_block_params{block_params[block_index]};
    min_block_params.dual_bitmap = 0ull;
    blk_stream.allocate_bits(min_block_params.encoding_bits);

    CompressionHeader header{create_compression_header(config, CompressionType::cmprsd, min_block_params)};

    blk_stream.write(*reinterpret_cast<uint64_t*>(&header), 10u);

    // Byte alignment
    if (config.arch_type != ArchType::NPU27) {
        blk_stream.write(0ull, 6u);
    }

    return block_index;
}

void Encoder::Impl::write_uncompressed_blk(const BitCompactorConfig& config, int blk,
                                           std::vector<BitStream>& block_stream,
                                           std::vector<AlgorithmParam>& block_params) {
    auto& blk_stream{block_stream[blk]};
    const auto input_offset{blk * BLOCK_SIZE};
    auto p_input_block{bit_stream_in_.get_byte_pointer(input_offset)};
    const auto algorithm_offset{blk * ALGORITHMS};
    blk_stream.allocate_bits(MAX_BLOCK_COMPRESSION_BITS);

    CompressionHeader header{};
    header.compression_type = static_cast<uint32_t>(CompressionType::uncmprsd);
    blk_stream.write(*reinterpret_cast<uint64_t*>(&header), 2u);

    // Byte alignment
    if (config.arch_type != ArchType::NPU27) {
        blk_stream.write(0ull, 6u);
    }

    AlgorithmParam params_uncomp{};
    params_uncomp.bit_length = 8u;
    params_uncomp.dual_bitmap = 0ull;
    params_uncomp.block_size = BLOCK_SIZE;

    std::memcpy(reinterpret_cast<void*>(params_uncomp.residual), reinterpret_cast<const void*>(p_input_block),
                BLOCK_SIZE);

    write_residual(config, blk_stream, params_uncomp);
}
void Encoder::Impl::write_compressed_blk(const BitCompactorConfig& config, int blk, const bool using_dual_encoder,
                                         std::vector<BitStream>& block_stream,
                                         std::vector<AlgorithmParam>& block_params, BestAlgorithm& best_algorithm) {
    uint32_t block_index{};
    auto& blk_stream{block_stream[blk]};

    if (using_dual_encoder) {
        block_index = write_dual_encode_header(config, blk, blk_stream, block_params, best_algorithm);
    } else {
        block_index = write_compressed_header(config, blk, blk_stream, block_params, best_algorithm);
    }

    auto& min_block_params{block_params[block_index]};

    switch (min_block_params.decoder) {
    case DecoderAlgorithm::ADDPROC:
    case DecoderAlgorithm::SIGNSHFTADDPROC:
        blk_stream.write(*reinterpret_cast<uint64_t*>(&min_block_params.minimum), 8u);
        break;
    }

    if (using_dual_encoder) {
        blk_stream.write(min_block_params.dual_bitmap, 64u);
    }
    write_residual(config, blk_stream, min_block_params);
}

bool Encoder::Impl::is_compression_better_than_uncompressed(BestAlgorithm& best_algorithm) {
    return best_algorithm.min_encoder_bits < MAX_BLOCK_COMPRESSION_BITS ||
           best_algorithm.dual_min_encoder_bits < MAX_BLOCK_COMPRESSION_BITS;
}

bool Encoder::Impl::using_dual_encoder(BestAlgorithm& best_algorithm) {
    return dual_encoder_enable_ && best_algorithm.dual_min_encoder_bits < best_algorithm.min_encoder_bits;
}

void Encoder::Impl::write_last_blk(uint32_t input_blocks, std::vector<BitStream>& block_stream,
                                   unsigned last_block_elements) {
    auto& last_blk_stream{block_stream[input_blocks]};
    last_blk_stream.allocate_bits((last_block_elements + 1u) << 3);

    LastBlockHeader header{};
    header.compression_type = static_cast<uint32_t>(CompressionType::lastblk);
    header.block_length = last_block_elements;
    last_blk_stream.write(*reinterpret_cast<uint64_t*>(&header), 8u);

    const auto input_offset{input_blocks * BLOCK_SIZE};

    auto p_input_block{bit_stream_in_.get_byte_pointer(input_offset)};

    for (uint32_t i{}; i < last_block_elements; i++) {
        uint64_t bits{static_cast<uint64_t>(p_input_block[i]) & 0xffull};
        last_blk_stream.write(bits, 8u);
    }
}

void Encoder::Impl::write_to_output(std::vector<uint8_t>& out, std::vector<BitStream>& block_stream) {
    uint32_t total_bit_count{};

    for (auto& stream : block_stream) {
        total_bit_count += stream.get_bit_count();
    }

    bit_stream_out_.allocate_bits(total_bit_count);

    for (auto& stream : block_stream) {
        bit_stream_out_.append(stream);
    }

    bit_stream_out_.write(0ull, 2);  // end of stream marker

    bit_stream_out_.read(out, output_byte_alignment_);
}

void Encoder::Impl::pack_sparse_data(const BitCompactorConfig& config) {
    const std::vector<uint8_t>& bitmap = config.bitmap;
    unsigned sparse_block_size = config.sparse_block_size;

    auto bitmap_size = bitmap.size();
    unsigned int type_bytes = (config.mode_fp16_enable) ? sizeof(uint16_t) : sizeof(uint8_t);
    unsigned int bitmap_block_size = sparse_block_size / (8 * type_bytes);

    unsigned int srcIdx = 0;
    unsigned int dstIdx = 0;
    unsigned int bitCnt = 0;
    unsigned int btmBlockStart = 0;
    unsigned int datablockStart = 0;

    uint32_t srcLen = bit_stream_in_.source_stream_length();
    uint8_t bitmapByte;

    uint8_t* src = bit_stream_in_.get_byte_pointer(0);

    std::vector<uint8_t> dst(bit_stream_in_.source_stream_length());

    if (sparse_block_size == 0 || sparse_block_size % 16 != 0) {
        throw std::invalid_argument{"(pack_sparse_data): Sparse block size invalid"};
    }
    if (bitmap_size == 0) {
        throw std::invalid_argument{"(pack_sparse_data): Bitmap size 0"};
    }
    if (srcLen % sparse_block_size != 0) {
        throw std::logic_error{"(pack_sparse_data): Invalid configuration"};
    }

    for (size_t btmIdx = 0; btmIdx < bitmap_size; btmIdx++) {
        // Count valid bytes in current block
        bitmapByte = bitmap[btmIdx];
        for (unsigned int bitPos = 0; bitPos < 8; bitPos++) {
            bitCnt += ((bitmapByte >> bitPos) & 1);
            srcIdx += type_bytes;
        }

        if ((srcIdx % sparse_block_size) == 0) {
            // Copy current bitmap block
            memcpy(&dst[dstIdx], &bitmap[btmBlockStart], bitmap_block_size);
            dstIdx += bitmap_block_size;

            if ((btmBlockStart + bitmap_block_size) > bitmap.size()) {
                throw std::logic_error{"(pack_sparse_data): Bitmap handling failed"};
            }

            btmBlockStart += bitmap_block_size;

            // Write the header containing the number of valid elements (corresponds
            // to bitCnt)
            if (bitCnt < 128) {
                // bitCnt is [0..127] => Use 1 byte {7'b<bitCnt>, 1'b0}
                dst[dstIdx] = ((bitCnt << 1) & 0xFF);
                dstIdx++;
            } else {
                // bitCnt is [128..8192]? => Use 2 bytes {14'h<bitCnt>, 2'b01}
                uint16_t header = (((bitCnt << 2) | 0x01) & 0xFFFF);
                dst[dstIdx] = static_cast<unsigned char>(header & 0xFF);
                dst[dstIdx + 1] = static_cast<unsigned char>((header >> 8) & 0xFF);
                dstIdx += 2;
            }

            memcpy(&dst[dstIdx], &src[datablockStart], (bitCnt * type_bytes));
            dstIdx += bitCnt * type_bytes;
            datablockStart = srcIdx;

            // Reset bit counter
            bitCnt = 0;
        }
    }
    dst.resize(dstIdx);
    bit_stream_in_ = BitStream{dst};
}

void Encoder::Impl::encode(const BitCompactorConfig& config, const std::vector<uint8_t>& in,
                           std::vector<uint8_t>& out) {
    init(config, in);
    if (config.bypass_compression) {
        out.resize(bit_stream_in_.source_stream_length());
        memcpy(out.data(), bit_stream_in_.get_byte_pointer(0), bit_stream_in_.source_stream_length());
        return;
    }

    uint64_t bits{};
    auto input_bytes{bit_stream_in_.source_stream_length()};
    auto input_blocks{input_bytes >> 6};

    const auto last_block_elements{input_bytes - (input_blocks << 6)};
    uint32_t last_block{static_cast<uint32_t>(last_block_elements > 0u)};

    std::size_t total_blocks = static_cast<std::size_t>(input_blocks) * static_cast<std::size_t>(ALGORITHMS);
    std::vector<AlgorithmParam> block_params(total_blocks);
    std::vector<BitStream> block_stream(input_blocks + last_block);

    if (config.mode_fp16_enable) {
        fp16_preprocess(bit_stream_in_.source_stream_length());
    }

    for (int blk = 0; blk < static_cast<int>(input_blocks); ++blk) {
        BestAlgorithm best_algorithm = get_best_algorithm(config, blk, block_params);

        if (is_compression_better_than_uncompressed(best_algorithm)) {
            write_compressed_blk(config, blk, using_dual_encoder(best_algorithm), block_stream, block_params,
                                 best_algorithm);
        } else {
            write_uncompressed_blk(config, blk, block_stream, block_params);
        }
    }

    if (last_block) {
        write_last_blk(input_blocks, block_stream, last_block_elements);
    }
    write_to_output(out, block_stream);
}

void Encoder::Impl::write_residual(const BitCompactorConfig& config, BitStream& stream, const AlgorithmParam& param) {
    uint64_t bits{};
    uint32_t bit_count{};
    uint64_t total_bit_count{};

    const uint64_t bit_mask{(1ull << param.bit_length) - 1ull};
    auto dual_bitmap{param.dual_bitmap};

    for (uint32_t i{}; i < param.block_size; ++i) {
        auto bit_length{param.bit_length};
        auto mask{bit_mask};

        if ((dual_bitmap & 0x1ull) == 0x1ull) {
            bit_length = 8u;
            mask = 0xffull;
        }
        dual_bitmap >>= 1;

        if ((bit_count + bit_length) > 64) {
            stream.write(bits, bit_count);
            bit_count = 0u;
            bits = 0ull;
        }

        bits |= ((static_cast<uint64_t>(param.residual[i]) & mask) << bit_count);
        bit_count += bit_length;
        total_bit_count += bit_length;
    }
    if (bit_count) {
        stream.write(bits, bit_count);
    }

    // Byte alignment
    if (config.arch_type != ArchType::NPU27) {
        stream.write(0ull, static_cast<uint32_t>(BYTE_ALIGN(total_bit_count) - total_bit_count));
    }
}
