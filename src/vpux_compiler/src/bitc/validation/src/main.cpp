//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <cassert>
#include <chrono>
#include <iostream>
#include <string>
#include "Decoder.hpp"
#include "bitc.hpp"
#include "config.hpp"
#include "fileIO.hpp"

using namespace std::chrono;
using namespace vpux;
using std::string_literals::operator""s;

bitc::ArchType string_to_arch(const std::string& arch_type) {
    if (arch_type == "NPU27"s) {
        return bitc::ArchType::NPU27;
    } else if (arch_type == "NPU4"s) {
        return bitc::ArchType::NPU4;
    } else if (arch_type == "NPU5"s) {
        return bitc::ArchType::NPU5;
    } else {
        return;
    }
}

int check_dataset_compression(const config_map& config_test, bitc::BitCompactorConfig& config) {
    std::cout << "\nEncoder running on dataset..." << std::endl;
    uint64_t count_size_diff{};
    uint64_t count_content_diff{};
    uint64_t count_runs{};
    if (config.sparse_mode_enable) {
        config.sparse_block_size = std::stoi(std::get<std::string>(config_test.at("sparse_block_size")));
    }
    std::string decompressed_data_path = std::get<std::string>(config_test.at("decompressed_data_path"));
    const string_vector& decompressed_data_set = std::get<string_vector>(config_test.at("decompressed_data"));
    const string_vector& compressed_data_set = std::get<string_vector>(config_test.at("compressed_data"));

    for (size_t idx = 0; idx < decompressed_data_set.size(); ++idx) {
        std::string decompressed_data_filename_path = decompressed_data_set[idx];
        decompressed_data_filename_path.insert(0, decompressed_data_path);

        std::vector<uint8_t> decompressed_data, compressed_data_out, bitmap;
        if (FileIO::read(decompressed_data_filename_path, decompressed_data)) {
            if (config.sparse_mode_enable) {
                const string_vector& bitmap_data_set = std::get<string_vector>(config_test.at("bitmap_data"));
                std::string bitmap_file_path = bitmap_data_set[idx];

                bitmap_file_path.insert(0, std::get<std::string>(config_test.at("bitmap_data_path")));

                if (!FileIO::read(bitmap_file_path, bitmap)) {
                    std::cerr << "Could not read bitmap\n";
                    return 1;
                }
                config.bitmap = bitmap;
            }
#ifdef __BITC__EN_DBG__
            std::cout << "Encoder read decompressed input: " << decompressed_data_filename << " ("
                      << decompressed_data.size() << " bytes)" << std::endl;
#endif
#ifdef __BITC__EN_PROFILING__
            auto start = steady_clock::now();
#endif
            bitc::Encoder encoder{};
            encoder.encode(config, decompressed_data, compressed_data_out);
#ifdef __BITC__EN_PROFILING__
            auto stop = steady_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);
            std::cout << "Encoder duration for " << decompressed_data_filename << ": " << duration.count()
                      << " microseconds" << std::endl;
#endif
#ifdef __BITC__EN_DBG__
            std::cout << "Encoder compressed output size: " << compressed_data_out.size() << " bytes" << std::endl;
#endif

            std::string golden_compressed_data_file_path = compressed_data_set[idx];
#ifdef __BITC__EN_OUT_WRITE__
            FileIO::write(golden_compressed_data_file_path, compressed_data_out.data(), compressed_data_out.size());
#endif
            golden_compressed_data_file_path.insert(0, std::get<std::string>(config_test.at("compressed_data_path")));

            std::vector<uint8_t> golden_compressed_data;
            if (FileIO::read(golden_compressed_data_file_path, golden_compressed_data)) {
                if (compressed_data_out.size() != golden_compressed_data.size()) {
                    count_size_diff++;
#ifdef __BITC__EN_DBG__
                    std::cout << "Encoder output: " << golden_compressed_data_filename << " size "
                              << compressed_data_out.size() << " doesn't match the golden size "
                              << golden_compressed_data.size() << std::endl;
#endif
                } else if (compressed_data_out != golden_compressed_data) {
                    count_content_diff++;
#ifdef __BITC__EN_DBG__
                    std::cout << "Encoder output: " << golden_compressed_data_filename
                              << " content doesn't match the golden one" << std::endl;
#endif
                } else {
#ifdef __BITC__EN_ENCODE_PERCENTAGE_RATE__
                    if (decompressed_data.size() > compressed_data_out.size()) {
                        double compress_percentage_rate = ((decompressed_data.size() - compressed_data_out.size()) /
                                                           static_cast<double>(decompressed_data.size())) *
                                                          100;
                        std::cout << "Encoder compression rate: -" << compress_percentage_rate << "%" << std::endl;
                    } else {
                        double compress_percentage_rate = ((compressed_data_out.size() - decompressed_data.size()) /
                                                           static_cast<double>(compressed_data_out.size())) *
                                                          100;
                        std::cout << "Encoder compression rate: +" << compress_percentage_rate << "%" << std::endl;
                    }
#endif
                }
                count_runs++;
            } else {
                std::cerr << "Wrong compressed dataset path: " << golden_compressed_data_file_path << std::endl;
                exit(1);
            }
        } else {
            std::cerr << "Wrong decompressed dataset path: " << decompressed_data_filename_path << std::endl;
            exit(1);
        }
    }

    std::cout << "\nEncoder results" << std::endl;
    std::cout << count_runs << " runs" << std::endl;
    std::cout << count_size_diff << " files from dataset doesn't match the golden one in terms of size" << std::endl;
    std::cout << count_content_diff << " files from dataset doesn't match the golden one in terms of content"
              << std::endl;

    return count_content_diff || count_size_diff;
}

int check_dataset_decompression(const config_map& config_test, const bitc::BitCompactorConfig& config) {
    std::cout << "\nDecoder running on dataset..." << std::endl;

    uint64_t count_size_diff{};
    uint64_t count_content_diff{};
    uint64_t count_runs{};

    int sparse_block_size = 0;
    if (config.sparse_mode_enable) {
        sparse_block_size = std::stoi(std::get<std::string>(config_test.at("sparse_block_size")));
    }
    std::string compressed_data_path = std::get<std::string>(config_test.at("compressed_data_path"));
    const string_vector& decompressed_data_set = std::get<string_vector>(config_test.at("decompressed_data"));
    const string_vector& compressed_data_set = std::get<string_vector>(config_test.at("compressed_data"));

    for (size_t idx = 0; idx < compressed_data_set.size(); ++idx) {
        std::string compressed_data_filename_path = compressed_data_set[idx];
        compressed_data_filename_path.insert(0, compressed_data_path);

        std::vector<uint8_t> compressed_data;
        if (FileIO::read(compressed_data_filename_path, compressed_data)) {
#ifdef __BITC__EN_DBG__
            std::cout << "Decoder read compressed input: " << compressed_data_filename << " (" << compressed_data.size()
                      << " bytes)" << std::endl;
#endif
            bitc::Decoder decoder{compressed_data, config};
            std::vector<uint8_t> decompressed_data_out;
            std::vector<uint8_t> bitmap;
            if (config.sparse_mode_enable) {
                const string_vector& bitmap_data_set = std::get<string_vector>(config_test.at("bitmap_data"));
                std::string bitmap_file_path = bitmap_data_set[idx];

                bitmap_file_path.insert(0, std::get<std::string>(config_test.at("bitmap_data_path")));

                if (!FileIO::read(bitmap_file_path, bitmap)) {
                    std::cerr << "Could not read bitmap\n";
                    return 1;
                }
            }
#ifdef __BITC__EN_PROFILING__
            auto start = steady_clock::now();
#endif
            if (config.sparse_mode_enable) {
                decoder.decode(decompressed_data_out, bitmap, sparse_block_size);
            } else {
                decoder.decode(decompressed_data_out);
            }
#ifdef __BITC__EN_PROFILING__
            auto stop = steady_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);
            std::cout << "Decoder duration for " << compressed_data_filename << ": " << duration.count()
                      << " microseconds" << std::endl;
#endif
#ifdef __BITC__EN_DBG__
            std::cout << "Decoder decompressed output size: " << decompressed_data_out.size() << " bytes" << std::endl;
#endif

            // Change name of the compress input to match it with the golden one
            std::string golden_decompressed_data_file_path = decompressed_data_set[idx];
#ifdef __BITC__EN_OUT_WRITE__
            FileIO::write(golden_decompressed_data_file_path, decompressed_data_out.data(),
                          decompressed_data_out.size());
#endif
            golden_decompressed_data_file_path.insert(0,
                                                      std::get<std::string>(config_test.at("decompressed_data_path")));

            std::vector<uint8_t> golden_decompressed_data;
            if (FileIO::read(golden_decompressed_data_file_path, golden_decompressed_data)) {
                if (decompressed_data_out.size() != golden_decompressed_data.size()) {
                    count_size_diff++;
#ifdef __BITC__EN_DBG__
                    std::cout << "Decoder output: " << golden_decompressed_data_filename << " size "
                              << decompressed_data_out.size() << " doesn't match the golden size "
                              << golden_decompressed_data.size() << std::endl;
#endif
                } else if (decompressed_data_out != golden_decompressed_data) {
                    count_content_diff++;
#ifdef __BITC__EN_DBG__
                    std::cout << "Decoder output: " << golden_decompressed_data_filename
                              << " content doesn't match the golden one" << std::endl;
#endif
                }
                count_runs++;
            } else {
                std::cerr << "Wrong decompressed dataset path: " << golden_decompressed_data_file_path << std::endl;
                exit(1);
            }
        } else {
            std::cerr << "Wrong compressed dataset path: " << compressed_data_filename_path << std::endl;
            exit(1);
        }
    }

    std::cout << "\nDecoder results" << std::endl;
    std::cout << count_runs << " runs" << std::endl;
    std::cout << count_size_diff << " files from dataset doesn't match the golden one in terms of size" << std::endl;
    std::cout << count_content_diff << " files from dataset doesn't match the golden one in terms of content"
              << std::endl;

    return count_content_diff || count_size_diff;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << ANSI_RED << "Usage ./bitc <config_file> \n";
        return 1;
    }
    std::ifstream config_stream{argv[1]};
    if (!config_stream.is_open()) {
        std::cerr << ANSI_RED << "File " << argv[1] << " couldn't be opened\n";
        return 1;
    }

    config_map config;

    try {
        config = parse_config(config_stream);
    } catch (const std::exception& e) {
        std::cerr << ANSI_RED << e.what() << '\n';
        return 1;
    }

    print_config(config);

    bitc::BitCompactorConfig config_bitc{string_to_arch(std::get<std::string>(config.at("arch_type"))),
                                         std::get<std::string>(config.at("sparse_mode_enable")) == "true"s,
                                         std::get<std::string>(config.at("weight_compress_enable")) == "true"s,
                                         std::get<std::string>(config.at("bypass_compression")) == "true"s,
                                         std::get<std::string>(config.at("mode_fp16_enable")) == "true"s};

    return check_dataset_compression(config, config_bitc) || check_dataset_decompression(config, config_bitc);
}
