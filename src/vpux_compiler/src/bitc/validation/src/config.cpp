//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "config.hpp"
#include <algorithm>

void verify_label(std::set<std::string>& labels, const std::string& key) {
    auto elem = labels.find(key);
    if (elem == end(labels)) {
        throw std::invalid_argument{"Unexpected key in config: " + key};
    } else {
        labels.erase(elem);
    }
}

void verify_line(std::string& line) {
    for (char c : line) {
        if (!isspace(c)) {
            std::cerr << ANSI_YELLOW << "Unexpected line in config: " << line << '\n' << ANSI_RESET;
            return;
        }
    }
}

void verify_labels(std::set<std::string>& labels, config_map& config) {
    // check boolean labels
    for (const auto& [key, value] : config) {
        if (key == "sparse_mode_enable" || key == "weight_compress_enable" || key == "bypass_compression" ||
            key == "mode_fp16_enable") {
            if (std::get<std::string>(value) != "true"s && std::get<std::string>(value) != "false"s) {
                throw std::invalid_argument{"Expected true or false for " + key +
                                            ", got: " + std::get<std::string>(value)};
            }
        }
    }

    // remove optional labels
    if (labels.find("sparse_mode_enable") != labels.end()) {
        labels.erase("sparse_mode_enable");
    }
    if (labels.find("weight_compress_enable") != labels.end()) {
        labels.erase("weight_compress_enable");
    }
    if (labels.find("bypass_compression") != labels.end()) {
        labels.erase("bypass_compression");
    }
    if (labels.find("mode_fp16_enable") != labels.end()) {
        labels.erase("mode_fp16_enable");
    }

    // Value validity
    if (config.find("arch_type") == config.end()) {
        std::cerr << ANSI_RED << "arch_type not specified\n" << ANSI_RESET;
        throw std::logic_error{"Needed values not specified"};
    } else {
        labels.erase("arch_type");
    }
    std::string arch_type = std::get<std::string>(config["arch_type"]);
    const std::vector<std::string> supported_archs = {
            "NPU27",
            "NPU4",
            "NPU5",
    };
    if (std::find(supported_archs.begin(), supported_archs.end(), arch_type) == supported_archs.end()) {
        std::string supported;
        for (const auto& a : supported_archs) {
            supported += (supported.empty() ? "" : "/") + a;
        }
        throw std::logic_error{"Expected " + supported + " for arch type, got: " + arch_type};
    }

    // Combination validity
    if (arch_type == "NPU27"s) {
        if (auto sparse_mode_enable_it = config.find("sparse_mode_enable"); sparse_mode_enable_it != config.end()) {
            if (std::get<std::string>(sparse_mode_enable_it->second) == "true"s) {
                throw std::logic_error{"NPU27 doesn't support sparse mode"};
            }
        }
        if (auto mode_fp16_enable_it = config.find("mode_fp16_enable"); mode_fp16_enable_it != config.end()) {
            if (std::get<std::string>(mode_fp16_enable_it->second) == "true"s) {
                throw std::logic_error{"NPU27 doesn't support fp16 mode"};
            }
        }
        if (auto weight_compress_enable_it = config.find("weight_compress_enable");
            weight_compress_enable_it != config.end()) {
            if (std::get<std::string>(weight_compress_enable_it->second) == "false"s) {
                throw std::logic_error{"NPU27 doesn't support activation compression"};
            }
        }
    }
    if (arch_type == "NPU4"s) {
        if (auto sparse_mode_enable_it = config.find("sparse_mode_enable"); sparse_mode_enable_it != config.end()) {
            if (std::get<std::string>(sparse_mode_enable_it->second) == "true"s) {
                throw std::logic_error{"NPU4 doesn't support sparse mode"};
            }
        }
    }

    // Sparse verifying
    bool sparse_mode_enable = false;
    if (auto sparse_mode_enable_it = config.find("sparse_mode_enable"); sparse_mode_enable_it != config.end()) {
        if (std::get<std::string>(sparse_mode_enable_it->second) == "true"s) {
            sparse_mode_enable = true;
        }
    }
    if (!sparse_mode_enable) {
        labels.erase("bitmap_data");
        if (config.find("bitmap_data") != config.end()) {
            std::cerr << ANSI_YELLOW
                      << "bitmap_data specified while sparse mode is not enabled, bitmap_data will be ignored\n"
                      << ANSI_RESET;
        }
        labels.erase("bitmap_data_path");
        if (config.find("bitmap_data_path") != config.end()) {
            std::cerr
                    << ANSI_YELLOW
                    << "bitmap_data_path specified while sparse mode is not enabled, bitmap_data_path will be ignored\n"
                    << ANSI_RESET;
        }
        labels.erase("sparse_block_size");
        if (config.find("sparse_block_size") != config.end()) {
            std::cerr << ANSI_YELLOW
                      << "sparse_block_size specified while sparse mode is not enabled, sparse_block_size will be "
                         "ignored\n"
                      << ANSI_RESET;
        }
    }

    // check for missing labels
    if (!labels.empty()) {
        std::cerr << ANSI_RED;
        for (const auto& elem : labels) {
            std::cerr << elem << " value not specified\n";
        }
        std::cerr << ANSI_RESET;
        throw std::logic_error{"Needed values not specified"};
    }
}

bool ends_with_data(const std::string& key) {
    if (key.size() < 4) {
        return false;
    }
    return key.compare(key.length() - 4, std::string::npos, "data") == 0;
}

bool ends_with_path(const std::string& key) {
    if (key.size() < 4) {
        return false;
    }
    return key.compare(key.length() - 4, std::string::npos, "path") == 0;
}

string_vector create_data(std::string& value) {
    string_vector data{};
    std::istringstream iss{value};
    std::string elem;

    while (std::getline(iss, elem, ',')) {
        int state = 0;
        int start = 0, end = 0;
        int start_idx = 0, end_idx = 0;
        bool match = false;

        size_t start_string = elem.find_first_not_of(" ");
        if (start_string != std::string::npos) {
            elem = elem.substr(start_string);
        }

        size_t end_string = elem.find_last_not_of(" \n\r");
        if (end_string != std::string::npos) {
            elem = elem.substr(0, end_string + 1);
        }

        for (int idx = 0; idx < elem.length(); ++idx) {
            switch (state) {
            case 0:
                if (elem[idx] == '[') {
                    start_idx = idx;
                    state = 1;
                }
                break;
            case 1:
                if (isdigit(elem[idx])) {
                    start = start * 10 + elem[idx] - '0';
                    state = 2;
                } else {
                    state = 0;
                }
                break;
            case 2:
                if (isdigit(elem[idx])) {
                    start = start * 10 + elem[idx] - '0';
                } else if (elem[idx] == '-') {
                    state = 3;
                } else {
                    state = 0;
                }
                break;
            case 3:
                if (isdigit(elem[idx])) {
                    end = end * 10 + elem[idx] - '0';
                    state = 4;
                } else {
                    state = 0;
                }
                break;
            case 4:
                if (isdigit(elem[idx])) {
                    end = end * 10 + elem[idx] - '0';
                } else if (elem[idx] == ']') {
                    match = true;
                    end_idx = idx;
                } else {
                    state = 0;
                }
                break;
            default:
                std::cout << "Unexpected\n";
            }
        }
        if (match) {
            for (int idx = start; idx <= end; ++idx) {
                std::string new_value = value;
                new_value.replace(start_idx, end_idx - start_idx + 1, std::to_string(idx));
                data.emplace_back(std::move(new_value));
            }
        } else {
            data.emplace_back(std::move(elem));
        }
    }
    return data;
}

config_map parse_config(std::ifstream& config_file) {
    config_map config;
    std::string line;
    std::set<std::string> labels{"arch_type",
                                 "sparse_mode_enable",
                                 "weight_compress_enable",
                                 "bypass_compression",
                                 "mode_fp16_enable",
                                 "compressed_data_path",
                                 "compressed_data",
                                 "decompressed_data_path",
                                 "decompressed_data",
                                 "bitmap_data_path",
                                 "bitmap_data",
                                 "sparse_block_size"};

    while (std::getline(config_file, line)) {
        std::istringstream iss(line);
        std::string key, value;
        if (std::getline(iss, key, '=') && std::getline(iss, value)) {
            verify_label(labels, key);

            if (ends_with_data(key)) {
                string_vector value_vector = create_data(value);
                config[key] = value_vector;
            } else {
                config[key] = value;
            }
        } else {
            verify_line(line);
        }
    }
    verify_labels(labels, config);

    return std::move(config);
}

void print_config(const config_map& config) {
    std::cout << "Configuration:\n";
    for (const auto& [key, value] : config) {
        if (!ends_with_data(key) && !ends_with_path(key)) {
            std::cout << "\t" << key << ": " << std::get<std::string>(value) << "\n";
        }
    }
}
