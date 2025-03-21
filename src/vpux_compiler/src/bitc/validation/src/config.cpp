//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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
    // All labels specified
    if (labels.size() != 0) {
        std::cerr << ANSI_RED;
        for (const auto& elem : labels) {
            std::cerr << elem << " value not specified\n";
        }
        std::cerr << ANSI_RESET;
        throw std::logic_error{"Needed values not specified"};
    }

    // Value validity
    std::string arch_type = std::get<std::string>(config["arch_type"]);
    if (arch_type != "NPU27"s && arch_type != "NPU4"s) {
        throw std::logic_error{"Expected NPU27/4 for arch type, got: " + arch_type};
    }
    std::string weight_compress_enable = std::get<std::string>(config["weight_compress_enable"]);
    if (weight_compress_enable != "true"s && weight_compress_enable != "false"s) {
        throw std::logic_error{"Expected true or false for weight compress enable, got: " + weight_compress_enable};
    }

    std::string bypass_compression = std::get<std::string>(config["bypass_compression"]);
    if (bypass_compression != "true"s && bypass_compression != "false"s) {
        throw std::logic_error{"Expected true or false for bypass compression, got: " + bypass_compression};
    }

    std::string mode_fp16_enable = std::get<std::string>(config["mode_fp16_enable"]);
    if (mode_fp16_enable != "true"s && mode_fp16_enable != "false"s) {
        throw std::logic_error{"Expected true or false for mode fp16 enable, got: " + mode_fp16_enable};
    }

    // Combination validity
    if (arch_type == "NPU27"s) {
        if (mode_fp16_enable == "true"s) {
            throw std::logic_error{"NPU27 doesn't support fp16 mode"};
        }
        if (weight_compress_enable == "false"s) {
            throw std::logic_error{"NPU27 doesn't support activation compression"};
        }
    }
}

bool ends_with_data(std::string& key) {
    return key.compare(key.length() - 4, std::string::npos, "data") == 0;
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
                } else
                    state = 0;
                break;
            case 2:
                if (isdigit(elem[idx]))
                    start = start * 10 + elem[idx] - '0';
                else if (elem[idx] == '-')
                    state = 3;
                else
                    state = 0;
                break;
            case 3:
                if (isdigit(elem[idx])) {
                    end = end * 10 + elem[idx] - '0';
                    state = 4;
                } else
                    state = 0;
                break;
            case 4:
                if (isdigit(elem[idx]))
                    end = end * 10 + elem[idx] - '0';
                else if (elem[idx] == ']') {
                    match = true;
                    end_idx = idx;
                } else
                    state = 0;
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
    std::cout << "Configuration: "
              << "\n\t >> Arch type: " << std::get<std::string>(config.at("arch_type"))
              << "\n\t >> Weight compress enabled: " << std::get<std::string>(config.at("weight_compress_enable"))
              << "\n\t >> Bypass compression: " << std::get<std::string>(config.at("bypass_compression"))
              << "\n\t >> FP16 Mode enabled: " << std::get<std::string>(config.at("mode_fp16_enable")) << "\n";
}
