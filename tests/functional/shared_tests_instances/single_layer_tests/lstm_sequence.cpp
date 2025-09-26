// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/lstm_sequence.hpp"
#include "shared_tests_instances/vpu_ov2_layer_test.hpp"

#include <random>
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "transformations/op_conversions/bidirectional_sequences_decomposition.hpp"
#include "transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp"

#include <fstream>

namespace ov {
namespace test {

using ov::test::utils::InputLayerType;
using ov::test::utils::SequenceTestsMode;

// prins statistics for every cell produced
// #define LSTM_PRINT_DEBUG_STATISTICS

class LSTMSequenceLayerTestCommon : public LSTMSequenceTest, virtual public VpuOv2LayerTest {
    void SetUp() override {
        SequenceTestsMode mode;
        size_t seq_lengths;
        size_t batch;
        size_t hidden_size;
        size_t input_size;
        std::vector<std::string> activations;
        std::vector<float> activations_alpha;
        std::vector<float> activations_beta;
        float clip;
        ov::op::RecurrentSequenceDirection direction;
        InputLayerType WRBType;
        ov::element::Type model_type;
        std::tie(mode, seq_lengths, batch, hidden_size, input_size, activations, clip, direction, WRBType, model_type,
                 targetDevice) = this->GetParam();

        max_seq_lengths = seq_lengths;
        size_t num_directions = direction == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL ? 2 : 1;
        std::vector<ov::Shape> inputShapes = {
                {batch, seq_lengths, input_size},
                {batch, num_directions, hidden_size},
                {batch, num_directions, hidden_size},
                {batch},
                {num_directions, 4 * hidden_size, input_size},
                {num_directions, 4 * hidden_size, hidden_size},
                {num_directions, 4 * hidden_size},
        };

        const auto& W_shape = inputShapes[4];
        const auto& R_shape = inputShapes[5];
        const auto& B_shape = inputShapes[6];

        std::vector<ov::Shape> param_shapes{inputShapes[0], inputShapes[1], inputShapes[2]};
        std::vector<ov::Shape> const_input_shapes;
        if (mode == SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_PARAM ||
            mode == SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_PARAM ||
            mode == SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM) {
            param_shapes.push_back(inputShapes[3]);
        }

        if (WRBType == InputLayerType::PARAMETER) {
            param_shapes.push_back(inputShapes[4]);
            param_shapes.push_back(inputShapes[5]);
            param_shapes.push_back(inputShapes[6]);
        }
        init_input_shapes(ov::test::static_shapes_to_test_representation(param_shapes));

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0]),
                                   std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[1]),
                                   std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[2])};

        std::shared_ptr<ov::Node> seq_lengths_node;
        if (mode == SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_PARAM ||
            mode == SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_PARAM ||
            mode == SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM) {
            auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, inputDynamicShapes[3]);
            seq_lengths_node = param;
            seq_lengths_node->set_friendly_name("seq_lengths");
            params.push_back(param);
        } else if (mode == SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_CONST ||
                   mode == SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_CONST) {
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = 0;
            in_data.range = seq_lengths;
            auto tensor = ov::test::utils::create_and_fill_tensor(ov::element::i64, inputShapes[3], in_data);
            seq_lengths_node = std::make_shared<ov::op::v0::Constant>(tensor);
        } else {
            std::vector<int64_t> lengths(inputShapes[3][0], seq_lengths);
            seq_lengths_node = ov::op::v0::Constant::create(ov::element::i64, inputShapes[3], lengths);
        }

        std::shared_ptr<ov::Node> W, R, B;
        if (WRBType == InputLayerType::PARAMETER) {
            auto param_num = inputDynamicShapes.size();
            const auto W_param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[param_num - 3]);
            const auto R_param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[param_num - 2]);
            const auto B_param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[param_num - 1]);
            W = W_param;
            R = R_param;
            B = B_param;
            params.push_back(W_param);
            params.push_back(R_param);
            params.push_back(B_param);
        } else {
            auto tensor_w = ov::test::utils::create_and_fill_tensor(model_type, W_shape);
            update_data_uniform_tensor(tensor_w, seedW, -0.5, 0.5);
            saveVector(tensor_w, "inputW_f32.bin");
            W = std::make_shared<ov::op::v0::Constant>(tensor_w);

            auto tensor_r = ov::test::utils::create_and_fill_tensor(model_type, R_shape);
            update_data_uniform_tensor(tensor_r, seedR, -0.5, 0.5);
            saveVector(tensor_r, "inputR_f32.bin");
            R = std::make_shared<ov::op::v0::Constant>(tensor_r);

            auto tensor_b = ov::test::utils::create_and_fill_tensor(model_type, B_shape);
            update_data_uniform_tensor(tensor_b, seedB, -0.5, 0.5);
            saveVector(tensor_b, "inputB_f32.bin");
            B = std::make_shared<ov::op::v0::Constant>(tensor_b);
        }

        auto lstm_sequence = std::make_shared<ov::op::v5::LSTMSequence>(
                params[0], params[1], params[2], seq_lengths_node, W, R, B, hidden_size, direction,
                std::vector<float>{}, std::vector<float>{}, activations, clip);

        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(lstm_sequence->output(0)),
                                 std::make_shared<ov::op::v0::Result>(lstm_sequence->output(1)),
                                 std::make_shared<ov::op::v0::Result>(lstm_sequence->output(2))};

        function = std::make_shared<ov::Model>(results, params, "lstm_sequence");
        bool is_pure_sequence = mode == SequenceTestsMode::PURE_SEQ ||
                                mode == SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM ||
                                mode == SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_CONST;

        if (!is_pure_sequence) {
            ov::pass::Manager manager;
            if (direction == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL) {
                manager.register_pass<ov::pass::BidirectionalLSTMSequenceDecomposition>();
            }
            manager.register_pass<ov::pass::ConvertLSTMSequenceToTensorIterator>();
            manager.run_passes(function);
            bool ti_found = ov::test::utils::is_tensor_iterator_exist(function);
            EXPECT_EQ(ti_found, true);
        } else {
            bool ti_found = ov::test::utils::is_tensor_iterator_exist(function);
            EXPECT_EQ(ti_found, false);
        }
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        // inputs:
        // 0 - X[batch_size, seq_length, input_size]
        // 1 - initial_hidden_state[batch_size, num_directions, hidden_size]
        // 2 - initial_cell_state[batch_size, num_directions, hidden_size]
        // 3 - sequence_lengths[batch_size]
        // 4 - W[num_directions, 4 * hidden_size, input_size]
        // 5 - R[num_directions, 4 * hidden_size, hidden_size]
        // 6 - B[num_directions, 4 * hidden_size]
        auto mode = std::get<0>(this->GetParam());
        auto seqLength = std::get<1>(this->GetParam());
        inputs.clear();
        const auto& funcInputs = function->inputs();

        size_t i = 0;  // X[batch_size, seq_length, input_size]
        auto totalSize = ov::shape_size(targetInputStaticShapes[i]);
        auto inputTensor = ov::Tensor{funcInputs[i].get_element_type(), targetInputStaticShapes[i]};
        auto batch_size = targetInputStaticShapes[i][0];
        auto seq_len = targetInputStaticShapes[i][1];
        auto input_size = targetInputStaticShapes[i][2];
        update_realistic_input_tensor(inputTensor, batch_size, seq_len, input_size, seedX);
        saveVector(inputTensor, "inputX_f32.bin");
        inputs.insert({funcInputs[i].get_node_shared_ptr(), inputTensor});

        i++;  // initial_hidden_state[batch_size, num_directions, hidden_size]
        inputTensor = ov::Tensor{funcInputs[i].get_element_type(), targetInputStaticShapes[i]};
        update_data_zero_tensor(inputTensor);
        inputs.insert({funcInputs[i].get_node_shared_ptr(), inputTensor});

        i++;  // initial_cell_state[batch_size, num_directions, hidden_size]
        inputTensor = ov::Tensor{funcInputs[i].get_element_type(), targetInputStaticShapes[i]};
        update_data_zero_tensor(inputTensor);
        inputs.insert({funcInputs[i].get_node_shared_ptr(), inputTensor});

        i++;  // sequence_lengths[batch_size]
        if (i == funcInputs.size()) {
            return;
        }
        if (i == 3 && (mode == utils::SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_PARAM ||
                       mode == utils::SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_PARAM ||
                       mode == utils::SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM)) {
            // sequence_lengths
            EXPECT_EQ(funcInputs[i].get_element_type(), ov::element::i64);
            auto inputData = inputTensor.data<ov::element_type_traits<ov::element::i64>::value_type>();
            for (size_t i = 0; i < totalSize; i++) {
                inputData[i] = seqLength;
            }
            inputs.insert({funcInputs[i].get_node_shared_ptr(), inputTensor});
        }

        i++;  // W[num_directions, 4 * hidden_size, input_size]
        if (i == funcInputs.size()) {
            return;
        }
        totalSize = ov::shape_size(targetInputStaticShapes[i]);
        inputTensor = ov::Tensor{funcInputs[i].get_element_type(), targetInputStaticShapes[i]};
        update_data_uniform_tensor(inputTensor, seedW, -0.5, 0.5);
        inputs.insert({funcInputs[i].get_node_shared_ptr(), inputTensor});

        i++;  // R[num_directions, 4 * hidden_size, hidden_size]
        if (i == funcInputs.size()) {
            return;
        }
        inputTensor = ov::Tensor{funcInputs[i].get_element_type(), targetInputStaticShapes[i]};
        update_data_uniform_tensor(inputTensor, seedR, -0.5, 0.5);
        inputs.insert({funcInputs[i].get_node_shared_ptr(), inputTensor});

        i++;  // B[num_directions, 4 * hidden_size]
        if (i == funcInputs.size()) {
            return;
        }
        inputTensor = ov::Tensor{funcInputs[i].get_element_type(), targetInputStaticShapes[i]};
        update_data_uniform_tensor(inputTensor, seedB, -0.5, 0.5);
        inputs.insert({funcInputs[i].get_node_shared_ptr(), inputTensor});
    }

    void validate() override {
        VpuOv2LayerTest::validate();
    }

    void compare(const std::vector<ov::Tensor>& expectedOutputs,
                 const std::vector<ov::Tensor>& actualOutputs) override {
        auto element_type = expectedOutputs[0].get_element_type();
        switch (element_type) {
        case ov::element::f16:
            compare_t<ov::float16>(expectedOutputs, actualOutputs);
            break;
        case ov::element::f32:
            compare_t<float>(expectedOutputs, actualOutputs);
            break;
        default:
            throw std::runtime_error("Unsupported element type.");
        }
    }

private:
    size_t max_seq_lengths;
    // Fixed seed for reproducibility
    unsigned int seedX = 42;
    unsigned int seedW = 52;
    unsigned int seedR = 53;
    unsigned int seedB = 54;
    // thresholds:
    double mse_thr = 0.001f;      // mean square error
    double abs_thr_hidden = 0.3;  // the real error will be catch by mse and cosine. Big absolute deference for this
                                  // filter not reflect an error
    double abs_thr_cell = 0.3;
    double cosine_thr = 0.995f;

    void update_data_uniform_tensor(ov::Tensor inputTensor, unsigned int seed, double low = -0.5, double high = 0.5) {
        auto totalSize = ov::shape_size(inputTensor.get_shape());
        auto element_type = inputTensor.get_element_type();
        switch (element_type) {
        case ov::element::f16: {
            auto b = inputTensor.data<ov::element_type_traits<ov::element::f16>::value_type>();
            generate_uniform_vector(b, totalSize, seedB, low, high);
        } break;
        case ov::element::f32: {
            auto b = inputTensor.data<ov::element_type_traits<ov::element::f32>::value_type>();
            generate_uniform_vector(b, totalSize, seedB, low, high);
        } break;
        default:
            throw std::runtime_error("Unsupported element type.");
        }
    }
    void update_data_zero_tensor(ov::Tensor inputTensor) {
        auto totalSize = ov::shape_size(inputTensor.get_shape());
        auto element_type = inputTensor.get_element_type();
        switch (element_type) {
        case ov::element::f16: {
            auto b = inputTensor.data<ov::element_type_traits<ov::element::f16>::value_type>();
            generate_zero_vector(b, totalSize);
        } break;
        case ov::element::f32: {
            auto b = inputTensor.data<ov::element_type_traits<ov::element::f32>::value_type>();
            generate_zero_vector(b, totalSize);
        } break;
        default:
            throw std::runtime_error("Unsupported element type.");
        }
    }
    void update_realistic_input_tensor(ov::Tensor inputTensor, int batch_size, int seq_length, int input_size,
                                       int seed) {
        auto totalSize = ov::shape_size(inputTensor.get_shape());
        auto element_type = inputTensor.get_element_type();
        switch (element_type) {
        case ov::element::f16: {
            auto X = inputTensor.data<ov::element_type_traits<ov::element::f16>::value_type>();
            generate_realistic_input(X, batch_size, seq_length, input_size, seed);
        } break;
        case ov::element::f32: {
            auto X = inputTensor.data<ov::element_type_traits<ov::element::f32>::value_type>();
            generate_realistic_input(X, batch_size, seq_length, input_size, seed);
        } break;
        default:
            throw std::runtime_error("Unsupported element type.");
        }
    }

    // Function to generate normally distributed values with a given seed
    template <typename T>
    void generate_normal_vector(T* values, size_t size, unsigned int seed, double mean = 0.1, double stddev = 0.2) {
        std::mt19937 gen(seed);  // Fixed seed for reproducibility
        std::normal_distribution<float> dist(mean, stddev);
        for (size_t i = 0; i < size; i++) {
            ov::float16 v = (ov::float16)dist(gen);
            values[i] = static_cast<T>(v);
        }
    }
    // Function to generate uniform values with a given seed
    template <typename T>
    void generate_uniform_vector(T* values, size_t size, unsigned int seed, double low = -0.5, double high = 0.5) {
        std::mt19937 gen(seed);  // Fixed seed
        std::uniform_real_distribution<float> dist(low, high);
        for (size_t i = 0; i < size; i++) {
            values[i] = static_cast<T>((ov::float16)dist(gen));
        }
    }
    // Function to generate uniform values with a given seed
    template <typename T>
    void generate_zero_vector(T* values, size_t size) {
        for (size_t i = 0; i < size; i++) {
            values[i] = static_cast<T>((ov::float16)(0));
        }
    }
    template <typename T>
    void generate_realistic_input(T* values, int batch_size, int seq_length, int input_size, int seed) {
        // Set the random seed
        const float pi = 3.14159265358979323846;
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> freq_dist(0.5, 3.0);  // Frequency range
        std::uniform_real_distribution<float> phase_dist(0, pi);    // Phase shift range
        std::normal_distribution<float> noise_dist(0, 0.2);         // Noise level
        seq_length = seq_length * batch_size;
        // Generate sinusoidal signals
        std::vector<float> time(seq_length);
        for (int t = 0; t < seq_length; t++) {
            time[t] = 10.0f * t / (seq_length - 1);  // Normalize time
        }

        for (int i = 0; i < input_size; i++) {
            float frequency = freq_dist(gen);
            float phase_shift = phase_dist(gen);

            for (int t = 0; t < seq_length; t++) {
                float signal = sin(frequency * time[t] + phase_shift) + noise_dist(gen);
                values[t * input_size + i] = static_cast<T>(
                        (ov::float16)(std::max(-1.0f, std::min(1.0f, signal))));  // Clip between -1 and 1
            }
        }
    }
    template <typename T>
    double get_max_abs(T* actual, T* expected, size_t size) {
        double max_abs = 0.0f;
        for (size_t i = 0; i < size; i++) {
            double abs = std::abs((double)actual[i] - (double)expected[i]);
            if (abs > max_abs) {
                max_abs = abs;
            }
        }
        return max_abs;
    }
    template <typename T>
    double get_mse(T* actual, T* expected, size_t size, bool reduction_mean = true) {
        double mse = 0.0f;
        for (size_t i = 0; i < size; i++) {
            double abs = std::abs((double)actual[i] - (double)expected[i]);
            mse += abs * abs;
        }
        return (reduction_mean ? (mse / size) : mse);
    }
    template <typename T>
    double get_cosine_similarity(T* actual, T* expected, size_t size) {
        double product = 0.0f, normA = 0.0f, normB = 0.0f;
        for (size_t i = 0; i < size; i++) {
            product += (double)actual[i] * (double)expected[i];
            normA += (double)actual[i] * (double)actual[i];
            normB += (double)expected[i] * (double)expected[i];
        }
        normA = sqrt(normA);
        normB = sqrt(normB);
        if (normA == 0 || normB == 0) {
            return 0.0;
        }
        return product / (normA * normB);
    }

    std::string getErrorMsg(double actual, double expected, std::string method, std::string name, size_t batch,
                            size_t direction, size_t sequence = 0) {
        std::ostringstream out;
        out << "[ COMPARATION ] COMPARATION IS FAILED! incorrect " << method << " for " << name << ": " << std::fixed
            << std::setprecision(10) << actual << " Expected: " << std::fixed << std::setprecision(10) << expected
            << " On positions: batch: " << batch << " directions: " << direction << " sequence: " << sequence;
        return out.str();
    }

    void saveVector(ov::Tensor inputTensor, const std::string& filename) {
#ifdef LSTM_PRINT_DEBUG_STATISTICS
        std::ofstream outFile(filename, std::ios::binary);  // Open in binary mode
        if (!outFile) {
            std::cerr << "Error opening file for writing: " << filename << std::endl;
            return;
        }
        auto totalSize = ov::shape_size(inputTensor.get_shape());
        auto element_type = inputTensor.get_element_type();
        switch (element_type) {
        case ov::element::f16: {
            auto x = inputTensor.data<ov::element_type_traits<ov::element::f16>::value_type>();
            std::vector<float> vec(x, x + totalSize);
            size_t size = vec.size();
            outFile.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(float));
        } break;
        case ov::element::f32: {
            auto x = inputTensor.data<ov::element_type_traits<ov::element::f32>::value_type>();
            std::vector<float> vec(x, x + totalSize);
            size_t size = vec.size();
            outFile.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(float));
        } break;
        default:
            throw std::runtime_error("Unsupported element type.");
        }
        outFile.close();
#endif
    }

    template <typename T>
    void compare_t(const std::vector<ov::Tensor>& expectedOutputs, const std::vector<ov::Tensor>& actualOutputs) {
        // outputs:
        // 0 - Y[batch_size, num_directions, seq_len, hidden_size]
        // 1 - Ho[batch_size, num_directions, hidden_size]
        // 2 - Co[batch_size, num_directions, hidden_size]
        size_t i = 0;  // Y[batch_size, num_directions, seq_len, hidden_size]
        EXPECT_EQ(actualOutputs[i].is_continuous(), true);
        EXPECT_EQ(expectedOutputs[i].is_continuous(), true);
        auto batch_size = actualOutputs[i].get_shape()[0];
        auto num_directions = actualOutputs[i].get_shape()[1];
        auto seq_len = actualOutputs[i].get_shape()[2];
        auto hidden_size = actualOutputs[i].get_shape()[3];
        auto* expectedBuffer = reinterpret_cast<T*>(expectedOutputs[i].data());
        auto* actualBuffer = reinterpret_cast<T*>(actualOutputs[i].data());
        std::vector<std::string> errors;
#ifdef LSTM_PRINT_DEBUG_STATISTICS
        std::cout << "[ COMPARATION Y] [  b,   d,   s] :         mse | max abs val | cos similarity |   min value |    "
                     "max "
                     "value |       mean  | standard deviation | "
                  << std::endl;
        std::vector<double> v_mse_hidden_vals;
        std::vector<double> v_max_abs_hidden_vals;
        std::vector<double> v_cosine_hidden_vals;
#endif
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t d = 0; d < num_directions; d++) {
                for (size_t s = 0; s < seq_len; s++) {
                    // Calculate per sequence and save inside a vector
                    double mse_hidden_vals = get_mse(actualBuffer, expectedBuffer, hidden_size);
                    double max_abs_hidden_vals = get_max_abs(actualBuffer, expectedBuffer, hidden_size);
                    double cosine_hidden_vals = get_cosine_similarity(actualBuffer, expectedBuffer, hidden_size);
#ifdef LSTM_PRINT_DEBUG_STATISTICS
                    v_mse_hidden_vals.push_back(mse_hidden_vals);
                    v_max_abs_hidden_vals.push_back(max_abs_hidden_vals);
                    v_cosine_hidden_vals.push_back(cosine_hidden_vals);

                    std::vector<float> vec(actualBuffer, actualBuffer + hidden_size);
                    auto [min_it, max_it] = std::minmax_element(vec.begin(), vec.end());
                    float min_val = *min_it;
                    float max_val = *max_it;
                    float sum = std::accumulate(vec.begin(), vec.end(), 0.0);
                    float mean = sum / vec.size();
                    double variance = 0.0;
                    for (const float& v : vec) {
                        variance += (v - mean) * (v - mean);
                    }
                    variance /= vec.size();
                    double std_dev = std::sqrt(variance);
                    std::cout << "[ COMPARATION Y] [" << std::setw(3) << b << ", " << std::setw(3) << d << ", "
                              << std::setw(3) << s << "] "
                              << ": " << std::fixed << std::setprecision(10) << mse_hidden_vals << " | " << std::fixed
                              << std::setprecision(10) << max_abs_hidden_vals << " | " << std::fixed
                              << std::setprecision(10) << cosine_hidden_vals << " | " << std::fixed
                              << std::setprecision(6) << std::setw(12) << min_val << " | " << std::fixed
                              << std::setprecision(6) << std::setw(12) << max_val << " | " << std::fixed
                              << std::setprecision(6) << std::setw(12) << mean << " | " << std::fixed
                              << std::setprecision(6) << std::setw(12) << std_dev << " | " << std::endl;
#endif
                    if ((mse_hidden_vals > mse_thr) || std::isnan(mse_hidden_vals)) {
                        errors.push_back(getErrorMsg(mse_hidden_vals, mse_thr, "MSE", "Y", b, d, s));
                    }
                    if ((max_abs_hidden_vals > abs_thr_hidden) || std::isnan(max_abs_hidden_vals)) {
                        errors.push_back(getErrorMsg(max_abs_hidden_vals, abs_thr_hidden, "Max absolute difference",
                                                     "Y", b, d, s));
                    }
                    if ((cosine_hidden_vals < cosine_thr) || std::isnan(cosine_hidden_vals)) {
                        errors.push_back(
                                getErrorMsg(cosine_hidden_vals, cosine_thr, "Cosine Similarity", "Y", b, d, s));
                    }

                    expectedBuffer += hidden_size;
                    actualBuffer += hidden_size;
                }
            }
        }
#ifdef LSTM_PRINT_DEBUG_STATISTICS
        std::cout << "[ COMPARATION Y] [" << "  b" << ", " << "  d" << ", " << "  s" << "] " << ": " << std::fixed
                  << std::setprecision(10) << *(std::max_element(v_mse_hidden_vals.begin(), v_mse_hidden_vals.end()))
                  << " | " << std::fixed << std::setprecision(10)
                  << *(std::max_element(v_max_abs_hidden_vals.begin(), v_max_abs_hidden_vals.end())) << " | "
                  << std::fixed << std::setprecision(10)
                  << *(std::min_element(v_cosine_hidden_vals.begin(), v_cosine_hidden_vals.end())) << std::endl;
#endif

        i = 1;  // Ho[batch_size, num_directions, hidden_size]
        // auto total_size = ov::shape_size(actualOutputs[i].get_shape());
        expectedBuffer = reinterpret_cast<T*>(expectedOutputs[i].data());
        actualBuffer = reinterpret_cast<T*>(actualOutputs[i].data());

        for (size_t b = 0; b < batch_size; b++) {
            for (size_t d = 0; d < num_directions; d++) {
                double mse_hidden_vals = get_mse(actualBuffer, expectedBuffer, hidden_size);
                double max_abs_hidden_vals = get_max_abs(actualBuffer, expectedBuffer, hidden_size);
                double cosine_hidden_vals = get_cosine_similarity(actualBuffer, expectedBuffer, hidden_size);
#ifdef LSTM_PRINT_DEBUG_STATISTICS
                std::cout << "[ COMPARATION Ho] [" << std::setw(3) << b << ", " << std::setw(3) << d << ", " << "] "
                          << ": " << std::fixed << std::setprecision(10) << mse_hidden_vals << " | " << std::fixed
                          << std::setprecision(10) << max_abs_hidden_vals << " | " << std::fixed
                          << std::setprecision(10) << cosine_hidden_vals << " | " << std::endl;
#endif
                // #define LSTM_THROW_EXCEPTION(NAME)
                if ((mse_hidden_vals > mse_thr) || std::isnan(mse_hidden_vals)) {
                    errors.push_back(getErrorMsg(mse_hidden_vals, mse_thr, "MSE", "Ho", b, d));
                }
                if ((max_abs_hidden_vals > abs_thr_hidden) || std::isnan(max_abs_hidden_vals)) {
                    errors.push_back(
                            getErrorMsg(max_abs_hidden_vals, abs_thr_hidden, "Max absolute difference", "Ho", b, d));
                }

                if ((cosine_hidden_vals < cosine_thr) || std::isnan(cosine_hidden_vals)) {
                    errors.push_back(getErrorMsg(cosine_hidden_vals, cosine_thr, "Cosine Similarity", "Ho", b, d));
                }
                expectedBuffer += hidden_size;
                actualBuffer += hidden_size;
            }
        }

        i = 2;  // Co[batch_size, num_directions, hidden_size]
        // total_size = ov::shape_size(actualOutputs[i].get_shape());
        expectedBuffer = reinterpret_cast<T*>(expectedOutputs[i].data());
        actualBuffer = reinterpret_cast<T*>(actualOutputs[i].data());

        for (size_t b = 0; b < batch_size; b++) {
            for (size_t d = 0; d < num_directions; d++) {
                double mse_hidden_vals = get_mse(actualBuffer, expectedBuffer, hidden_size);
                double max_abs_hidden_vals = get_max_abs(actualBuffer, expectedBuffer, hidden_size);
                double cosine_hidden_vals = get_cosine_similarity(actualBuffer, expectedBuffer, hidden_size);
#ifdef LSTM_PRINT_DEBUG_STATISTICS
                std::cout << "[ COMPARATION Co] [" << std::setw(3) << b << ", " << std::setw(3) << d << ", "
                          << "] "
                          << ": " << std::fixed << std::setprecision(10) << mse_hidden_vals << " | " << std::fixed
                          << std::setprecision(10) << max_abs_hidden_vals << " | " << std::fixed
                          << std::setprecision(10) << cosine_hidden_vals << " | " << std::endl;
#endif
                if ((mse_hidden_vals > mse_thr) || std::isnan(mse_hidden_vals)) {
                    errors.push_back(getErrorMsg(mse_hidden_vals, mse_thr, "MSE", "Co", b, d));
                }
                if ((max_abs_hidden_vals > abs_thr_cell) || std::isnan(max_abs_hidden_vals)) {
                    errors.push_back(
                            getErrorMsg(max_abs_hidden_vals, abs_thr_cell, "Max absolute difference", "Co", b, d));
                }

                if ((cosine_hidden_vals < cosine_thr) || std::isnan(cosine_hidden_vals)) {
                    errors.push_back(getErrorMsg(cosine_hidden_vals, cosine_thr, "Cosine Similarity", "Co", b, d));
                }
                expectedBuffer += hidden_size;
                actualBuffer += hidden_size;
            }
        }

        if (errors.size()) {
            for (const std::string& word : errors) {
                std::cout << word << std::endl;
            }
            std::string msg = "[ COMPARATION ] COMPARATION IS FAILED!";
            throw std::runtime_error(msg);
        }
        return;
    }

};  // namespace test

class LSTMSequenceLayerTestCommonSwDpu : public LSTMSequenceLayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-dpu-from-shave-control=true";
    }
};

TEST_P(LSTMSequenceLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(LSTMSequenceLayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(LSTMSequenceLayerTestCommonSwDpu, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;
namespace {
std::vector<utils::SequenceTestsMode> mode = {
        utils::SequenceTestsMode::PURE_SEQ,
};

std::vector<std::vector<std::string>> activations = {{"sigmoid", "tanh", "tanh"}};
std::vector<float> clip{0.f};
std::vector<ov::op::RecurrentSequenceDirection> direction = {ov::op::RecurrentSequenceDirection::FORWARD,
                                                             ov::op::RecurrentSequenceDirection::REVERSE,
                                                             ov::op::RecurrentSequenceDirection::BIDIRECTIONAL};
std::vector<ov::element::Type> modelTypes = {ov::element::f32};

std::vector<size_t> seq_lengths_zero_clip{3};
std::vector<size_t> batch{3};
std::vector<size_t> hidden_size{64};
std::vector<size_t> input_size{67};

const auto lstmConfig = ::testing::Combine(
        ::testing::ValuesIn(mode), ::testing::ValuesIn(seq_lengths_zero_clip), ::testing::ValuesIn(batch),
        ::testing::ValuesIn(hidden_size), ::testing::ValuesIn(input_size), ::testing::ValuesIn(activations),
        ::testing::ValuesIn(clip), ::testing::ValuesIn(direction), ::testing::Values(utils::InputLayerType::CONSTANT),
        ::testing::ValuesIn(modelTypes), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_LSTMSequenceCommonZeroClip, LSTMSequenceLayerTestCommon, lstmConfig,
                         LSTMSequenceLayerTestCommon::getTestCaseName);

// --------- NPU4000 Target speed up scenario---------
std::vector<size_t> seq_lengthsPt{2};  // 160 real case reduced for speed reason
std::vector<size_t> batchPt{1, 2};
std::vector<size_t> hidden_sizePt{16, 17, 64, 128, 144};
std::vector<size_t> input_sizePt{64};
std::vector<float> clipPt{0.f};
std::vector<ov::op::RecurrentSequenceDirection> directionPt = {ov::op::RecurrentSequenceDirection::BIDIRECTIONAL,
                                                               ov::op::RecurrentSequenceDirection::REVERSE};
// FORWARD BIDIRECTIONAL
const auto lstmConfigPt = ::testing::Combine(
        ::testing::ValuesIn(mode), ::testing::ValuesIn(seq_lengthsPt), ::testing::ValuesIn(batchPt),
        ::testing::ValuesIn(hidden_sizePt), ::testing::ValuesIn(input_sizePt), ::testing::ValuesIn(activations),
        ::testing::ValuesIn(clipPt), ::testing::ValuesIn(directionPt),
        ::testing::Values(ov::test::utils::InputLayerType::CONSTANT), ::testing::ValuesIn(modelTypes),
        ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_LSTMSequencePt, LSTMSequenceLayerTestCommon, lstmConfigPt,
                         LSTMSequenceLayerTestCommon::getTestCaseName);

// --------- NPU4000 Accuracy check scenario---------
std::vector<size_t> seqLengthsAccuracy{25};  // 160 real case reduced for speed reason
std::vector<size_t> batchAccuracy{1};
std::vector<size_t> hiddenSizeAccuracy{128};
std::vector<size_t> inputSizeAccuracy{64};
std::vector<float> clipAccuracy{0.f};
std::vector<ov::op::RecurrentSequenceDirection> directionAccuracy = {ov::op::RecurrentSequenceDirection::BIDIRECTIONAL};
// FORWARD BIDIRECTIONAL
const auto lstmConfigAccuracy = ::testing::Combine(
        ::testing::ValuesIn(mode), ::testing::ValuesIn(seqLengthsAccuracy), ::testing::ValuesIn(batchAccuracy),
        ::testing::ValuesIn(hiddenSizeAccuracy), ::testing::ValuesIn(inputSizeAccuracy),
        ::testing::ValuesIn(activations), ::testing::ValuesIn(clipAccuracy), ::testing::ValuesIn(directionAccuracy),
        ::testing::Values(ov::test::utils::InputLayerType::CONSTANT), ::testing::ValuesIn(modelTypes),
        ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_LSTMSequenceAccuracy, LSTMSequenceLayerTestCommon, lstmConfigAccuracy,
                         LSTMSequenceLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_LSTMSequenceAccuracy, LSTMSequenceLayerTestCommonSwDpu, lstmConfigAccuracy,
                         LSTMSequenceLayerTestCommon::getTestCaseName);

}  // namespace
