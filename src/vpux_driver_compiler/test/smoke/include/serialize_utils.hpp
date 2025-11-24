//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <streambuf>
#include <string>
#include <unordered_set>
#include <vector>

#include "npu_driver_compiler.h"
#include "openvino/core/model.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/core.hpp"
#include "transformations/op_conversions/convert_interpolate11_downgrade.hpp"
#include "vcl_logger.hpp"

const std::string INPUTS_PRECISIONS_KEY = "--inputs_precisions";
const std::string INPUTS_LAYOUTS_KEY = "--inputs_layouts";
const std::string OUTPUTS_PRECISIONS_KEY = "--outputs_precisions";
const std::string OUTPUTS_LAYOUTS_KEY = "--outputs_layouts";

// <option key>="<option value>"
const std::string KEY_VALUE_SEPARATOR = "=";
const std::string VALUE_DELIMITER = "\"";  // marks beginning and end of value
const std::string NAME_VALUE_SEPARATOR = ":";
const std::string VALUES_SEPARATOR = " ";

using supported_log_level = std::unordered_map<std::string, vpux::LogLevel>;

vpux::LogLevel getLogLevel(const std::string& lvl) {
    static const supported_log_level supported_log_levels = {
            {"LOG_NONE", vpux::LogLevel::None},       {"LOG_ERROR", vpux::LogLevel::Error},
            {"LOG_WARNING", vpux::LogLevel::Warning}, {"LOG_INFO", vpux::LogLevel::Info},
            {"LOG_DEBUG", vpux::LogLevel::Debug},     {"LOG_TRACE", vpux::LogLevel::Trace},
    };

    const auto log_level = supported_log_levels.find(lvl);
    if (log_level == supported_log_levels.end()) {
        throw std::invalid_argument(
                "Invalid log level: " + lvl +
                ". Expected levels are: LOG_NONE, LOG_ERROR, LOG_WARNING, LOG_INFO, LOG_DEBUG, LOG_TRACE.");
    }
    return log_level->second;
}

std::string ovPrecisionToLegacyPrecisionString(const ov::element::Type& precision) {
    switch (precision) {
    case ov::element::Type_t::f16:
        return "FP16";
    case ov::element::Type_t::f32:
        return "FP32";
    case ov::element::Type_t::f64:
        return "FP64";
    case ov::element::Type_t::bf16:
        return "BF16";
    case ov::element::Type_t::i4:
        return "I4";
    case ov::element::Type_t::i8:
        return "I8";
    case ov::element::Type_t::i16:
        return "I16";
    case ov::element::Type_t::i32:
        return "I32";
    case ov::element::Type_t::i64:
        return "I64";
    case ov::element::Type_t::u4:
        return "U4";
    case ov::element::Type_t::u8:
        return "U8";
    case ov::element::Type_t::u16:
        return "U16";
    case ov::element::Type_t::u32:
        return "U32";
    case ov::element::Type_t::u64:
        return "U64";
    case ov::element::Type_t::u1:
        return "BIN";
    case ov::element::Type_t::boolean:
        return "BOOL";
    case ov::element::Type_t::dynamic:
        return "DYNAMIC";
    default:
        OPENVINO_THROW("Incorrect precision: ", precision);
    }
}

std::string rankToLegacyLayoutString(const size_t rank) {
    switch (rank) {
    case 0:
        return "**SCALAR**";
    case 1:
        return "C";
    case 2:
        return "NC";
    case 3:
        return "CHW";
    case 4:
        return "NCHW";
    case 5:
        return "NCDHW";
    default:
        return "BLOCKED";
    }
}

std::string serializeIOInfo(const std::shared_ptr<const ov::Model>& model, const bool useIndices = false) {
    const ov::ParameterVector& parameters = model->get_parameters();
    const ov::ResultVector& results = model->get_results();

    std::stringstream inputsPrecisionSS;
    std::stringstream inputsLayoutSS;
    std::stringstream outputsPrecisionSS;
    std::stringstream outputsLayoutSS;

    inputsPrecisionSS << INPUTS_PRECISIONS_KEY << KEY_VALUE_SEPARATOR << VALUE_DELIMITER;
    inputsLayoutSS << INPUTS_LAYOUTS_KEY << KEY_VALUE_SEPARATOR << VALUE_DELIMITER;
    const auto getRankOrThrow = [](const ov::PartialShape& shape) -> size_t {
        if (shape.rank().is_dynamic()) {
            throw std::runtime_error("Dynamic rank is not supported for NPU plugin");
        }
        return shape.rank().get_length();
    };

    if (!parameters.empty()) {
        size_t parameterIndex = 0;

        for (const std::shared_ptr<ov::op::v0::Parameter>& parameter : parameters) {
            const auto precision = parameter->get_element_type();
            const auto rank = getRankOrThrow(parameter->get_partial_shape());

            if (parameterIndex != 0) {
                inputsPrecisionSS << VALUES_SEPARATOR;
                inputsLayoutSS << VALUES_SEPARATOR;
            }

            if (useIndices) {
                inputsPrecisionSS << parameterIndex;
                inputsLayoutSS << parameterIndex;
            } else {
                const std::string& name = parameter->get_friendly_name();

                inputsPrecisionSS << name;
                inputsLayoutSS << name;
            }

            inputsPrecisionSS << NAME_VALUE_SEPARATOR << ovPrecisionToLegacyPrecisionString(precision);
            inputsLayoutSS << NAME_VALUE_SEPARATOR << rankToLegacyLayoutString(rank);

            ++parameterIndex;
        }
    }

    inputsPrecisionSS << VALUE_DELIMITER;
    inputsLayoutSS << VALUE_DELIMITER;

    outputsPrecisionSS << OUTPUTS_PRECISIONS_KEY << KEY_VALUE_SEPARATOR << VALUE_DELIMITER;
    outputsLayoutSS << OUTPUTS_LAYOUTS_KEY << KEY_VALUE_SEPARATOR << VALUE_DELIMITER;

    size_t resultIndex = 0;
    for (const std::shared_ptr<ov::op::v0::Result>& result : results) {
        const auto precision = result->get_element_type();
        const auto rank = getRankOrThrow(result->get_output_partial_shape(0));

        if (resultIndex != 0) {
            outputsPrecisionSS << VALUES_SEPARATOR;
            outputsLayoutSS << VALUES_SEPARATOR;
        }

        if (useIndices) {
            outputsPrecisionSS << resultIndex;
            outputsLayoutSS << resultIndex;
        } else {
            const std::string& name = result->get_input_node_ptr(0)->get_friendly_name();

            outputsPrecisionSS << name;
            outputsLayoutSS << name;
        }

        outputsPrecisionSS << NAME_VALUE_SEPARATOR << ovPrecisionToLegacyPrecisionString(precision);
        outputsLayoutSS << NAME_VALUE_SEPARATOR << rankToLegacyLayoutString(rank);

        ++resultIndex;
    }

    outputsPrecisionSS << VALUE_DELIMITER;
    outputsLayoutSS << VALUE_DELIMITER;

    // One line without spaces to avoid parsing as config option inside CID
    return inputsPrecisionSS.str() + VALUES_SEPARATOR.data() + inputsLayoutSS.str() + VALUES_SEPARATOR.data() +
           outputsPrecisionSS.str() + VALUES_SEPARATOR.data() + outputsLayoutSS.str();
}

/**
 *  @brief Counter stream buffer, just counts the written bytes.
 *  Reads will result in EOF and no seek is supported.
 */
class CounterStreamBuf final : public std::streambuf {
public:
    /// Return the number of bytes written to the stream
    std::streamsize size() {
        return m_size;
    }

private:
    virtual int overflow(int c) override {
        ++m_size;
        return c;
    }

    virtual std::streamsize xsputn(const char* s, std::streamsize n) override {
        m_size += n;
        return n;
    }

    virtual std::streampos seekoff(std::streamoff off, std::ios_base::seekdir way,
                                   std::ios_base::openmode which) override {
        // Return current stream position
        if (off == 0 && way == std::ios_base::cur && which == std::ios_base::out) {
            return m_size;
        } else {
            // No seek support
            throw std::runtime_error("Seek operation is not supported for CounterStreamBuf");
        }
    }

    std::streamsize m_size = 0;
};

/**
 *  @brief Writer stream buffer, writes data to target iterator.
 *  Reads will result in EOF and no seek is supported.
 */
template <typename OutputIt>
class WriterStreamBuf final : public std::streambuf {
public:
    WriterStreamBuf(const OutputIt& it): startIt(it), writeIt(it) {
    }

private:
    int overflow(int c) override {
        return *writeIt++ = c;
    }

    std::streamsize xsputn(const char* s, std::streamsize n) override {
        writeIt = std::copy_n(s, n, writeIt);
        return n;
    }

    std::streampos seekoff(std::streamoff off, std::ios_base::seekdir way, std::ios_base::openmode which) override {
        // Return current stream position
        if (off == 0 && way == std::ios_base::cur && which == std::ios_base::out) {
            return std::distance(startIt, writeIt);
        } else {
            // No seek support
            throw std::runtime_error("Seek operation is not supported for WriterStreamBuf");
        }
    }

    OutputIt startIt;
    OutputIt writeIt;
};

class IRSerializer {
public:
    IRSerializer(const std::shared_ptr<const ov::Model>& origModel, const std::string& logLevelStr,
                 const uint32_t supportedOpset = 11);

    size_t getXmlSize() const {
        return _xmlSize;
    }

    size_t getWeightsSize() const {
        return _weightsSize;
    }

    /**
     * @brief Serialize OpenVINO model to target buffer
     */
    void serializeModelToBuffer(uint8_t* xml, uint8_t* weights);

    /**
     * @brief Save serialized model to ir format.
     */
    void saveSerializeModel() {
        if (const auto env = std::getenv("CID_GET_SERIALIZED_MODEL")) {
            std::string savePath(env);
            _vclLogger->info("start to save serialized model");
            ov::serialize(_model, savePath + "serialized.xml", savePath + "serialized.bin");
            _vclLogger->info("end of saving serialized model");
        }
    }

private:
    /**
     * @brief Serialize OpenVINO model to target stream
     */
    void serializeModelToStream(std::ostream& xml, std::ostream& weights);

    /**
     * @brief Get size of xml and weights from model
     */
    void countModelSize();

    std::shared_ptr<ov::Model> _model = nullptr;
    uint32_t _supportedOpset = 11;
    size_t _xmlSize = 0;
    size_t _weightsSize = 0;
    std::shared_ptr<VPUXDriverCompiler::VCLLogger> _vclLogger = nullptr;
};

IRSerializer::IRSerializer(const std::shared_ptr<const ov::Model>& origModel, const std::string& logLevelStr,
                           const uint32_t supportedOpset)
        : _supportedOpset(supportedOpset) {
    _vclLogger = std::make_shared<VPUXDriverCompiler::VCLLogger>(llvm::StringLiteral("serializeIR"),
                                                                 getLogLevel(logLevelStr), false);
    // There is no const variant of run_passes so use const_cast here
    // as model serialization does not mutate the model
    _model = std::const_pointer_cast<ov::Model>(origModel);

    if (supportedOpset < 11) {
        // Need to clone to modify the model and remain thread safe
        _model = _model->clone();
        _vclLogger->warning("Clone model for opset smaller than 11");
    }

    countModelSize();
}

void IRSerializer::serializeModelToStream(std::ostream& xml, std::ostream& weights) {
    _vclLogger->info("serializeModelToStream start");
    const auto passConfig = std::make_shared<ov::pass::PassConfig>();
    ov::pass::Manager manager(std::move(passConfig), "NPU:serializeModelToStream");

    if (_supportedOpset < 11) {
        // Downgrade to opset10
        manager.register_pass<ov::pass::ConvertInterpolate11ToInterpolate4>();
        _vclLogger->warning("Downgrade op for opset smaller than 11\n");
    }

    manager.register_pass<ov::pass::Serialize>(xml, weights);

    // Depending on the driver version, the compiler attached to it may request this information as an indicator of the
    // precision/layout preprocessing requirement. We are setting this value to "true" since the API version is no
    // longer a cause for altering the metadata. This is due to the preprocessing performed in the OpenVINO framework's
    // implementaion, the "ov::Model" object is preprocessed before reaching the NPU plugin.
    const auto newAPIKey = "is_new_api";

    // Flag used for indicating an NPU plugin version which switched the I/O identification convention from names to
    // indices. The flag is required in order to inform the driver-compiler adapter to expect indices when attempting to
    // deserialize the I/O metadata.
    const auto useIndicesForIOMetadata = "use_indices_for_io_metadata";

    // We modify the original model object here therefore a mutex is required
    static std::mutex rtInfoMutex;

    {
        std::lock_guard<std::mutex> lock(rtInfoMutex);

        _model->set_rt_info(true, newAPIKey);
        _model->set_rt_info(true, useIndicesForIOMetadata);

        manager.run_passes(_model);

        auto& rtInfo = _model->get_rt_info();
        rtInfo.erase(newAPIKey);
        rtInfo.erase(useIndicesForIOMetadata);
    }

    // save intermediate serialized model into ir.
    saveSerializeModel();

    _vclLogger->info("serializeModelToStream end");
}

void IRSerializer::countModelSize() {
    _vclLogger->info("countModelSize start");

    CounterStreamBuf xmlStreamBuf;
    CounterStreamBuf weightsStreamBuf;
    std::ostream xmlStream(&xmlStreamBuf);
    std::ostream weightsStream(&weightsStreamBuf);

    serializeModelToStream(xmlStream, weightsStream);

    _xmlSize = xmlStreamBuf.size();
    _weightsSize = weightsStreamBuf.size();

    _vclLogger->info("countModelSize completed, xml size: {0}, weights size: {1}", _xmlSize, _weightsSize);
}

void IRSerializer::serializeModelToBuffer(uint8_t* xml, uint8_t* weights) {
    _vclLogger->info("serializeModelToBuffer start");

    WriterStreamBuf xmlStreamBuf(xml);
    WriterStreamBuf weightsStreamBuf(weights);
    std::ostream xmlStream(&xmlStreamBuf);
    std::ostream weightsStream(&weightsStreamBuf);

    serializeModelToStream(xmlStream, weightsStream);

    _vclLogger->info("serializeModelToBuffer end");
}

void checkedMemcpy(void* destination, size_t destinationSize, void const* source, size_t numberOfBytes) {
    if (numberOfBytes == 0) {
        return;
    }

    if (destination == nullptr) {
        throw std::runtime_error("Memcpy: received a null destination address");
    }
    if (source == nullptr) {
        throw std::runtime_error("Memcpy: received a null source address");
    }
    if (numberOfBytes > destinationSize) {
        throw std::runtime_error("Memcpy: the source buffer does not fit inside the destination one");
    }

    memmove(destination, source, numberOfBytes);
}

using SerializedIR = std::pair<size_t, std::shared_ptr<uint8_t>>;
vcl_result_t serializeIR(const std::shared_ptr<const ov::Model>& model, vcl_version_info_t compilerVersion,
                         const uint32_t supportedOpsetVersion, vcl_compiler_handle_t compiler,
                         SerializedIR& returnSerializedIR, const std::string& logLevelStr) {
    VPUXDriverCompiler::VCLLogger vclLogger("serializeIR", getLogLevel(logLevelStr), false);
    IRSerializer irSerializer(model, logLevelStr, supportedOpsetVersion);

    // Contract between adapter and compiler in driver
    const uint32_t maxNumberOfElements = 10;
    const uint64_t maxSizeOfXML = std::numeric_limits<uint64_t>::max() / 3;
    const uint64_t maxSizeOfWeights = maxSizeOfXML * 2;

    const uint32_t numberOfInputData = 2;
    const uint64_t xmlSize = static_cast<uint64_t>(irSerializer.getXmlSize());
    const uint64_t weightsSize = static_cast<uint64_t>(irSerializer.getWeightsSize());

    OPENVINO_ASSERT(numberOfInputData < maxNumberOfElements);
    if (xmlSize >= maxSizeOfXML) {
        vclLogger.error("Xml file is too big to process. xmlSize: {0} >= maxSizeOfXML: {1} !!!", xmlSize, maxSizeOfXML);
        return VCL_RESULT_ERROR_IO;
    }
    if (weightsSize >= maxSizeOfWeights) {
        vclLogger.error("Bin file is too big to process. xmlSize: {0} >= maxSizeOfWeights: {1} !!!", weightsSize,
                        maxSizeOfWeights);
        return VCL_RESULT_ERROR_IO;
    }

    const uint64_t sizeOfSerializedIR = sizeof(compilerVersion) + sizeof(numberOfInputData) + sizeof(xmlSize) +
                                        xmlSize + sizeof(weightsSize) + weightsSize;

    // use array to avoid vector's memory zeroing overhead
    std::shared_ptr<uint8_t> buffer(new uint8_t[sizeOfSerializedIR], std::default_delete<uint8_t[]>());
    uint8_t* serializedIR = buffer.get();

    uint64_t offset = 0;
    checkedMemcpy(serializedIR + offset, sizeOfSerializedIR - offset, &compilerVersion, sizeof(compilerVersion));
    offset += sizeof(compilerVersion);
    checkedMemcpy(serializedIR + offset, sizeOfSerializedIR - offset, &numberOfInputData, sizeof(numberOfInputData));
    offset += sizeof(numberOfInputData);

    checkedMemcpy(serializedIR + offset, sizeOfSerializedIR - offset, &xmlSize, sizeof(xmlSize));
    offset += sizeof(xmlSize);
    // xml data is filled in serializeModel()
    uint64_t xmlOffset = offset;
    offset += xmlSize;

    checkedMemcpy(serializedIR + offset, sizeOfSerializedIR - offset, &weightsSize, sizeof(weightsSize));
    offset += sizeof(weightsSize);
    // weights data is filled in serializeModel()
    uint64_t weightsOffset = offset;
    offset += weightsSize;

    irSerializer.serializeModelToBuffer(serializedIR + xmlOffset, serializedIR + weightsOffset);

    if (offset != sizeOfSerializedIR) {
        vclLogger.error("Short read on IR file!!!");
        return VCL_RESULT_ERROR_IO;
    }

    returnSerializedIR = std::make_pair(sizeOfSerializedIR, buffer);

    return VCL_RESULT_SUCCESS;
}
