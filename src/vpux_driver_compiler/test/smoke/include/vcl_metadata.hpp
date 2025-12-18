//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <intel_npu/utils/logger/logger.hpp>
#include "openvino/core/layout.hpp"
#include "openvino/core/version.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/util/variant_visitor.hpp"

namespace vcl {

namespace utils {
static inline size_t align_size_to_standard_page_size(size_t size) {
    constexpr std::size_t STANDARD_PAGE_SIZE = 4096;
    return (size + STANDARD_PAGE_SIZE - 1) & ~(STANDARD_PAGE_SIZE - 1);
}
}  // namespace utils

class MetadataBase {
public:
    MetadataBase(uint32_t version, uint64_t blobDataSize);

    using uninitialized_source = void*;
    using Source = std::variant<uninitialized_source, std::reference_wrapper<std::istream>,
                                std::reference_wrapper<const ov::Tensor>>;

    void read(std::istream& tensor);
    void read(const ov::Tensor& tensor);
    virtual void read() = 0;
    virtual void write(std::ostream& stream) = 0;
    virtual bool is_compatible() = 0;
    virtual uint64_t get_blob_size() const;
    virtual std::optional<std::vector<uint64_t>> get_init_sizes() const;
    virtual std::optional<std::vector<ov::Layout>> get_input_layouts() const;
    virtual std::optional<std::vector<ov::Layout>> get_output_layouts() const;
    virtual std::optional<int64_t> get_batch_size() const;
    virtual ~MetadataBase() = default;
    static std::streampos getFileSize(std::istream& stream);
    virtual size_t get_metadata_size() const = 0;

    static constexpr uint32_t make_version(uint16_t major, uint16_t minor) {
        return major << 16 | (minor & 0x0000ffff);
    }
    static constexpr uint16_t get_major(uint32_t version) {
        return static_cast<uint16_t>(version >> 16);
    }
    static constexpr uint16_t get_minor(uint32_t version) {
        return static_cast<uint16_t>(version);
    }

protected:
    void read_data_from_source(char* destination, const size_t size);
    void append_padding_blob_size_and_magic(std::ostream& stream);

    uint32_t _version;
    uint64_t _blobDataSize;
    intel_npu::Logger _logger;
    Source _source;
    size_t _cursorOffset = 0;
};

constexpr std::string_view MAGIC_BYTES = "OVNPU";
constexpr uint32_t METADATA_VERSION_2_0{MetadataBase::make_version(2, 0)};
constexpr uint32_t METADATA_VERSION_2_1{MetadataBase::make_version(2, 1)};
constexpr uint32_t METADATA_VERSION_2_2{MetadataBase::make_version(2, 2)};
constexpr uint32_t METADATA_VERSION_2_3{MetadataBase::make_version(2, 3)};
constexpr uint32_t CURRENT_METADATA_VERSION{METADATA_VERSION_2_3};
constexpr uint16_t CURRENT_METADATA_MAJOR_VERSION{MetadataBase::get_major(CURRENT_METADATA_VERSION)};
constexpr uint16_t CURRENT_METADATA_MINOR_VERSION{MetadataBase::get_minor(CURRENT_METADATA_VERSION)};

class OpenvinoVersion final {
public:
    constexpr OpenvinoVersion(uint16_t major, uint16_t minor, uint16_t patch)
            : _major(major), _minor(minor), _patch(patch) {
    }

    OpenvinoVersion(const OpenvinoVersion& version);
    OpenvinoVersion& operator=(const OpenvinoVersion& other);
    ~OpenvinoVersion() = default;
    void read(std::istream& istream);
    void read(const ov::Tensor& tensor);
    void write(std::ostream& stream);
    uint16_t get_major() const;
    uint16_t get_minor() const;
    uint16_t get_patch() const;
    size_t get_openvino_version_size() const;
    bool operator!=(const OpenvinoVersion& version);

private:
    uint16_t _major;
    uint16_t _minor;
    uint16_t _patch;
};

constexpr OpenvinoVersion CURRENT_OPENVINO_VERSION(OPENVINO_VERSION_MAJOR, OPENVINO_VERSION_MINOR,
                                                   OPENVINO_VERSION_PATCH);

template <uint32_t version>
struct Metadata : public MetadataBase {};

template <>
class Metadata<METADATA_VERSION_2_0> : public MetadataBase {
public:
    Metadata(uint64_t blobSize, const std::optional<OpenvinoVersion>& ovVersion = std::nullopt);
    void read() override;
    void write(std::ostream& stream) override;
    bool is_compatible() override;
    size_t get_metadata_size() const override;

protected:
    OpenvinoVersion _ovVersion;
};

template <>
class Metadata<METADATA_VERSION_2_1> : public Metadata<METADATA_VERSION_2_0> {
public:
    Metadata(uint64_t blobSize, const std::optional<OpenvinoVersion>& ovVersion = std::nullopt,
             const std::optional<std::vector<uint64_t>>& initSizes = std::nullopt);
    void read() override;
    void write(std::ostream& stream) override;
    std::optional<std::vector<uint64_t>> get_init_sizes() const override;
    size_t get_metadata_size() const override;

private:
    std::optional<std::vector<uint64_t>> _initSizes;
    uint64_t _numberOfInits = 0;
};

template <>
class Metadata<METADATA_VERSION_2_2> : public Metadata<METADATA_VERSION_2_1> {
public:
    Metadata(uint64_t blobSize, std::optional<OpenvinoVersion> ovVersion = std::nullopt,
             const std::optional<std::vector<uint64_t>> initSizes = std::nullopt,
             const std::optional<int64_t> batchSize = std::nullopt);
    void read() override;
    void write(std::ostream& stream) override;
    std::optional<int64_t> get_batch_size() const override;
    size_t get_metadata_size() const override;

private:
    std::optional<int64_t> _batchSize;
};

template <>
class Metadata<METADATA_VERSION_2_3> : public Metadata<METADATA_VERSION_2_2> {
public:
    Metadata(uint64_t blobSize, const std::optional<OpenvinoVersion>& ovVersion = std::nullopt,
             const std::optional<std::vector<uint64_t>>& initSizes = std::nullopt,
             const std::optional<int64_t> batchSize = std::nullopt,
             const std::optional<std::vector<ov::Layout>>& inputLayouts = std::nullopt,
             const std::optional<std::vector<ov::Layout>>& outputLayouts = std::nullopt);
    void read() override;
    void write(std::ostream& stream) override;
    size_t get_metadata_size() const override;
    std::optional<std::vector<ov::Layout>> get_input_layouts() const override;
    std::optional<std::vector<ov::Layout>> get_output_layouts() const override;

private:
    std::optional<std::vector<ov::Layout>> _inputLayouts;
    std::optional<std::vector<ov::Layout>> _outputLayouts;
};

// Implementations

inline uint16_t OpenvinoVersion::get_major() const {
    return _major;
}
inline uint16_t OpenvinoVersion::get_minor() const {
    return _minor;
}
inline uint16_t OpenvinoVersion::get_patch() const {
    return _patch;
}
inline bool OpenvinoVersion::operator!=(const OpenvinoVersion& version) {
    return this->_major != version._major || this->_minor != version._minor || this->_patch != version._patch;
}
inline OpenvinoVersion::OpenvinoVersion(const OpenvinoVersion& version)
        : _major(version.get_major()), _minor(version.get_minor()), _patch(version.get_patch()) {
}

inline OpenvinoVersion& OpenvinoVersion::operator=(const OpenvinoVersion& other) {
    if (this != &other) {
        _major = other.get_major();
        _minor = other.get_minor();
        _patch = other.get_patch();
    }
    return *this;
}

inline void OpenvinoVersion::read(std::istream& stream) {
    stream.read(reinterpret_cast<char*>(&_major), sizeof(_major));
    stream.read(reinterpret_cast<char*>(&_minor), sizeof(_minor));
    stream.read(reinterpret_cast<char*>(&_patch), sizeof(_patch));
}
inline void OpenvinoVersion::read(const ov::Tensor& tensor) {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    _major = *reinterpret_cast<const decltype(_major)*>(tensor.data<const char>());
    _minor = *reinterpret_cast<const decltype(_minor)*>(tensor.data<const char>() + sizeof(_major));
    _patch = *reinterpret_cast<const decltype(_patch)*>(tensor.data<const char>() + sizeof(_major) + sizeof(_minor));
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
}
inline void OpenvinoVersion::write(std::ostream& stream) {
    stream.write(reinterpret_cast<const char*>(&_major), sizeof(_major));
    stream.write(reinterpret_cast<const char*>(&_minor), sizeof(_minor));
    stream.write(reinterpret_cast<const char*>(&_patch), sizeof(_patch));
}
inline size_t OpenvinoVersion::get_openvino_version_size() const {
    return sizeof(_major) + sizeof(_minor) + sizeof(_patch);
}

inline MetadataBase::MetadataBase(uint32_t version, uint64_t blobDataSize)
        : _version(version),
          _blobDataSize(blobDataSize),
          _logger("NPUBlobMetadata", intel_npu::Logger::global().level()),
          _source() {
}

inline Metadata<METADATA_VERSION_2_0>::Metadata(uint64_t blobSize, const std::optional<OpenvinoVersion>& ovVersion)
        : MetadataBase{METADATA_VERSION_2_0, blobSize}, _ovVersion{ovVersion.value_or(CURRENT_OPENVINO_VERSION)} {
}

inline Metadata<METADATA_VERSION_2_1>::Metadata(uint64_t blobSize, const std::optional<OpenvinoVersion>& ovVersion,
                                                const std::optional<std::vector<uint64_t>>& initSizes)
        : Metadata<METADATA_VERSION_2_0>{blobSize, ovVersion}, _initSizes{initSizes} {
    _version = METADATA_VERSION_2_1;
}

inline Metadata<METADATA_VERSION_2_2>::Metadata(uint64_t blobSize, std::optional<OpenvinoVersion> ovVersion,
                                                const std::optional<std::vector<uint64_t>> initSizes,
                                                const std::optional<int64_t> batchSize)
        : Metadata<METADATA_VERSION_2_1>{blobSize, ovVersion, initSizes}, _batchSize{batchSize} {
    _version = METADATA_VERSION_2_2;
}

inline Metadata<METADATA_VERSION_2_3>::Metadata(uint64_t blobSize, const std::optional<OpenvinoVersion>& ovVersion,
                                                const std::optional<std::vector<uint64_t>>& initSizes,
                                                const std::optional<int64_t> batchSize,
                                                const std::optional<std::vector<ov::Layout>>& inputLayouts,
                                                const std::optional<std::vector<ov::Layout>>& outputLayouts)
        : Metadata<METADATA_VERSION_2_2>{blobSize, ovVersion, initSizes, batchSize},
          _inputLayouts{inputLayouts},
          _outputLayouts{outputLayouts} {
    _version = METADATA_VERSION_2_3;
}

inline void MetadataBase::read(std::istream& tensor) {
    _source = Source(tensor);
    read();
}
inline void MetadataBase::read(const ov::Tensor& tensor) {
    _source = Source(tensor);
    read();
}
inline void MetadataBase::read_data_from_source(char* destination, const size_t size) {
    if (const std::reference_wrapper<std::istream>* stream =
                std::get_if<std::reference_wrapper<std::istream>>(&_source)) {
        stream->get().read(destination, size);
    } else if (const std::reference_wrapper<const ov::Tensor>* tensor =
                       std::get_if<std::reference_wrapper<const ov::Tensor>>(&_source)) {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
        std::memcpy(destination, tensor->get().data<const char>() + _cursorOffset, size);
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
        _cursorOffset += size;
    } else {
        OPENVINO_THROW("No blob has been provided to NPU plugin's metadata reader.");
    }
}
inline void MetadataBase::append_padding_blob_size_and_magic(std::ostream& stream) {
    size_t metadataSize = get_metadata_size() + sizeof(_blobDataSize) + MAGIC_BYTES.size();
    size_t size = utils::align_size_to_standard_page_size(metadataSize);
    size_t paddingSize = size - metadataSize;
    if (paddingSize > 0) {
        std::fill_n(std::ostream_iterator<char>(stream), paddingSize, 0);
    }

    stream.write(reinterpret_cast<const char*>(&_blobDataSize), sizeof(_blobDataSize));
    stream.write(MAGIC_BYTES.data(), MAGIC_BYTES.size());
}

inline void Metadata<METADATA_VERSION_2_0>::read() {
    if (const std::reference_wrapper<std::istream>* source =
                std::get_if<std::reference_wrapper<std::istream>>(&_source)) {
        _ovVersion.read(*source);
    } else if (const std::reference_wrapper<const ov::Tensor>* source =
                       std::get_if<std::reference_wrapper<const ov::Tensor>>(&_source)) {
        _ovVersion.read(*source);
        _cursorOffset = _ovVersion.get_openvino_version_size();
    } else {
        OPENVINO_THROW("No blob has been provided to NPU plugin's metadata reader.");
    }
}

inline void Metadata<METADATA_VERSION_2_1>::read() {
    Metadata<METADATA_VERSION_2_0>::read();

    uint64_t numberOfInits;
    read_data_from_source(reinterpret_cast<char*>(&numberOfInits), sizeof(numberOfInits));

    if (numberOfInits) {
        _initSizes = std::vector<uint64_t>(numberOfInits);
        for (uint64_t initIndex = 0; initIndex < numberOfInits; ++initIndex) {
            read_data_from_source(reinterpret_cast<char*>(&_initSizes->at(initIndex)),
                                  sizeof(_initSizes->at(initIndex)));
        }
    }
}

inline void Metadata<METADATA_VERSION_2_2>::read() {
    Metadata<METADATA_VERSION_2_1>::read();

    int64_t batchSize;
    read_data_from_source(reinterpret_cast<char*>(&batchSize), sizeof(batchSize));

    _batchSize = batchSize != 0 ? std::optional(batchSize) : std::nullopt;
}

inline void Metadata<METADATA_VERSION_2_3>::read() {
    Metadata<METADATA_VERSION_2_2>::read();

    uint64_t numberOfInputLayouts, numberOfOutputLayouts;
    read_data_from_source(reinterpret_cast<char*>(&numberOfInputLayouts), sizeof(numberOfInputLayouts));
    read_data_from_source(reinterpret_cast<char*>(&numberOfOutputLayouts), sizeof(numberOfOutputLayouts));

    const auto readNLayouts = [&](const uint64_t numberOfLayouts, const char* loggerAddition) {
        std::optional<std::vector<ov::Layout>> layouts = std::nullopt;
        if (!numberOfLayouts) {
            return layouts;
        }

        uint16_t stringLength;
        layouts = std::vector<ov::Layout>();
        layouts->reserve(numberOfLayouts);
        for (uint64_t layoutIndex = 0; layoutIndex < numberOfLayouts; ++layoutIndex) {
            read_data_from_source(reinterpret_cast<char*>(&stringLength), sizeof(stringLength));

            std::string layoutString(stringLength, 0);
            read_data_from_source(const_cast<char*>(layoutString.c_str()), stringLength);

            try {
                layouts->push_back(ov::Layout(std::move(layoutString)));
            } catch (const ov::Exception&) {
                _logger.warning("Error encountered while constructing an ov::Layout object. %s index: %d. Value "
                                "read from blob: %s. A default value will be used instead.",
                                loggerAddition, (int)layoutIndex, layoutString.c_str());
                layouts->push_back(ov::Layout());
            }
        }
        return layouts;
    };

    _inputLayouts = readNLayouts(numberOfInputLayouts, "Input");
    _outputLayouts = readNLayouts(numberOfOutputLayouts, "Output");
}

inline void Metadata<METADATA_VERSION_2_0>::write(std::ostream& stream) {
    stream.write(reinterpret_cast<const char*>(&_version), sizeof(_version));
    _ovVersion.write(stream);
}

inline void Metadata<METADATA_VERSION_2_1>::write(std::ostream& stream) {
    Metadata<METADATA_VERSION_2_0>::write(stream);

    _numberOfInits = _initSizes.has_value() ? _initSizes->size() : 0;
    stream.write(reinterpret_cast<const char*>(&_numberOfInits), sizeof(_numberOfInits));

    if (_initSizes.has_value()) {
        for (uint64_t initSize : _initSizes.value()) {
            stream.write(reinterpret_cast<const char*>(&initSize), sizeof(initSize));
        }
    }
}

inline void Metadata<METADATA_VERSION_2_2>::write(std::ostream& stream) {
    Metadata<METADATA_VERSION_2_1>::write(stream);

    int64_t batchValue = _batchSize.value_or(0);
    stream.write(reinterpret_cast<const char*>(&batchValue), sizeof(batchValue));
}

inline void Metadata<METADATA_VERSION_2_3>::write(std::ostream& stream) {
    Metadata<METADATA_VERSION_2_2>::write(stream);

    const uint64_t numberOfInputLayouts = _inputLayouts.has_value() ? _inputLayouts->size() : 0;
    const uint64_t numberOfOutputLayouts = _outputLayouts.has_value() ? _outputLayouts->size() : 0;
    stream.write(reinterpret_cast<const char*>(&numberOfInputLayouts), sizeof(numberOfInputLayouts));
    stream.write(reinterpret_cast<const char*>(&numberOfOutputLayouts), sizeof(numberOfOutputLayouts));

    const auto writeLayouts = [&](const std::optional<std::vector<ov::Layout>>& layouts) {
        if (layouts.has_value()) {
            for (const ov::Layout& layout : layouts.value()) {
                const std::string layoutString = layout.to_string();
                const uint16_t stringLength = static_cast<uint16_t>(layoutString.size());
                stream.write(reinterpret_cast<const char*>(&stringLength), sizeof(stringLength));
                stream.write(layoutString.c_str(), stringLength);
            }
        }
    };

    writeLayouts(_inputLayouts);
    writeLayouts(_outputLayouts);

    append_padding_blob_size_and_magic(stream);
}

inline bool Metadata<METADATA_VERSION_2_0>::is_compatible() {
    if (_ovVersion != CURRENT_OPENVINO_VERSION) {
        _logger.error("Imported blob OpenVINO version: %d.%d.%d, but the current OpenVINO version is: %d.%d.%d",
                      _ovVersion.get_major(), _ovVersion.get_minor(), _ovVersion.get_patch(),
                      (int)OPENVINO_VERSION_MAJOR, (int)OPENVINO_VERSION_MINOR, (int)OPENVINO_VERSION_PATCH);
        return false;
    }
    return true;
}

inline std::streampos MetadataBase::getFileSize(std::istream& stream) {
    if (!stream) {
        OPENVINO_THROW("Stream is in bad status! Please check the passed stream status!");
    }

    if (dynamic_cast<ov::SharedStreamBuffer*>(stream.rdbuf()) != nullptr) {
        return stream.rdbuf()->in_avail();
    }
    const std::streampos streamStart = stream.tellg();
    stream.seekg(0, std::ios_base::end);
    const std::streampos streamEnd = stream.tellg();
    stream.seekg(streamStart, std::ios_base::beg);

    if (streamEnd < streamStart) {
        OPENVINO_THROW("Invalid stream size: streamEnd (", streamEnd, ") is not larger than streamStart (", streamStart,
                       ")!");
    }

    return streamEnd - streamStart;
}

inline uint64_t MetadataBase::get_blob_size() const {
    return _blobDataSize;
}
inline std::optional<std::vector<uint64_t>> MetadataBase::get_init_sizes() const {
    return std::nullopt;
}
inline std::optional<int64_t> MetadataBase::get_batch_size() const {
    return std::nullopt;
}
inline std::optional<std::vector<ov::Layout>> MetadataBase::get_input_layouts() const {
    return std::nullopt;
}
inline std::optional<std::vector<ov::Layout>> MetadataBase::get_output_layouts() const {
    return std::nullopt;
}
inline std::optional<std::vector<uint64_t>> Metadata<METADATA_VERSION_2_1>::get_init_sizes() const {
    return _initSizes;
}
inline std::optional<int64_t> Metadata<METADATA_VERSION_2_2>::get_batch_size() const {
    return _batchSize;
}
inline std::optional<std::vector<ov::Layout>> Metadata<METADATA_VERSION_2_3>::get_input_layouts() const {
    return _inputLayouts;
}
inline std::optional<std::vector<ov::Layout>> Metadata<METADATA_VERSION_2_3>::get_output_layouts() const {
    return _outputLayouts;
}

inline size_t Metadata<METADATA_VERSION_2_0>::get_metadata_size() const {
    return sizeof(_version) + _ovVersion.get_openvino_version_size();
}
inline size_t Metadata<METADATA_VERSION_2_1>::get_metadata_size() const {
    size_t metadataSize = Metadata<METADATA_VERSION_2_0>::get_metadata_size() + sizeof(_numberOfInits);
    if (_initSizes.has_value()) {
        metadataSize += _initSizes->size() * sizeof(uint64_t);
    }
    return metadataSize;
}
inline size_t Metadata<METADATA_VERSION_2_2>::get_metadata_size() const {
    size_t metadataSize = Metadata<METADATA_VERSION_2_1>::get_metadata_size() + sizeof(int64_t);
    return metadataSize;
}
inline size_t Metadata<METADATA_VERSION_2_3>::get_metadata_size() const {
    size_t metadataSize = Metadata<METADATA_VERSION_2_2>::get_metadata_size();
    metadataSize += 2 * sizeof(uint64_t);  // Number of input layouts & number of output layouts
    if (_inputLayouts.has_value()) {
        for (const ov::Layout& layout : _inputLayouts.value()) {
            metadataSize +=
                    sizeof(uint16_t) + layout.to_string().size();  // Length followed by the layout value as string
        }
    }
    if (_outputLayouts.has_value()) {
        for (const ov::Layout& layout : _outputLayouts.value()) {
            metadataSize += sizeof(uint16_t) + layout.to_string().size();
        }
    }
    return metadataSize;
}

}  // namespace vcl
