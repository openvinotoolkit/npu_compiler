//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/hash.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/core/string_ref.hpp"
#include "vpux/utils/core/string_utils.hpp"

#include <ie_parameter.hpp>

#include <llvm/ADT/FunctionExtras.h>
#include <llvm/Support/TypeName.h>

#include <cassert>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace vpux {

//
// OptionParser
//

template <typename T>
struct OptionParser;

template <>
struct OptionParser<std::string> final {
    static std::string parse(StringRef val) {
        return val.str();
    }
};

template <>
struct OptionParser<bool> final {
    static bool parse(StringRef val);
};

template <>
struct OptionParser<int64_t> final {
    static int64_t parse(StringRef val);
};

template <>
struct OptionParser<double> final {
    static double parse(StringRef val);
};

template <>
struct OptionParser<LogLevel> final {
    static LogLevel parse(StringRef val);
};

template <typename T>
struct OptionParser<std::vector<T>> final {
    static std::vector<T> parse(StringRef val) {
        std::vector<T> res;
        splitStringList(val, ',', [&](StringRef item) {
            res.push_back(OptionParser<T>::parse(item));
        });
        return res;
    }
};

//
// OptionMode
//

enum class OptionMode {
    Both,
    CompileTime,
    RunTime,
};

StringLiteral stringifyEnum(OptionMode val);

//
// OptionBase
//

// Actual Option description must inherit this class and pass itself as template parameter.
template <class ActualOpt, typename T>
struct OptionBase {
    using ValueType = T;

    // `ActualOpt` must implement the following method:
    // static StringRef key()

#ifdef VPUX_DEVELOPER_BUILD
    // Overload this to provide environment variable support.
    static StringRef envVar() {
        return {};
    }
#endif

    // Overload this to provide deprecated keys names.
    static SmallVector<StringRef> deprecatedKeys() {
        return {};
    }

    // Overload this to provide default value if it wasn't specified by user.
    // If it is None - exception will be thrown in case of missing option access.
    static Optional<T> defaultValue() {
        return None;
    }

    // Overload this to provide more specific parser.
    static ValueType parse(StringRef val) {
        return OptionParser<ValueType>::parse(val);
    }

    // Overload this to provide more specific validation
    static void validateValue(const ValueType&) {
    }

    // Overload this to provide more specific implementation.
    static OptionMode mode() {
        return OptionMode::Both;
    }

    // Overload this for private options.
    static bool isPublic() {
        return true;
    }
};

//
// OptionValue
//

namespace details {

class OptionValue {
public:
    virtual ~OptionValue();

    virtual StringRef getTypeName() const = 0;
};

template <typename T>
class OptionValueImpl final : public OptionValue {
public:
    template <typename U>
    OptionValueImpl(U&& val): _val(std::forward<U>(val)) {
    }

    StringRef getTypeName() const final {
        return llvm::getTypeName<T>();
    }

    const T& getValue() const {
        return _val;
    }

private:
    T _val;
};

}  // namespace details

//
// OptionConcept
//

namespace details {

struct OptionConcept final {
    StringRef (*key)() = nullptr;
#ifdef VPUX_DEVELOPER_BUILD
    StringRef (*envVar)() = nullptr;
#endif
    OptionMode (*mode)() = nullptr;
    bool (*isPublic)() = nullptr;
    std::shared_ptr<OptionValue> (*validateAndParse)(StringRef val) = nullptr;
};

template <class Opt>
std::shared_ptr<OptionValue> validateAndParse(StringRef val) {
    using ValueType = typename Opt::ValueType;

    try {
        auto parsedVal = Opt::parse(val);
        Opt::validateValue(parsedVal);
        return std::make_shared<OptionValueImpl<ValueType>>(std::move(parsedVal));
    } catch (const std::exception& e) {
        VPUX_THROW("Failed to parse '{0}' option : {1}", Opt::key(), e.what());
    }
}

template <class Opt>
OptionConcept makeOptionModel() {
    return {
            &Opt::key,
#ifdef VPUX_DEVELOPER_BUILD
            &Opt::envVar,
#endif
            &Opt::mode,              //
            &Opt::isPublic,          //
            &validateAndParse<Opt>,  //
    };
}

}  // namespace details

//
// OptionsDesc
//

class OptionsDesc final {
public:
    OptionsDesc() = default;

    // Destructor preserves unload order of implementation object and reference to library.
    // To preserve destruction order inside default generated assignment operator we store `_impl` before `_so`.
    // And use destructor to remove implementation object before reference to library explicitly.
    ~OptionsDesc() {
        _impl.clear();
    }

public:
    template <class Opt>
    void add();

    void addSharedObject(const std::shared_ptr<void>& so) {
        _so.push_back(so);
    }

public:
    std::vector<std::string> getSupported(bool includePrivate = false) const;

public:
    details::OptionConcept get(StringRef key, OptionMode mode) const;
    void walk(FuncRef<void(const details::OptionConcept&)> cb) const;

private:
    std::unordered_map<StringRef, details::OptionConcept> _impl;
    std::unordered_map<StringRef, StringRef> _deprecated;

    // Keep pointer to `_so` to avoid shared library unloading prior destruction of the `_impl` object.
    SmallVector<std::shared_ptr<void>> _so;
};

template <class Opt>
void OptionsDesc::add() {
    VPUX_THROW_UNLESS(_impl.count(Opt::key()) == 0, "Option '{0}' was already registered", Opt::key());
    _impl.insert({Opt::key(), details::makeOptionModel<Opt>()});

    for (const auto& deprecatedKey : Opt::deprecatedKeys()) {
        VPUX_THROW_UNLESS(_deprecated.count(deprecatedKey) == 0, "Option '{0}' was already registered", deprecatedKey);
        _deprecated.insert({deprecatedKey, Opt::key()});
    }
}

//
// Config
//

class Config final {
public:
    using ConfigMap = std::map<std::string, std::string>;
    using ImplMap = std::unordered_map<StringRef, std::shared_ptr<details::OptionValue>>;

public:
    explicit Config(const std::shared_ptr<const OptionsDesc>& desc);

public:
    void update(const ConfigMap& options, OptionMode mode = OptionMode::Both);

#ifdef VPUX_DEVELOPER_BUILD
    void parseEnvVars();
#endif

public:
    template <class Opt>
    bool has() const;

    template <class Opt>
    typename Opt::ValueType get() const;

private:
    std::shared_ptr<const OptionsDesc> _desc;
    ImplMap _impl;
};

template <class Opt>
bool Config::has() const {
    return _impl.count(Opt::key()) != 0;
}

template <class Opt>
typename Opt::ValueType Config::get() const {
    using ValueType = typename Opt::ValueType;

    auto log = Logger::global().nest("Config", 0);
    log.trace("Get value for the option '{0}'", Opt::key());

    const auto it = _impl.find(Opt::key());

    if (it == _impl.end()) {
        const Optional<ValueType> optional = Opt::defaultValue();
        log.nest().trace("The option was not set by user, try default value");

        VPUX_THROW_UNLESS(optional.hasValue(), "Option '{0}' was not provided, no default value is available",
                          Opt::key());
        return optional.getValue();
    }

    VPUX_THROW_WHEN(it->second == nullptr, "Got NULL OptionValue for '{0}'", Opt::key());

    const auto optVal = std::dynamic_pointer_cast<details::OptionValueImpl<ValueType>>(it->second);
    VPUX_THROW_WHEN(optVal == nullptr, "Option '{0}' has wrong parsed type: expected '{1}', got '{2}'", Opt::key(),
                    llvm::getTypeName<ValueType>(), it->second->getTypeName());

    return optVal->getValue();
}

bool envVarStrToBool(const char* varName, const char* varValue);

}  // namespace vpux
