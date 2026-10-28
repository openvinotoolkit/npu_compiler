//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/HostExec/transforms/wrap_func_attr.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/utils/core/common_string_utils.hpp"

#include <charconv>
#include <functional>
#include <iterator>

using namespace vpux;

WrapFuncData::WrapFuncData(const mlir::DenseMap<StringRef, StringRef>& params) {
    auto wrappedFuncIt = params.find(WrapFuncData::getWrapperFunctionFieldName());
    VPUX_THROW_WHEN(wrappedFuncIt == params.end(),
                    "Cannot create WrapFuncData from params, as the mandatory field: \"{0}\" is missing, total params "
                    "count: {1}",
                    WrapFuncData::getWrapperFunctionFieldName(), params.size());
    init(wrappedFuncIt->second, params);
}

WrapFuncData::WrapFuncData(StringRef wrapperFunctionName): wrapperFuncName(wrapperFunctionName.str()), deleteWrapped() {
}

WrapFuncData::WrapFuncData(StringRef wrapperFunctionName, bool needToDeleteWrappedFunc)
        : wrapperFuncName(wrapperFunctionName.str()), deleteWrapped(needToDeleteWrappedFunc) {
}

WrapFuncData::WrapFuncData(StringRef wrapperFunctionName, const mlir::DenseMap<StringRef, StringRef>& optionalParams) {
    init(wrapperFunctionName, optionalParams);
}

void WrapFuncData::init(StringRef wrapperFunctionName, const mlir::DenseMap<StringRef, StringRef>& optionalParams) {
    wrapperFuncName = wrapperFunctionName.str();
    for (auto [attr, value] : optionalParams) {
        if (attr == WrapFuncData::getNeedToDeleteWrapperDeclarationFieldName()) {
            VPUX_THROW_WHEN(deleteWrapped.has_value(),
                            "WrapFuncData: the mandatory parameter: {0} must be unique, got: {1}, previous: {2}",
                            WrapFuncData::getNeedToDeleteWrapperDeclarationFieldName(), value, deleteWrapped.value());
            const auto loweredValue = value.lower();
            bool parsedDeleteWrapped = false;
            bool deleteWrappedValue = false;

            if (loweredValue == "1" || loweredValue == "true") {
                parsedDeleteWrapped = true;
                deleteWrappedValue = true;
            } else if (loweredValue == "0" || loweredValue == "false") {
                parsedDeleteWrapped = true;
                deleteWrappedValue = false;
            }

            VPUX_THROW_WHEN(!parsedDeleteWrapped,
                            "WrapFuncData: invalid value for param: {0}, got: \"{1}\". Supported values are: "
                            "\"true\", \"false\", \"1\", \"0\"",
                            WrapFuncData::getNeedToDeleteWrapperDeclarationFieldName(), value);

            deleteWrapped = deleteWrappedValue;
        }
    }
}

StringRef WrapFuncData::getWrapperFunctionNameValue() const {
    return wrapperFuncName;
}

bool WrapFuncData::needToDeleteWrapped() const {
    return deleteWrapped.has_value() ? *deleteWrapped : false;
}

WrapFuncData WrapFuncData::deserialize(const SmallVector<StringRef>& array) {
    constexpr uint32_t minElementsInArray = 1;  // only "wrapper" is a mandatory parameter
    VPUX_THROW_WHEN(
            array.size() < minElementsInArray,
            "Cannot deserialize WrapFuncData from an array of: {0} elements, as the mandatory parameter is: {1}",
            array.size(), minElementsInArray);

    mlir::DenseMap<StringRef, StringRef> params;
    for (StringRef str : array) {
        auto pos = str.find('=');
        VPUX_THROW_WHEN(pos == std::string::npos,
                        "Cannot deserialize WrapFuncData due to an incorrect format of the attribute value pair: {0}, "
                        "must have the separator: '='",
                        str);
        VPUX_THROW_WHEN(
                pos == 0,
                "Cannot deserialize WrapFuncData as the attribute value pair: {0}, hasn't got any attribute name", str);
        VPUX_THROW_WHEN(pos == str.size(),
                        "Cannot deserialize WrapFuncData as the attribute value pair: {0}, hasn't got any value", str);
        auto attrValue = str.split('=');
        params.insert(std::make_pair(attrValue.first.trim(), attrValue.second.trim()));
    }
    return WrapFuncData(params);
}

SmallVector<std::string> WrapFuncData::serialize() const {
    return SmallVector<std::string>(
            {std::string(WrapFuncData::getWrapperFunctionFieldName().data()) + "=" + wrapperFuncName,
             std::string(WrapFuncData::getNeedToDeleteWrapperDeclarationFieldName().data()) + "=" +
                     (needToDeleteWrapped() ? "true" : "false")});
}

std::string WrapFuncData::to_string() const {
    auto paramList = serialize();
    return std::accumulate(paramList.begin(), paramList.end(), std::string(""), [](std::string res, const auto& v) {
        bool wasEmpty = res.empty();
        return std::move(res) + (wasEmpty ? "" : ", ") + v;
    });
}

const WrapFuncData& WrapFuncDataAttributeView::getFuncData() const {
    return data;
}

std::optional<WrapFuncDataAttributeView> WrapFuncDataAttributeView::extract(mlir::func::FuncOp funcOp) {
    if (!funcOp->hasAttr(WrapFuncDataAttributeView::name())) {
        return {};
    }
    auto attr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(funcOp->getAttr(WrapFuncDataAttributeView::name()));
    VPUX_THROW_WHEN(attr == nullptr, "Unexpected type for \"{0}\", only \"mlir::ArrayAttr\" supported",
                    WrapFuncDataAttributeView::name());
    auto attrData = parseCustomAttrArray<mlir::StringAttr>(attr);
    SmallVector<StringRef> attrStringData;
    std::transform(attrData.begin(), attrData.end(), std::back_inserter(attrStringData),
                   [](const mlir::StringAttr& attr) {
                       return attr.getValue();
                   });
    return WrapFuncDataAttributeView(WrapFuncData::deserialize(attrStringData));
}

void WrapFuncDataAttributeView::injectImpl(mlir::func::FuncOp funcOp) const {
    auto serializedArray = data.serialize();

    SmallVector<mlir::Attribute> strAttrs;
    auto ctx = funcOp.getContext();
    for (const auto& v : serializedArray) {
        strAttrs.push_back(mlir::StringAttr::get(ctx, v));
    }

    auto wrapAttrs = mlir::ArrayAttr::get(ctx, strAttrs);
    VPUX_THROW_WHEN(wrapAttrs == nullptr, "Cannot create 'WrapFuncDataAttributeView' attribute \"{0}\"",
                    WrapFuncDataAttributeView::name());
    funcOp->setAttr(WrapFuncDataAttributeView::name(), wrapAttrs);
}
