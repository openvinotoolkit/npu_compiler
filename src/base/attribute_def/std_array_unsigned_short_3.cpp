#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include <array>

namespace mv
{

    namespace attr
    {

        static mv::json::Value toJSON(const Attribute& a)
        {
            json::Array output;
            auto arr = a.get<std::array<unsigned short, 3>>();
            output.append(static_cast<long long>(arr[0]));
            output.append(static_cast<long long>(arr[1]));
            output.append(static_cast<long long>(arr[2]));
            return output;
        }

        static Attribute fromJSON(const json::Value& v)
        {
            if (v.valueType() != json::JSONType::Array)
                throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) + 
                    " to std::array<unsigned short, 3>");

            if (v.size() != 3)
                throw AttributeError(v, "Unable to convert json::Array of size " + std::to_string(v.size()) +  
                    " to std::array<unsigned short, 3>");
            
            std::array<unsigned short, 3> output;
            for (std::size_t i = 0; i < v.size(); ++i)
            {
                if (v[i].valueType() != json::JSONType::NumberInteger)
                    throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v[i].valueType()) + 
                    " to unsigned short (during the conversion to std::array<unsigned short, 3>)");
                output[i] = static_cast<unsigned short>(v[i].get<long long>());
            }

            return output;
        }

        static std::string toString(const Attribute& a)
        {
            auto arr = a.get<std::array<unsigned short, 3>>();
            std::string output = "{" + std::to_string(arr[0]) + ", " + std::to_string(arr[1]) + ", " + 
                std::to_string(arr[2]) + "}";
            return output;
        }

        #define COMMA ,

        MV_REGISTER_ATTR(std::array<unsigned short COMMA 3>)
            .setToJSONFunc(toJSON)
            .setFromJSONFunc(fromJSON)
            .setToStringFunc(toString);

    }

}