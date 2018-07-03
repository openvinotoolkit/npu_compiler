#ifndef MV_JSON_ARRAY_HPP_
#define MV_JSON_ARRAY_HPP_

#include <vector>
#include <memory>
#include "include/mcm/base/json/value.hpp"
#include "include/mcm/base/json/exception/index_error.hpp"

namespace mv
{

    namespace json
    {

        class Array : public ValueContent
        {

            std::vector<Value> elements_;

        public:

            Array();
            Array(const Array& other);
            Array(std::initializer_list<Value> l);
            void push_back(const Value& value);
            void erase(unsigned idx);
            unsigned size() const;
            Value& operator[](unsigned idx);
            std::string stringify() const override;
            Array& operator=(const Array& other);

        };  

    }

}

#endif // MV_JSON_ARRAY_HPP_