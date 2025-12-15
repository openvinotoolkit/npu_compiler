# MLIR Primer

MLIR is a framework for building compilers. It consists of multiple concepts which can be combined to offer an extensible infrastructure: dialects, operations, types, attributes, interfaces, passes etc.

This guide aims to introduce you to these concepts in the context of VPUX compiler. The code snippets present in the guide are written to be valid changes which can be applied to your local build of the project, so that you can follow along. If you are new to MLIR, it is recommended to try these changes locally, alter them and observe the effects.

For the purpose of this guide, most of the code snippets will be made to TableGen files (i.e. `.td` files). These files are used to generate C++ headers and sources that contain the definition of the concept that is being represented, such as an operation. This is a widely used feature in VPUX compiler. For the purpose of following this guide, it is sufficient to know that changes to `.td` files are reflected in the generated C++ headers and sources whenever you build the project. If you want to know more about this MLIR feature, you can read this document on [Operation Definition Specification (ODS)](https://mlir.llvm.org/docs/DefiningDialects/Operations/).

---

## Intermediate Representation (IR)

Before diving into specific concepts of MLIR, it is important to get familiar with the Intermediate Representation (IR) that it utilizes. This IR follows the [Static Single-Assignment (SSA)](https://en.wikipedia.org/wiki/Static_single-assignment_form) paradigm, where each value has to be assigned only once and defined before it is used. The following is an example IR:

```MLIR
func.func @main(%arg0: tensor<10xf16>) -> tensor<1xf16> {
    %0 = "dialect.operation"(%arg0) : (tensor<10xf16>) -> tensor<1xf16>
    "return" %0 : tensor<1xf16>
}
```

This example contains one main function operation that has one input value, one output value and two inner operations: `dialect.operation` and `return`. Breaking down the first inner operation, we have:
- `"dialect.operation"` as the operation name
- `%arg0` as the operand of the operation, whose type is `tensor<10xf16>`
- `%0` as the result of the operation, whose type is `tensor<1xf16>`; this result value is defined before it is used by the `result` user.

All of these concepts will be explained in the following sections. For now, it is sufficient to understand the general format of the IR and its SSA constraints.

---

## Dialects

Dialects represent a method of enclosing multiple concepts under one namespace. Let's take the IE dialect as an example. Its definition can be found in [IE/dialect.td](../../tblgen/vpux/compiler/dialect/IE/dialect.td):

```MLIR
include "mlir/IR/OpBase.td"

def IE_Dialect : Dialect {
    let summary = "InferenceEngine IR Dialect";

    let description = [{
        // ...
    }];

    let name = "IE";

    let cppNamespace = "vpux::IE";

    let dependentDialects = [
        // ...
    ];

    // ...
}
```

The snippet above shows two main things:
- `OpBase.td` is included, in order to have the concept of `Dialect` known;
- `IE_Dialect` is defined as being of the `Dialect` class, which comes with a number of fields: `summary`, `description`, `name`, `cppNamespace`, etc.

The IE dialect is placed into its own unique namespace called `vpux::IE`. The generated sources will ensure all of the concepts that are part of the dialect will also be placed into this namespace. For this particular dialect, the sources that are generated can be found in the `IE/generated/dialect.hpp.inc` and `IE/generated/dialect.cpp.inc` files created in your local build directory. It is generally a good idea to get familiar with the generated sources, in order to understand how MLIR actually works.

Dialects are used to separate parts or components of the compilation. In our case, the IE dialect is meant to be a 1-to-1 mapping with the OpenVINO opset and it is the highest level of representation in the compiler. As the compilation progresses, the concepts that are defined in IE dialect will be converted to another dialect, such as VPU, in order to represent more hardware-specific information. However, concepts from different dialects can co-exist into one module (e.g. IE attributes and VPU operations).

Beside the custom dialects that are created by users of MLIR, there are also a series of dialects that come with it. One of these dialects is [Builtin](https://mlir.llvm.org/docs/Dialects/Builtin), which offers a multitude of features that are commonly used.

We are going to continue using the IE and Builtin dialect for the majority of the following sections.

---

## Operations

Once a dialect is created, operations can be added to it. Operations are a core MLIR concept whose semantics can be user-defined - they can represent functions, instructions, layers etc. One practical example for this project is a Convolution. It generally has two inputs (data and weights), an output and attributes (padding, kernel size, kernel strides). MLIR is capable of representing the concept of a Convolution as an operation.

For IE dialect, all operation are defined in the [IE/ops.td](../../tblgen/vpux/compiler/dialect/IE/ops.td) file. This dialect already offers a helper class for generating operations into the dialect:

```MLIR
class IE_Op<string mnemonic, list<Trait> traits = []> :
        Op<IE_Dialect, mnemonic, traits>;
```

This class inherits from the base `Op` class that MLIR provides via [OpBase.td](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-definition), but makes sure the operations created using it end up in the IE dialect. Every operation that is defined using the `Op` TableGen class will result in a C++ class generated for the operation that implement the `mlir::Op` interface. Let's see this in action by creating a new `GettingStarted` operation:

```MLIR
def IE_GettingStartedOp : IE_Op<"GettingStarted"> {
    let summary = "Simple layer used as an example";

    let arguments = (ins
        AnyRankedTensor:$input
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}
```

The operation has one input and one output which are of `AnyRankedTensor` type. We will discuss types in a following section, but for the moment it is sufficient to understand that this type represents the input and output data as a tensor. After building the project, you can find the following classes generated in the `IE/generated/ops.hpp.inc` file from your build directory, accompanied by the `IE/generated/ops.cpp.inc` file, the latter containing the definitions of some methods:

```C++
namespace vpux {
namespace IE {

// Helper class that can be used to make use of accessors without creating an operation
// In large part used internally in MLIR and sparingly in the project
class GettingStartedOpAdaptor {
  // ...
};

// Inherits from `mlir::Op`, which also receives a number of OpTraits.
// These traits are automatically added during the generation of the operation, based on the characteristics found in the TableGen file.
// For example, the `mlir::OpTrait::OneResult` trait is added since the operation has only one result. A verifier is also present in the
// trait, which will ensure only valid instances of GettingStartedOp are created with regards to the number of results, as well as other
// helper methods.
class GettingStartedOp : public ::mlir::Op<GettingStartedOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GettingStartedOpAdaptor;
public:
  // The operation has no attributes, so this is empty
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  // Each operation has one unique name that represents the combination between the dialect and mnemonic used in TableGen
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("IE.GettingStarted");
  }

  // Accessors for the operands of the operation
  // For this operation, the input is the only operand. Based on the name of the operand used in TableGen, helper methods
  // are created to work with it: `input` and `inputMutable`. Internally, these helper methods make use of the generic
  // `getODSOperandIndexAndLength` and `getODSOperands` methods (see the associated `ops.cpp.inc` file from the same directory)
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::MutableOperandRange inputMutable();

  // Similar accessors are created for the results of the operation, which in this case is the output value
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value output();

  // Builders for the operation
  // These will be called when a GettingStarted operation is created. Multiple builders are created for an operation; the one
  // with the matching argument types and numbers will be used.
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});

  // Verifiers created automatically for the operation, to ensure the constraints specified in TableGen are met
  // For example, the operands and results have to be of tensor type
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
public:
};
} // namespace IE
} // namespace vpux
MLIR_DECLARE_EXPLICIT_TYPE_ID(vpux::IE::GettingStartedOp)
```

All of the operands and results of an operation will be represented by the `mlir::Value` class. The main features provided by this class are:
- the type of the value (of class `mlir::Type`)
- an accessor for the operation that generated the value (i.e. the defining operation)
- accessors to the users of the value; i.e. operations consuming the value

Along with the generated sources, operation-specific documentation is also generated in [docs/dialect/_IE.md](../../docs/dialect/_IE.md):

```md
### `IE.GettingStarted` (vpux::IE::GettingStartedOp)

Simple layer used as an example


#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | ranked tensor of any type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | ranked tensor of any type values
```

### Printing and parsing

Let's see how this operation looks like into an IR. Create a new file called `getting_started.mlir` with the following content:

```MLIR
module {
    func.func @main(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
        %0 = "IE.GettingStarted"(%arg0) : (tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16>
        return %0 : tensor<1x16x3x3xf16>
    }
}
```

Running the IR with `vpux-opt` will validate the parsing and printing logic of the operation:

```sh
# The `vpu-arch` argument is necessary for the tool to work, but it will have no effect for this small sample execution
./vpux-opt --vpu-arch=NPU37XX getting_started.mlir
```

The tool will print the same IR, showing that the printer and parser of the operation are aligned. Every concept in MLIR, including operations, have a printing and parsing logic which describe the concept's presence into an IR. By default, the generic form will be used, unless a custom printer / parser is provided. The printing / parsing format is also called an assembly format. One way a custom assembly format is in by setting the `assemblyFormat` value in TableGen:

```MLIR
def IE_GettingStartedOp : IE_Op<"GettingStarted"> {
    let summary = "Simple layer used as an example";

    let arguments = (ins
        AnyRankedTensor:$input
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    // Custom operation assembly format
    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
}
```

The effect of the custom assembly format is the presence of the following methods in the operation's class declaration from `IE/generated/ops.hpp.inc`:

```C++
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
```

Along with the following definitions of the methods in `IE/generated/ops.cpp.inc`:

```C++
// Describes how the operation is parsed when read from an IR, based on the format provided in `assemblyFormat`:
//   `(` operands `)` attr-dict `:` type(operands) `->` type(results)
// Returns mlir::success() if the parsing is successful or mlir::failure() on the first parsing failure
::mlir::ParseResult GettingStartedOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allOperands;
  ::llvm::SmallVector<::mlir::Type, 1> allOperandTypes;
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;
  // Parses `(`
  if (parser.parseLParen())
    return ::mlir::failure();
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  // Parses the list of operands (e.g. `%0`) into the `allOperands` vector
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  // Parses `)`
  if (parser.parseRParen())
    return ::mlir::failure();
  // Parses an optional list of attributes into a dictionary (e.g. `{my_attr = value}`)
  // This will be detailed in a later section, since the operation has no attributes yet
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  // Parses `:`
  if (parser.parseColon())
    return ::mlir::failure();

  // Parses the list of operand types (e.g. `tensor<1x16x3x3xf16>`) into the `allOperandTypes` vector
  if (parser.parseTypeList(allOperandTypes))
    return ::mlir::failure();
  // Parses `->`
  if (parser.parseArrow())
    return ::mlir::failure();

  // Parses the list of result types  (e.g. `tensor<1x16x3x3xf16>`) into the `allResultTypes` vector
  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  // Adds the result type(s) to the operation state
  result.addTypes(allResultTypes);
  // Resolves the parsed operands and operand types into actual mlir::Values and adds them to the operation state
  if (parser.resolveOperands(allOperands, allOperandTypes, allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

// The counterpart to the parser above, which describes how the operation is printed into an IR
void GettingStartedOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << "(";
  _odsPrinter << getOperation()->getOperands();
  _odsPrinter << ")";
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getOperandTypes();
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}
```

As can be seen, the logic of the printer and parser match: what gets printed by the `print` method can be parsed by the `parse` method. In IR format, the example from before becomes:

```MLIR
module {
    func.func @main(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
        %0 = IE.GettingStarted(%arg0) : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>
        return %0 : tensor<1x16x3x3xf16>
    }
}
```

It is also possible to manually define the `parse` / `print` methods of an operation, by setting the `hasCustomAssemblyFormat` flag to 1. This will generate the declaration of the methods as part of the class, but without the implementation.

### Custom methods

MLIR also offers a way to create custom methods in the operation's declaration. This can be done using the `extraClassDeclarations` option:

```MLIR
def IE_GettingStartedOp : IE_Op<"GettingStarted"> {
    let summary = "Simple layer used as an example";

    let arguments = (ins
        AnyRankedTensor:$input
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];

    // Custom methods in the operation's declaration
    let extraClassDeclaration = [{
        static void customStaticMethod() {
            // Do something interesting
        }

        bool customMethod(int x);
    }];
}
```

This extra code will be copied to the operation's declaration in `IE/generated/ops.hpp.inc`, with public access:

```C++
public:
  static void customStaticMethod() {
      // Do something interesting
  }

  bool customMethod(int x);
```

The definition of `customMethod` can be provided externally. For example, the following definition can be set in `src/vpux_compiler/src/dialect/IE/IR/ops/getting_started.cpp`:

```C++
#include "vpux/compiler/dialect/IE/IR/ops.hpp"

bool vpux::IE::GettingStartedOp::customMethod(int x) {
    return x >= 0;
}
```

### Regions

Operations can also have nested operations inside them. One example is the Module and Function operations that was seen in the IR examples (i.e. `module { ... }` and `func.func ... { ... }`). In MLIR terms, the inner lists of operations are called regions. Regions are represented by a list of blocks and each of these blocks contains a list of operations.

More information on regions can be found [here](https://mlir.llvm.org/docs/LangRef/#regions) and [here](https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/#traversing-the-ir-nesting).

---

## Attributes

One important concept in MLIR is attributes, as they allow adding data to operations. Every operation contains a dictionary of attributes, whose state can differ for each instance of the operation.

Attributes can be added to an operation in TableGen:

```MLIR
def IE_GettingStartedOp : IE_Op<"GettingStarted"> {
    let summary = "Simple layer used as an example";

    let arguments = (ins
        AnyRankedTensor:$input,

        // Attributes of the operation
        IntAttr:$myInt,
        I64ArrayAttr:$myIntArray,
        OptionalAttr<F64Attr>:$myOptionalFloat,
        DefaultValuedAttr<IntAttr, "10">:$myDefaultValuedInt
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];

    let extraClassDeclaration = [{
        static void customStaticMethod() {
            // Do something interesting
        }

        bool customMethod(int x);
    }];
}
```

There are many types of attributes that can be created. In this example, we can see integer, integer arrays and float attributes, but there are multiple builtin ones that can be used (see `OpBase.td` and `BuiltinAttributes.td` for more). Additionally, specifiers can also be used to mark optional attributes, those that have default values when unset or to add different constraints over its value.

That being said, these are the main changes to an operation's declaration when attributes are present:

```C++
class GettingStartedOp : // ... {
  // ...

  // Getters for each attribute, returning either attribute objects (e.g. `mlir::IntegerAttr`) or the underlying object (e.g. int64_t)
  mlir::IntegerAttr myIntAttr();
  int64_t myInt();
  ::mlir::ArrayAttr myIntArrayAttr();
  ::mlir::ArrayAttr myIntArray();
  ::mlir::FloatAttr myOptionalFloatAttr();
  ::llvm::Optional< ::llvm::APFloat > myOptionalFloat();
  mlir::IntegerAttr myDefaultValuedIntAttr();
  int64_t myDefaultValuedInt();

  // Setters for each attribute
  void myIntAttr(mlir::IntegerAttr attr);
  void myIntArrayAttr(::mlir::ArrayAttr attr);
  void myOptionalFloatAttr(::mlir::FloatAttr attr);
  void myDefaultValuedIntAttr(mlir::IntegerAttr attr);
  ::mlir::Attribute removeMyOptionalFloatAttr();

  // Builders for the operation, updated to be able to set the values for the attributes
  // Variants are also created accepting the underlying type for attributes directly, where the wrapper `mlir::Attribute` class is handled inside
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, mlir::IntegerAttr myInt, ::mlir::ArrayAttr myIntArray, /*optional*/::mlir::FloatAttr myOptionalFloat, mlir::IntegerAttr myDefaultValuedInt);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, mlir::IntegerAttr myInt, ::mlir::ArrayAttr myIntArray, /*optional*/::mlir::FloatAttr myOptionalFloat, mlir::IntegerAttr myDefaultValuedInt);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, int64_t myInt, ::mlir::ArrayAttr myIntArray, /*optional*/::mlir::FloatAttr myOptionalFloat, int64_t myDefaultValuedInt = 10);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, int64_t myInt, ::mlir::ArrayAttr myIntArray, /*optional*/::mlir::FloatAttr myOptionalFloat, int64_t myDefaultValuedInt = 10);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});

  // Method to populate unset attributes that have a the DefaultValuedAttr specifier
  // Not present if there are no attributes with this specifier present
  static void populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes);
```

Every type of attribute will inherit from the base `mlir::Attribute` class. This is true for both builtin attributes as well as custom attributes. Let's see this in practice with a custom attribute. Introduce the following code in [IE/attributes.td](../../tblgen/vpux/compiler/dialect/IE/attributes.td):

```MLIR
// // The `dialect.td` file should already be included and `IE_Attr` should already be defined, which is why they are added as comments here
// // Similar to how operations are defined, attributes also inherit from a specific class; in this case, `AttrDef`
// // Using a helper class like `IE_Attr` is helpful in making sure every attribute created in this file ends up in the IE dialect and is formatted the same way by default
//
// include "vpux/compiler/dialect/IE/dialect.td"
//
// class IE_Attr<string name, list<Trait> traits = []> :
//         AttrDef<IE_Dialect, name, traits> {
//     let mnemonic = name;
//     let assemblyFormat = "`<` struct(params) `>`";
// }

def IE_GettingStartedAttr : IE_Attr<"GettingStarted"> {
    let parameters = (ins
        "mlir::BoolAttr":$boolParameter,
        "mlir::FloatAttr":$floatParameter
    );
}
```

The `parameters` list contains the types of values that are part of the attribute. It uses a string to identify the base type of the parameter, making it compatible with any attribute type - even ones that are defined manually in C++ instead of using TableGen. There are also specifiers for parameters, similar to those we have seen before for operation attributes, which allow specifying optional ones, default-valued ones etc.

After building the project, the following code will be generated in `IE/generated/attributes.hpp.inc`:

```C++
// GettingStartedAttrStorage will actually contain the values inside the attribute, while GettingStartedAttr will only contain
// a pointer to a GettingStartedAttrStorage object
namespace detail {
struct GettingStartedAttrStorage;
} // namespace detail

// CRTP class, it uses `::mlir::Attribute` as a base class inside
class GettingStartedAttr : public ::mlir::Attribute::AttrBase<GettingStartedAttr, ::mlir::Attribute, detail::GettingStartedAttrStorage> {
public:
  using Base::Base;
public:
  // Method to create a GettingStartedAttr object with the given parameters
  static GettingStartedAttr get(::mlir::MLIRContext *context, mlir::BoolAttr boolParameter, mlir::FloatAttr floatParameter);

  // Used in printing / parsing the attribute
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"GettingStarted"};
  }

  // Auto-generated printer and parser methods for the attribute
  // The implementation can also be controlled using `assemblyFormat` or manually defined with `hasCustomAssemblyFormat`
  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;

  // Getter methods for the parameters of the attribute
  mlir::BoolAttr getBoolParameter() const;
  mlir::FloatAttr getFloatParameter() const;
};
```

Feel free to also explore the implementations for the methods and storage class in `IE/generated/attributes.cpp.inc`.

With the custom attribute created, it can now be used in our operation:

```MLIR
// // This header should already be included, but the line can be uncommented otherwise
// include "vpux/compiler/dialect/IE/attributes.td"

def IE_GettingStartedOp : IE_Op<"GettingStarted"> {
    let summary = "Simple layer used as an example";

    let arguments = (ins
        AnyRankedTensor:$input,

        IntAttr:$myInt,
        I64ArrayAttr:$myIntArray,
        OptionalAttr<F64Attr>:$myOptionalFloat,
        DefaultValuedAttr<IntAttr, "10">:$myDefaultValuedInt,
        // Using the custom attribute in the operation
        IE_GettingStartedAttr:$gettingStarted
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];

    let extraClassDeclaration = [{
        static void customStaticMethod() {
            // Do something interesting
        }

        bool customMethod(int x);
    }];
}
```

This is how the attributes are used in the IR:

```MLIR
module {
    func.func @main(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
        %0 = IE.GettingStarted(%arg0, %arg1) {
            gettingStarted = #IE.GettingStarted<boolParameter = true, floatParameter = 1.0>,
            myInt = 5 : i64,
            myIntArray = [1, 2, 3]
        } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>
        return %0 : tensor<1x16x3x3xf16>
    }
}
```

Optional and default-valued attributes do not have to always be present for the operation. For example, `gettingStarted`'s `floatParameter` is present, while `myOptionalFloat` is not.

### Custom builders

As seen in the previous examples, the builders generated for an operation will change depending on the definition of the operation: list of operands, results, attributes etc. When an operation has optional operands or attributes, it is often useful to have builders that do not need these values to be passed as parameters (even with explicit `nullptr`). To achieve this, custom builders can be added to an operation. Add the following lines to the operation's TableGen definition:

```MLIR
def IE_GettingStartedOp : IE_Op<"GettingStarted"> {
    // ...

    let builders = [
        // Builder containing the input value and only the required attributes
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::IntegerAttr":$myInt, "mlir::ArrayAttr":$myIntArray,
                 "vpux::IE::GettingStartedAttr":$gettingStarted)
        >
    ];
}
```

The following method declaration will be added to the operation's generated class in `IE/generated/ops.hpp.inc`:

```C++
static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::Type type, mlir::Value input, mlir::IntegerAttr myInt, mlir::ArrayAttr myIntArray, vpux::IE::GettingStartedAttr gettingStarted);
```

The method should also be defined in `src/vpux_compiler/src/dialect/IE/IR/ops/getting_started.cpp`:

```C++
void vpux::IE::GettingStartedOp::build(mlir::OpBuilder& /*odsBuilder*/, mlir::OperationState& odsState, mlir::Type output,
                                       mlir::Value input, mlir::IntegerAttr myInt, mlir::ArrayAttr myIntArray,
                                       vpux::IE::GettingStartedAttr gettingStarted) {
    odsState.addOperands(input);
    odsState.addAttribute(myIntAttrName(odsState.name), myInt);
    odsState.addAttribute(myIntArrayAttrName(odsState.name), myIntArray);
    odsState.addAttribute(gettingStartedAttrName(odsState.name), gettingStarted);
    odsState.addTypes(output);
}
```

---

## Types

Every operand and result (i.e. value) of an operation has a type. These values are represented by the `mlir::Value` class, which contains the type as a `mlir::Type` component. The `mlir::Type` class is used as a base class for all types, similar to the way all attributes inherit from the base `mlir::Attribute` class.

Types are used to describe the operation's values. MLIR offers an open type system, which allows the creation of any type. It also offers some builtin types. In the previous examples, the `AnyRankedTensor` type has been used, which materializes into a `mlir::RankedTensorType`. In IR form, this was seen as `tensor<d1xd2x...xdNxDTYPE>` (e.g. `tensor<1x16x3x3xf16>`), where `d1xd2x...xdN` represents the shape of the value and `DTYPE` represents the element type (e.g. float16, int32). Let's take a look over how the builtin `mlir::RankedTensorType` is defined:

```C++
// Inherits from `mlir::TensorType` and implements multiple interfaces, such as `mlir::ShapedType`
class RankedTensorType : public ::mlir::Type::TypeBase<RankedTensorType, TensorType, detail::RankedTensorTypeStorage, ::mlir::ShapedType::Trait> {
public:
  // ...

  // Method to create a `mlir::RankedTensorType` by providing the shape, element type and an optional encoding attribute
  static RankedTensorType get(ArrayRef<int64_t> shape, Type elementType, Attribute encoding = {});

  // Getters for the parameters of the type
  ::llvm::ArrayRef<int64_t> getShape() const;
  Type getElementType() const;
  Attribute getEncoding() const;
};
```

`mlir::RankedTensorType` also contains another `mlir::Type` inside, the element type, which is usually instantiated to a class like `mlir::IntegerType`, `mlir::FloatType` or `mlir::quant::QuantizedType`. This shows the open nature of the type system in MLIR: the semantics of the types are given by the way they are created and utilized.

When `AnyRankedTensor` is used for values of an operation, constraints are added to the operation itself. For `GettingStartedOp`, the `IE/generated/ops.cpp.inc` file contains the following code which ensures the type constraints are satisfied:

```C++
// Returns success if a specific constraint is satisfied
// This constraint corresponds to the TableGen `AnyRankedTensor` item
static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ops4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  // The type has to be a `mlir::TensorType` and it must have a rank (i.e. the number of dimensions is known)
  // Note: `mlir::TensorType` inherits from `mlir::ShapedType`, which is why the cast to the latter can be executed
  if (!((((type.isa<::mlir::TensorType>())) && ((mlir::cast<::mlir::ShapedType>(type).hasRank()))) && ([](::mlir::Type elementType) { return (true); }(mlir::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be ranked tensor of any type values, but got " << type;
  }
  return ::mlir::success();
}

// Verifies whether the constraints of the operation described in TableGen are met
::mlir::LogicalResult GettingStartedOp::verifyInvariantsImpl() {
  // ...
  {
    // Iterates over all operands and checks whether the constraints generated in `__mlir_ods_local_type_constraint_ops4` are satisfied
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  // ..
}
```

Values can also be created to support any type, by using `AnyType`. Constraints can also be added to the element type only (e.g. `AnyTypeOf<[F16, F32]>`, `RankedTensorOf<[I8, F64]>`, `F16Tensor`), to the rank only (e.g. `1DTensorOf<F32>`) etc. `OpBase.td` contains all of the builtin constraints.

Similar to attributes, custom types can also be created. Let's see this in practice with an example. Introduce the following code in [IE/types.td](../../tblgen/vpux/compiler/dialect/IE/types.td) (create the file if it does not exist):

```MLIR
// // In case you are creating a new file, these lines will also be necessary
// // A helper IE_Type class is created, similar to what we did for attributes
//
// include "vpux/compiler/dialect/IE/dialect.td"

// include "mlir/IR/AttrTypeBase.td"

// class IE_Type<string name, list<Trait> traits = []>
//     : TypeDef<IE_Dialect, name, traits> {
//     let mnemonic = name;
//     let assemblyFormat = "`<` struct(params) `>`";
// }

def IE_GettingStartedTensor : IE_Type<"GettingStartedTensor"> {
    let parameters = (ins
        ArrayRefParameter<"int64_t">:$shape,
        "mlir::Type":$elementType
    );
}
```

Let's use this new type for our operation, by adding a new operand:

```MLIR
// // This line is only necessary if the file is not already included
// include "vpux/compiler/dialect/IE/types.td"

def IE_GettingStartedOp : IE_Op<"GettingStarted"> {
    let summary = "Simple layer used as an example";

    let arguments = (ins
        AnyRankedTensor:$input,
        // New operand using the new type
        IE_GettingStartedTensor:$secondInput,

        IntAttr:$myInt,
        I64ArrayAttr:$myIntArray,
        OptionalAttr<F64Attr>:$myOptionalFloat,
        DefaultValuedAttr<IntAttr, "10">:$myDefaultValuedInt,
        IE_GettingStartedAttr:$gettingStarted
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];

    let extraClassDeclaration = [{
        static void customStaticMethod() {
            // Do something interesting
        }

        bool customMethod(int x);
    }];

    let builders = [
        OpBuilder<
            (ins "mlir::Type":$type, "mlir::Value":$input, "mlir::IntegerAttr":$myInt,
                 "mlir::ArrayAttr":$myIntArray, "vpux::IE::GettingStartedAttr":$gettingStarted)
        >
    ];
}
```

If the `types.td` file was created manually, it is also necessary to make the generated C++ type symbols visible to the C++ code of the operation. The way this generated code is included in compilation is by including the generated files (i.e. `types.hpp.inc` and `types.cpp.inc`) in compile targets and initializing the types:
```C++
/*
 * In `src/vpux_compiler/tblgen/vpux/compiler/dialect/IE/CMakeLists.txt`
 * Ensures the TypeDefs in the types.td file are used to generate the C++ classes for the types
 */
add_vpux_type(IE)


/*
 * In `src/vpux_compiler/include/vpux/compiler/dialect/IE/types.hpp`
 * Includes the class declaration for the new type
 */
#pragma once

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/dialect/IE/types.hpp.inc>


/*
 * In `src/vpux_compiler/src/dialect/IE/types.cpp`
 * Includes the class definition of the new type
 *
 * The `types.cpp.inc` file contains two sections that can be included which are guarded by `GET_TYPEDEF_CLASSES` and `GET_TYPEDEF_LIST`
 * The first one contains the definitions of the type classes, while the second one contains a list of all the type symbols, which can be used
 * to register the types into the dialect
 */
#include "vpux/compiler/dialect/IE/types.hpp"

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/dialect/IE/types.cpp.inc>

void vpux::IE::IEDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include <vpux/compiler/dialect/IE/types.cpp.inc>
            >();
}

/*
 * In `src/vpux_compiler/tblgen/vpux/compiler/dialect/IE/dialect.td`:
 * Introduces method in dialect for registering custom types into the dialect
 */
def IE_Dialect : Dialect {
    // ...
    let extraClassDeclaration = [{
        // ...
        // Add the following method
        void registerTypes();
    }];
    // ...
    // Generates printers / parsers for dialect types
    let useDefaultTypePrinterParser = 1;
}


/*
 * In `src/vpux_compiler/src/dialect/IE/dialect.cpp`:
 * Calls the registration method defined before
 */
void vpux::IE::IEDialect::initialize() {
    // ...
    registerTypes();
}

/*
 * In `src/vpux_compiler/src/dialect/IE/ops.cpp`:
 * Since the new type is now used for the operation's operand, a constraint will be generated
 * in the operation's source file to ensure the type is correct
 * The type symbol has to be visible for the compilation to succeed
 */
#include "vpux/compiler/dialect/IE/types.hpp"
```

The same process applies for attributes. However, this is already done in the compiler, which is why it was not necessary to handle this when we added a custom attribute before.

Now that the new type is used in the operation, let's see how the operation looks in an IR:

```MLIR
module {
    func.func @main(%arg0: tensor<1x16x3x3xf16>, %arg1: !IE.GettingStartedTensor<numElements = 10, elementType = f16>) -> tensor<1x16x3x3xf16> {
        %0 = IE.GettingStarted(%arg0, %arg1) {
            gettingStarted = #IE.GettingStarted<boolParameter = true, floatParameter = 1.0>,
            myInt = 5 : i64,
            myIntArray = [1, 2, 3]
        } : tensor<1x16x3x3xf16>, !IE.GettingStartedTensor<numElements = 10, elementType = f16> -> tensor<1x16x3x3xf16>
        return %0 : tensor<1x16x3x3xf16>
    }
}
```

The second operand `%arg1` is introduced with the `!IE.GettingStartedTensor` type. Custom types are printed with an exclamation point as a prefix. The type of the operand is present in both appearances of the value: in the function's arguments and in the operations's list of operand types.

Tensor types are not the only commonly-used types in MLIR. One useful category of types is buffer types. Their semantics are generally meant to represent regions of memory, which can be allocated, deallocated etc. You can read more about this by checking the official documentation of [MemRefType](https://mlir.llvm.org/docs/Dialects/Builtin/#memreftype), which is one such buffer type.

---

## Interfaces

Another powerful concept in MLIR is interfaces. It offers a layer of abstraction over concepts in IR which allows handling them without the knowledge of which specific concept is being treated. Interfaces can be added to dialects, operations, types and attributes. For the purpose of this guide, only operation interfaces will be detailed, but interfaces for other concepts follow the same idea.

Operation interfaces represent interfaces that can be added to operations. These operations can then be converted to the interface and make use of the methods it offers. For example, you might not care if you are dealing with a Convolution or a Depthwise Convolution in an IR, just that they are both operations that have weights as an operand. An interface could be created and added to both operations which offers a method to return the weights operand. You could then iterate over all of the operations which implement this interface and make use of the method to work with the weights, without having to adapt the code explicitly for both operations. The same code would also be functional for other operations that implement the interface, without any changes.

Let's see how an operation interface works in practice. Introduce the following code into [IE/ops_interfaces.td](../../tblgen/vpux/compiler/dialect/IE/ops_interfaces.td):

```MLIR
def IE_GettingStartedOpInterface : OpInterface<"GettingStartedOpInterface"> {
    let description = "Simple interface used as an example";

    let cppNamespace = "vpux::IE";

    let methods = [
        InterfaceMethod<"The first method of the interface",
            "int", "firstMethod",
            (ins "int":$value)
        >,
        InterfaceMethod<"The second method of the interface, having a default implementation",
            "bool", "secondMethod", (ins),
            [{}],
            [{
                return true;
            }]
        >
    ];
}
```

The following code is generated in `IE/generated/ops_interfaces.hpp.inc` after building the project:

```C++
namespace vpux {
namespace IE {

// Forward declarations of interface classes as well as some inner details regarding the interface implementation
class GettingStartedOpInterface;
namespace detail {
struct GettingStartedOpInterfaceInterfaceTraits {
  // Contains multiple inner classes (concepts and models) that have specific uses
  // These are outside of the scope of this guide, but more information can be found here: https://mlir.llvm.org/docs/Interfaces/#external-models-for-attribute-operation-and-type-interfaces
  // ...
};
template <typename ConcreteOp>
struct GettingStartedOpInterfaceTrait;
} // namespace detail

// The main class of the interface, which inherits from `mlir::OpInterface`
class GettingStartedOpInterface : public ::mlir::OpInterface<GettingStartedOpInterface, detail::GettingStartedOpInterfaceInterfaceTraits> {
public:
  // ...

  // Useful structure that is used by operations when they implement interfaces
  template <typename ConcreteOp>
  struct Trait : public detail::GettingStartedOpInterfaceTrait<ConcreteOp> {};

  // The methods of the interface
  void firstMethod(int value);
  bool secondMethod();
};

// Contains the default implementations for the relevant methods
namespace detail {
  template <typename ConcreteOp>
  struct GettingStartedOpInterfaceTrait : public ::mlir::OpInterface<GettingStartedOpInterface, detail::GettingStartedOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    bool secondMethod() {
      return true;
    }
  };
}// namespace detail

// More definition are present here for the concepts and models that were mentioned above
// ...

} // namespace IE
} // namespace vpux
```

The code might seem a bit confusing, but in essence it contains the methods that were specified in TableGen and the default implementations for the methods that have them. Let's see how this interface looks when it is attached to an operation. Add the interface to the GettingStarted operation in [IE/ops.td](../../tblgen/vpux/compiler/dialect/IE/ops.td):

``` MLIR
def IE_GettingStartedOp :
        IE_Op<
            "GettingStarted",
            // The list of interfaces is added here
            // `DeclareOpInterfaceMethods` is used in order to have the declarations of the interface methods present in the class of the operation
            // Note: methods that have a default implementation will not be present unless explicitly included; e.g.:
            //    DeclareOpInterfaceMethods<IE_GettingStartedOpInterface, ["secondMethod"]>
            [
                DeclareOpInterfaceMethods<IE_GettingStartedOpInterface>
            ]
        > {
    let summary = "Simple layer used as an example";

    let arguments = (ins
        AnyRankedTensor:$input,
        IE_GettingStartedTensor:$secondInput,

        IntAttr:$myInt,
        I64ArrayAttr:$myIntArray,
        OptionalAttr<F64Attr>:$myOptionalFloat,
        DefaultValuedAttr<IntAttr, "10">:$myDefaultValuedInt,
        IE_GettingStartedAttr:$gettingStarted
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];

    let extraClassDeclaration = [{
        static void customStaticMethod() {
            // Do something interesting
        }

        bool customMethod(int x);
    }];

    let builders = [
        OpBuilder<
            (ins "mlir::Type":$type, "mlir::Value":$input, "mlir::IntegerAttr":$myInt,
                 "mlir::ArrayAttr":$myIntArray, "vpux::IE::GettingStartedAttr":$gettingStarted)
        >
    ];
}
```

When looking at the generated class of the operation in `IE/generated/ops.hpp.inc`, the following changes can be seen:

```C++
// The operation inherits from `IE::GettingStartedOpInterface::Trait`
class GettingStartedOp : public ::mlir::Op<GettingStartedOp, ..., vpux::IE::GettingStartedOpInterface::Trait> {
  // ...

  // The declaration of the interface method that has no default implementation
  int firstMethod(int value);

  // ...
};
```

In order for the project to build, it is also necessary to provide an implementation to the method without a default implementation. This can be done in `src/vpux_compiler/src/dialect/IE/ops/getting_started.cpp`:

```C++
int vpux::IE::GettingStartedOp::firstMethod(int value) {
    return value * 2;
}
```

The interface mechanism also allows adding static methods (using `StaticInterfaceMethod`), custom code (using `extraClassDeclaration`) or custom verifiers (using `verify`). Interfaces can also inherit from one or more interfaces as well. They can also be attached dynamically to existing concepts, without modifying their definition, by making use of the model classes generated in the interface trait; more information can be found [here](https://mlir.llvm.org/docs/Interfaces/#external-models-for-attribute-operation-and-type-interfaces).

Many useful interfaces come builtin with MLIR. One of the most useful ones is called `InferTypeOpInterface` (and its counterpart `InferShapedTypeOpInterface` for `mlir::ShapedType`). This interface allows the operation to infer its return type(s) upon creation, based on the given operands and attributes. Without this interface, the return type would have to be manually computed / specified every time an operation is created. It also comes with methods to check the inferred type against the one set into the return value, in case the type was set manually, as well as a method to decide if and what discrepancy between the inferred and actual return type is allowed.

Let's add this interface to our operation in [IE/ops.td](../../tblgen/vpux/compiler/dialect/IE/ops.td):

```MLIR
def IE_GettingStartedOp :
        IE_Op<
            "GettingStarted",
            [
                DeclareOpInterfaceMethods<IE_GettingStartedOpInterface>,
                // Add the interface to the operation
                DeclareOpInterfaceMethods<InferTypeOpInterface>
            ]
        > {
    let summary = "Simple layer used as an example";

    let arguments = (ins
        AnyRankedTensor:$input,
        IE_GettingStartedTensor:$secondInput,

        IntAttr:$myInt,
        I64ArrayAttr:$myIntArray,
        OptionalAttr<F64Attr>:$myOptionalFloat,
        DefaultValuedAttr<IntAttr, "10">:$myDefaultValuedInt,
        IE_GettingStartedAttr:$gettingStarted
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];

    let extraClassDeclaration = [{
        static void customStaticMethod() {
            // Do something interesting
        }

        bool customMethod(int x);
    }];

    let builders = [
        OpBuilder<
            (ins "mlir::Type":$type, "mlir::Value":$input, "mlir::IntegerAttr":$myInt,
                 "mlir::ArrayAttr":$myIntArray, "vpux::IE::GettingStartedAttr":$gettingStarted)
        >
    ];
}
```

The `inferReturnTypes` method can then be defined in `src/vpux_compiler/src/dialect/IE/ops/getting_started.cpp`:

```C++
mlir::LogicalResult vpux::IE::GettingStartedOp::inferReturnTypes(
        mlir::MLIRContext* context, llvm::Optional<mlir::Location> location, mlir::ValueRange operands,
        mlir::DictionaryAttr attributes, mlir::RegionRange /*regions*/,
        llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    // Locations are a useful debugging feature that allows tracing the history of an operation
    // Since the parameter is optional, an unknown location object is created in case no value is set
    const auto loc = location.value_or(mlir::UnknownLoc::get(context));

    // Every operation has an adaptor class created that can be used in instances such as this one,
    // where not all of the operation's information is known (here, the return types)
    // The adaptor offers an interface similar to the main operation class, with accessors to the operands
    // and attributes of the operation. Additionally, it contains a verify method that checks whether
    // the constraints of the operation are satisfied. If the constraints are not satisfied, the return
    // type might not be inferrable so a failure is returned.
    IE::GettingStartedOpAdaptor gettingStartedOp(operands, attributes);
    if (mlir::failed(gettingStartedOp.verify(loc))) {
        return mlir::failure();
    }

    // `input()` returns a `mlir::Value` object from which we extract the `mlir::Type`
    // This type is then passed to the output for this particular example operation, but the creation of
    // the output type varies from operation to operation
    const auto inputType = gettingStartedOp.input().getType();
    inferredReturnTypes.push_back(inputType);

    // Success is returned when the return type has been inferred
    return mlir::success();
}
```

After the interface has been added to the operation, variants of builders have also been created that receive no return type as a parameter. An example from the `IE/generated/ops.hpp.inc` file:

```C++
  // Builder with `mlir::Type output` as a parameter
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::Value secondInput, mlir::IntegerAttr myInt, ::mlir::ArrayAttr myIntArray, /*optional*/::mlir::FloatAttr myOptionalFloat, mlir::IntegerAttr myDefaultValuedInt, vpux::IE::GettingStartedAttr gettingStarted);
  // Builder without `mlir::Type output` as a parameter
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value secondInput, mlir::IntegerAttr myInt, ::mlir::ArrayAttr myIntArray, /*optional*/::mlir::FloatAttr myOptionalFloat, mlir::IntegerAttr myDefaultValuedInt, vpux::IE::GettingStartedAttr gettingStarted);
```

These builders internally call the `inferReturnTypes` method of the operation, as can be seen in the `IE/generated/ops.cpp.inc` file:

```C++
void GettingStartedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value secondInput, mlir::IntegerAttr myInt, ::mlir::ArrayAttr myIntArray, /*optional*/::mlir::FloatAttr myOptionalFloat, mlir::IntegerAttr myDefaultValuedInt, vpux::IE::GettingStartedAttr gettingStarted) {
  // ...
  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(GettingStartedOp::inferReturnTypes(odsBuilder.getContext(),
                odsState.location, odsState.operands,
                odsState.attributes.getDictionary(odsState.getContext()),
                /*regions=*/{}, inferredReturnTypes)))
    odsState.addTypes(inferredReturnTypes);
  else
    ::llvm::report_fatal_error("Failed to infer result type(s).");
}
```

We will see in the following section how these builders are used.

---

## Passes

Passes are a fundamental part of any compiler, as they perform the transformations and optimizations over IRs. In MLIR, passes run on operations and they can only change the operations they target. In other words, they should not affect the surrounding IR of the operation. For example, a Module pass would be able to affect anything in the IR since the module is the outer-most part of the IR (i.e. `module { ... }` in IR). A Function pass however should only affect the function operation it targets and its inner operations, not the operations outside (i.e. only `func.func ... { ... }` in IR).

Let's see an example of a Function pass in practice. Add the following section in the [IE/passes.td](../../tblgen/vpux/compiler/dialect/IE/passes.td) file:

```MLIR
// Creates a `HandleGettingStarted` pass with the argument name `handle-getting-started`, which operates
// over operations of type `mlir::func::FuncOp`
// The argument name will be used when working with tools such as `vpux-opt`, to invoke the pass
def HandleGettingStarted : PassBase<"handle-getting-started", "mlir::OperationPass<mlir::func::FuncOp>"> {
    let summary = "Example pass that works with the GettingStarted operation";

    // Every pass has a constructor which describes how a default instance of the pass can be created
    let constructor = "vpux::IE::createHandleGettingStartedPass()";

    // The pass works with operations, attributes and types from the IE dialect, so they are a dependency
    // that has to be registered for the pass to function
    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}
```

The constructor that was mentioned in TableGen format also has to be declared & defined. It can be declared in the [IE/transforms/passes.hpp](../../src/vpux_compiler/include/vpux/compiler/dialect/IE/transforms/passes.hpp) file:

```C++
// This declaration has to be added before `IE/passes.hpp.inc` is included in the same file,
// since the generated pass code makes use of it (see `IE/generated/passes.hpp.inc` for details)
std::unique_ptr<mlir::Pass> createHandleGettingStartedPass();
```

Now, let's define the pass itself. The TableGen code provided the base class (`HandleGettingStartedBase`) which can be used to define the logic of the pass. Create a file `src/vpux_compiler/src/dialect/IE/passes/handle_getting_started.cpp` with the following content:

```C++
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include <mlir/IR/BuiltinAttributes.h>

using namespace vpux;

namespace {

// The main class of the pass, inheriting from the base class that was generated
class HandleGettingStartedPass : public IE::HandleGettingStartedBase<HandleGettingStartedPass> {
public:
    explicit HandleGettingStartedPass() {
    }

private:
    // The entrypoint into the pass. Since this is a Function pass (i.e. specialized for `mlir::func::FuncOp`),
    // this function will be called on every instance of `mlir::func::FuncOp` in the IR
    void runOnOperation() final {
        // Get the `mlir::func::FuncOp` operation that the pass was called on
        auto func = getOperation();

        // Iterate over all `IE::GettingStartedOp` operations in the function op
        func.walk([](IE::GettingStartedOp op) {
            // Get the value of the `myInt` attribute and print it
            auto intValue = op.myInt();
            std::cout << intValue << std::endl;

            // Create a new mlir::IntegerAttr object with a 64-bit type and value 100
            auto intType = mlir::IntegerType::get(op.getContext(), 64);
            int64_t newIntValue = 100;
            auto newIntAttr = mlir::IntegerAttr::get(intType, newIntValue);

            // Set the new value attribute into the operation
            op.myIntAttr(newIntAttr);
        });
    }
};

}  // namespace

// The implementation for the previously-declared default constructor
std::unique_ptr<mlir::Pass> vpux::IE::createHandleGettingStartedPass() {
    return std::make_unique<HandleGettingStartedPass>();
}
```

With the pass defined, we can now execute it over an IR. Set the following content to a `getting_started.mlir` file:

```MLIR
module {
    func.func @main(%arg0: tensor<1x16x3x3xf16>, %arg1: !IE.GettingStartedTensor<numElements = 10, elementType = f16>) -> tensor<1x16x3x3xf16> {
        %0 = IE.GettingStarted(%arg0, %arg1) {
            gettingStarted = #IE.GettingStarted<boolParameter = true, floatParameter = 1.0>,
            myInt = 5 : i64,
            myIntArray = [1, 2, 3]
        } : tensor<1x16x3x3xf16>, !IE.GettingStartedTensor<numElements = 10, elementType = f16> -> tensor<1x16x3x3xf16>
        return %0 : tensor<1x16x3x3xf16>
    }
}
```

As can be seen, `myInt` has value 5. Now let's run the pass using `vpux-opt` by passing the `--handle-getting-started` option (the argument name set in TableGen for the pass):

```sh
./vpux-opt --vpu-arch=NPU37XX --handle-getting-started debug/getting_started.mlir
```

The following content will be printed:

```
5
module {
  func.func @main(%arg0: tensor<1x16x3x3xf16>, %arg1: !IE.GettingStartedTensor<numElements = 10, elementType = f16>) -> tensor<1x16x3x3xf16> {
    %0 = IE.GettingStarted(%arg0, %arg1) {gettingStarted = #IE.GettingStarted<boolParameter = true, floatParameter = 1.000000e+00 : f64>, myInt = 100 : i64, myIntArray = [1, 2, 3]} : tensor<1x16x3x3xf16>, !IE.GettingStartedTensor<numElements = 10, elementType = f16> -> tensor<1x16x3x3xf16>
    return %0 : tensor<1x16x3x3xf16>
  }
}
```

The input IR has only one `IE.GettingStarted` operation, so the `walk` function will only iterate over it. Value `5` is printed, showing the original value of the `myInt` attribute. Then, the output IR is printed, showing that the value of `myInt` has changed to 100.

### Rewriters

The previous example was a small pass where we manually iterate over some operations. But MLIR offers more powerful features for working with IRs. One such feature is pattern rewriters. They consist of two parts:
- pattern definition: specifies what operations should be matched and how to transform them;
- pattern application: drives how the pattern will be applied over the operations in the IR.

Let's see this with an example. Replace the code pass with the following:

```C++
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

// Rewriter for the IE::GettingStarted operation
// For this example, `mlir::OpRewritePattern` is used to match one operation type. There is however the option to match
// an interface using `OpInterfaceRewritePattern`
class GettingStartedOpRewriter final : public mlir::OpRewritePattern<IE::GettingStartedOp> {
public:
    GettingStartedOpRewriter(mlir::MLIRContext* ctx): mlir::OpRewritePattern<IE::GettingStartedOp>(ctx) {
    }

private:
    // The main logic of the pattern rewriter, containing the logic that matches the relevant operations
    // and applies some transformations on top of it
    // The purpose of this rewriter is to identify instances of `IE::GettingStartedOp` where the `myInt` attribute has a
    // positive value and decrement the value (creating a new operation each time) until the value zero is reached
    mlir::LogicalResult matchAndRewrite(IE::GettingStartedOp origOp, mlir::PatternRewriter& rewriter) const final {
        // Extracts the value of the `myInt` attribute
        auto intValue = origOp.myInt();

        // In case the value is already zero or negative, a failure is returned
        // Failure in the context of the pattern rewriter means that this operation is not a match - this pattern should
        // not apply any transformation over the operation, so the pattern applicator should not call this pattern over
        // the operation again
        // Note: `rewriter.notifyMatchFailure` can also be used to add a message with the reasoning
        if (intValue <= 0) {
            return mlir::failure();
        }

        // Decrements the value by one and prints it
        auto newIntValue = intValue - 1;
        std::cout << newIntValue << std::endl;

        // Creates a new integer attribute with the new value
        auto intType = mlir::IntegerType::get(origOp.getContext(), 64);
        auto newIntAttr = mlir::IntegerAttr::get(intType, newIntValue);

        // Creates a new `IE::GettingStarted` operation that replaces the original one. Internally, `replaceOpWithNewOp`
        // will call the builder of the operation that matches the given argument list
        rewriter.replaceOpWithNewOp<IE::GettingStartedOp>(origOp, origOp.input(), origOp.secondInput(), newIntAttr,
                                                          origOp.myIntArrayAttr(), origOp.myOptionalFloatAttr(),
                                                          origOp.myDefaultValuedIntAttr(), origOp.gettingStartedAttr());

        // The pattern rewriter has successfully transformed the IR
        return mlir::success();
    }
};

class HandleGettingStartedPass : public IE::HandleGettingStartedBase<HandleGettingStartedPass> {
public:
    explicit HandleGettingStartedPass() {
    }

private:
    void runOnOperation() final {
        auto func = getOperation();

        // Obtain the mlir::MLIRContext object; will be described in a follow-up section
        auto& ctx = getContext();

        // Creates a set of pattern rewriters that will be applied over the IR
        // In this case, only one rewriter is added
        mlir::RewritePatternSet patterns(&ctx);
        patterns.add<GettingStartedOpRewriter>(&ctx);

        // Configures the pattern applicator to apply the pattern in a top-down order
        mlir::GreedyRewriteConfig config;
        config.useTopDownTraversal = true;

        // Calls the pattern applicator over the function operation with the pattern set and
        // the previously-defined configuration
    if (mlir::failed(applyPatternsGreedily(
        func, std::move(patterns), config))) {
            // In case something went wrong, signal a failure to stop compilation
            signalPassFailure();
        }
    }
};

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createHandleGettingStartedPass() {
    return std::make_unique<HandleGettingStartedPass>();
}
```

By running `vpux-opt` over the same input IR, the following output is produced:

```
4
3
2
1
0
module {
  func.func @main(%arg0: tensor<1x16x3x3xf16>, %arg1: !IE.GettingStartedTensor<numElements = 10, elementType = f16>) -> tensor<1x16x3x3xf16> {
    %0 = IE.GettingStarted(%arg0, %arg1) {gettingStarted = #IE.GettingStarted<boolParameter = true, floatParameter = 1.000000e+00 : f64>, myInt = 0 : i64, myIntArray = [1, 2, 3]} : tensor<1x16x3x3xf16>, !IE.GettingStartedTensor<numElements = 10, elementType = f16> -> tensor<1x16x3x3xf16>
    return %0 : tensor<1x16x3x3xf16>
  }
}
```

This shows that the pattern has been applied a total of five times over the IR. It matched the original `IE::GettingStarted` operation that had `myInt = 5` and replaced it with a new operation containing `myInt = 4`. The same thing is repeated until an the operation with `myInt = 1` is replaced by an operation with `myInt = 0`. The pattern rewriter will then return failure for this case, since this operation is no longer a match for this pattern.

That is the way the greedy pattern rewrite driver works: it applies the registered patterns until a target point is reached or until a maximum number of iterations is reached (configurable using `mlir::GreedyRewriteConfig`). It also provides other features, such as making use of the benefit of the pattern rewriter to decide the application order. More information on this driver can be found [here](https://mlir.llvm.org/docs/PatternRewriter/#greedy-pattern-rewrite-driver).

As shown in the example, the pattern rewriters also come with an API which allows the user to more easily transform the IR. We have seen `replaceOpWithNewOp`, which creates a new operation and replaces an existing one with it, but there are more methods available: `eraseOp` to remove an operation that has no uses, `create` to create an operation, `replaceOp` to change the uses of an operation to some new values etc. More information on these methods can be found [here](https://mlir.llvm.org/docs/PatternRewriter/#pattern-rewriter). One important thing to remember is that all of the transformations inside a pattern rewriter should be done using the `mlir::PatternRewriter&` parameter of the `matchAndRewrite` function.

As a note, the example above uses the `matchAndRewrite` method of the pattern rewriter. It is also possible to use separate `match` and `rewrite` methods if that is desired. When using a single function, no change should take place until the match is considered successful, otherwise the IR might end up in an invalid state.

With that covered, let's see another type of pattern application driver: dialect conversion. This driver introduces the concept of legality of operations to identify what operations should be handled by the pattern rewriters. It can be used for converting between dialects or even within a dialect.

In order to use the dialect conversion driver, it is necessary to specify the conversion target which marks operations (or entire dialects) as:
- `illegal`: this type of operation should not be present in the IR and they must be converted if present for the pass to succeed
- `legal`: this type of operation is legal, so there is no need to apply any conversion for it
- `dynamically legal`: only some instances of a given operation type are legal, based on the given condition

Beside the conversion target, the conversion mode is also something that has to be determined. Two important conversion modes are:
- `full`: all operations in the IR have to be marked as legal for the conversion to be successful; after the IR conversion has completed, it should only contain known operations
- `partial`: only handles operations that are explicitly marked as legal or illegal; in other words, not all operations have to be added to the target as they can remain uncovered

To see this in practice, replace the pass code with the following:

```C++
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include <mlir/IR/BuiltinAttributes.h>

using namespace vpux;

namespace {

// Conversion rewriter for the IE::GettingStarted operation
// Similar to `mlir::OpPatternRewriter`, an equivalent for matching interfaces exists: `OpInterfaceConversionPattern`
class GettingStartedOpConversionRewriter final : public mlir::OpConversionPattern<IE::GettingStartedOp> {
public:
    GettingStartedOpConversionRewriter(mlir::MLIRContext* ctx): mlir::OpConversionPattern<IE::GettingStartedOp>(ctx) {
    }

public:
    // The main logic of the pattern rewriter, containing the transformations that should legalize the operation
    mlir::LogicalResult matchAndRewrite(IE::GettingStartedOp origOp, OpAdaptor /*newArgs*/,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        // Replaces the use of the operation's output with the operation's input, thus leaving the operation without
        // uses and foldable (i.e. removable in this case)
        rewriter.replaceOp(origOp, origOp.input());
        return mlir::success();
    }
};

class HandleGettingStartedPass : public IE::HandleGettingStartedBase<HandleGettingStartedPass> {
public:
    explicit HandleGettingStartedPass() {
    }

private:
    void runOnOperation() final {
        auto func = getOperation();

        // Obtain the mlir::MLIRContext object; will be described in a follow-up section
        auto& ctx = getContext();

        // Creates a conversion target and marks `IE::GettingStartedOp` as being legal only when the `myInt` attribute
        // is less or equal to zero
        mlir::ConversionTarget target(ctx);
        target.addDynamicallyLegalOp<IE::GettingStartedOp>([](IE::GettingStartedOp op) -> bool {
            return op.myInt() <= 0;
        });

        // Adds the conversion pattern to the set
        mlir::RewritePatternSet patterns(&ctx);
        patterns.add<GettingStartedOpConversionRewriter>(&ctx);

        // Applies a partial conversion for the given target using the pattern set
        if (mlir::failed(applyPartialConversion(func, target, std::move(patterns)))) {
            // In case something went wrong, signal a failure to stop compilation
            signalPassFailure();
        }
    }
};

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createHandleGettingStartedPass() {
    return std::make_unique<HandleGettingStartedPass>();
}
```

Executing the pass over the same input IR will produce the following output IR:

```
module {
  func.func @main(%arg0: tensor<1x16x3x3xf16>, %arg1: !IE.GettingStartedTensor<numElements = 10, elementType = f16>) -> tensor<1x16x3x3xf16> {
    return %arg0 : tensor<1x16x3x3xf16>
  }
}
```

Since the operation has a `myInt` value greater than zero, it is marked as illegal. The pattern applicator will run and remove the operation. In case the input IR has a `myInt` value smaller or equal than zero, the operation will be left untouched since it is already legal. The general way a pattern applicator works for conversion passes is by first identifying the illegal operations, apply the relevant patterns from the set and re-check the legality.

When executing `vpux-opt`, it is also possible to pass the `-debug-only=dialect-conversion` option, which can help trace all of the transformations that are done during conversions.

The dialect conversion driver also supports converting the types that an operation works with. To achieve this, it introduces a `TypeConverter` component which can describe when and how to convert types. For example, it might be necessary to introduce an operation that does the type conversion explicitly (e.g. from one element type to another) - this can be done using materialization. The `newArgs` parameter that is present `matchAndRewrite` methods for conversion pattern rewriters already contains the correct types in the values which is why they should be used in the transformations. More information on this topic can be found [here](https://mlir.llvm.org/docs/DialectConversion/#type-conversion).

### Canonicalizers

Pattern rewriters can also be applied to operations directly, in the form of canonicalizers. The canonicalizer pass is a commonly-used builtin pass which applies some globally applied rules and the operation pattern rewriters using a greedy driver. Some of the global rules are the elimination of operations that have no uses, constant folding etc.

To add a canonicalizer to an operation, the following flag should be set in TableGen. For the GettingStarted operation, this can be done in [IE/ops.td](../../tblgen/vpux/compiler/dialect/IE/ops.td):
```MLIR
def IE_GettingStartedOp :
        // ...
        > {
    // ...
    let hasCanonicalizer = 1;
}
```

This will add a `getCanonicalizationPatterns` method in the class declaration of the operation which has to be manually defined. Let's try using the same greedy rewriter from the previous example as a canonicalizer. Add the following code in `src/vpux_compiler/src/dialect/IE/IR/ops/getting_started.cpp`:

```C++
// The same rewriter used in a previous example, which decrements the `myInt` value until number zero is reached
// A new IE::GettingStarted operation is created for each decrement
class GettingStartedOpRewriter final : public mlir::OpRewritePattern<vpux::IE::GettingStartedOp> {
public:
    GettingStartedOpRewriter(mlir::MLIRContext* ctx): mlir::OpRewritePattern<vpux::IE::GettingStartedOp>(ctx) {
    }

private:
    mlir::LogicalResult matchAndRewrite(vpux::IE::GettingStartedOp origOp,
                                        mlir::PatternRewriter& rewriter) const final {
        auto intValue = origOp.myInt();
        if (intValue <= 0) {
            return mlir::failure();
        }

        auto newIntValue = intValue - 1;
        auto intType = mlir::IntegerType::get(origOp.getContext(), 64);
        auto newIntAttr = mlir::IntegerAttr::get(intType, newIntValue);

        rewriter.replaceOpWithNewOp<vpux::IE::GettingStartedOp>(
                origOp, origOp.input(), origOp.secondInput(), newIntAttr, origOp.myIntArrayAttr(),
                origOp.myOptionalFloatAttr(), origOp.myDefaultValuedIntAttr(), origOp.gettingStartedAttr());
        return mlir::success();
    }
};

// The implementation of the method declared by `hasCanonicalizer = 1`, in which the desired patterns are added to the set
void vpux::IE::GettingStartedOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                             mlir::MLIRContext* ctx) {
    patterns.add<GettingStartedOpRewriter>(ctx);
}
```

After building the project, the canonicalizer can be run by running the `canonicalize` pass:

```sh
./vpux-opt --vpu-arch=NPU37XX --canonicalize debug/getting_started.mlir
```

When using the same input IR as the other examples (with `myInt = 5`), the following IR will be generated which has `myInt = 0`.

```
module {
  func.func @main(%arg0: tensor<1x16x3x3xf16>, %arg1: !IE.GettingStartedTensor<numElements = 10, elementType = f16>) -> tensor<1x16x3x3xf16> {
    %0 = IE.GettingStarted(%arg0, %arg1) {gettingStarted = #IE.GettingStarted<boolParameter = true, floatParameter = 1.000000e+00 : f64>, myInt = 0 : i64, myIntArray = [1, 2, 3]} : tensor<1x16x3x3xf16>, !IE.GettingStartedTensor<numElements = 10, elementType = f16> -> tensor<1x16x3x3xf16>
    return %0 : tensor<1x16x3x3xf16>
  }
}
```

Canonicalizers are generally used for small changes that are operation specific and that might need to be applied multiple times (e.g. when simplifying an operation which reaches the non-simplified state multiple times as the compilation progresses).

Another specific type of canonicalizer is folding. It represents a more limited type of canonicalization which cannot create or erase operations, but it might replace the associated operation. In other words, folding can only affect the local operation that it is defined for. Compared to general canonicalizers, folding occurs in multiple places during compilation, not just when the canonicalizer pass executes. For example, when the `createOrFold` method runs or during dialect conversion. It can be enabled by using the `hasFolder` flag of the operation in TableGen and by defining the resulting `fold` method.

More information on canonicalization can be found [here](https://mlir.llvm.org/docs/Canonicalization/).

### Pipelines

So far, passes have been described in isolation. But during normal compilation, multiple passes are executed in a generally specific order. In MLIR, passes are grouped into pipelines and compilation can be done by executing pipelines over input IRs.

A pipeline may contain either passes or other pipelines. This allows related passes to be grouped by some feature (e.g. optimization pipeline containing only optimizations).

More information on pipelines can be found [here](https://mlir.llvm.org/docs/PassManagement/#pass-manager).

---

## MLIR Context

The MLIR Context is a core component of MLIR. It contains a registry of the dialects and their inner concepts, making them available for parts of the compilation by loading them as necessary. It also contains the actual storage data for immutable objects such as attributes or types.

In MLIR, an attribute or type cannot be changed after creation. New attributes or types can be created with the desired values and used instead of old ones, but the actual values of the existing ones are immutable. Once an attribute or type is created, it is stored into the context and referenced as needed for the duration of the compilation. Generally, the context exists for the entire duration of a compilation. This means that it will keep increasing in capacity as the compilation continues - that is why it is not recommended to create intermediate attributes or types that are not actually used in IRs.

---

## Implementations of concepts

In MLIR, all concepts use a wrapper class whose only member value is a pointer. In a previous example, we have seen `IE::detail::GettingStartedAttrStorage` being generated for a custom attribute; this class contains the actual data in the attribute. Internally, the `IE::GettingStartedAttr` class that is used throughout the code has a pointer to a storage object. The methods used in the wrapper class will dereference the pointer and access the data from the storage.

This detail is hidden from an MLIR user. But the implication is that copying these objects is very cheap. Therefore, it is recommended to pass such objects by-value instead of by-reference.

---

## More resources

After completing this guide, it is recommended to explore the [Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/) from MLIR.
The [official documentation](https://mlir.llvm.org/docs/) is also a great resource for the concepts that come with MLIR, providing more information on the items detailed above and those not covered.
