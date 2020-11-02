// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shape_inference.h"

#include "core/graph/model.h"
#include "core/common/logging/logging.h"
#include "test/framework/model_builder_utils.h"
#include "test/test_environment.h"

#include <string>
#include <unordered_map>

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::logging;

namespace onnxruntime {
namespace test {

using namespace modelbuilder;

ShapeInferenceTest::ShapeInferenceTest() : model_("Test", false, DefaultLoggingManager().DefaultLogger()), node_count_(0) {}

void ShapeInferenceTest::Input(const std::string& name, const Type& type) {
  name_to_arg_[name] = onnxruntime::make_unique<onnxruntime::NodeArg>(name, &type.value);
}

onnxruntime::NodeArg* ShapeInferenceTest::Arg(const std::string& name) {
  if (name_to_arg_.count(name) == 0)
    name_to_arg_[name] = onnxruntime::make_unique<onnxruntime::NodeArg>(name, nullptr);
  return name_to_arg_[name].get();
}

onnxruntime::Node& ShapeInferenceTest::Node(const std::string& op, const std::vector<std::string>& inputs, const std::vector<std::string>& outputs) {
  std::vector<onnxruntime::NodeArg*> input_args;
  std::vector<onnxruntime::NodeArg*> output_args;
  for (const std::string& input : inputs) {
    input_args.push_back(Arg(input));
  }
  for (const std::string& output : outputs) {
    output_args.push_back(Arg(output));
  }

  int num = node_count_++;
  return model_.MainGraph().AddNode("node" + std::to_string(num), op, "test op", input_args, output_args);
}

void ShapeInferenceTest::DoShapeInference() {
  auto status = model_.MainGraph().Resolve();
  EXPECT_TRUE(status.IsOK()) << "Graph resolve failed: " << status.ErrorMessage();
}

const TensorShapeProto* ShapeInferenceTest::InputShape(onnxruntime::Node& node, int arg_num) {
  return node.InputDefs()[arg_num]->Shape();
}

const TensorShapeProto* ShapeInferenceTest::OutputShape(onnxruntime::Node& node, int arg_num) {
  return node.OutputDefs()[arg_num]->Shape();
}

void ShapeInferenceTest::CheckShapeEquality(const TensorShapeProto* shape1, const TensorShapeProto* shape2) {
  EXPECT_NE(shape1, nullptr);
  EXPECT_NE(shape2, nullptr);
  if ((shape1 != nullptr) && (shape2 != nullptr)) {
    EXPECT_EQ(shape1->dim_size(), shape2->dim_size()) << "Shapes do not have same rank";
    auto min_dims = std::min(shape1->dim_size(), shape2->dim_size());
    for (int i = 0; i < min_dims; ++i) {
      auto dim1 = shape1->dim(i);
      auto dim2 = shape2->dim(i);
      EXPECT_EQ(dim1.has_dim_value(), dim2.has_dim_value());
      if (dim1.has_dim_value()) {
        EXPECT_EQ(dim1.dim_value(), dim2.dim_value());
      }
      EXPECT_EQ(dim1.has_dim_param(), dim2.has_dim_param());
      if (dim1.has_dim_param()) {
        EXPECT_EQ(dim1.dim_param(), dim2.dim_param());
      }
    }
  }
}

}  // namespace test
}  // namespace onnxruntime
