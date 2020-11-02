// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>
#include <memory>

#include "gtest/gtest.h"
#include "core/graph/model.h"
#include "test/framework/model_builder_utils.h"

namespace onnxruntime {
namespace test {

class ShapeInferenceTest : public ::testing::Test {
 protected:
  onnxruntime::Model model_;
  int node_count_ = 0;
  std::unordered_map<std::string, std::unique_ptr<onnxruntime::NodeArg>> name_to_arg_;

 public:
  ShapeInferenceTest();

  void Input(const std::string& name, const modelbuilder::Type& type);

  onnxruntime::NodeArg* Arg(const std::string& name);

  onnxruntime::Node& Node(const std::string& op, const std::vector<std::string>& input, const std::vector<std::string>& output);

  void DoShapeInference();

  const ONNX_NAMESPACE::TensorShapeProto* InputShape(onnxruntime::Node& node, int arg_num = 0);

  const ONNX_NAMESPACE::TensorShapeProto* OutputShape(onnxruntime::Node& node, int arg_num = 0);

  void CheckShapeEquality(const ONNX_NAMESPACE::TensorShapeProto* shape1, const ONNX_NAMESPACE::TensorShapeProto* shape2);
};

}  // namespace test
}  // namespace onnxruntime
