// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shape_inference.h"

#include "test/framework/model_builder_utils.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

using namespace modelbuilder;

TEST_F(ShapeInferenceTest, BasicTest) {
  Type type1({1, 50, 100});
  Input("X1", type1);

  auto& node = Node("Cast", {"X1"}, {"Y1"});
  //AttributeProto squeezed_axes;
  //squeezed_axes.set_name("axes");
  //squeezed_axes.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INTS);
  //squeezed_axes.add_ints(0);
  //p_node->AddAttribute("axes", squeezed_axes);
  AttributeProto cast_to;
  cast_to.set_name("to");
  cast_to.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  cast_to.set_i(ONNX_NAMESPACE::TensorProto_DataType_INT32);
  //cast_to.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
  //cast_to.set_s("INT16");
  node.AddAttribute("to", cast_to);

  DoShapeInference();
  // check inferred shapes
  Shape expected_shape({1, 50, 100});
  CheckShapeEquality(OutputShape(node), &expected_shape.value);
  CheckShapeEquality(InputShape(node), OutputShape(node));
}

}  // namespace test
}  // namespace onnxruntime
