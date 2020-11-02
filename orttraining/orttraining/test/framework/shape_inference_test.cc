// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shape_inference.h"

#include "test/framework/model_builder_utils.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

using namespace modelbuilder;

TEST_F(ShapeInferenceTest, SoftmaxCrossEntropyMeanTest) {
  const int kNumClasses = 20;
  const int kBatchSize = 10;

  Type logits_type({kBatchSize, kNumClasses});
  Input("logits", logits_type);
  Type labels_type({kBatchSize, kNumClasses});
  Input("labels", labels_type);

  auto& node = Node("SoftmaxCrossEntropy", {"logits", "labels"}, {"loss", "log_prob"});
  AttributeProto reduction;
  reduction.set_name("reduction");
  reduction.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
  reduction.set_s("mean");
  node.AddAttribute("reduction", reduction);

  DoShapeInference();
  Shape loss_expected_shape;
  Shape log_prob_expected_shape({kBatchSize, kNumClasses});

  CheckShapeEquality(OutputShape(node, 0), &loss_expected_shape.value);
  CheckShapeEquality(OutputShape(node, 1), &log_prob_expected_shape.value);
}

TEST_F(ShapeInferenceTest, SoftmaxCrossEntropyNoReductionTest) {
  const int kNumClasses = 20;
  const int kBatchSize = 10;

  Type logits_type({kBatchSize, kNumClasses});
  Input("logits", logits_type);
  Type labels_type({kBatchSize, kNumClasses});
  Input("labels", labels_type);

  auto& node = Node("SoftmaxCrossEntropy", {"logits", "labels"}, {"loss", "log_prob"});
  AttributeProto reduction;
  reduction.set_name("reduction");
  reduction.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
  reduction.set_s("none");
  node.AddAttribute("reduction", reduction);

  DoShapeInference();
  Shape loss_expected_shape({kBatchSize});
  Shape log_prob_expected_shape({kBatchSize, kNumClasses});

  CheckShapeEquality(OutputShape(node, 0), &loss_expected_shape.value);
  CheckShapeEquality(OutputShape(node, 1), &log_prob_expected_shape.value);
}

TEST_F(ShapeInferenceTest, SoftmaxCrossEntropyGradTest) {
  const int kNumClasses = 20;
  const int kBatchSize = 10;

  Type dy_type; // Scalar;
  Input("dy", dy_type);
  Type logprob_type({kBatchSize, kNumClasses});
  Input("log_prob", logprob_type);
  Type labels_type({kBatchSize, kNumClasses});
  Input("labels", labels_type);

  auto& node = Node("SoftmaxCrossEntropyGrad", {"dy", "log_prob", "labels"}, {"d_logits"});

  DoShapeInference();
  Shape d_logits_expected_shape({kBatchSize, kNumClasses});

  CheckShapeEquality(OutputShape(node, 0), &d_logits_expected_shape.value);
}

TEST_F(ShapeInferenceTest, SoftmaxCrossEntropyLossGradTest) {
  const int kNumClasses = 20;
  const int kBatchSize = 10;

  Type dy_type;  // Scalar.
  Input("dy", dy_type);
  Type logprob_type({kBatchSize, kNumClasses});
  Input("log_prob", logprob_type);
  Type labels_type({kBatchSize}, TensorProto_DataType_INT32);
  Input("labels", labels_type);

  auto& node = Node("SoftmaxCrossEntropyLossGrad", {"dy", "log_prob", "labels"}, {"d_logits"});

  DoShapeInference();
  Shape d_logits_expected_shape({kBatchSize, kNumClasses});

  CheckShapeEquality(OutputShape(node, 0), &d_logits_expected_shape.value);
}

TEST_F(ShapeInferenceTest, LayerNormalizationGradTest) {
  Type y_grad_type({10, 20});
  Input("y_grad", y_grad_type);
  Type x_type({10, 20});
  Input("x", x_type);
  Type weight_type({20});
  Input("scale", weight_type);
  Input("mean", weight_type);
  Type inv_std_var_type;
  Input("inv_std_var", inv_std_var_type);

  auto& node = Node("LayerNormalizationGrad", {"y_grad", "x", "scale", "mean", "inv_std_var"}, {"X_grad", "scale_grad", "bias_grad"});

  DoShapeInference();
  Shape x_grad_expected_shape({10, 20});
  Shape scale_grad_expected_shape({20});
  Shape bias_grad_expected_shape({20});

  CheckShapeEquality(OutputShape(node, 0), &x_grad_expected_shape.value);
  CheckShapeEquality(OutputShape(node, 1), &scale_grad_expected_shape.value);
  CheckShapeEquality(OutputShape(node, 2), &bias_grad_expected_shape.value);
}

}  // namespace test
}  // namespace onnxruntim
