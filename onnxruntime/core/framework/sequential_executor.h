// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <unordered_map>
#include "core/common/common.h"
#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/framework/iexecutor.h"
#include "core/framework/framework_common.h"
#include "core/framework/ml_value.h"
#include "core/framework/session_state.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/op_kernel_context_internal.h"

namespace onnxruntime {
class SequentialExecutor : public IExecutor {
 public:
  SequentialExecutor(const bool& terminate_flag = false, const bool only_execute_path_to_fetches = false)
      : terminate_flag_{terminate_flag}, only_execute_path_to_fetches_(only_execute_path_to_fetches), start_range_times_(100000, 0) {}

  common::Status Execute(const SessionState& session_state, const std::vector<int>& feed_mlvalue_idxs,
                         const std::vector<OrtValue>& feeds, const std::vector<int>& fetch_mlvalue_idxs,
                         std::vector<OrtValue>& fetches,
                         const std::unordered_map<size_t, CustomAllocator>& fetch_allocators,
                         const logging::Logger& logger) override;

  ~SequentialExecutor() {
    std::cout << "Start Range timings: " << std::endl;
    for (int i = 0; i <= counter_; ++i) {
      std::cout << i << " "  << start_range_times_[i] << std::endl;
    }
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SequentialExecutor);
  const bool& terminate_flag_;
  const bool only_execute_path_to_fetches_;

  int counter_ = -1;
  std::vector<int64_t> start_range_times_;

};
}  // namespace onnxruntime
