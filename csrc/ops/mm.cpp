#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include "../vulkan_allocator.h"
#include "../vulkan_context.h"
#include <ATen/Functions.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/CPUFallback.h>
#include <torch/library.h>

namespace torch_vulkan {
namespace ops {

// aten::mm — matrix multiplication via matmul.spv
// This is the first operator. Get this working and the rest follow the pattern.
at::Tensor mm(const at::Tensor& self, const at::Tensor& mat2) {
  TORCH_CHECK(self.dim() == 2, "mm: self must be 2D");
  TORCH_CHECK(mat2.dim() == 2, "mm: mat2 must be 2D");
  TORCH_CHECK(
      self.size(1) == mat2.size(0),
      "mm: self.size(1) must match mat2.size(0)");
  TORCH_CHECK(self.dtype() == at::kFloat, "mm: only float32 supported (for now)");

  int64_t M = self.size(0);
  int64_t K = self.size(1);
  int64_t N = mat2.size(1);

  // For the square matmul shader, K must equal N must equal M.
  // TODO: Generalize the shader for non-square matrices.
  // For now, pad to max dim and use the square kernel.
  int64_t dim = std::max({M, K, N});

  auto& ctx = VulkanContext::instance();
  auto& mgr = ctx.manager();

  // Get input data (ensure contiguous)
  auto self_c = self.contiguous();
  auto mat2_c = mat2.contiguous();

  // Create Kompute tensors from input data
  std::vector<float> a_data(dim * dim, 0.0f);
  std::vector<float> b_data(dim * dim, 0.0f);
  std::vector<float> c_data(dim * dim, 0.0f);

  // Copy input data into padded buffers
  const float* a_ptr = self_c.data_ptr<float>();
  const float* b_ptr = mat2_c.data_ptr<float>();
  for (int64_t i = 0; i < M; i++) {
    for (int64_t j = 0; j < K; j++) {
      a_data[i * dim + j] = a_ptr[i * K + j];
    }
  }
  for (int64_t i = 0; i < K; i++) {
    for (int64_t j = 0; j < N; j++) {
      b_data[i * dim + j] = b_ptr[i * N + j];
    }
  }

  auto tensor_a = mgr.tensor(a_data);
  auto tensor_b = mgr.tensor(b_data);
  auto tensor_c = mgr.tensor(c_data);

  // Load the matmul shader
  auto spirv = ctx.load_shader("matmul");

  // Push constant: N dimension
  uint32_t n_val = static_cast<uint32_t>(dim);

  // Workgroup dispatch: ceil(dim/16) x ceil(dim/16)
  uint32_t wg_x = (dim + 15) / 16;
  uint32_t wg_y = (dim + 15) / 16;

  // Record and execute
  auto algorithm = mgr.algorithm(
      {tensor_a, tensor_b, tensor_c},
      spirv,
      kp::Workgroup({wg_x, wg_y, 1}),
      {},                                    // specialization constants
      std::vector<uint32_t>{n_val});         // push constants

  auto seq = mgr.sequence();
  seq->record<kp::OpTensorSyncDevice>({tensor_a, tensor_b});
  seq->record<kp::OpAlgoDispatch>(algorithm);
  seq->record<kp::OpTensorSyncLocal>({tensor_c});
  seq->eval();

  // Extract result into a new PyTorch tensor
  auto output = at::empty({M, N}, self.options());
  float* out_ptr = output.data_ptr<float>();
  const float* c_ptr = tensor_c->data();
  for (int64_t i = 0; i < M; i++) {
    for (int64_t j = 0; j < N; j++) {
      out_ptr[i * N + j] = c_ptr[i * dim + j];
    }
  }

  return output;
}

// aten::add.Tensor — elementwise addition
at::Tensor add(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  // TODO: Implement via Vulkan compute shader
  // For now, fall back to CPU
  auto cpu_result =
      at::add(self.to(at::kCPU), other.to(at::kCPU), alpha);
  return cpu_result.to(self.device());
}

// aten::empty.memory_format — tensor creation on Vulkan device
at::Tensor empty_memory_format(
    at::IntArrayRef size,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory,
    std::optional<at::MemoryFormat> memory_format) {
  auto actual_dtype = dtype.value_or(at::kFloat);
  auto allocator = &VulkanAllocator::instance();

  int64_t nelements = 1;
  for (auto s : size) nelements *= s;
  size_t nbytes = nelements * at::elementSize(actual_dtype);

  auto storage = c10::Storage(
      c10::Storage::use_byte_size_t(),
      nbytes,
      allocator->allocate(nbytes),
      allocator,
      true);

  auto tensor = at::detail::make_tensor<c10::TensorImpl>(
      std::move(storage),
      c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
      at::scalarTypeToTypeMeta(actual_dtype));

  tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);

  return tensor;
}

// CPU fallback for unimplemented ops
void cpu_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  at::native::cpu_fallback(op, stack);
}

// Register all operators with the PrivateUse1 dispatch key
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("mm", &mm);
  m.impl("add.Tensor", &add);
  m.impl("empty.memory_format", &empty_memory_format);
}

// Catch-all: any unregistered op falls back to CPU
TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&cpu_fallback>());
}

} // namespace ops
} // namespace torch_vulkan
