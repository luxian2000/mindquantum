import mindspore as ms
import mindspore.numpy as mnp
from mindspore import Tensor, ops
import numpy as np

# --- 创建复数张量的可靠方法 ---

# 方法 1: 直接从 Python 复数列表或 NumPy 复数数组创建
complex_list = [1.0 + 0.5j, 2.0 + 1.5j]
complex_tensor_1 = Tensor(complex_list, dtype=ms.complex64) # 明确指定 dtype
print(f"Method 1 - From list: {complex_tensor_1}, dtype: {complex_tensor_1.dtype}")

np_complex_array = np.array([3.0 + 0.2j, 4.0 - 1.1j], dtype=np.complex64)
complex_tensor_2 = Tensor(np_complex_array) # dtype 会自动推断
print(f"Method 2 - From numpy array: {complex_tensor_2}, dtype: {complex_tensor_2.dtype}")

# 方法 2: 分别创建实部和虚部，然后组合成复数
real_part = Tensor([1.0, 2.0], ms.float32)
imag_part = Tensor([0.5, 1.5], ms.float32)
# 使用 1j (MindSpore 中的虚数单位) 来构造复数
complex_tensor_3 = real_part + 1j * imag_part
print(f"Method 3 - Combine real and imag: {complex_tensor_3}, dtype: {complex_tensor_3.dtype}")

# --- 自动微分示例 ---
from mindspore import value_and_grad

# 定义一个简单的复数函数，例如 f(z) = |z|^2 的总和 (输出为实数)
def complex_func(z):
    # 计算复数张量的模长平方 (返回实数)
    # mnp.abs(z) 会返回实数张量
    modulus_sq = mnp.sum(mnp.abs(z)**2)
    return modulus_sq

# 使用其中一个创建的复数张量作为输入
z_input = complex_tensor_1 # 或 complex_tensor_2 或 complex_tensor_3

# 计算函数值和梯度
func_value, gradient = value_and_grad(complex_func, grad_position=0)(z_input)

print(f"\nInput Tensor: {z_input}")
print(f"Function Value (real scalar): {func_value}")
print(f"Gradient w.r.t. input (complex tensor): {gradient}")
print(f"Gradient Dtype: {gradient.dtype}")

# 验证梯度是否也是复数
assert gradient.dtype in [ms.complex64, ms.complex128], "Gradient should be complex!"
