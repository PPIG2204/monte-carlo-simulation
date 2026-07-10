import numpy as np
from typing import Callable

class ImportanceSampler:
    """Class triển khai phương pháp Importance Sampling để giảm phương sai."""
    
    def __init__(self, target_func: Callable[[float], float], a: float, b: float):
        self.f = target_func
        self.a = a
        self.b = b

    def crude_monte_carlo(self, num_samples: int):
        """Tính tích phân bằng phân phối đều (Crude MC) để làm đối chứng."""
        x = np.random.uniform(self.a, self.b, num_samples)
        f_values = self.f(x)
        estimate = (self.b - self.a) * np.mean(f_values)
        variance = np.var((self.b - self.a) * f_values) / num_samples
        return estimate, variance, f_values

    def importance_sampling_exp(self, num_samples: int):
        """
        Tính tích phân bằng tầm quan trọng (Importance Sampling).
        Chọn hàm đề xuất g(x) là phân phối Mũ (Exponential Distribution) truncated trên [a, b].
        """
        # Sinh mẫu tuân theo phân phối mũ g(x) trong đoạn [a, b]
        # Sử dụng phương pháp biến đổi nghịch đảo (Inverse Transform Sampling)
        u = np.random.uniform(0, 1, num_samples)
        # Hàm mật độ tích lũy của phân phối mũ f(x)=e^-x trên [0, 5]
        # X = -ln(1 - u * (1 - e^-5))
        x_samples = -np.log(1 - u * (1 - np.exp(-self.b)))
        
        # Hàm mật độ xác suất g(x) đã chuẩn hóa để tổng diện tích = 1
        g_values = np.exp(-x_samples) / (1 - np.exp(-self.b))
        
        # Tính toán giá trị hàm mục tiêu f(x)
        f_values = self.f(x_samples)
        
        # Trọng số tầm quan trọng w(x) = f(x) / g(x)
        weights = f_values / g_values
        
        # Giá trị ước lượng tích phân và phương sai mẫu của toán tử
        estimate = np.mean(weights)
        variance = np.var(weights) / num_samples
        
        return estimate, variance, weights