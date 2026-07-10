import numpy as np
from typing import Callable

class MonteCarloIntegrator:
    """Class thực hiện tính tích phân bằng phương pháp Monte Carlo thông thường."""
    
    def __init__(self, func: Callable, a: float, b: float, c: float = None, d: float = None):
        """
        Args:
            func: Hàm số cần tính tích phân. 1 chiều: f(x), 2 chiều: f(x, y).
            a, b: Cận dưới và cận trên của biến x.
            c, d: Cận dưới và cận trên của biến y (chỉ dùng cho tích phân 2 chiều).
        """
        self.func = func
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def integrate_1d(self, num_samples: int):
        """Tính tích phân 1 chiều bằng phương pháp Mean Value (Giữ nguyên như cũ)."""
        x_samples = np.random.uniform(self.a, self.b, num_samples)
        f_values = self.func(x_samples)
        integral_value = (self.b - self.a) * np.mean(f_values)
        
        check_points = np.unique(np.logspace(1, np.log10(num_samples), num=200, dtype=int))
        history_n = []
        history_estimates = []
        cumulative_f = np.cumsum(f_values)
        for n in check_points:
            integral_n = (self.b - self.a) * (cumulative_f[n-1] / n)
            history_n.append(n)
            history_estimates.append(integral_n)
            
        return integral_value, history_n, history_estimates

    def integrate_2d(self, num_samples: int):
        """
        [BỔ SUNG MỚI] Tính tích phân 2 chiều trên miền [a, b] x [c, d].
        Công thức: I = (b - a) * (d - c) * E[f(X, Y)]
        """
        if self.c is None or self.d is None:
            raise ValueError("Cần cung cấp cận c và d cho tích phân 2 chiều!")
            
        # Sinh mẫu ngẫu nhiên độc lập cho x và y tuân theo phân phối đều
        x_samples = np.random.uniform(self.a, self.b, num_samples)
        y_samples = np.random.uniform(self.c, self.d, num_samples)
        
        # Tính giá trị hàm f(x, y) trên toàn bộ các cặp mẫu (Vectorized)
        f_values = self.func(x_samples, y_samples)
        
        # Diện tích miền lấy tích phân (Vùng đáy hình hộp)
        area = (self.b - self.a) * (self.d - self.c)
        integral_value = area * np.mean(f_values)
        
        # Ghi nhận lịch sử hội tụ để vẽ đồ thị
        check_points = np.unique(np.logspace(1, np.log10(num_samples), num=200, dtype=int))
        history_n = []
        history_estimates = []
        cumulative_f = np.cumsum(f_values)
        
        for n in check_points:
            integral_n = area * (cumulative_f[n-1] / n)
            history_n.append(n)
            history_estimates.append(integral_n)
            
        return integral_value, history_n, history_estimates