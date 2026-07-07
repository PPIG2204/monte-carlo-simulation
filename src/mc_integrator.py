import numpy as np
from typing import Callable

class MonteCarloIntegrator:
    """Class thực hiện tính tích phân bằng phương pháp Monte Carlo thông thường."""
    
    def __init__(self, func: Callable[[float], float], a: float, b: float):
        """
        Args:
            func: Hàm số cần tính tích phân f(x).
            a: Cận dưới.
            b: Cận trên.
        """
        self.func = func
        self.a = a
        self.b = b

    def integrate_1d(self, num_samples: int):
        """
        Tính tích phân 1 chiều bằng phương pháp Mean Value.
        Returns:
            integral_value (float): Giá trị tích phân ước lượng.
            history_n (list): Danh sách số lượng mẫu tại các mốc kiểm tra.
            history_errors (list): Sai số của tích phân qua các mốc mẫu.
        """
        # Lấy mẫu ngẫu nhiên x tuân theo phân phối đều trên [a, b]
        x_samples = np.random.uniform(self.a, self.b, num_samples)
        
        # Tính giá trị của hàm tại các điểm mẫu
        f_values = self.func(x_samples)
        
        # Giá trị ước lượng tổng thể
        integral_value = (self.b - self.a) * np.mean(f_values)
        
        # Sinh lịch sử hội tụ để vẽ đồ thị
        check_points = np.unique(np.logspace(1, np.log10(num_samples), num=200, dtype=int))
        history_n = []
        history_estimates = []
        
        cumulative_f = np.cumsum(f_values)
        for n in check_points:
            integral_n = (self.b - self.a) * (cumulative_f[n-1] / n)
            history_n.append(n)
            history_estimates.append(integral_n)
            
        return integral_value, history_n, history_estimates