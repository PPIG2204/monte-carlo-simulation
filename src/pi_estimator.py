import numpy as np

class PiEstimator:
    """Class thực hiện mô phỏng Monte Carlo để ước lượng số Pi."""
    
    def __init__(self, num_samples: int):
        self.num_samples = num_samples
        self.true_pi = np.pi

    def estimate(self):
        """
        Ước lượng số Pi bằng cách sinh mẫu ngẫu nhiên trên không gian [0, 1] x [0, 1].
        Returns:
            estimated_pi (float): Giá trị Pi ước lượng.
            history_n (list): Danh sách số lượng mẫu tại các mốc kiểm tra.
            history_errors (list): Sai số tuyệt đối tương ứng với các mốc.
        """
        # Sinh ngẫu nhiên tọa độ x, y cho toàn bộ số mẫu
        x = np.random.uniform(0, 1, self.num_samples)
        y = np.random.uniform(0, 1, self.num_samples)
        
        # Kiểm tra điều kiện nằm trong 1/4 đường tròn: x^2 + y^2 <= 1
        inside_circle = (x**2 + y**2) <= 1
        
        # Tính toán ước lượng cuối cùng
        k = np.sum(inside_circle)
        estimated_pi = 4 * k / self.num_samples
        
        # Ghi nhận lịch sử hội tụ để phục vụ yêu cầu vẽ đồ thị sai số
        # Kiểm tra sai số tại các mốc lũy thừa để đồ thị mượt mà hơn
        check_points = np.unique(np.logspace(1, np.log10(self.num_samples), num=200, dtype=int))
        history_n = []
        history_errors = []
        
        cumulative_inside = np.cumsum(inside_circle)
        for n in check_points:
            pi_n = 4 * cumulative_inside[n-1] / n
            error_n = np.abs(pi_n - self.true_pi)
            history_n.append(n)
            history_errors.append(error_n)
            
        return estimated_pi, history_n, history_errors
        