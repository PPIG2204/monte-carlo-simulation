import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.importance_sampler.py import ImportanceSampler # Sửa lại cho chính xác tên file
# Hoặc import trực tiếp từ file
from src.importance_sampler import ImportanceSampler

def target_sharp_function(x):
    # Hàm Gauss biến động mạnh: f(x) = exp(-x^2)
    return np.exp(-x**2)

def main():
    a, b = 0, 5
    num_samples = 50000
    
    # Tính nghiệm chính xác bằng phương pháp số Gauss để làm chuẩn đối sánh
    true_val, _ = quad(target_sharp_function, a, b)
    
    sampler = ImportanceSampler(target_func=target_sharp_function, a=a, b=b)
    
    crude_est, crude_var, crude_w = sampler.crude_monte_carlo(num_samples)
    is_est, is_var, is_w = sampler.importance_sampling_exp(num_samples)
    
    print("=== Thử nghiệm Nâng cao: Importance Sampling ===")
    print(f"Hàm số f(x) = exp(-x^2) trên đoạn [{a}, {b}]")
    print(f"Giá trị chính xác (Ground Truth): {true_val:.8f}")
    print(f"1. Crude MC (Uniform)        : {crude_est:.8f} (Phương sai toán tử: {crude_var:.12f})")
    print(f"2. Importance Sampling (Exp) : {is_est:.8f} (Phương sai toán tử: {is_var:.12f})")
    print(f"-> Tỷ lệ giảm phương sai: {crude_var / is_var:.2f} lần!")
    
    # Vẽ đồ thị phân phối trọng số mẫu để minh họa trực quan hiện tượng giảm biến động
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(crude_w * (b-a), bins=100, alpha=0.6, color='orange', label='Crude MC Weights')
    plt.title("Phân phối Trọng số mẫu - Crude MC\n(Biến động rất rộng)")
    plt.xlabel("Giá trị hàm hiệu chỉnh")
    plt.ylabel("Tần suất mẫu")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.hist(is_w, bins=100, alpha=0.6, color='teal', label='Importance Sampling Weights')
    plt.title("Phân phối Trọng số mẫu - Importance Sampling\n(Cực kỳ tập trung -> Phương sai thấp)")
    plt.xlabel("Giá trị hàm hiệu chỉnh")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()