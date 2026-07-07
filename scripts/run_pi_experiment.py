import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Thêm thư mục gốc vào path để import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pi_estimator import PiEstimator

def main():
    num_samples = 100000
    estimator = PiEstimator(num_samples=num_samples)
    
    print("=== Thử nghiệm ước lượng số Pi ===")
    estimated_pi, history_n, history_errors = estimator.estimate()
    print(f"Số lượng mẫu: {num_samples}")
    print(f"Giá trị Pi ước lượng: {estimated_pi}")
    print(f"Sai số tuyệt đối: {np.abs(estimated_pi - np.pi):.6f}")
    
    # Vẽ đồ thị sai số
    plt.figure(figsize=(10, 6))
    plt.plot(history_n, history_errors, label="Sai số thực nghiệm Monte Carlo", color='blue', alpha=0.7)
    
    # Vẽ đường lý thuyết O(1/sqrt(N)) để đối chiếu
    history_n = np.array(history_n)
    theoretical_bound = history_errors[0] * np.sqrt(history_n[0]) / np.sqrt(history_n)
    plt.plot(history_n, theoretical_bound, '--', label=r"Đường tiệm cận lý thuyết $\mathcal{O}(1/\sqrt{N})$", color='red')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Số lượng mẫu (N) - Thang Logarithm")
    plt.ylabel("Sai số tuyệt đối - Thang Logarithm")
    plt.title("Đồ thị Hội tụ Sai số Ước lượng Số Pi")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()