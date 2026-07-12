import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mc_integrator import MonteCarloIntegrator

def target_function_2d(x, y):
    return np.sin(x) * np.cos(y)

def main():
    a, b = 0, np.pi       # Miền x
    c, d = 0, np.pi / 2   # Miền y
    num_samples = 100000
    true_value = 2.0      # Giá trị giải tích chuẩn toán học
    
    integrator = MonteCarloIntegrator(func=target_function_2d, a=a, b=b, c=c, d=d)
    estimated_val, history_n, history_estimates = integrator.integrate_2d(num_samples)
    
    print("=== Thử nghiệm tính Tích phân 2 chiều ===")
    print(f"Hàm số f(x, y) = sin(x) * cos(y) trên [0, pi] x [0, pi/2]")
    print(f"Giá trị thực tế lý thuyết: {true_value}")
    print(f"Giá trị Monte Carlo 2D ước lượng: {estimated_val:.6f}")
    print(f"Sai số tuyệt đối cuối cùng: {np.abs(estimated_val - true_value):.6f}")
    
    # Vẽ đồ thị đường cong hội tụ tích phân 2 chiều
    plt.figure(figsize=(10, 6))
    plt.plot(history_n, history_estimates, color='purple', label='Ước lượng Monte Carlo 2D')
    plt.axhline(y=true_value, color='red', linestyle='--', label=f'Giá trị thực tế ({true_value})')
    plt.xscale('log')
    plt.xlabel("Số lượng mẫu phối hợp (N) - Thang Log")
    plt.ylabel("Giá trị Tích phân ước lượng")
    plt.title("Quá trình hội tụ của thuật toán Monte Carlo trong không gian 2 chiều")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()