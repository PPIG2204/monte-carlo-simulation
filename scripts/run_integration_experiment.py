import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mc_integrator import MonteCarloIntegrator

def my_function(x):
    # Thử nghiệm với hàm số f(x) = x * sin(x)
    return x * np.sin(x)

def main():
    a, b = 0, np.pi
    num_samples = 100000000
    
    # Tính giá trị chính xác bằng Scipy giải tích (để làm ground-truth so sánh)
    true_val, _ = quad(my_function, a, b)
    
    integrator = MonteCarloIntegrator(func=my_function, a=a, b=b)
    estimated_val, history_n, history_estimates = integrator.integrate_1d(num_samples)
    
    history_errors = [np.abs(est - true_val) for est in history_estimates]
    
    print("=== Thử nghiệm tính Tích phân 1 chiều ===")
    print(f"Hàm số: x * sin(x) trên [{a}, {b}]")
    print(f"Giá trị giải tích thực tế (Scipy): {true_val}")
    print(f"Giá trị Monte Carlo ước lượng: {estimated_val}")
    
    # Vẽ đồ thị quá trình hội tụ của giá trị tích phân
    plt.figure(figsize=(10, 5))
    plt.plot(history_n, history_estimates, color='green', label='Ước lượng Monte Carlo')
    plt.axhline(y=true_val, color='r', linestyle='-', label=f'Giá trị thực tế ({true_val:.4f})')
    plt.xscale('log')
    plt.xlabel("Số lượng mẫu (N)")
    plt.ylabel("Giá trị tích phân")
    plt.title("Quá trình hội tụ giá trị Tích phân theo số mẫu N")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()