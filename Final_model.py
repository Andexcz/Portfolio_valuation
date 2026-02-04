import numpy as np

# Portfolio value function
def portfolio_value(s1, s2, w1=100, w2=80, k=50):
  linear_part = w1 * s1 + w2 * s2
  phi = k * np.sin(s1 / 20) * np.cos(s2 / 20)
  return linear_part + phi

# First order approximation (from previous exercise)
def first_order_approximation(s1, s2, s1_star=100, s2_star=100):
  V_star = portfolio_value(s1_star, s2_star)
  dV_dS1 = 100 + (50 / 20) * np.cos(s1_star / 20) * np.cos(s2_star / 20)
  dV_dS2 = 80 - (50 / 20) * np.sin(s1_star / 20) * np.sin(s2_star / 20)
  delta_s1 = s1 - s1_star
  delta_s2 = s2 - s2_star
  return V_star + dV_dS1 * delta_s1 + dV_dS2 * delta_s2

def second_order_approximation(s1, s2, s1_star=100, s2_star=100):
  V_star = portfolio_value(s1_star, s2_star)

  # Gradient
  dV_dS1 = 100 + (50 / 20) * np.cos(s1_star / 20) * np.cos(s2_star / 20)
  dV_dS2 = 80 - (50 / 20) * np.sin(s1_star / 20) * np.sin(s2_star / 20)

  # Hessian
  h11 = -(50 / 400) * np.sin(s1_star / 20) * np.cos(s2_star / 20)
  h22 = -(50 / 400) * np.sin(s1_star / 20) * np.cos(s2_star / 20)
  h12 = -(50 / 400) * np.cos(s1_star / 20) * np.sin(s2_star / 20)

  delta_s1 = s1 - s1_star
  delta_s2 = s2 - s2_star

  # First order
  linear_term = dV_dS1 * delta_s1 + dV_dS2 * delta_s2

  quadratic_term = 0.5 * (h11 * delta_s1**2 + 2 * h12 * delta_s1 * delta_s2 + h22 * delta_s2**2)

  return V_star + linear_term + quadratic_term

# Compare approximations
s1_test, s2_test = 110, 95
V_true = portfolio_value(s1_test, s2_test)
V_linear = first_order_approximation(s1_test, s2_test)
V_quadratic = second_order_approximation(s1_test, s2_test)

print(f"True V(110, 95)     = {V_true:.2f}")
print(f"Linear approx       = {V_linear:.2f} (error: {abs(V_true - V_linear):.2f})")
print(f"Quadratic approx    = {V_quadratic:.2f} (error: {abs(V_true - V_quadratic):.2f})")