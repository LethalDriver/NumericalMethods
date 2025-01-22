import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2
from sklearn.metrics import mean_squared_error

x_data = np.array([0.75, 2, 3, 4, 6, 8, 8.5])
y_data = np.array([1.2, 1.95, 2, 2.4, 2.5, 2.7, 2.6])


def saturation_growth(x, alpha, beta):
    return (alpha * x) / (beta + x)


popt_sg, pcov_sg = curve_fit(saturation_growth, x_data, y_data)
alpha_sg, beta_sg = popt_sg

x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit_sg = saturation_growth(x_fit, alpha_sg, beta_sg)


plt.figure(figsize=(8, 5))
plt.plot(x_data, y_data, "o", label="Data")
plt.plot(x_fit, y_fit_sg, "-", label=f"Fitted: y = {alpha_sg:.2f}x/({beta_sg:.2f}+x)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Saturation Growth Rate Fit")
plt.legend()
plt.grid(True)
plt.show()


def power_equation(x, a, b):
    return a * (x**b)


popt_pow, pcov_pow = curve_fit(power_equation, x_data, y_data)
a_pow, b_pow = popt_pow

x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit_pow = power_equation(x_fit, a_pow, b_pow)

plt.figure(figsize=(8, 5))
plt.plot(x_data, y_data, "o", label="Data")
plt.plot(x_fit, y_fit_pow, "-", label=f"Fitted: y = {a_pow:.2f}x^{b_pow:.2f}")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Power Equation Fit")
plt.legend()
plt.grid(True)
plt.show()


def polynomial(x, *coeffs):
    y = np.zeros_like(x)
    for i, c in enumerate(coeffs):
        y += c * x**i
    return y


x_ground_truth = np.linspace(-5, 5, 50)
y_ground_truth = 2 + 0.5 * x_ground_truth**2 + 0.1 * x_ground_truth**3

np.random.seed(42)
y_noise = np.random.normal(loc=0, scale=2, size=len(x_ground_truth))
y_data_p2 = y_ground_truth + y_noise


def calculate_aic_bic(y_true, y_pred, num_params):
    n = len(y_true)
    mse = mean_squared_error(y_true, y_pred)
    aic = n * np.log(mse) + 2 * num_params
    bic = n * np.log(mse) + num_params * np.log(n)
    return aic, bic


# Define a function to compute Log-Likelihood Ratio (LLR) and p-value
def calculate_llr_test(y_true, y_pred_small, y_pred_big, k_small, k_big):
    n = len(y_true)

    rss_small = np.sum((y_true - y_pred_small) ** 2)
    rss_big = np.sum((y_true - y_pred_big) ** 2)

    log_like_small = -n / 2 * (np.log(2 * np.pi) + np.log(rss_small / n) + 1)
    log_like_big = -n / 2 * (np.log(2 * np.pi) + np.log(rss_big / n) + 1)

    llr = 2 * (log_like_big - log_like_small)

    df = k_big - k_small

    p_value = 1 - chi2.cdf(llr, df)

    return llr, p_value


max_order = 5
results = {}

for order in range(1, max_order + 1):
    popt, _ = curve_fit(polynomial, x_ground_truth, y_data_p2, p0=[1] * (order + 1))
    y_pred = polynomial(x_ground_truth, *popt)

    aic, bic = calculate_aic_bic(y_data_p2, y_pred, order + 1)
    results[order] = {"coeffs": popt, "y_pred": y_pred, "aic": aic, "bic": bic}

    print(f"Order {order}: AIC={aic:.2f}, BIC={bic:.2f}")

print("\nLikelihood Ratio test:")
for order in range(1, max_order):
    y_pred_small = results[order]["y_pred"]
    y_pred_big = results[order + 1]["y_pred"]
    llr, p_value = calculate_llr_test(
        y_data_p2, y_pred_small, y_pred_big, order + 1, order + 2
    )
    print(
        f"Order {order} vs {order + 1}:  Log Likelihood Ratio={llr:.2f},  p-value={p_value:.3f}"
    )


# Select the best order
best_order = min(results, key=lambda k: results[k]["aic"])
print(f"\nBest order based on AIC: {best_order}")

best_order = min(results, key=lambda k: results[k]["bic"])
print(f"Best order based on BIC: {best_order}")


# Plot the best-fit polynomial
plt.figure(figsize=(8, 5))
plt.plot(x_ground_truth, y_data_p2, "o", label="Data")
plt.plot(
    x_ground_truth,
    results[best_order]["y_pred"],
    "-",
    label=f"Fitted Polynomial (Order {best_order})",
)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Polynomial Fit")
plt.legend()
plt.grid(True)
plt.show()

colors = ["b", "g", "r", "c", "m", "y", "k"]

# Plot all polynomial fits
plt.figure(figsize=(8, 5))
plt.plot(x_ground_truth, y_data_p2, "o", label="Data")
for i, order in enumerate(results):
    plt.plot(
        x_ground_truth,
        results[order]["y_pred"],
        "-",
        label=f"Order {order}: AIC={results[order]['aic']:.2f}, BIC={results[order]['bic']:.2f}",
        color=colors[i % len(colors)]  # Cycle through colors
    )
plt.xlabel("x")
plt.ylabel("y")
plt.title("Polynomial Fit")
plt.legend()
plt.grid(True)
plt.show()
