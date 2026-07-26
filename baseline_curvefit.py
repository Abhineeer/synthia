import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator
import json
import torch
from solvers.heat_fd import solve_heat_fd
import os

x_in, t_in, u_in = solve_heat_fd(alpha_new=0.1, n_x=100, n_t=200)
T_max = t_in.max()
interpolator = RegularGridInterpolator((t_in, x_in), u_in)

def generate_observations(sigma, seed):
    torch.manual_seed(seed)
    x_obs = torch.rand(500, 1)
    t_obs = torch.rand(500, 1) * T_max

    combined_obs = torch.cat((t_obs, x_obs), dim=1)
    u_obs_array = interpolator(combined_obs.numpy())

    noise = torch.randn(500, ) * sigma
    u_obs = torch.from_numpy(u_obs_array)
    u_ex_data = (u_obs + noise).reshape(-1, 1)

    return x_obs.numpy().flatten(), t_obs.numpy().flatten(), u_ex_data.numpy().flatten()

noise_vals = [0.01, 0.05, 0.1, 0.2]
seed_vals = [40, 10, 21, 102, 93]
count = len(noise_vals)

results_dict = {}

def f(combined_obs, alpha):
    result = np.sin(np.pi*combined_obs[0])*np.exp(-1*alpha*(np.pi)**2*combined_obs[1])
    return result
    # this is what will get compared to the noisy u values.
    # This is the model function

for i in range(count):

    print("------- sigma = " + str(noise_vals[i]) + " -------")
    alpha_results = []
    fail_count = 0

    for j in range(len(seed_vals)):
        x_np, t_np, u_np = generate_observations(noise_vals[i], seed_vals[j])
        combined_obs = np.vstack((x_np, t_np))

        popt, pcov = curve_fit(f, combined_obs, u_np, p0=[0.5])
        # popt: paramater optimal values - Its an arrys containing the best fit parameters that curve_fit found, values that minimized the sum of squared residuals.
        # For this there is only one unknown alpha = we can pull it out using popt[0]
        # pcov: parameter covariance matrix and that tells us the uncertaintiy of the fitted parameters, basically how conficent the fit is.
        # pcov helps in understading the curvature of S(alpha), specifically at the minimum
        # AGain for this we have only one entry is a square matrix, we extract it usign pcov[0][0] ---> This is the variance of the alpha (popt[0]) quantity
        # f is the model function
        # combined_obs is the single data set of x and t vals that go into the model function to find the true values
        # u_ex_data is the messy data, the made up experimental data that is compared against the model function's data.
        # p0 is the parameter we are nudging (alpha in this case). and we set it to an initial value 0.5 in this case
        alpha_recovered = popt[0]

        if not np.isfinite(alpha_recovered) or alpha_recovered <= 0:
            print(f"  seed {seed_vals[j]}: FAILED (alpha={alpha_recovered})")
            fail_count += 1
            continue

        alpha_results.append(alpha_recovered)

    alpha_array = np.array(alpha_results)
    errors_array = np.abs(alpha_array - 0.1)

    mean_alpha = float(alpha_array.mean())
    std_alpha = float(alpha_array.std())
    mean_error_pct = float(errors_array.mean() / 0.1 * 100)
    std_error_pct = float(errors_array.std() / 0.1 * 100)

    print("mean alpha recovered: " + str(mean_alpha))
    print("mean error: " + str(mean_error_pct) + "%")
    print()

    key = f"sigma_{noise_vals[i]}"
    results_dict[key] = {
        "curve_fit": {
            "alpha_values": [float(a) for a in alpha_results],
            "mean_alpha_recovered": mean_alpha,
            "std_alpha_recovered": std_alpha,
            "mean_error_pct": mean_error_pct,
            "std_error_pct": std_error_pct,
            "fail_count": fail_count,
            "n_seeds": len(seed_vals)
        }
    }

json_path = "benchmarks/benchmark_inverse.json"
if os.path.exists(json_path):
    with open(json_path, "r") as f:
        full_results = json.load(f)
else:
    full_results = {}

for key, value in results_dict.items():
    if key not in full_results:
        full_results[key] = {}
    full_results[key]["curve_fit"] = value["curve_fit"]

os.makedirs("benchmarks", exist_ok=True)
with open(json_path, "w") as f:
    json.dump(full_results, f, indent=2)