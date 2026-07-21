import numpy as np
from scipy.optimize import curve_fit
from solvers.heat_fd import u_exact

rng = np.random.default_rng()
# this is the random number generator initialized for np

x_obs = rng.uniform(low=0, high=1, size=150)
t_obs = rng.uniform(low=0, high=1, size=150)
u_obs = u_exact(x_obs, t_obs)
# I am using the exact solution instead of teh finite solver because I want to isolate teh variable I am acutally testing. Instead of having uncertainty in everything I choose to only test teh curve_fit mechanism.

noise_vals = [0.01, 0.05, 0.1, 0.2]
count = len(noise_vals)

ave_alpha_recovered = 0
ave_alpha_stderr = 0

def f(combined_obs, alpha):
    result = np.sin(np.pi*combined_obs[0])*np.exp(-1*alpha*(np.pi)**2*combined_obs[1])
    return result
    # this is what will get compared to the noisy u values.
    # This is the model function

for i in range(count):
    noise = np.random.randn(150, )*noise_vals[i]
    u_ex_data = u_obs + noise

    # print(u_obs.min(), u_obs.max())
    # print(u_ex_data.min(), u_ex_data.max())
    # SAnity check

    combined_obs = np.vstack((x_obs, t_obs))

    for j in range(20):
        popt, pcov = curve_fit(f, combined_obs, u_ex_data, p0=[0.5])
        # popt: paramater optimal values - Its an arrys containing the best fit parameters that curve_fit found, values that minimized the sum of squared residuals.
        # For this there is only one unknown alpha = we can pull it out using popt[0]
        # pcov: parameter covariance matrix and that tells us the uncertaintiy of the fitted parameters, basically how conficent the fit is.
        # pcov helps in understading the curvature of S(alpha), specifically at the minimum
        # AGain for this we have only one entry is a square matrix, we extract it usign pcov[0][0] ---> This is the variance of the alpha (popt[0]) quantity
        # f is the model function
        # combined_obs is the single data set of x and t vals that go into the model function to find the true values
        # u_ex_data is the messy data, the made up experimental data that is compared against the model function's data.
        # p0 is the parameter we are nudging (alpha in this case). and we set it to an initial value 0.5 in this case
        ave_alpha_recovered += popt[0]

        ave_alpha_stderr += np.sqrt(pcov[0][0])
        # we sqrt the variance term because variance itself is in squared units
        #  std deviation = sqrt(variance)

    print("------- sigma = " + str(noise_vals[i]) + " -------")
    ave_alpha_recovered /= 20
    ave_alpha_stderr /= 20

    alpha_error = abs(ave_alpha_recovered - 0.1)
    error_percentage = (alpha_error/0.1)*100   
    print("alpha recovered: " + str(ave_alpha_recovered))
    print("alpha error: " + str(alpha_error))
    print("Error: " + str(error_percentage) + "%")
    print(ave_alpha_stderr)
    print()

    ave_alpha_recovered = 0;
    ave_alpha_stderr = 0;
