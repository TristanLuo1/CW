import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import jarque_bera

# Load the data
data = pd.read_csv("CW data.csv")

# Define dependent and independent variables
Y = data['return']
X = sm.add_constant(data['FFR'])  # Adding a constant for the intercept

# Fit the OLS regression model
model = sm.OLS(Y, X)
results = model.fit()

# Display the regression results
print("OLS Regression Results:")
print(results.summary())

# Plot the data and the regression line
plt.figure(figsize=(10, 6))
sns.regplot(x='FFR', y='return', data=data, ci=None, line_kws={"color": "red"})
plt.xlabel("Benchmark Interest Rate (Fed Funds Rate)")
plt.ylabel("S&P500 Return")
plt.title("Linear Relationship between S&P500 Return and Benchmark Interest Rate")
plt.show()

# Diagnostic Tests

# 1. Breusch-Pagan test for homoskedasticity
_, bp_pvalue, _, _ = het_breuschpagan(results.resid, X)
print(f"Breusch-Pagan p-value: {bp_pvalue}")

# 2. Q-Q Plot for Normality of Residuals
sm.qqplot(results.resid, line='45')
plt.title("Q-Q Plot of Residuals")
plt.show()

# 3. Durbin-Watson test for autocorrelation in residuals
dw_statistic = durbin_watson(results.resid)
print(f"Durbin-Watson Statistic: {dw_statistic}")

# 4. Confidence Intervals for the Coefficients
print("Confidence intervals for the coefficients:")
print(results.conf_int())

# 5. Robust Standard Errors
results_robust = model.fit(cov_type='HC3')
print("OLS Regression Results with Robust Standard Errors:")
print(results_robust.summary())

# 6. Jarque-Bera test for normality of errors
jb_stat, jb_p_value = jarque_bera(results.resid)
print(f"Jarque-Bera Test Statistic: {jb_stat}")
print(f"Jarque-Bera p-value: {jb_p_value}")

# Interpretation of Jarque-Bera Test
if jb_p_value < 0.05:
    print("Residuals are not normally distributed (reject null hypothesis of normality).")
else:
    print("Residuals are normally distributed (fail to reject null hypothesis of normality).")

# 7. Residuals vs Fitted Values Plot
plt.figure(figsize=(10, 6))
plt.scatter(results.fittedvalues, results.resid)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
plt.show()

# Additional Analysis: Adding a Nonlinear Term (Quadratic) for FFR
data['FFR_squared'] = data['FFR'] ** 2
X_nonlin = sm.add_constant(data[['FFR', 'FFR_squared']])
model_nonlin = sm.OLS(Y, X_nonlin)
results_nonlin = model_nonlin.fit()

print("Nonlinear Model (Including FFR Squared) Regression Results:")
print(results_nonlin.summary())
