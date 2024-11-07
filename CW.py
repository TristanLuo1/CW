import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv("CW data.csv")

# Define dependent and independent variables
Y = data['return']
X = sm.add_constant(data['FFR'])  # Adding a constant for the intercept

# Fit the OLS regression model
model = sm.OLS(Y, X)
results = model.fit()

# Display the regression results
print(results.summary())

# Plot the data and the regression line
plt.figure(figsize=(10, 6))
sns.regplot(x='FFR', y='return', data=data, ci=None, line_kws={"color": "red"})
plt.xlabel("Benchmark Interest Rate (Fed Funds Rate)")
plt.ylabel("S&P500 Return")
plt.title("Linear Relationship between S&P500 Return and Benchmark Interest Rate")
plt.show()


from statsmodels.stats.diagnostic import het_breuschpagan
test_statistic, p_value, _, _ = het_breuschpagan(results.resid, X)
print(f"Breusch-Pagan p-value: {p_value}")

sm.qqplot(results.resid, line='45')
plt.title("Q-Q Plot of Residuals")
plt.show()

from statsmodels.stats.stattools import durbin_watson
dw_statistic = durbin_watson(results.resid)
print(f"Durbin-Watson Statistic: {dw_statistic}")

print("Confidence intervals for the coefficients:")
print(results.conf_int())

from scipy.stats import jarque_bera

# Perform the Jarque-Bera test on the residuals
jb_stat, jb_p_value = jarque_bera(results.resid)
print(f"Jarque-Bera Test Statistic: {jb_stat}")
print(f"Jarque-Bera p-value: {jb_p_value}")

# Interpretation
if jb_p_value < 0.05:
    print("Residuals are not normally distributed (reject null hypothesis of normality).")
else:
    print("Residuals are normally distributed (fail to reject null hypothesis of normality).")
