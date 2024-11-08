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
data['inflation_squared'] = data['inflation'] ** 2
X_nonlin = data[['FFR', 'GDP_growth', 'inflation', 'inflation_squared']]
X_nonlin = sm.add_constant(X_nonlin)
model_nonlin = sm.OLS(Y, X_nonlin).fit()
print(model_nonlin.summary())



from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate correlation matrix
correlation_matrix = data[['FFR', 'GDP_growth', 'inflation']].corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Calculate VIF for each predictor
X = data[['FFR', 'GDP_growth', 'inflation']]
X = sm.add_constant(X)  # add constant for intercept
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nVariance Inflation Factors (VIF):")
print(vif_data)


# Define dependent and independent variables
Y = data['return']
X_extended = data[['FFR', 'GDP_growth', 'inflation']]
X_extended = sm.add_constant(X_extended)  # Add intercept

# Fit the extended model with FFR, GDP growth, and inflation as predictors
model_extended = sm.OLS(Y, X_extended)
results_extended = model_extended.fit()

# Print the extended model results
print("Extended Model Regression Results:")
print(results_extended.summary())


# Assessing new model:
# 1. Breusch-Pagan test for homoskedasticity
_, bp_pvalue, _, _ = het_breuschpagan(results_extended.resid, X_extended)
print(f"Breusch-Pagan p-value: {bp_pvalue}")

# 2. Durbin-Watson test for autocorrelation in residuals
dw_statistic = durbin_watson(results_extended.resid)
print(f"Durbin-Watson Statistic: {dw_statistic}")

# 3. Confidence Intervals for the Coefficients
print("Confidence intervals for the coefficients:")
print(results_extended.conf_int())

# 4. Robust Standard Errors
results_robust_extended = model_extended.fit(cov_type='HC3')
print("Extended Model with Robust Standard Errors:")
print(results_robust_extended.summary())

# 5. Jarque-Bera test for normality of errors
jb_stat, jb_p_value = jarque_bera(results_extended.resid)
print(f"Jarque-Bera Test Statistic: {jb_stat}")
print(f"Jarque-Bera p-value: {jb_p_value}")

# Interpretation of Jarque-Bera Test
if jb_p_value < 0.05:
    print("Residuals are not normally distributed (reject null hypothesis of normality).")
else:
    print("Residuals are normally distributed (fail to reject null hypothesis of normality).")

# 6. Residuals vs Fitted Values Plot
plt.figure(figsize=(10, 6))
plt.scatter(results_extended.fittedvalues, results_extended.resid)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values (Extended Model)")
plt.show()

import pandas as pd
import statsmodels.api as sm

# Load data
data = pd.read_csv("CW data.csv")

# Define the dependent and independent variables for the model
Y = data['return']
X_fgls = data[['FFR', 'GDP_growth', 'inflation']]
X_fgls = sm.add_constant(X_fgls)  # Add intercept

# Fit the initial OLS model to obtain residuals
initial_model = sm.OLS(Y, X_fgls)
initial_results = initial_model.fit()

# Obtain residuals from the initial OLS model
residuals = initial_results.resid

# Estimate a model of the residuals (heteroskedasticity) to use in FGLS
# Here we assume the variance of residuals may be related to one of the predictors
# e.g., fit a model regressing squared residuals on FFR to get an estimate of heteroskedasticity
weights_model = sm.OLS(residuals**2, sm.add_constant(data['FFR'])).fit()

# Use fitted values from this model as weights for GLS
weights = 1 / weights_model.fittedvalues  # Inverse variance weighting

# Fit the FGLS model with weights
fgls_model = sm.WLS(Y, X_fgls, weights=weights).fit()

# Print the FGLS model summary
print("Feasible Generalized Least Squares (FGLS) Model Results:")
print(fgls_model.summary())


