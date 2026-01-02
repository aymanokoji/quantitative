import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


# --- CONFIGURATION ---
PATH_STOCK = "adjusted_data/QQQ.csv"
PATH_VIX = "adjusted_data/^VIX.csv"
TEST_DAYS = 252  # last year
EPOCHS = 400     # number of learning steps
LR = 0.01        #  learning rate

print("=== 1. PREPARING DATA ===")

# 1. Loading data and merging
df_s = pd.read_csv(PATH_STOCK, parse_dates=['Date'], index_col='Date')
df_v = pd.read_csv(PATH_VIX, parse_dates=['Date'], index_col='Date')
df_v = df_v[['Close', 'High']] 
df_v = df_v.rename(columns={'Close': 'VIX', 'High': 'VIX_High'})
df = df_s.join(df_v, how='inner')
#df = df.iloc[-3000:] # 

# 2. Target
df['Return'] = 100.0 * np.log(df['Close'] / df['Close'].shift(1))
df['Realized_Vol'] = np.abs(df['Return'])

# 3. Inputs for EGARCH (PyTorch)
# A. Garman-Klass : Intraday volatility + gaps
log_hl = np.log(df['High'] / df['Low'])
log_co = np.log(df['Close'] / df['Open'])
gk = 0.5 * log_hl**2 - (2*np.log(2)-1) * log_co**2
df['GK_Log'] = np.log(gk * 10000 + 1e-6)

# B. VIX Macro
df['VIX_Log'] = np.log(df['VIX']**2 / 252 + 1e-6) # we need to normalize, so that the algo doesn't need to manage too large spreads (daily volatility with annualized volatility)

# C. sign of the return
df['Sign'] = np.sign(df['Return'])

# 4. XGBoosst inputs
# RSI (relative strength index) calculation
df['RSI'] = 100 - (100 / (1 + df['Return'].rolling(14).apply(lambda x: x[x>0].sum()/abs(x[x<0].sum()) if abs(x[x<0].sum()) > 0 else 1)))
df['Vol_MA_20'] = df['Realized_Vol'].rolling(20).mean()

# --- lookahead bias management---
# XGboost can only have access to yesterday data, so we need to create lagged columns
df['XGB_RSI'] = df['RSI'].shift(1)
df['XGB_VIX'] = df['VIX'].shift(1)
df['XGB_Vol_MA'] = df['Vol_MA_20'].shift(1)
df['VIX_Panic'] = (df['VIX_High'] - df['VIX']) / df['VIX']
df['XGB_VIX_Panic'] = df['VIX_Panic'].shift(1)
df.dropna(inplace=True)

# tensor convertion for pytorch
gk_tensor = torch.tensor(df['GK_Log'].values, dtype=torch.float32)
vix_tensor = torch.tensor(df['VIX_Log'].values, dtype=torch.float32)
sign_tensor = torch.tensor(df['Sign'].values, dtype=torch.float32)
returns_tensor = torch.tensor(df['Return'].values, dtype=torch.float32)

print(f"Loaded data : {len(df)} days.")


print("\n=== 2. STEP 1 : (PYTORCH) ===")

# --- A. defining parameters  ---
# requires_grad=True 
omega = torch.tensor(-0.1, requires_grad=True) # constant
beta  = torch.tensor(0.95, requires_grad=True) # memory
alpha = torch.tensor(0.1,  requires_grad=True) # intraday impact
theta = torch.tensor(-0.1, requires_grad=True) # asymetry 
gamma = torch.tensor(0.1,  requires_grad=True) # vix impact

# grouping parameters for the optimizer
params = [omega, beta, alpha, theta, gamma]
optimizer = optim.Adam(params, lr=LR)

# --- B. variance calculation function
def calculate_egarch_variance(gk, vix, sign):
    log_sigma2 = []
    curr = torch.mean(gk)
    log_sigma2.append(curr)
    
    for t in range(1, len(gk)):
        prev = log_sigma2[-1]
        
        nxt = omega + \
              beta * prev + \
              alpha * gk[t-1] + \
              theta * sign[t-1] * gk[t-1] + \
              gamma * vix[t-1]
        
        log_sigma2.append(nxt)
    
    return torch.stack(log_sigma2)

# --- C. Trainingloop
for i in range(EPOCHS):
    optimizer.zero_grad() # Reset des gradients
    log_variance_pred = calculate_egarch_variance(gk_tensor, vix_tensor, sign_tensor)
    variance_pred = torch.exp(log_variance_pred) # On repasse en échelle réelle
    
    # 2. Loss : Log-Likelihood
    # gaussian formula of log likelihood, we use the mean of the observation rather than the sum for stability (so that the model doesn't diverge)
    loss = 0.5 * (log_variance_pred + (returns_tensor**2 / variance_pred)).mean()
    
    # 3. Backward : gradient calculation through the chain rules
    '''after line 115, we have our gradient for each parameter.
    loss.backward()
    
 
    # gradient descent: omega_old = omega_new - Learning Rate * Gradient
    # Adam rather than stochastic gradient descent : Adam remembers the direction of its steps, and adds momentum to avoid the gradient getting stucked
    optimizer.step()
    
    if i % 100 == 0:
        print(f"Epoch {i}: Loss = {loss.item():.5f}")

# --- D. BASE PREDICTION GENERATION ---
with torch.no_grad(): # we stop calculating gradient for the inference
    log_var_final = calculate_egarch_variance(gk_tensor, vix_tensor, sign_tensor)
    # sqroot for volatility
    pred_base = torch.sqrt(torch.exp(log_var_final)).numpy()

df['Pred_Base'] = pred_base


print("\n=== 3. Step 2 : XGBoost corrector ===")

# 1. garch error calculation (residual)
# XGBOOST target = what happened minus what GARCH predicted
df['Residual'] = df['Realized_Vol'] - df['Pred_Base']

# 2. data split for in and out of sample
split_index = len(df) - TEST_DAYS

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:].copy()


# 3. feature prep for xgboost
features = ['XGB_RSI', 'XGB_VIX', 'XGB_Vol_MA', 'XGB_VIX_Panic','Pred_Base']
target = 'Residual'

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]

# 4. booster training, depth=3 to avoid overfitting 
xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05)
xgb.fit(X_train, y_train)

print("\n--- IMPORTANCE DES VARIABLES ---")
importance = pd.DataFrame({
    'Feature': features,
    'Importance': xgb.feature_importances_
}).sort_values(by='Importance', ascending=False)
print(importance)
print("--------------------------------\n")

# 5.  predict correction
correction = xgb.predict(X_test)


print("\n=== 4. RESULTS ===")

# CHIMERA = Base + Correction
test_df['Correction'] = correction
test_df['Pred_Final'] = test_df['Pred_Base'] + test_df['Correction']

# in the case where XGboost correction is higher or equal to the base_pred of garch, which is pretty unlikely (volatility equal to 0 might be problematic, and negative volatility doesn't exist)
test_df['Pred_Final'] = test_df['Pred_Final'].clip(lower=0.1)

# performance (MAE rather than MSE because MSE can skew the results with days where volatility spikes)
mae_base = mean_absolute_error(test_df['Realized_Vol'], test_df['Pred_Base'])
mae_final = mean_absolute_error(test_df['Realized_Vol'], test_df['Pred_Final'])
gain = (mae_base - mae_final) / mae_base * 100

print(f"Testing period :{TEST_DAYS}")
print("-" * 40)
print(f"Mean error (GARCH alone) : {mae_base:.5f}")
print(f"Mean error (Chimera (garch +  xgboost))    : {mae_final:.5f}")
print("-" * 40)
print(f"acurracy gain : +{gain:.2f}%")

# save to excel
test_df.to_csv("chimera_results.csv")
print("\nFichier 'chimera_results.csv' généré.")



print("\n=== 5. predicting following day ===")

# last inputs
last_gk = gk_tensor[-1]
last_vix = vix_tensor[-1]
last_sign = sign_tensor[-1]

# last GARCH state
with torch.no_grad():
    log_vars = calculate_egarch_variance(gk_tensor, vix_tensor, sign_tensor)
    last_log_var = log_vars[-1]


# predict next day volatility
next_log_var = omega + \
               beta * last_log_var + \
               alpha * last_gk + \
               theta * last_sign * last_gk + \
               gamma * last_vix

next_pred_base = torch.sqrt(torch.exp(next_log_var)).item()

# xgboost correction

last_features = pd.DataFrame([{
    'XGB_RSI': df['RSI'].iloc[-1],    
    'XGB_VIX': df['VIX'].iloc[-1],      
    'XGB_Vol_MA': df['Vol_MA_20'].iloc[-1], 
    'XGB_VIX_Panic': df['VIX_Panic'].iloc[-1], 
    'Pred_Base': next_pred_base         
}])

next_correction = xgb.predict(last_features)[0]

# E. Résultat Final
next_pred_final = next_pred_base + next_correction
next_pred_final = max(0.1, next_pred_final) # Sécurité

print(f"--- NEXT DAY PREDICTION ---")
print(f"Base GARCH   : {next_pred_base:.4f}%")
print(f"Ajustment   : {next_correction:.4f}%")
print(f"Estimated Volatility: {next_pred_final:.4f}%")
print(f"likelihood zone (1 sigma) : +/- {next_pred_final:.2f}%")
