import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


# --- CONFIGURATION ---
PATH_STOCK = "adjusted_data/QQQ.csv"
PATH_VIX = "adjusted_data/^VIX.csv"
TEST_DAYS = 252  # On teste sur la dernière année
EPOCHS = 400     # Nombre de tours d'apprentissage PyTorch
LR = 0.01        # Vitesse d'apprentissage

print("=== 1. PRÉPARATION DES DONNÉES ===")

# 1. Chargement et Fusion
df_s = pd.read_csv(PATH_STOCK, parse_dates=['Date'], index_col='Date')
df_v = pd.read_csv(PATH_VIX, parse_dates=['Date'], index_col='Date')
# On sélectionne les deux colonnes vitales
df_v = df_v[['Close', 'High']] 
# IMPORTANT : On renomme pour éviter la collision avec le High du Stock
df_v = df_v.rename(columns={'Close': 'VIX', 'High': 'VIX_High'})
df = df_s.join(df_v, how='inner')
#df = df.iloc[-3000:] # on coupe les données. Les changements de régime consécutifs (non-stationnarité) peuvent compromettre les résultats

# 2. La Cible (Ce qu'on veut prédire)
# On prédit la volatilité réalisée (Proxy = Valeur Absolue du Rendement Log)
df['Return'] = 100.0 * np.log(df['Close'] / df['Close'].shift(1))
df['Realized_Vol'] = np.abs(df['Return'])

# 3. Inputs pour le Moteur EGARCH (PyTorch)
# A. Garman-Klass : Capture la volatilité Intraday + les Gaps
log_hl = np.log(df['High'] / df['Low'])
log_co = np.log(df['Close'] / df['Open'])
gk = 0.5 * log_hl**2 - (2*np.log(2)-1) * log_co**2
df['GK_Log'] = np.log(gk * 10000 + 1e-6) # Log pour stabiliser l'EGARCH, 1e-6 pour pas  que ça plante si gk = 0

# B. VIX Macro
df['VIX_Log'] = np.log(df['VIX']**2 / 252 + 1e-6) # faut tout normaliser, pour pas que l'algo ait a traité des écarts trop grands entre nombres très petits (variance journalière), et très grands (variance anualisée)

# C. Signe du rendement (Pour l'effet de levier)
df['Sign'] = np.sign(df['Return'])

# 4. Inputs pour le Correcteur (XGBoost)
# On calcule des indicateurs techniques contextuels
df['RSI'] = 100 - (100 / (1 + df['Return'].rolling(14).apply(lambda x: x[x>0].sum()/abs(x[x<0].sum()) if abs(x[x<0].sum()) > 0 else 1)))
df['Vol_MA_20'] = df['Realized_Vol'].rolling(20).mean()

# --- POINT CRITIQUE : GESTION DU BIAIS DU FUTUR ---
# Pour prédire l'erreur de demain (J), XGBoost ne peut utiliser que les infos d'hier (J-1).
# On crée des colonnes décalées (Lag).
df['XGB_RSI'] = df['RSI'].shift(1)
df['XGB_VIX'] = df['VIX'].shift(1)
df['XGB_Vol_MA'] = df['Vol_MA_20'].shift(1)
df['VIX_Panic'] = (df['VIX_High'] - df['VIX']) / df['VIX']
df['XGB_VIX_Panic'] = df['VIX_Panic'].shift(1)
# Nettoyage des NaN (dus aux moyennes mobiles et shift)
df.dropna(inplace=True)

print("\n--- VÉRIFICATION VIX PANIC ---")
print(df[['VIX', 'VIX_High', 'VIX_Panic']].tail(5))
print(f"Moyenne de la panique : {df['VIX_Panic'].mean():.5f}")
print("------------------------------\n")

# Conversion en Tensors pour PyTorch
gk_tensor = torch.tensor(df['GK_Log'].values, dtype=torch.float32)
vix_tensor = torch.tensor(df['VIX_Log'].values, dtype=torch.float32)
sign_tensor = torch.tensor(df['Sign'].values, dtype=torch.float32)
returns_tensor = torch.tensor(df['Return'].values, dtype=torch.float32)

print(f"Données chargées : {len(df)} jours.")


print("\n=== 2. ÉTAGE 1 : MOTEUR EGARCH (PYTORCH) ===")

# --- A. Définition des Paramètres (Variables à optimiser) ---
# On n'utilise pas de classe, juste des variables brutes.
# requires_grad=True dit à PyTorch : "Tu devras calculer la dérivée pour celles-ci"
omega = torch.tensor(-0.1, requires_grad=True) # Constante
beta  = torch.tensor(0.95, requires_grad=True) # Mémoire (Persistance)
alpha = torch.tensor(0.1,  requires_grad=True) # Impact GK (Intraday)
theta = torch.tensor(-0.1, requires_grad=True) # Effet Levier (Asymétrie)
gamma = torch.tensor(0.1,  requires_grad=True) # Impact VIX

# On groupe les paramètres pour l'optimiseur
params = [omega, beta, alpha, theta, gamma]
optimizer = optim.Adam(params, lr=LR)

# --- B. Fonction de calcul de la variance (La Recette) ---
def calculate_egarch_variance(gk, vix, sign):
    """Calcule toute la série de variance temporelle."""
    log_sigma2 = []
    # Initialisation avec la moyenne
    curr = torch.mean(gk)
    log_sigma2.append(curr)
    
    # Boucle Temporelle
    for t in range(1, len(gk)):
        # Formule EGARCH : Log Variance de demain dépend d'aujourd'hui
        # On utilise les inputs à t-1 pour prédire t
        prev = log_sigma2[-1]
        
        nxt = omega + \
              beta * prev + \
              alpha * gk[t-1] + \
              theta * sign[t-1] * gk[t-1] + \
              gamma * vix[t-1]
        
        log_sigma2.append(nxt)
    
    return torch.stack(log_sigma2)

# --- C. Boucle d'Entraînement ---
for i in range(EPOCHS):
    optimizer.zero_grad() # Reset des gradients
    
    # 1. Forward : On calcule les variances prédites
    log_variance_pred = calculate_egarch_variance(gk_tensor, vix_tensor, sign_tensor)
    variance_pred = torch.exp(log_variance_pred) # On repasse en échelle réelle
    
    # 2. Loss : Log-Vraisemblance (On cherche à maximiser la probabilité des données)

    # Formule gaussienne de la log vraisemblance, à laquelle on a enlevé le terme constant. On fait la moyenne des observations car c'est une histoire de stabilité, pour être fidèle à la formule on aurait du utiliser la somme, mais de cette manière là les paramètres font des trop gros bonds et le modèle diverge
    loss = 0.5 * (log_variance_pred + (returns_tensor**2 / variance_pred)).mean()
    
    # 3. Backward : PyTorch calcule les gradients (Règle de la chaîne)
    # On dérive loss en fonction de chaque paramètre, ça nous donne une valeur pour chacune d'entre elle ce qui nous fournit notre gradient, par exemple d(L)/d(Beta) = volatilité t-1...
    '''A la suite de cette ligne, on a notre gradient pour chaque paramètre, calculé à l'aide de tous les jours de données, et des paramtères alpha beta omega... précédents
    optimizer.step() fait la descente de gradient : un calcul simple, il prend le gradient (obtenu à l'aide des paramètres de l'itération précédente)
    Pour obtenir les nouveaux paramètres, il soustrait les anciens paramètres (vecteur de paramètre) au learning rate (l'hyperparamètre) multiplié par le gradient (vecteur des dérivées partielles)'''
    loss.backward()
    
 
    # applique la descente de gradient : omega_old = omega_new - Learning Rate * Gradient
    # On utilise Adam plutôt SGD (Stochastic gradient descent). Adam se souvient de la direction de ses derniers pas, et rajoute du momentum
    optimizer.step()
    
    if i % 100 == 0:
        print(f"Epoch {i}: Loss = {loss.item():.5f}")

# --- D. Génération de la prédiction de base ---
with torch.no_grad(): # On arrête de calculer les gradients pour l'inférence
    log_var_final = calculate_egarch_variance(gk_tensor, vix_tensor, sign_tensor)
    # Racine carrée pour avoir la Volatilité (et pas la Variance)
    pred_base = torch.sqrt(torch.exp(log_var_final)).numpy()

# On ajoute la prédiction au DataFrame
df['Pred_Base'] = pred_base


print("\n=== 3. ÉTAGE 2 : CORRECTEUR XGBOOST ===")

# 1. Calcul de l'erreur du GARCH (Le Résidu)
# Cible XGBoost = Ce qui s'est passé - Ce que GARCH a prédit
df['Residual'] = df['Realized_Vol'] - df['Pred_Base']

# 2. Découpage Temporel (Respecter la chronologie !)
split_index = len(df) - TEST_DAYS

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:].copy()


# 3. Préparation des Features pour XGBoost
# IMPORTANT : On utilise les colonnes 'XGB_...' qui sont déjà décalées (Shift-1)
# On ajoute aussi la prédiction de base comme info pour le booster
features = ['XGB_RSI', 'XGB_VIX', 'XGB_Vol_MA', 'XGB_VIX_Panic','Pred_Base']
target = 'Residual'

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]

# 4. Entraînement du Booster
# Il apprend à prédire l'erreur du GARCH en fonction du contexte (RSI, VIX...)
xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05)
xgb.fit(X_train, y_train)

print("\n--- IMPORTANCE DES VARIABLES ---")
importance = pd.DataFrame({
    'Feature': features,
    'Importance': xgb.feature_importances_
}).sort_values(by='Importance', ascending=False)
print(importance)
print("--------------------------------\n")

# 5. Prédiction de la Correction
correction = xgb.predict(X_test)


print("\n=== 4. ASSEMBLAGE FINAL ET RÉSULTATS ===")

# La Chimère = Base + Correction
test_df['Correction'] = correction
test_df['Pred_Final'] = test_df['Pred_Base'] + test_df['Correction']

# Sécurité : Pas de volatilité négative
test_df['Pred_Final'] = test_df['Pred_Final'].clip(lower=0.1)

# Calcul de la performance (MAE)
mae_base = mean_absolute_error(test_df['Realized_Vol'], test_df['Pred_Base'])
mae_final = mean_absolute_error(test_df['Realized_Vol'], test_df['Pred_Final'])
gain = (mae_base - mae_final) / mae_base * 100

print(f"Période de Test : Les {TEST_DAYS} derniers jours.")
print("-" * 40)
print(f"Erreur Moyenne (GARCH Seul) : {mae_base:.5f}")
print(f"Erreur Moyenne (Chimera)    : {mae_final:.5f}")
print("-" * 40)
print(f"GAIN DE PRÉCISION : +{gain:.2f}%")

# Sauvegarde pour analyse Excel
test_df.to_csv("chimera_results.csv")
print("\nFichier 'chimera_results.csv' généré.")



print("\n=== 5. PRÉDICTION POUR JOUR SUIVANT ===")

# last inputs
last_gk = gk_tensor[-1]
last_vix = vix_tensor[-1]
last_sign = sign_tensor[-1]

# last GARCH state
with torch.no_grad():
    log_vars = calculate_egarch_variance(gk_tensor, vix_tensor, sign_tensor)
    last_log_var = log_vars[-1]

# C. Faire un pas vers le futur (Formule GARCH manuelle)
# On utilise les params optimisés (omega, beta...)
next_log_var = omega + \
               beta * last_log_var + \
               alpha * last_gk + \
               theta * last_sign * last_gk + \
               gamma * last_vix

next_pred_base = torch.sqrt(torch.exp(next_log_var)).item()

# Correction XGboost, on prend les valeurs brutes de Vendredi (pas les colonnes Shiftées XGB_...)

last_features = pd.DataFrame([{
    'XGB_RSI': df['RSI'].iloc[-1],      # RSI de Vendredi
    'XGB_VIX': df['VIX'].iloc[-1],      # VIX de Vendredi
    'XGB_Vol_MA': df['Vol_MA_20'].iloc[-1], # Vol MA de Vendredi
    # AJOUT DE LA NOUVELLE FEATURE :
    # On prend le 'VIX_Panic' brut de Vendredi, car pour Lundi, c'est l'info "d'hier"
    'XGB_VIX_Panic': df['VIX_Panic'].iloc[-1], 
    'Pred_Base': next_pred_base         
}])

next_correction = xgb.predict(last_features)[0]

# E. Résultat Final
next_pred_final = next_pred_base + next_correction
next_pred_final = max(0.1, next_pred_final) # Sécurité

print(f"--- PRÉVISION POUR LUNDI ---")
print(f"Base GARCH   : {next_pred_base:.4f}%")
print(f"Ajustement   : {next_correction:.4f}%")
print(f"VOLATILITÉ ESTIMÉE : {next_pred_final:.4f}%")
print(f"Zone Probable (1 sigma) : +/- {next_pred_final:.2f}%")