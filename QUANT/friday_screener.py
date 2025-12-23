import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import warnings

warnings.filterwarnings("ignore")

# --- PARAMÈTRES 
date_start_test = "2025-04-21" 
REPERTORY = "adjusted_data/"      
OUTPUT_FILE = "REPORT_FRIDAY.csv"
TICKER_FILE = "ndx.txt" 
# ------------------------------

def friday_screener(repertory, date_start, output_file):
    
    try:
        with open(TICKER_FILE, "r", encoding="utf-8") as f:
            files = [l.strip() + ".csv" for l in f.readlines() if l.strip()]
    except FileNotFoundError:
        files = [f for f in os.listdir(repertory) if f.endswith('.csv')]

    results_data = []

    print(f"--- ANALYSIS OF {len(files)} STOCKS ---")

    for f in files:
        stock_ticker = f.replace('.csv', '')
        file = os.path.join(repertory, f)
        
        if not os.path.exists(file):
            continue

        try:
            df = pd.read_csv(file)
            df.columns = df.columns.str.replace(' ', '')
            df["Date"] = pd.to_datetime(df["Date"])
            df["PctChange"] = df["Close"].pct_change() * 100 
            
            # Gap (Open - PrevClose) / PrevClose
            df["Gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1) * 100

            # Features
            df["Friday_Close_Return"] = df["PctChange"].shift(1)
            df["Friday_Gap"] = df["Gap"].shift(1)

            # Strategies
            strategies = {
                "1_Friday_Red_Close": df["Friday_Close_Return"] < 0,
                "2_Friday_Green_Gap":      df["Friday_Gap"] > 0,
                "3_Friday_Red_Gap":     df["Friday_Gap"] < 0,
                "4_Friday_Strong_Green_Gap":      df["Friday_Gap"] > 1,
                "5_Friday_Green_Close" : df["Friday_Close_Return"] > 0,
                "6_Friday_Strong_Red_Close": df["Friday_Close_Return"] < -2,
                "7_Friday_Red_Gap & Red Close":     (df["Friday_Gap"] < 0) & (df["Friday_Close_Return"] < 0),
                "8_Green_Gap_Red_Close": (df["Friday_Gap"] > 0) & (df["Friday_Close_Return"] < 0),
                "9_Red_Gap_Green_Close": (df["Friday_Gap"] < 0) & (df["Friday_Close_Return"] > 0)
            }

            # 2. ANALYSIS LOOP
            for strat_name, condition_check in strategies.items():
                
                df["Signal"] = ((df["Date"].dt.dayofweek == 0) & condition_check).astype(int)
                dfbis = df[(df["Date"] >= date_start)].copy()
                dfbis.dropna(subset=['PctChange', 'Friday_Close_Return', 'Friday_Gap'], inplace=True)

            

                # 3. REGRESSION
                y = dfbis["PctChange"]
                X = dfbis.loc[:, ["Signal"]] 
                X = sm.add_constant(X) 

                sm_model = sm.OLS(y, X).fit()
                
                # 4. results extraction
                alpha = sm_model.params.get('Signal', 0.0)  #alpha
                beta_zero = sm_model.params.get('const', 0.0)   # overall trend (overall, how does behave the said asset ?)
                p_value = sm_model.pvalues.get('Signal', 1.0) #p-value (really important as it tells us how statistically significant our results are as opposed to random)
                std_dev = y.std()

                # expected profit in % 
                total_expected = beta_zero + alpha

                strategy_nb_days = y.loc[dfbis['Signal'] == 1]
            
                if len(strategy_nb_days) > 0:
                    median_return = strategy_nb_days.median() 
                    win_rate = (strategy_nb_days > 0).mean() * 100 
                    worst_day = strategy_nb_days.min() 
                    best_day = strategy_nb_days.max() 
                    
                
                    # The closer it is from 0, the more significant the alpha
                    spread_mean_median = abs(alpha - median_return)
                else:
                    median_return, win_rate, worst_day, best_day, spread_mean_median = 0, 0, 0, 0, 0
                
                
                results_data.append({
                        "Asset": stock_ticker,
                        "EV (%)": round(total_expected, 4), 
                        "P_Value": round(p_value, 5),
                        "Strategy": strat_name,
                        "Alpha (%)": round(alpha, 4), 
                        "Alpha_Median (%)": round(median_return - beta_zero, 4), 
                        "Suspicious_spread": round(spread_mean_median, 4),
                        "Beta_Zero (%)": round(beta_zero, 4),   
                        "Is_Significant": p_value < 0.05,
                        "N_Events": int(dfbis['Signal'].sum()),
                        "Volatility (%)": round(std_dev, 4),
                        "Winrate (%)": round(win_rate, 2),
                        "Worst_Day (%)": round(worst_day, 2),
                        "Best_Day (%)" : round(best_day, 2)
                })

        except Exception as e:
            continue

    if not results_data: return

    results_df = pd.DataFrame(results_data)
    results_df = results_df[results_df["Winrate (%)"] > 70]
    #results_df = results_df[results_df["Suspicious_spread"] < 0.5]

    
    
    
    # p_value sorting
    results_df_sorted = results_df.sort_values(by=['P_Value', 'EV (%)'], ascending=[True, False])
    results_df_sorted.to_csv(output_file, index=False)

    print(f"\n--- ANALYSIS ---")
    print(results_df_sorted.head(5).to_string(index=False))

if __name__ == "__main__":
    friday_screener(REPERTORY, date_start_test, OUTPUT_FILE)