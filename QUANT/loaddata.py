from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import requests
import os
from datetime import datetime, timedelta
import threading
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('EODHD_API_KEY')
# Noms interdits par Windows
WINDOWS_RESERVED = ['CON', 'PRN', 'AUX', 'NUL', 
                    'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
                    'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9']

# Lock pour affichage thread-safe
print_lock = threading.Lock()

def safe_filename(ticker):
    """Convertit un ticker en nom de fichier sûr pour Windows"""
    if ticker.upper() in WINDOWS_RESERVED:
        return f"{ticker}_ticker"
    return ticker

def thread_print(message):
    """Print thread-safe"""
    with print_lock:
        print(message)

def download_via_yfinance(ticker):
    """Télécharge les données via yfinance en fallback"""
    try:
        safe_name = safe_filename(ticker)
        
        # Télécharger les données
        stock = yf.Ticker(ticker)
        hist = stock.history(period="max")
        
        if hist.empty:
            return False, "pas de données"
        
        # Sauvegarder les données OHLCV (non ajustées)
        hist_reset = hist.reset_index()
        with open(f'adjusted_data/{safe_name}.csv', 'w') as f:
            f.write("Date,Open,High,Low,Close,Volume\n")
            for _, row in hist_reset.iterrows():
                date = row['Date'].strftime('%Y-%m-%d')
                f.write(f"{date},{row['Open']},{row['High']},{row['Low']},{row['Close']},{int(row['Volume'])}\n")
        
        # Récupérer les splits
        splits = stock.splits
        if not splits.empty:
            with open(f'split/{safe_name}.csv', 'w') as f:
                f.write('Date,"Stock Splits"\n')
                for date, ratio in splits.items():
                    date_str = date.strftime('%Y-%m-%d')
                    # yfinance donne le ratio directement (ex: 2.0 pour 2:1)
                    f.write(f'{date_str},{ratio:.6f}/1.000000\n')
        
        # Récupérer les dividendes
        dividends = stock.dividends
        if not dividends.empty:
            with open(f'dividend/{safe_name}.csv', 'w') as f:
                f.write('Date,Dividends\n')
                for date, div in dividends.items():
                    date_str = date.strftime('%Y-%m-%d')
                    f.write(f'{date_str},{div}\n')
        
        return True, "yfinance"
        
    except Exception as e:
        return False, str(e)

def download_full_history(ticker, download_splits, download_dividends):
    """Télécharge l'historique complet pour un ticker manquant"""
    safe_name = safe_filename(ticker)
    
    try:
        # Télécharger les données OHLCV
        url = f"https://eodhd.com/api/eod/{ticker}.US?period=d&api_token={API_KEY}=csv"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200 or len(response.text) < 50:
            # Fallback vers yfinance
            return download_via_yfinance(ticker)
        
        # Sauvegarder les données brutes
        lines = response.text.strip().split('\n')
        if len(lines) <= 1:
            return download_via_yfinance(ticker)
        
        with open(f'adjusted_data/{safe_name}.csv', 'w') as f:
            f.write("Date,Open,High,Low,Close,Volume\n")
            for line in lines[1:]:
                parts = line.split(',')
                if len(parts) >= 6:
                    f.write(f"{parts[0]},{parts[1]},{parts[2]},{parts[3]},{parts[4]},{parts[6]}\n")
        
        # Télécharger les splits si demandé
        if download_splits:
            url = f"https://eodhd.com/api/splits/{ticker}.US?period=d&api_token={API_KEY}=csv"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(f'split/{safe_name}.csv', 'w') as f:
                    f.write(response.text)
        
        # Télécharger les dividendes si demandé
        if download_dividends:
            url = f"https://eodhd.com/api/div/{ticker}.US?period=d&api_token={API_KEY}=csv"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(f'dividend/{safe_name}.csv', 'w') as f:
                    f.write(response.text)
        
        return True, "téléchargé"
        
    except Exception as e:
        # Dernier recours: yfinance
        return download_via_yfinance(ticker)

def update_ticker(args):
    """Fonction pour mettre à jour un ticker"""
    i, ticker, download_splits, download_dividends = args
    safe_name = safe_filename(ticker)
    file_path = f'adjusted_data/{safe_name}.csv'
    
    # Si le fichier n'existe pas, télécharger l'historique complet
    if not os.path.exists(file_path):
        thread_print(f"[{i}] {ticker}: NOUVEAU - téléchargement complet...")
        success, msg = download_full_history(ticker, download_splits, download_dividends)
        if success:
            thread_print(f"[{i}] {ticker}: ✓ historique complet téléchargé ({msg})")
            return ticker, 'downloaded_full'
        else:
            thread_print(f"[{i}] {ticker}: ERREUR téléchargement - {msg}")
            return ticker, f'error_{msg}'
    
    # Sinon, mise à jour normale
    try:
        df = pd.read_csv(file_path)
        last_date = df['Date'].iloc[-1]
        last_date_obj = datetime.strptime(last_date, '%Y-%m-%d')
        
        days_diff = (datetime.now() - last_date_obj).days
        if days_diff <= 1:
            thread_print(f"[{i}] {ticker}: à jour (dernière date: {last_date})")
            return ticker, 'up_to_date'
        
        start_date = (last_date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
        today = datetime.now().strftime('%Y-%m-%d')
        
        thread_print(f"[{i}] {ticker}: mise à jour depuis {start_date}...")
        
        # Mettre à jour les données OHLCV
        url = f"https://eodhd.com/api/eod/{ticker}.US?from={start_date}&to={today}&period=d&api_token={API_KEY}=csv"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            thread_print(f"[{i}] {ticker}: erreur HTTP {response.status_code}, essai avec yfinance...")
            # Fallback yfinance pour mise à jour
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date)
                if not hist.empty:
                    hist_reset = hist.reset_index()
                    with open(file_path, 'a') as f:
                        for _, row in hist_reset.iterrows():
                            date = row['Date'].strftime('%Y-%m-%d')
                            f.write(f"{date},{row['Open']},{row['High']},{row['Low']},{row['Close']},{int(row['Volume'])}\n")
                    return ticker, f'updated_{len(hist)}_yfinance'
            except:
                pass
            return ticker, f'error_http_{response.status_code}'
        
        if len(response.text) < 50:
            thread_print(f"[{i}] {ticker}: pas de nouvelles données")
            return ticker, 'no_new_data'
        
        new_lines = response.text.strip().split('\n')
        if len(new_lines) <= 1:
            thread_print(f"[{i}] {ticker}: réponse vide")
            return ticker, 'no_new_data'
        
        new_data_lines = new_lines[1:]
        
        with open(file_path, 'a') as f:
            for line in new_data_lines:
                parts = line.split(',')
                if len(parts) >= 6:
                    f.write(f"{parts[0]},{parts[1]},{parts[2]},{parts[3]},{parts[4]},{parts[6]}\n")
        
        # Mettre à jour splits si demandé
        if download_splits:
            url = f"https://eodhd.com/api/splits/{ticker}.US?from={start_date}&to={today}&period=d&api_token={API_KEY}=csv"
            response = requests.get(url, timeout=10)
            if response.status_code == 200 and len(response.text) > 50:
                split_file = f'split/{safe_name}.csv'
                # Ajouter ou créer le fichier split
                if os.path.exists(split_file):
                    with open(split_file, 'a') as f:
                        split_lines = response.text.strip().split('\n')[1:]
                        for line in split_lines:
                            f.write(line + '\n')
                else:
                    with open(split_file, 'w') as f:
                        f.write(response.text)
        
        # Mettre à jour dividendes si demandé
        if download_dividends:
            url = f"https://eodhd.com/api/div/{ticker}.US?from={start_date}&to={today}&period=d&api_token={API_KEY}=csv"
            response = requests.get(url, timeout=10)
            if response.status_code == 200 and len(response.text) > 50:
                div_file = f'dividend/{safe_name}.csv'
                if os.path.exists(div_file):
                    with open(div_file, 'a') as f:
                        div_lines = response.text.strip().split('\n')[1:]
                        for line in div_lines:
                            f.write(line + '\n')
                else:
                    with open(div_file, 'w') as f:
                        f.write(response.text)
        
        thread_print(f"[{i}] {ticker}: ✓ mis à jour (+{len(new_data_lines)} jours)")
        return ticker, f'updated_{len(new_data_lines)}'
        
    except Exception as e:
        thread_print(f"[{i}] {ticker}: ERREUR - {str(e)}")
        return ticker, f'error_{str(e)}'

# ============= MENU PRINCIPAL =============

print("="*60)
print("MISE À JOUR DES DONNÉES DE MARCHÉ")
print("="*60)

# Question 0: Mode single ticker ou tous les tickers
while True:
    choice = input("\nMODE:\n  1. Mettre à jour TOUS les tickers\n  2. Télécharger UN SEUL ticker\nChoix (1/2): ").strip()
    if choice in ['1', '2']:
        single_mode = (choice == '2')
        break
    print("Entrez 1 ou 2.")

if single_mode:
    ticker_input = input("\nEntrez le ticker à télécharger (ex: AAPL): ").strip().upper()
    tickers = [ticker_input]
    print(f"\nTéléchargement de {ticker_input}...")
else:
    # Lire tickers depuis le fichier
    print("\nLecture des tickers...")
    with open('tickers.txt', 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]
    print(f"{len(tickers)} tickers trouvés")

# Question 1: Télécharger les splits ?
while True:
    choice = input("\nTélécharger/mettre à jour les SPLITS ? (o/n): ").lower().strip()
    if choice in ['o', 'n', 'oui', 'non', 'y', 'yes']:
        download_splits = choice in ['o', 'oui', 'y', 'yes']
        break
    print("Réponse invalide. Entrez 'o' pour oui ou 'n' pour non.")

# Question 2: Télécharger les dividendes ?
while True:
    choice = input("Télécharger/mettre à jour les DIVIDENDES ? (o/n): ").lower().strip()
    if choice in ['o', 'n', 'oui', 'non', 'y', 'yes']:
        download_dividends = choice in ['o', 'oui', 'y', 'yes']
        break
    print("Réponse invalide. Entrez 'o' pour oui ou 'n' pour non.")

# Question 3: Nombre de threads (seulement en mode multi)
if not single_mode:
    while True:
        try:
            num_threads = input("\nNombre de threads (10-200, recommandé: 50-100): ").strip()
            num_threads = int(num_threads)
            if 1 <= num_threads <= 200:
                break
            print("Entrez un nombre entre 1 et 200.")
        except:
            print("Entrez un nombre valide.")
else:
    num_threads = 1

print("\n" + "="*60)
print(f"Configuration:")
print(f"  - Mode: {'Single ticker' if single_mode else 'Tous les tickers'}")
print(f"  - Splits: {'OUI' if download_splits else 'NON'}")
print(f"  - Dividendes: {'OUI' if download_dividends else 'NON'}")
if not single_mode:
    print(f"  - Threads: {num_threads}")
print("="*60)

# Créer les dossiers s'ils n'existent pas
os.makedirs('adjusted_data', exist_ok=True)
if download_splits:
    os.makedirs('split', exist_ok=True)
if download_dividends:
    os.makedirs('dividend', exist_ok=True)

print(f"\nTraitement de {len(tickers)} ticker(s)...")
print("="*60 + "\n")

# Créer des tuples (index, ticker, options)
ticker_args = [(i+1, ticker, download_splits, download_dividends) for i, ticker in enumerate(tickers)]

# Exécution
if single_mode:
    results = [update_ticker(ticker_args[0])]
else:
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(update_ticker, ticker_args))

# Comptage des résultats
updated = sum(1 for _, status in results if status.startswith('updated'))
downloaded = sum(1 for _, status in results if status == 'downloaded_full')
up_to_date = sum(1 for _, status in results if status == 'up_to_date')
no_data = sum(1 for _, status in results if status == 'no_new_data')
errors = sum(1 for _, status in results if status.startswith('error'))

print("\n" + "="*60)
print("RÉSUMÉ")
print("="*60)
print(f"Nouveaux téléchargés : {downloaded}")
print(f"Mis à jour           : {updated}")
print(f"Déjà à jour          : {up_to_date}")
print(f"Pas de données       : {no_data}")
print(f"Erreurs              : {errors}")
print(f"Total                : {len(tickers)}")

# Afficher quelques erreurs si présentes
if errors > 0:
    print("\nPremières erreurs :")
    error_list = [(ticker, status) for ticker, status in results if status.startswith('error')]
    for ticker, status in error_list[:20]:
        print(f"  {ticker}: {status}")

# Afficher nouveaux tickers téléchargés
if downloaded > 0:
    print("\nNouveaux tickers téléchargés :")
    download_list = [(ticker, status) for ticker, status in results if status == 'downloaded_full']
    for ticker, status in download_list[:20]:
        print(f"  {ticker}")