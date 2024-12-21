import pandas as pd
import numpy as np
from pathlib import Path

def load_datasets(data_dir):
    """Load all datasets from the data directory"""
    try:
        ex_rates = pd.read_csv(f"{data_dir}/ex_rates.csv")
        payments_1 = pd.read_csv(f"{data_dir}/payments_1.csv")
        payments_2 = pd.read_csv(f"{data_dir}/payments_2.csv")
        providers_1 = pd.read_csv(f"{data_dir}/providers_1.csv")
        providers_2 = pd.read_csv(f"{data_dir}/providers_2.csv")
        
        # Convert timestamps
        payments_1['eventTimeRes'] = pd.to_datetime(payments_1['eventTimeRes'])
        payments_2['eventTimeRes'] = pd.to_datetime(payments_2['eventTimeRes'])
        providers_1['TIME'] = pd.to_datetime(providers_1['TIME'])
        providers_2['TIME'] = pd.to_datetime(providers_2['TIME'])
        
        return {
            'ex_rates': ex_rates,
            'payments_1': payments_1,
            'payments_2': payments_2,
            'providers_1': providers_1,
            'providers_2': providers_2
        }
    except Exception as e:
        print(f"Error loading datasets: {str(e)}")
        raise

def analyze_payments(payments_1, payments_2, ex_rates):
    """Analyze payment patterns and characteristics"""
    all_payments = pd.concat([payments_1, payments_2])
    
    # Add USD amount column
    rate_dict = dict(zip(ex_rates['destination'], ex_rates['rate']))
    all_payments['amount_usd'] = all_payments.apply(
        lambda x: x['amount'] * rate_dict.get(x['cur'], 1), axis=1
    )
    
    # Basic statistics
    basic_stats = {
        'total_transactions': len(all_payments),
        'unique_cards': all_payments['cardToken'].nunique(),
        'currencies': all_payments['cur'].unique().tolist(),
        'total_volume_usd': all_payments['amount_usd'].sum(),
        'avg_transaction_usd': all_payments['amount_usd'].mean()
    }
    
    # Currency-specific analysis
    currency_stats = all_payments.groupby('cur').agg({
        'amount': ['count', 'sum', 'mean', 'std'],
        'amount_usd': ['sum', 'mean', 'std'],
        'cardToken': 'nunique'
    }).round(2)
    
    # Card usage patterns - modified approach
    card_transactions = all_payments.groupby('cardToken').agg({
        'amount_usd': ['count', 'sum', 'mean']
    })
    
    # Calculate currencies per card separately
    currencies_per_card = all_payments.groupby('cardToken')['cur'].unique()
    multi_currency_cards = currencies_per_card[currencies_per_card.apply(len) > 1]
    
    return {
        'basic_stats': basic_stats,
        'currency_stats': currency_stats,
        'card_stats': {
            'total_cards': len(card_transactions),
            'multi_currency_cards': len(multi_currency_cards),
            'avg_transactions_per_card': card_transactions['amount_usd']['count'].mean(),
            'max_transactions_per_card': card_transactions['amount_usd']['count'].max()
        }
    }

def analyze_providers(providers_1, providers_2):
    """Analyze provider characteristics and changes"""
    all_providers = pd.concat([providers_1, providers_2])
    
    # Basic provider statistics
    provider_stats = {
        'day1_providers': len(providers_1),
        'day2_providers': len(providers_2),
        'currencies_supported': all_providers['CURRENCY'].unique().tolist()
    }
    
    # Provider changes analysis
    provider_changes = {}
    for provider_id in all_providers['ID'].unique():
        day1 = providers_1[providers_1['ID'] == provider_id]
        day2 = providers_2[providers_2['ID'] == provider_id]
        
        if not day1.empty and not day2.empty:
            provider_changes[provider_id] = {
                'conversion_change': day2['CONVERSION'].iloc[0] - day1['CONVERSION'].iloc[0],
                'commission_change': day2['COMMISSION'].iloc[0] - day1['COMMISSION'].iloc[0],
                'time_change': day2['AVG_TIME'].iloc[0] - day1['AVG_TIME'].iloc[0]
            }
    
    # Currency support analysis
    currency_support = all_providers.groupby(['CURRENCY']).agg({
        'ID': 'count',
        'CONVERSION': ['mean', 'std', 'min', 'max'],
        'COMMISSION': ['mean', 'std', 'min', 'max'],
        'AVG_TIME': ['mean', 'std', 'min', 'max'],
        'LIMIT_MIN': ['mean', 'min', 'max'],
        'LIMIT_MAX': ['mean', 'min', 'max']
    }).round(4)
    
    return {
        'provider_stats': provider_stats,
        'provider_changes': provider_changes,
        'currency_support': currency_support
    }

def analyze_volume_requirements(providers_1, providers_2, payments_1, payments_2, ex_rates):
    """Analyze volume requirements and limits"""
    all_providers = pd.concat([providers_1, providers_2])
    all_payments = pd.concat([payments_1, payments_2])
    
    # Convert amounts to USD
    rate_dict = dict(zip(ex_rates['destination'], ex_rates['rate']))
    all_payments['amount_usd'] = all_payments.apply(
        lambda x: x['amount'] * rate_dict.get(x['cur'], 1), axis=1
    )
    
    # Daily volume analysis
    daily_volumes = all_payments.groupby(['cur', all_payments['eventTimeRes'].dt.date]).agg({
        'amount': ['sum', 'count'],
        'amount_usd': ['sum', 'mean']
    }).round(2)
    
    # Provider limit analysis
    limit_analysis = []
    for currency in all_providers['CURRENCY'].unique():
        curr_providers = all_providers[all_providers['CURRENCY'] == currency]
        curr_volumes = daily_volumes.loc[currency] if currency in daily_volumes.index else pd.DataFrame()
        
        if not curr_volumes.empty:
            avg_daily_volume = curr_volumes['amount_usd']['sum'].mean()
            
            for _, provider in curr_providers.iterrows():
                limit_min_usd = provider['LIMIT_MIN'] * rate_dict.get(currency, 1)
                limit_max_usd = provider['LIMIT_MAX'] * rate_dict.get(currency, 1)
                
                limit_analysis.append({
                    'currency': currency,
                    'provider_id': provider['ID'],
                    'avg_daily_volume': avg_daily_volume,
                    'limit_min_usd': limit_min_usd,
                    'limit_max_usd': limit_max_usd,
                    'meets_min': avg_daily_volume >= limit_min_usd,
                    'within_max': avg_daily_volume <= limit_max_usd,
                    'potential_penalty': 0 if avg_daily_volume >= limit_min_usd else limit_min_usd * 0.01
                })
    
    return {
        'daily_volumes': daily_volumes,
        'limit_analysis': pd.DataFrame(limit_analysis)
    }

def main():
    try:
        # Load datasets
        datasets = load_datasets('data')
        
        # Perform analysis
        analysis_results = {
            'payments': analyze_payments(
                datasets['payments_1'],
                datasets['payments_2'],
                datasets['ex_rates']
            ),
            'providers': analyze_providers(
                datasets['providers_1'],
                datasets['providers_2']
            ),
            'volume': analyze_volume_requirements(
                datasets['providers_1'],
                datasets['providers_2'],
                datasets['payments_1'],
                datasets['payments_2'],
                datasets['ex_rates']
            )
        }
        
        # Save results as JSON-like structure
        with open('docs/eda_report.md', 'w') as f:
            f.write("# Payment System Analysis Results\n\n")
            f.write("```python\n")
            f.write(str(analysis_results))
            f.write("\n```")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 