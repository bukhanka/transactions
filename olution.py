import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.model_selection import train_test_split
import joblib
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool
import os  # For CPU count
import cupy as cp  # Add at top with other imports
from numba import cuda  # For custom CUDA kernels if needed
import time
from tqdm import tqdm  # Add at top with other imports
from functools import partial
from collections import deque
from random import random

# At the top of the file after imports
ex_rates = None

# Make a short-horizon record of recent events for each provider
recent_success_tracker = defaultdict(lambda: deque(maxlen=500))

# Add near the top with other constants
max_fallback_attempts = 2  # Maximum number of providers to try in a chain

def initialize_ex_rates(ex_rates_df):
    """Initialize exchange rates globally"""
    global ex_rates
    ex_rates = ex_rates_df

# Load data
def load_data(filepath):
    """Load data from CSV file"""
    return pd.read_csv(filepath)

# --- 1. Data Preparation and Transformation ---
def prepare_provider_data(providers_1, providers_2):
    """Simplified provider data preparation"""
    providers = pd.concat([providers_1, providers_2], ignore_index=True)
    providers['TIME'] = pd.to_datetime(providers['TIME'])
    providers.sort_values(by=['TIME', 'ID'], inplace=True)
    return providers

def convert_currency_to_usd(amount, currency, ex_rates):
    """Converts amount to USD using provided exchange rates with improved error handling"""
    try:
        if currency == 'USD':
            return amount
        rate_row = ex_rates[ex_rates['destination'] == currency]
        if rate_row.empty:
            print(f"Warning: No exchange rate found for {currency}, using 1.0")
            return amount
        rate = rate_row.iloc[0]['rate']
        return amount * rate
    except Exception as e:
        print(f"Error converting currency: {str(e)}")
        return amount  # Return original amount if conversion fails

def get_provider_data_at_time(providers, time):
    """Optimized provider lookup using binary search"""
    time = pd.to_datetime(time)
    
    # Get indices where TIME <= target_time using binary search
    idx = providers['TIME'].searchsorted(time, side='right') - 1
    if idx < 0:
        return pd.DataFrame()
        
    # Get latest data for each provider up to this time
    latest_providers = (providers.iloc[:idx+1]
                       .groupby('ID')
                       .last()
                       .reset_index())
    
    return latest_providers


def calculate_penalty(amount, min_limit):
  """calculates penalty for provider based on minimum limit. """
  if amount < min_limit :
    return (min_limit - amount) * 0.01
  else:
    return 0

def apply_sliding_average(providers, window_size=3):
    """Applies a simple moving average to providers' numeric columns"""
    numeric_cols = providers.select_dtypes(include=np.number).columns
    numeric_cols = numeric_cols.drop('ID') if 'ID' in numeric_cols else numeric_cols
    
    # Group by ID and apply rolling average
    for col in numeric_cols:
        providers[col] = providers.groupby('ID')[col].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).mean()
        )
    
    return providers

def initialize_providers(providers_1_df, providers_2_df):
    """Initialize and prepare provider data"""
    providers = prepare_provider_data(providers_1_df, providers_2_df)
    
    # Skip sliding average for now as it's causing issues
    # providers = apply_sliding_average(providers, 3)
    
    # Verify we have data
    if providers.empty:
        raise ValueError("No provider data available after initialization")
        
    return providers


# --- 2. Scoring Model ---

def calculate_penalty_vectorized(amounts, min_limits):
    """Vectorized version of penalty calculation"""
    penalties = np.zeros_like(amounts, dtype=float)
    mask = amounts < min_limits
    penalties[mask] = (min_limits[mask] - amounts[mask]) * 0.01
    return penalties

def score_provider_vectorized(providers_df, amounts_usd, daily_amounts):
    """Vectorized scoring for multiple providers"""
    w1, w2, w3, w4 = 1.0, 0.5, 1.0, 1.0
    
    # Ensure inputs are numpy arrays
    amounts_usd = np.asarray(amounts_usd).reshape(-1, 1)
    
    # Calculate components
    profits = amounts_usd * (1 - providers_df['COMMISSION'].values)
    time_scores = 1/providers_df['AVG_TIME'].values
    conversion_scores = providers_df['CONVERSION'].values
    
    # Calculate penalties using daily amounts
    penalties = calculate_penalty_vectorized(
        daily_amounts + amounts_usd.flatten(),
        providers_df['LIMIT_MIN'].values
    )
    
    # Combine scores
    scores = (w1 * profits.flatten() + 
             w2 * time_scores + 
             w3 * conversion_scores - 
             w4 * penalties)
    
    return scores

# --- 3. Provider Selection ---
def select_providers(providers_at_time, payment_amount, currency, previous_daily_amounts_used, hyperparams=None):
    """Enhanced provider selection with time constraints"""
    if hyperparams is None:
        hyperparams = {
            'penalty_weight': 1.3,
            'balance_factor': 0.25,
            'conversion_weight': 1.0,
            'time_weight': 0.5,
            'utilization_boost': 0.2,
            'max_chain_time': 180,  # Maximum allowed chain time in seconds
            'max_chain_length': 3   # Maximum allowed chain length
        }
    
    scored_providers = []
    
    try:
        payment_amount_usd = convert_currency_to_usd(payment_amount, currency, ex_rates)
    except ValueError:
        return []
    
    total_daily_volume = sum(previous_daily_amounts_used.values())
    
    for _, provider in providers_at_time.iterrows():
        if (payment_amount >= provider['MIN_SUM'] and 
            payment_amount <= provider['MAX_SUM'] and
            provider['CURRENCY'] == currency):

            base_score = score_provider(
                provider,
                payment_amount_usd,
                previous_daily_amounts_used[provider['ID']],
                hyperparams
            )

            # Calculate utilization metrics with configurable boost
            current_volume = previous_daily_amounts_used[provider['ID']]
            min_limit = provider['LIMIT_MIN']
            max_limit = provider['LIMIT_MAX']
            
            utilization_factor = 1.0
            
            if current_volume < min_limit:
                remaining_to_min = min_limit - current_volume
                utilization_factor += hyperparams['utilization_boost'] * (remaining_to_min / min_limit)
            
            if current_volume > max_limit * 0.8:
                utilization_factor -= hyperparams['utilization_boost'] * ((current_volume - max_limit * 0.8) / (max_limit * 0.2))

            combined_score = (
                base_score * (1 - hyperparams['balance_factor']) +
                utilization_factor * hyperparams['balance_factor'] -
                hyperparams['penalty_weight'] * calculate_penalty(
                    current_volume + payment_amount_usd,
                    min_limit
                )
            )

            scored_providers.append({
                'ID': provider['ID'],
                'score': combined_score,
                'provider': provider
            })

    # Filter out providers that would exceed time constraints
    current_chain_time = sum(p['provider']['AVG_TIME'] for p in scored_providers[:len(flow)])
    if current_chain_time > hyperparams['max_chain_time']:
        return []
        
    # Limit chain length
    if len(flow) >= hyperparams['max_chain_length']:
        return []

    return sorted(scored_providers, key=lambda x: x['score'], reverse=True)

def score_provider(provider, payment_amount_usd, previous_daily_amount_used, hyperparams):
    """Calculate base score for provider"""
    w1 = hyperparams['conversion_weight']  # profit weight
    w2 = hyperparams['time_weight']  # time weight 
    w3 = 1.0  # conversion weight
    
    # Get dynamic conversion estimate
    dynamic_conversion = get_recent_conversion_estimate(provider['ID'])
    base_conversion = provider['CONVERSION']
    effective_conversion = 0.5 * base_conversion + 0.5 * dynamic_conversion

    # Calculate components
    profit = payment_amount_usd * (1 - provider['COMMISSION'])
    time_score = 1 / provider['AVG_TIME']
    
    # Combine scores
    score = (w1 * profit + 
             w2 * time_score + 
             w3 * effective_conversion)
    
    return score

# --- 4. Transaction Simulation ---
def simulate_transactions_parallel(providers, transactions_file, num_processes=None, use_gpu=True):
    """Process transactions in parallel chunks"""
    try:
        if num_processes is None:
            num_processes = max(1, os.cpu_count() - 1)  # Leave one CPU free
            
        transactions = load_data(transactions_file)
        
        # Validate required columns
        required_columns = {'amount', 'cur', 'eventTimeRes', 'payment'}
        missing_columns = required_columns - set(transactions.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns in transactions file: {missing_columns}")
        
        transactions['eventTimeRes'] = pd.to_datetime(transactions['eventTimeRes'])
        transactions = transactions.sort_values('eventTimeRes')
        
        # Split into evenly sized chunks
        chunk_size = 5000  # Smaller chunks for better parallelization
        chunks = split_into_chunks(transactions, chunk_size)
        total_chunks = len(chunks)
        total_transactions = len(transactions)
        
        print(f"\nProcessing {total_transactions:,} transactions in {total_chunks} chunks...")
        print(f"Using {num_processes} processes")
        print(f"Average chunk size: {total_transactions // total_chunks:,} transactions")
        print(f"GPU acceleration: {'enabled' if use_gpu else 'disabled'}\n")
        
        # Process chunks in parallel with progress bar
        results = []
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # Submit all tasks
            future_to_chunk = {
                executor.submit(process_transaction_chunk, chunk, providers, use_gpu): chunk 
                for chunk in chunks
            }
            
            # Process results as they complete with progress bar
            with tqdm(total=total_transactions, desc="Processing transactions") as pbar:
                for future in as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    try:
                        result = future.result()
                        if result is not None and not result.empty:
                            results.append(result)
                        pbar.update(len(chunk))
                    except Exception as e:
                        print(f"\nError processing chunk: {str(e)}")
                        continue
            
        if not results:
            print("Warning: No results were processed successfully")
            return create_empty_results()
            
        final_results = pd.concat(results, ignore_index=True)
        print(f"\nSuccessfully processed {len(final_results):,} transactions")
        return final_results
        
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        return create_empty_results()

def process_chunk_wrapper(args):
    """Wrapper for chunk processing with improved error handling"""
    chunk, providers = args
    try:
        return process_transaction_chunk(chunk, providers)
    except Exception as e:
        print(f"Error processing chunk: {str(e)}")
        return pd.DataFrame()

def process_transaction_chunk(chunk, providers, use_gpu=True):
    """Process chunk with improved fallback chain logic"""
    try:
        # Create a copy of providers to avoid sharing memory between processes
        providers = providers.copy()
        
        results = []
        daily_amounts = defaultdict(float)
        
        # Convert amounts to USD upfront
        chunk = chunk.copy()
        chunk['amount_usd'] = chunk.apply(
            lambda x: convert_currency_to_usd(x['amount'], x['cur'], ex_rates),
            axis=1
        )
        
        # Get provider data for the chunk's time range
        chunk_end_time = chunk['eventTimeRes'].max()
        provider_data = get_provider_data_at_time(providers, chunk_end_time)
        
        if provider_data.empty:
            return pd.DataFrame([{
                'payment': row['payment'],
                'flow': 'failed',
                'status': 'FAILED',
                'provider': None,
                'eventTimeRes': row['eventTimeRes'],
                'amount_usd': row['amount_usd'],
                'amount': row['amount'],  # Preserve original amount
                'cur': row['cur']  # Preserve original currency
            } for _, row in chunk.iterrows()])
        
        # Process transactions
        for _, txn in chunk.iterrows():
            try:
                # Filter valid providers
                valid_providers = provider_data[
                    (txn['amount'] >= provider_data['MIN_SUM']) & 
                    (txn['amount'] <= provider_data['MAX_SUM']) & 
                    (txn['cur'] == provider_data['CURRENCY'])
                ].copy()
                
                if valid_providers.empty:
                    results.append({
                        'payment': txn['payment'],
                        'flow': 'failed',
                        'status': 'FAILED',
                        'provider': None,
                        'eventTimeRes': txn['eventTimeRes'],
                        'amount_usd': txn['amount_usd'],
                        'amount': txn['amount'],  # Preserve original amount
                        'cur': txn['cur']  # Preserve original currency
                    })
                    continue
                
                # Score providers
                provider_daily_amounts = np.array([daily_amounts[pid] for pid in valid_providers['ID']])
                
                if use_gpu and len(valid_providers) > 10:
                    scores = score_provider_vectorized_gpu(
                        valid_providers,
                        np.array([txn['amount_usd']]),
                        provider_daily_amounts
                    )
                else:
                    scores = score_provider_vectorized(
                        valid_providers,
                        np.array([txn['amount_usd']]),
                        provider_daily_amounts
                    )
                
                valid_providers['score'] = scores
                scored_providers = valid_providers.sort_values('score', ascending=False)
                
                selected = None
                flow = []
                attempt_count = 0
                
                # Try providers in order of score until success or max attempts reached
                for _, provider in scored_providers.iterrows():
                    if attempt_count >= max_fallback_attempts:
                        break
                        
                    provider = provider.copy()  # Avoid modifying original
                    provider['chain_index'] = attempt_count
                    flow.append(str(provider['ID']))
                    
                    # Calculate success probability
                    dynamic_conv = get_recent_conversion_estimate(provider['ID'])
                    base_conv = provider['CONVERSION']
                    effective_conv = 0.5 * base_conv + 0.5 * dynamic_conv
                    
                    # Check if we can use this provider (within limits)
                    can_use = (daily_amounts[provider['ID']] + txn['amount_usd'] <= provider['LIMIT_MAX'])
                    success_prob = effective_conv if can_use else 0.0
                    
                    # Simulate success/failure
                    if random() <= success_prob:
                        selected = provider
                        daily_amounts[provider['ID']] += txn['amount_usd']
                        update_recent_success(provider['ID'], True)
                        break
                    else:
                        update_recent_success(provider['ID'], False)
                    
                    attempt_count += 1
                
                # Record result
                results.append({
                    'payment': txn['payment'],
                    'flow': '-'.join(flow) if flow else 'failed',
                    'status': 'CAPTURED' if selected is not None else 'FAILED',
                    'provider': selected['ID'] if selected is not None else None,
                    'eventTimeRes': txn['eventTimeRes'],
                    'amount_usd': txn['amount_usd'],
                    'amount': txn['amount'],  # Preserve original amount
                    'cur': txn['cur']  # Preserve original currency
                })
                
            except Exception as e:
                print(f"Error processing transaction: {str(e)}")
                results.append({
                    'payment': txn['payment'],
                    'flow': 'failed',
                    'status': 'FAILED',
                    'provider': None,
                    'eventTimeRes': txn['eventTimeRes'],
                    'amount_usd': txn.get('amount_usd', 0),
                    'amount': txn.get('amount', 0),  # Preserve original amount
                    'cur': txn.get('cur', '')  # Preserve original currency
                })
                continue
        
        return pd.DataFrame(results)
        
    except Exception as e:
        print(f"Error processing chunk: {str(e)}")
        return create_empty_results()

# --- Main Simulation ---

# Remove all the global execution code and print statements
# Keep only the function definitions

def calculate_metrics(transactions_results, providers):
    """Calculate metrics with improved error handling"""
    try:
        # Calculate basic metrics
        total_payments = len(transactions_results)
        successful_mask = transactions_results['status'] == 'CAPTURED'
        successful_payments = successful_mask.sum()
        total_conversion_rate = successful_payments / total_payments if total_payments else 0
        
        # Calculate profits and penalties
        total_profit = 0
        total_penalties = 0
        
        if successful_payments > 0:
            successful_txns = transactions_results[successful_mask]
            
            # Group by date for daily calculations
            successful_txns['date'] = pd.to_datetime(successful_txns['eventTimeRes']).dt.date
            
            for date, group in successful_txns.groupby('date'):
                provider_data = get_provider_data_at_time(
                    providers, 
                    pd.Timestamp(date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                )
                
                if not provider_data.empty:
                    # Calculate profits
                    merged = group.merge(
                        provider_data[['ID', 'COMMISSION']], 
                        left_on='provider', 
                        right_on='ID'
                    )
                    total_profit += (merged['amount_usd'] * (1 - merged['COMMISSION'])).sum()
                    
                    # Calculate penalties
                    daily_volumes = group.groupby('provider')['amount_usd'].sum()
                    for provider_id, volume in daily_volumes.items():
                        provider = provider_data[provider_data['ID'] == provider_id].iloc[0]
                        total_penalties += calculate_penalty(volume, provider['LIMIT_MIN'])
        
        return {
            'profit': total_profit,
            'conversion_rate': total_conversion_rate,
            'penalties': total_penalties,
            'total_transactions': total_payments
        }
        
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return {
            'profit': 0,
            'conversion_rate': 0,
            'penalties': 0,
            'total_transactions': 0
        }

class EnhancedMLScorer:
    def __init__(self):
        # Add trained flag
        self.is_trained = False
        # Create a stacking ensemble
        self.base_models = [
            ('rf', RandomForestRegressor(n_estimators=100)),
            ('gb', GradientBoostingRegressor()),
            ('xgb', xgb.XGBRegressor(objective='binary:logistic'))
        ]
        self.meta_model = LogisticRegression()
        self.model = StackingRegressor(
            estimators=self.base_models,
            final_estimator=self.meta_model
        )
        self.scaler = MinMaxScaler()
        self.feature_importance = {}
        self.model_performance = []
        self.last_retrain = None
        self.retrain_interval = timedelta(hours=1)
        
    def prepare_features(self, provider, payment_amount, history_metrics, time_features=None):
        """Enhanced feature engineering"""
        base_features = [
            provider['CONVERSION'],
            provider['AVG_TIME'],
            provider['COMMISSION'],
            payment_amount,
            provider['LIMIT_MAX'],
            provider['LIMIT_MIN'],
            payment_amount / provider['LIMIT_MAX'],  # Utilization ratio
            provider['LIMIT_MIN'] / payment_amount,  # Risk ratio
        ]
        
        if history_metrics:
            historical_features = [
                history_metrics['real_conversion'],
                history_metrics['avg_processing_time'],
                history_metrics['total_amount'],
                history_metrics['success_trend'],
                history_metrics['volume_trend'],
                history_metrics['recent_failures']
            ]
        else:
            historical_features = [
                provider['CONVERSION'], 
                provider['AVG_TIME'],
                0, 0, 0, 0
            ]
            
        if time_features:
            time_based_features = [
                time_features['hour'],
                time_features['day_of_week'],
                time_features['is_weekend'],
                time_features['is_peak_hour']
            ]
        else:
            time_based_features = [0, 0, 0, 0]
            
        return np.array(base_features + historical_features + time_based_features).reshape(1, -1)
        
    def train(self, historical_data):
        """Enhanced training with feature importance analysis"""
        if len(historical_data) < 10:  # Don't train with too little data
            return
            
        X = self._prepare_training_features(historical_data)
        y = historical_data['success']
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        # Calculate feature importance
        feature_names = self._get_feature_names()
        importances = np.mean([model.feature_importances_ 
                             for name, model in self.base_models 
                             if hasattr(model, 'feature_importances_')], axis=0)
        
        self.feature_importance = dict(zip(feature_names, importances))
        self.last_retrain = datetime.now()
        self.is_trained = True
        
        # Track model performance
        y_pred = self.model.predict_proba(X_scaled)[:, 1]
        auc_score = roc_auc_score(y, y_pred)
        self.model_performance.append({
            'timestamp': datetime.now(),
            'auc_score': auc_score,
            'data_size': len(y)
        })
        
    def predict_success_probability(self, provider, payment_amount, history_metrics, 
                                  current_time=None):
        """Enhanced prediction with confidence scoring"""
        # Return default prediction if not trained
        if not self.is_trained:
            return provider['CONVERSION']  # Use provider's stated conversion rate as fallback
            
        if self._should_retrain(current_time):
            recent_data = self._get_recent_data()
            if len(recent_data) >= 10:  # Only train if we have enough data
                self.train(recent_data)
            
        features = self.prepare_features(
            provider, 
            payment_amount, 
            history_metrics,
            self._extract_time_features(current_time) if current_time else None
        )
        features_scaled = self.scaler.transform(features)
        
        # Get predictions from all base models
        base_predictions = [model.predict_proba(features_scaled)[:, 1] 
                          for _, model in self.base_models]
        
        # Calculate prediction confidence
        prediction_mean = np.mean(base_predictions)
        prediction_std = np.std(base_predictions)
        confidence_score = 1 - (prediction_std / prediction_mean if prediction_mean > 0 else 1)
        
        return prediction_mean * confidence_score
    
    def _should_retrain(self, current_time):
        """Check if model should be retrained"""
        if not self.last_retrain or not current_time:
            return True
        return current_time - self.last_retrain > self.retrain_interval
    
    def _extract_time_features(self, timestamp):
        """Extract time-based features"""
        return {
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'is_weekend': 1 if timestamp.weekday() >= 5 else 0,
            'is_peak_hour': 1 if 9 <= timestamp.hour <= 18 else 0
        }
    
    def _get_feature_names(self):
        """Get feature names for importance tracking"""
        return [
            'conversion', 'avg_time', 'commission', 'amount', 
            'limit_max', 'limit_min', 'utilization_ratio', 'risk_ratio',
            'historical_conversion', 'historical_avg_time', 'historical_amount',
            'success_trend', 'volume_trend', 'recent_failures',
            'hour', 'day_of_week', 'is_weekend', 'is_peak_hour'
        ]

    def _prepare_training_features(self, historical_data):
        """Prepare features for model training"""
        features = []
        for _, row in historical_data.iterrows():
            provider_features = self.prepare_features(
                row['provider_data'],
                row['amount'],
                row['history_metrics'],
                self._extract_time_features(row['timestamp']) if 'timestamp' in row else None
            )
            features.append(provider_features.ravel())
        return np.array(features)

    def _get_recent_data(self):
        """Get recent transaction data for retraining"""
        # Return empty DataFrame with correct structure
        return pd.DataFrame({
            'provider_data': [],
            'amount': [],
            'success': [],
            'timestamp': [],
            'history_metrics': []
        })

class EnhancedPerformanceTracker:
    def __init__(self):
        self.history = defaultdict(lambda: {
            'success_count': 0,
            'total_count': 0,
            'processing_times': [],
            'amounts': [],
            'last_update': None,
            'success_history': [],
            'failure_history': [],
            'volume_history': [],
            'recent_window': 100  # Track last N transactions
        })
        
    def update(self, provider_id, success, amount, processing_time):
        """Enhanced update with trend analysis"""
        data = self.history[provider_id]
        data['total_count'] += 1
        
        if success:
            data['success_count'] += 1
            data['success_history'].append(1)
        else:
            data['failure_history'].append(1)
            
        data['amounts'].append(amount)
        data['processing_times'].append(processing_time)
        data['volume_history'].append(amount)
        data['last_update'] = datetime.now()
        
        # Maintain fixed window size
        for key in ['success_history', 'failure_history', 'volume_history']:
            if len(data[key]) > data['recent_window']:
                data[key] = data[key][-data['recent_window']:]
                
    def get_metrics(self, provider_id):
        """Enhanced metrics with trend analysis"""
        data = self.history[provider_id]
        if data['total_count'] == 0:
            return None
            
        recent_success_rate = np.mean(data['success_history'][-20:]) if data['success_history'] else 0
        historical_success_rate = data['success_count'] / data['total_count']
        
        return {
            'real_conversion': data['success_count'] / data['total_count'],
            'avg_processing_time': np.mean(data['processing_times']),
            'total_amount': sum(data['amounts']),
            'transaction_count': data['total_count'],
            'success_trend': recent_success_rate - historical_success_rate,
            'volume_trend': self._calculate_trend(data['volume_history']),
            'recent_failures': sum(1 for x in data['failure_history'][-10:] if x == 0)
        }
        
    def _calculate_trend(self, history):
        """Calculate trend using simple linear regression"""
        if len(history) < 2:
            return 0
        x = np.arange(len(history)).reshape(-1, 1)
        y = np.array(history).reshape(-1, 1)
        slope = np.polyfit(x.ravel(), y.ravel(), 1)[0]
        return slope

class ProviderCache:
    def __init__(self):
        self._cache = {}
        
    def get_provider_data(self, time, providers):
        date_key = pd.Timestamp(time).date()
        if date_key not in self._cache:
            self._cache[date_key] = get_provider_data_at_time(providers, time)
        return self._cache[date_key]

def score_provider_vectorized_gpu(providers_df, amounts_usd, daily_amounts):
    """GPU-accelerated provider scoring with chain index support"""
    w1, w2, w3, w4, w5 = 1.0, 0.5, 1.0, 1.0, 0.2
    
    # Move data to GPU
    amounts_gpu = cp.asarray(amounts_usd).reshape(-1, 1)
    commission_gpu = cp.asarray(providers_df['COMMISSION'].values)
    time_gpu = cp.asarray(providers_df['AVG_TIME'].values)
    daily_amounts_gpu = cp.asarray(daily_amounts)
    limit_min_gpu = cp.asarray(providers_df['LIMIT_MIN'].values)
    chain_index_gpu = cp.asarray(providers_df['chain_index'].values if 'chain_index' in providers_df else cp.zeros_like(commission_gpu))
    
    # Calculate components on GPU
    profits = amounts_gpu * (1 - commission_gpu)
    time_scores = 1/time_gpu
    
    # Get conversion estimates
    conversion_scores = cp.zeros_like(commission_gpu)
    for i, provider in enumerate(providers_df.itertuples()):
        dynamic_conv = get_recent_conversion_estimate(provider.ID)
        base_conv = provider.CONVERSION
        conversion_scores[i] = 0.5 * base_conv + 0.5 * dynamic_conv
    
    # Calculate penalties
    penalties = cp.zeros_like(daily_amounts_gpu, dtype=cp.float32)
    mask = (daily_amounts_gpu + amounts_gpu.flatten()) < limit_min_gpu
    penalties[mask] = (limit_min_gpu[mask] - (daily_amounts_gpu[mask] + amounts_gpu.flatten()[mask])) * 0.01
    
    # Combine scores with chain penalty
    scores = (w1 * profits.flatten() + 
             w2 * conversion_scores + 
             w3 * time_scores - 
             w4 * penalties - 
             w5 * chain_index_gpu)
    
    # Move result back to CPU
    return cp.asnumpy(scores)

def create_empty_results():
    """Create an empty DataFrame with the correct structure for results"""
    return pd.DataFrame(columns=[
        'payment', 'flow', 'status', 'provider', 
        'eventTimeRes', 'amount_usd', 'amount', 'cur'
    ])

def split_into_chunks(df, chunk_size=10000):
    """Split DataFrame into roughly equal sized chunks"""
    return [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

def evaluate_solution(results_df, providers_df):
    """Detailed evaluation of the solution with improved error handling"""
    start_time = time.time()
    
    print("\n=== Solution Evaluation ===")
    
    # Verify DataFrame structure
    required_columns = {'status', 'flow', 'provider'}
    missing_columns = required_columns - set(results_df.columns)
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return
    
    # Basic metrics
    total_transactions = len(results_df)
    if total_transactions == 0:
        print("No transactions to evaluate")
        return
        
    successful_transactions = len(results_df[results_df['status'] == 'CAPTURED'])
    conversion_rate = successful_transactions / total_transactions if total_transactions > 0 else 0
    
    print(f"\nBasic Metrics:")
    print(f"Total Transactions: {total_transactions}")
    print(f"Successful Transactions: {successful_transactions}")
    print(f"Overall Conversion Rate: {conversion_rate:.4f}")
    
    # Flow analysis
    print("\nFlow Analysis:")
    flow_lengths = results_df['flow'].str.count('-') + 1
    print(f"Average Chain Length: {flow_lengths.mean():.2f}")
    print(f"Max Chain Length: {flow_lengths.max()}")
    
    # Provider utilization
    print("\nProvider Utilization:")
    successful_mask = results_df['status'] == 'CAPTURED'
    if successful_mask.any():
        provider_counts = results_df[successful_mask]['provider'].value_counts()
        for provider_id, count in provider_counts.items():
            print(f"Provider {provider_id}: {count} successful transactions")
    else:
        print("No successful transactions to analyze")
    
    # Calculate detailed metrics
    try:
        calculate_metrics(results_df, providers_df)
    except Exception as e:
        print(f"Error calculating detailed metrics: {str(e)}")
    
    elapsed = time.time() - start_time
    print(f"\nEvaluation time: {elapsed:.2f} seconds")

def save_results(results, filepath):
    """Save processed results to a file"""
    try:
        results.to_parquet(filepath, compression='gzip')
        print(f"Results saved to {filepath}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")

def load_results(filepath):
    """Load processed results from a file"""
    try:
        if os.path.exists(filepath):
            return pd.read_parquet(filepath)
        return None
    except Exception as e:
        print(f"Error loading results: {str(e)}")
        return None

def update_recent_success(provider_id, success):
    """Update short-horizon success metric."""
    recent_success_tracker[provider_id].append(1 if success else 0)

def get_recent_conversion_estimate(provider_id):
    """Compute short-horizon conversion estimate for a provider."""
    data = recent_success_tracker[provider_id]
    if len(data) == 0:
        # fallback if no recent data
        return 1.0
    return sum(data) / len(data)

def format_final_output(results_df):
    """Format results to match required output format"""
    # Get required columns for evaluation and output
    required_cols = ['payment', 'eventTimeRes', 'flow', 'status', 'provider', 'amount', 'cur', 'amount_usd']
    
    # Create final dataframe with required columns
    final_df = results_df[required_cols].copy()
    
    # Clean up flow column (empty string for failures)
    final_df.loc[final_df['status'] != 'CAPTURED', 'flow'] = ''
    
    return final_df

def reset_daily_states():
    """Reset all daily counters and states"""
    global recent_success_tracker
    
    # Reset provider tracking
    recent_success_tracker = defaultdict(lambda: deque(maxlen=500))
    
    # Reset ML model states if needed
    if hasattr(EnhancedMLScorer, 'instance'):
        EnhancedMLScorer.instance.reset_state()

class MetricsTracker:
    def __init__(self):
        self.metrics = {
            'total_transactions': 0,
            'successful_transactions': 0,
            'total_profit': 0,
            'total_penalties': 0,
            'chain_lengths': [],
            'processing_times': [],
            'provider_usage': defaultdict(float)
        }
        
    def update(self, transaction_result):
        """Update metrics with new transaction result"""
        self.metrics['total_transactions'] += 1
        
        if transaction_result['status'] == 'CAPTURED':
            self.metrics['successful_transactions'] += 1
            self.metrics['chain_lengths'].append(
                len(transaction_result['flow'].split('-'))
            )
            # Add other metric updates...
    
    def get_summary(self):
        """Get summary of current metrics"""
        return {
            'conversion_rate': self.metrics['successful_transactions'] / 
                             self.metrics['total_transactions'],
            'avg_chain_length': np.mean(self.metrics['chain_lengths']),
            'avg_processing_time': np.mean(self.metrics['processing_times']),
            'total_profit': self.metrics['total_profit'],
            'total_penalties': self.metrics['total_penalties']
        }
