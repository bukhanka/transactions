import time
import pandas as pd
import numpy as np
from olution import *
import os
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import argparse

def setup_paths():
    """Setup data paths based on dataset.md"""
    base_dir = Path(os.getcwd())
    data_dir = base_dir / 'data'
    
    return {
        'ex_rates': data_dir / 'ex_rates.csv',
        'payments_1': data_dir / 'payments_1.csv',
        'payments_2': data_dir / 'payments_2.csv',
        'providers_1': data_dir / 'providers_1.csv',
        'providers_2': data_dir / 'providers_2.csv'
    }

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

def main(force_reprocess=False, hyperparams=None):
    """Modified main function with separate day handling"""
    if hyperparams is None:
        hyperparams = {
            'penalty_weight': 1.3,
            'balance_factor': 0.25,
            'conversion_weight': 1.0,
            'time_weight': 0.5,
            'utilization_boost': 0.2
        }
    
    total_start = time.time()
    
    # Setup paths
    paths = setup_paths()
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Define cache paths
    cache_paths = {
        'results_1': results_dir / 'payment_set_1_results.parquet',
        'results_2': results_dir / 'payment_set_2_results.parquet'
    }
    
    # Load data
    print("Loading data...")
    load_start = time.time()
    ex_rates_df = pd.read_csv(paths['ex_rates'])
    initialize_ex_rates(ex_rates_df)
    
    providers_1 = pd.read_csv(paths['providers_1'])
    providers_2 = pd.read_csv(paths['providers_2'])
    load_time = time.time() - load_start
    print(f"Data loading time: {load_time:.2f} seconds")
    
    # Initialize providers data
    print("\nInitializing providers...")
    init_start = time.time()
    providers = initialize_providers(providers_1, providers_2)
    init_time = time.time() - init_start
    print(f"Provider initialization time: {init_time:.2f} seconds")
    
    # Process payment set 1 (Day 1)
    print("\nProcessing Day 1 transactions...")
    results_1 = None if force_reprocess else load_results(cache_paths['results_1'])
    if results_1 is None:
        # Reset all daily counters and states
        reset_daily_states()
        results_1 = simulate_transactions_parallel(providers, str(paths['payments_1']), use_gpu=True)
        results_1 = format_final_output(results_1)
        save_results(results_1, cache_paths['results_1'])
    
    # Reset all states before Day 2
    reset_daily_states()
    
    # Process payment set 2 (Day 2)
    print("\nProcessing Day 2 transactions...")
    results_2 = None if force_reprocess else load_results(cache_paths['results_2'])
    if results_2 is None:
        results_2 = simulate_transactions_parallel(providers, str(paths['payments_2']), use_gpu=True)
        results_2 = format_final_output(results_2)
        save_results(results_2, cache_paths['results_2'])
    
    # Calculate metrics for both days
    metrics_day1 = calculate_metrics(results_1, providers)
    metrics_day2 = calculate_metrics(results_2, providers)
    
    # Print evaluation results
    print("\n=== Results for Payment Set 1 ===")
    evaluate_solution(results_1, providers)
    
    print("\n=== Results for Payment Set 2 ===")
    evaluate_solution(results_2, providers)
    
    # Combine metrics from both days
    total_metrics = {
        'profit': metrics_day1.get('profit', 0) + metrics_day2.get('profit', 0),
        'conversion_rate': (metrics_day1.get('conversion_rate', 0) + metrics_day2.get('conversion_rate', 0)) / 2,
        'penalties': metrics_day1.get('penalties', 0) + metrics_day2.get('penalties', 0),
        'total_transactions': metrics_day1.get('total_transactions', 0) + metrics_day2.get('total_transactions', 0)
    }
    
    total_time = time.time() - total_start
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    
    return total_metrics

def simulate_transactions_parallel(providers, transactions_file, num_processes=None, use_gpu=True):
    """Process transactions in parallel chunks"""
    try:
        if num_processes is None:
            num_processes = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
            
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
    """Wrapper function to handle chunk processing"""
    if len(args) == 3:
        chunk, providers, use_gpu = args
    else:
        chunk, providers = args
        use_gpu = False
    try:
        return process_transaction_chunk(chunk, providers, use_gpu)
    except Exception as e:
        print(f"Error processing chunk: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force-reprocess', action='store_true', 
                       help='Force reprocessing even if cached results exist')
    return parser.parse_args()

def save_results(results, filepath):
    """Save processed results to a file with CSV fallback"""
    try:
        # Try parquet first
        results.to_parquet(filepath, compression='gzip')
        print(f"Results saved to {filepath}")
    except Exception as e:
        # Fallback to CSV if parquet fails
        csv_path = str(filepath).replace('.parquet', '.csv')
        results.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path} (CSV fallback)")

if __name__ == "__main__":
    args = parse_args()
    main(force_reprocess=args.force_reprocess) 