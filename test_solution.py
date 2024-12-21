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

def reset_daily_states():
    """Reset all provider states and counters for a new day"""
    print("\nResetting daily states...")
    
    # Reset provider tracking from olution.py
    from olution import recent_success_tracker, reset_daily_states as reset_solution_states
    
    # Reset the solution states
    reset_solution_states()
    
    print("Daily states reset complete")

def process_day(day_number, payments_file, providers, cache_path, force_reprocess=False):
    """Process a single day's transactions"""
    print(f"\nProcessing Day {day_number} transactions...")
    
    # Try to load cached results first
    results = None if force_reprocess else load_results(cache_path)
    
    if results is None:
        # Reset states before processing
        reset_daily_states()
        
        # Process transactions
        results = simulate_transactions_parallel(providers, str(payments_file), use_gpu=True)
        
        # Format results keeping evaluation columns
        results = format_final_output(results, for_output=False)
        
        # Cache results
        save_results(results, cache_path)
    
    return results

def main(force_reprocess=False, hyperparams=None):
    """Modified main function with clear day separation"""
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
    
    # Process each day separately
    results_1 = process_day(1, paths['payments_1'], providers, cache_paths['results_1'], force_reprocess)
    results_2 = process_day(2, paths['payments_2'], providers, cache_paths['results_2'], force_reprocess)
    
    # Calculate metrics for both days
    metrics_day1 = calculate_metrics(results_1, providers)
    metrics_day2 = calculate_metrics(results_2, providers)
    
    # Print evaluation results
    print("\n=== Results for Payment Set 1 ===")
    evaluate_solution(results_1, providers)
    
    print("\n=== Results for Payment Set 2 ===")
    evaluate_solution(results_2, providers)
    
    # Save final output format
    final_results_1 = format_final_output(results_1, for_output=True)
    final_results_2 = format_final_output(results_2, for_output=True)
    
    # Save final results in required format
    save_results(final_results_1, results_dir / 'final_payment_set_1.csv')
    save_results(final_results_2, results_dir / 'final_payment_set_2.csv')
    
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
        
        # Format results before returning
        final_results = format_final_output(final_results)
        
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
    """Save processed results to a file"""
    try:
        if filepath.suffix == '.csv':
            results.to_csv(filepath, index=False)
        else:
            results.to_parquet(filepath, compression='gzip')
        print(f"Results saved to {filepath}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")

def format_final_output(results_df, for_output=False):
    """Format results to match required output format
    
    Args:
        results_df: DataFrame with results
        for_output: If True, only keep columns required for final output
    """
    # Create a copy to avoid modifying the original
    df = results_df.copy()
    
    # Clean up flow column (empty string for failures) before removing status column
    df['flow'] = df.apply(
        lambda row: row['flow'] if row.get('status') == 'CAPTURED' else '',
        axis=1
    )
    
    if for_output:
        # For final output, keep only required columns
        payment_columns = ['payment', 'amount', 'cur', 'eventTimeRes', 'flow']
        final_df = df[payment_columns]
    else:
        # For evaluation, keep additional columns
        required_cols = [
            'payment', 'amount', 'cur', 'eventTimeRes', 
            'flow', 'status', 'provider', 'amount_usd'
        ]
        final_df = df[required_cols]
    
    return final_df

if __name__ == "__main__":
    args = parse_args()
    main(force_reprocess=args.force_reprocess) 