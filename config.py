"""Configuration settings for the payment routing solution"""

# Time-related parameters
MAX_CHAIN_LENGTH = 3  # Maximum providers in chain
MAX_TIME_PER_TRANSACTION = 300  # Maximum processing time (seconds)
TIME_PENALTY_WEIGHT = 0.8  # Weight for time penalties in scoring

# Conversion optimization
MIN_ACCEPTABLE_CONVERSION = 0.80  # Minimum acceptable conversion rate
CONVERSION_BOOST_THRESHOLD = 0.85  # Threshold for conversion rate bonus

# Provider chain parameters
OPTIMAL_CHAIN_LENGTH = 2  # Target number of providers in chain
CHAIN_LENGTH_PENALTY = 0.15  # Penalty per additional provider

# Economic parameters
PROFIT_WEIGHT = 1.0  # Base weight for profit calculations
PENALTY_WEIGHT = 1.3  # Weight for limit violation penalties
BALANCE_FACTOR = 0.25  # Factor for balancing between metrics

# Learning parameters
HISTORY_WINDOW = 1000  # Number of transactions to consider for learning
MIN_SAMPLES_FOR_UPDATE = 100  # Minimum samples before model update
LEARNING_RATE = 0.1  # Rate of model updates

def get_default_hyperparams():
    """Get default hyperparameter configuration"""
    return {
        'penalty_weight': PENALTY_WEIGHT,
        'balance_factor': BALANCE_FACTOR,
        'conversion_weight': 1.0,
        'time_weight': TIME_PENALTY_WEIGHT,
        'utilization_boost': 0.2
    } 