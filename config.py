"""
Configuration file for Federated Learning Smart Home Privacy Project
"""
import os

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Data generation parameters
NUM_HOMES = 50
NUM_SAMPLES_PER_HOME = 100
TRAIN_TEST_SPLIT = 0.7  # 70% training, 30% test

# Smart home data parameters
ELDERLY_TEMP_MEAN = 71.0  # Fahrenheit
ELDERLY_TEMP_STD = 2.0
ELDERLY_USAGE_MEAN = 0.8
ELDERLY_USAGE_STD = 0.1

NON_ELDERLY_TEMP_MEAN = 65.0
NON_ELDERLY_TEMP_STD = 3.0
NON_ELDERLY_USAGE_MEAN = 0.5
NON_ELDERLY_USAGE_STD = 0.2

# Federated Learning parameters
FL_ROUNDS = 20
LOCAL_EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 0.01
CLIENT_PARTICIPATION_RATE = 0.4  # 40% of clients per round

# Model architecture
INPUT_DIM = 2  # temperature, usage
HIDDEN_DIM = 16
OUTPUT_DIM = 1  # energy prediction

# Attack parameters
ATTACK_TRAIN_SPLIT = 0.7  # 70% for training attack model
RANDOM_FOREST_ESTIMATORS = 100
RANDOM_FOREST_MAX_DEPTH = 10
RANDOM_FOREST_MIN_SAMPLES_LEAF = 5

# Defense parameters - Differential Privacy
DP_EPSILONS = [0.1, 0.5, 1.0, 5.0, 10.0]
DP_SENSITIVITY = 0.5  # Gradient clipping norm

# Defense parameters - Gradient Compression
COMPRESSION_RATES = [0.1, 0.25, 0.5, 0.75]  # Keep top 10%, 25%, 50%, 75%

# Defense parameters - Secure Aggregation
SECURE_AGG_ENABLED = True  # Simulate secure aggregation

# Visualization parameters
FIGURE_SIZE = (10, 6)
DPI = 300
FONT_SIZE = 12

# Random seed for reproducibility
RANDOM_SEED = 42

# Verbose output
VERBOSE = True
