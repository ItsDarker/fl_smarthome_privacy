"""
Generate synthetic smart home sensor data with property labels
"""
import numpy as np
import pandas as pd
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def generate_smart_home_data():
    """
    Generate synthetic smart home data for all homes
    Each home has temperature, usage patterns, and has_elderly label
    """
    np.random.seed(config.RANDOM_SEED)
    
    data = []
    home_labels = {}
    
    for home_id in range(config.NUM_HOMES):
        # Randomly assign property: has_elderly (0 or 1)
        has_elderly = np.random.choice([0, 1])
        home_labels[home_id] = has_elderly
        
        # Generate sensor readings based on property
        if has_elderly:
            # Elderly homes: warmer temperature, consistent usage
            temps = np.random.normal(
                config.ELDERLY_TEMP_MEAN, 
                config.ELDERLY_TEMP_STD, 
                config.NUM_SAMPLES_PER_HOME
            )
            usage = np.random.normal(
                config.ELDERLY_USAGE_MEAN, 
                config.ELDERLY_USAGE_STD, 
                config.NUM_SAMPLES_PER_HOME
            )
        else:
            # Non-elderly homes: cooler temperature, variable usage
            temps = np.random.normal(
                config.NON_ELDERLY_TEMP_MEAN, 
                config.NON_ELDERLY_TEMP_STD, 
                config.NUM_SAMPLES_PER_HOME
            )
            usage = np.random.normal(
                config.NON_ELDERLY_USAGE_MEAN, 
                config.NON_ELDERLY_USAGE_STD, 
                config.NUM_SAMPLES_PER_HOME
            )
        
        # Clip values to realistic ranges
        temps = np.clip(temps, 60, 80)
        usage = np.clip(usage, 0, 1)
        
        # Generate energy consumption (target variable for FL task)
        # Simple model: energy depends on temperature deviation from 68F and usage
        baseline_temp = 68.0
        energy = (
            0.5 * np.abs(temps - baseline_temp) +  # Heating/cooling cost
            0.3 * usage * 10 +                      # Usage-based cost
            np.random.normal(0, 0.5, config.NUM_SAMPLES_PER_HOME)  # Noise
        )
        energy = np.clip(energy, 0, 15)  # Realistic energy range
        
        for i in range(config.NUM_SAMPLES_PER_HOME):
            data.append({
                'home_id': home_id,
                'temperature': temps[i],
                'usage': usage[i],
                'energy': energy[i],
                'has_elderly': has_elderly
            })
    
    df = pd.DataFrame(data)
    
    # Save dataset
    data_file = os.path.join(config.DATA_DIR, 'smart_home_data.csv')
    df.to_csv(data_file, index=False)
    
    # Save home labels separately
    labels_file = os.path.join(config.DATA_DIR, 'home_labels.pkl')
    with open(labels_file, 'wb') as f:
        pickle.dump(home_labels, f)
    
    if config.VERBOSE:
        print(f"Generated data for {config.NUM_HOMES} homes")
        print(f"Total samples: {len(df)}")
        print(f"Homes with elderly: {sum(home_labels.values())}")
        print(f"Homes without elderly: {config.NUM_HOMES - sum(home_labels.values())}")
        print(f"\nData saved to: {data_file}")
        print(f"Labels saved to: {labels_file}")
        print(f"\nData statistics:")
        print(df.groupby('has_elderly')[['temperature', 'usage', 'energy']].mean())
    
    return df, home_labels


def load_data():
    """Load previously generated data"""
    data_file = os.path.join(config.DATA_DIR, 'smart_home_data.csv')
    labels_file = os.path.join(config.DATA_DIR, 'home_labels.pkl')
    
    if not os.path.exists(data_file) or not os.path.exists(labels_file):
        print("Data not found. Generating new data...")
        return generate_smart_home_data()
    
    df = pd.read_csv(data_file)
    with open(labels_file, 'rb') as f:
        home_labels = pickle.load(f)
    
    if config.VERBOSE:
        print(f"Loaded data for {config.NUM_HOMES} homes")
        print(f"Total samples: {len(df)}")
    
    return df, home_labels


def get_client_data(df, home_id):
    """Get data for a specific home/client"""
    client_df = df[df['home_id'] == home_id].copy()
    
    X = client_df[['temperature', 'usage']].values
    y = client_df['energy'].values.reshape(-1, 1)
    has_elderly = client_df['has_elderly'].iloc[0]
    
    return X, y, has_elderly


def split_train_test_clients(num_homes):
    """Split homes into training and test sets"""
    np.random.seed(config.RANDOM_SEED)
    all_homes = np.arange(num_homes)
    np.random.shuffle(all_homes)
    
    split_idx = int(num_homes * config.TRAIN_TEST_SPLIT)
    train_homes = all_homes[:split_idx]
    test_homes = all_homes[split_idx:]
    
    return train_homes.tolist(), test_homes.tolist()


if __name__ == '__main__':
    print("Generating smart home dataset...")
    df, labels = generate_smart_home_data()
    print("\nDataset generation complete!")
    
    # Show sample data
    print("\nSample data (first 5 rows):")
    print(df.head())
