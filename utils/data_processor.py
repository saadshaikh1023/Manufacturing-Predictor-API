import pandas as pd
import numpy as np
import os

class DataProcessor:
    @staticmethod
    def generate_sample_data(n_samples=1000):
        np.random.seed(42)
        data = {
            'Machine_ID': [f'M{i}' for i in range(n_samples)],
            'Temperature': np.random.normal(80, 10, n_samples),
            'Run_Time': np.random.gamma(4, 50, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        high_temp = df['Temperature'] > 85
        long_runtime = df['Run_Time'] > 200
        normal_wear = np.random.random(n_samples) < (df['Run_Time'] / 1000)
        
        probability = (
            0.7 * (high_temp & long_runtime).astype(float) +
            0.2 * high_temp.astype(float) +
            0.3 * long_runtime.astype(float) +
            0.1 * normal_wear
        )
        
        df['Downtime_Flag'] = (np.random.random(n_samples) < probability).astype(int)
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Save to CSV
        df.to_csv('data/sample_data.csv', index=False)
        return df

if __name__ == "__main__":
    DataProcessor.generate_sample_data()