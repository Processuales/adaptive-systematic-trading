import pandas as pd
import matplotlib.pyplot as plt
import os

# Configuration
DATA_DIR = "data"
SYMBOLS = ["SPY", "QQQ"]
COLORS = {"SPY": "#ff0000", "QQQ": "#0000ff"}
OUTPUT_PLOT = "spy_qqq_close_chart.png"

def plot_symbols():
    plt.figure(figsize=(12, 6))
    
    for symbol in SYMBOLS:
        csv_path = os.path.join(DATA_DIR, f"{symbol.lower()}_1h_rth.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found. Skipping {symbol}.")
            continue

        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)

        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date for chronological plotting
        df = df.sort_values('date')

        print(f"Plotting {len(df)} rows for {symbol}...")
        plt.plot(df['date'], df['close'], label=f'{symbol} Close', color=COLORS[symbol], linewidth=1.5)
    
    # Formatting
    plt.title('SPY vs QQQ Closing Prices (Hourly RTH)', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plt.savefig(OUTPUT_PLOT)
    print(f"Plot saved to {OUTPUT_PLOT}")

    # Show the plot
    print("Displaying plot window...")
    plt.show()

if __name__ == "__main__":
    plot_symbols()
