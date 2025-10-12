import pandas as pd
import matplotlib.pyplot as plt
from ..core.utils import ensure_dir


def plot_equity_curve(equity_curve: pd.DataFrame, save_path: str):
    """
    Plots the equity curve and saves it to a file.
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot equity
    ax.plot(equity_curve.index,
            equity_curve['equity'], label='Equity', color='blue')
    ax.set_title('Portfolio Equity Curve', fontsize=16)
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity (USDT)')

    # Formatting
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    # Ensure the directory exists and save the figure
    ensure_dir(save_path.rsplit('/', 1)[0])
    plt.savefig(save_path)
    print(f"\nPerformance chart saved to: {save_path}")
