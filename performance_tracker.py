import pandas as pd

# Records trades and stores them in a vector. Includes summary report

class PerformanceTracker:
    def __init__(self):
        self.trades = []

    def record_trade(self, ticker, price, profit):
        self.trades.append({"Ticker": ticker, "Price": price, "Profit": profit})

    def get_summary(self):
        df = pd.DataFrame(self.trades)
        
        summary = df.groupby("Ticker").agg(
            Total_Trades=("Profit", "count"),
            Volume_Traded=("Price", "sum"),
            Total_Profit=("Profit", "sum"),
            Average_Profit=("Profit", "mean"),
            Success_Rate=("Profit", lambda x: (x > 0).mean()),  
        ).reset_index()

        summary["Result (%)"] = 100 + (summary["Total_Profit"] / summary["Volume_Traded"]) * 100
        summary["Return (%)"] = summary["Result (%)"] - 100
        return summary
