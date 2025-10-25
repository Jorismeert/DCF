import os
import pandas as pd
import time
import numpy as np
from yahooquery import Ticker 
from scipy.stats import gmean
import json

pd.set_option('display.max_rows', None)

def get_financial_data(symbol):
    """Get and clean financial data from Yahoo Finance"""
    stock = Ticker(symbol)
    
    # Get cash flow statement
    df_cash = pd.DataFrame(stock.cash_flow())
    df_cash['asOfDate'] = pd.to_datetime(df_cash['asOfDate'])
    df_cash.set_index('asOfDate', inplace=True)
    cash_period = df_cash['periodType'].iloc[0]
    df_cash = df_cash.iloc[:,2:]
    
    # Get balance sheet
    df_balance = pd.DataFrame(stock.balance_sheet())
    df_balance['asOfDate'] = pd.to_datetime(df_balance['asOfDate'])
    df_balance.set_index('asOfDate', inplace=True)
    df_balance = df_balance.iloc[:,2:]
    
    # Get current share price and shares outstanding
    price_data = stock.price
    current_price = price_data[symbol]['regularMarketPrice']
    shares_outstanding = price_data[symbol]['marketCap'] / current_price
    
    return df_cash, df_balance, current_price, shares_outstanding

def column_to_list(df, column_name):
    """Extract column data as cleaned list"""
    data_list = df[column_name].tolist()
    data_list = [x for x in data_list if pd.notnull(x)]
    return data_list

def calculate_fcf_growth_rate(historic_fcf, method='geometric'):
    """
    Calculate average FCF growth rate with robust error handling
    """
    if len(historic_fcf) < 2:
        return 0
    
    historic_fcf = np.array(historic_fcf, dtype=float)
    
    try:
        if method == 'arithmetic':
            growth_rates = (historic_fcf[1:] - historic_fcf[:-1]) / np.where(historic_fcf[:-1] != 0, historic_fcf[:-1], np.nan)
            valid_rates = growth_rates[np.isfinite(growth_rates)]
            return np.mean(valid_rates) if len(valid_rates) > 0 else 0
        
        elif method == 'geometric':
            growth_factors = historic_fcf[1:] / historic_fcf[:-1]
            valid_factors = growth_factors[np.isfinite(growth_factors) & (growth_factors > 0)]
            if len(valid_factors) > 0:
                return gmean(valid_factors) - 1
            else:
                return 0
    except Exception as e:
        print(f"Error calculating growth rate: {e}")
        return 0

def dcf_valuation(historic_fcf, net_debt, shares_outstanding, discount_rate=0.08, 
                  projection_years=5, terminal_growth_rate=0.025, growth_decay=0.1):
    """
    Complete DCF valuation with equity value calculation
    """
    # Input validation
    if not historic_fcf or len(historic_fcf) < 2:
        raise ValueError("Need at least 2 historical FCF values")
    
    if discount_rate <= terminal_growth_rate:
        raise ValueError("Discount rate must be greater than terminal growth rate")
    
    # Calculate growth rate
    fcf_avg_growth_rate = calculate_fcf_growth_rate(historic_fcf, method='geometric')
    
    # Cap unrealistic growth rates for stability
    realistic_base_growth = np.clip(fcf_avg_growth_rate, -0.2, 0.5)
    
    # Project explicit forecast period
    projected_fcf = []
    discount_factors = []
    current_fcf = historic_fcf[-1]
    
    for year in range(1, projection_years + 1):
        # Apply decaying growth rate
        decayed_growth = realistic_base_growth * (1 - growth_decay) ** (year - 1)
        
        # Ensure minimum growth
        final_growth_rate = max(decayed_growth, terminal_growth_rate)
        
        current_fcf = current_fcf * (1 + final_growth_rate)
        projected_fcf.append(current_fcf)
        
        # Calculate discount factor
        discount_factor = 1 / (1 + discount_rate) ** year
        discount_factors.append(discount_factor)
    
    # Calculate terminal value
    terminal_value = (current_fcf * (1 + terminal_growth_rate)) / (discount_rate - terminal_growth_rate)
    
    # Discount terminal value to present
    terminal_discount_factor = 1 / (1 + discount_rate) ** projection_years
    present_terminal_value = terminal_value * terminal_discount_factor
    
    # Calculate present value of projected FCF
    present_value_fcf = np.sum(np.array(projected_fcf) * np.array(discount_factors))
    
    # Total enterprise value
    enterprise_value = present_value_fcf + present_terminal_value
    
    # Calculate equity value (Enterprise Value - Net Debt)
    equity_value = enterprise_value - net_debt
    
    # Calculate intrinsic value per share
    intrinsic_value_per_share = equity_value / shares_outstanding
    
    return {
        'enterprise_value': enterprise_value,
        'equity_value': equity_value,
        'intrinsic_value_per_share': intrinsic_value_per_share,
        'projected_fcf': projected_fcf,
        'terminal_value': terminal_value,
        'present_terminal_value': present_terminal_value,
        'present_value_fcf': present_value_fcf,
        'base_growth_rate': fcf_avg_growth_rate,
        'used_growth_rate': realistic_base_growth,
        'net_debt': net_debt,
        'shares_outstanding': shares_outstanding
    }

def sensitivity_analysis(historic_fcf, net_debt, shares_outstanding, base_discount_rate=0.08):
    """Perform sensitivity analysis on key assumptions"""
    scenarios = {}
    
    # Test different discount rates and terminal growth rates
    for dr in [0.07, 0.08, 0.09, 0.10]:
        for tg in [0.02, 0.025, 0.03]:
            try:
                scenario_key = f"DR_{dr}_TG_{tg}"
                scenarios[scenario_key] = dcf_valuation(
                    historic_fcf, 
                    net_debt,
                    shares_outstanding,
                    discount_rate=dr,
                    terminal_growth_rate=tg
                )
            except Exception as e:
                print(f"Scenario failed: {scenario_key}, Error: {e}")
    
    return scenarios

def print_valuation_results(results, current_price, symbol):
    """Print formatted valuation results"""
    print("=" * 60)
    print(f"DCF VALUATION ANALYSIS - {symbol}")
    print("=" * 60)
    
    print(f"\nHistorical Analysis:")
    print(f"Base FCF Growth Rate: {results['base_growth_rate']:.2%}")
    print(f"Used Growth Rate: {results['used_growth_rate']:.2%}")
    print(f"Net Debt: ${results['net_debt']/1e9:.2f}B")
    print(f"Shares Outstanding: {results['shares_outstanding']/1e6:.0f}M")
    
    print(f"\nProjected Free Cash Flow (next 5 years):")
    for i, fcf in enumerate(results['projected_fcf'], 1):
        print(f"  Year {i}: ${fcf/1e9:.2f}B")
    
    print(f"\nValuation Results:")
    print(f"Enterprise Value: ${results['enterprise_value']/1e9:.2f}B")
    print(f"Equity Value: ${results['equity_value']/1e9:.2f}B")
    print(f"Intrinsic Value per Share: ${results['intrinsic_value_per_share']:.2f}")
    print(f"Current Market Price: ${current_price:.2f}")
    
    margin_of_safety = ((results['intrinsic_value_per_share'] - current_price) / current_price) * 100
    print(f"Margin of Safety: {margin_of_safety:+.1f}%")
    
    if results['intrinsic_value_per_share'] > current_price:
        print("RECOMMENDATION: UNDERVALUED")
    else:
        print("RECOMMENDATION: OVERVALUED")

# Main execution
if __name__ == "__main__":
    symbol = 'CAT'
    
    try:
        print(f"Fetching financial data for {symbol}...")
        df_cash, df_balance, current_price, shares_outstanding = get_financial_data(symbol)
        
        # Extract key financial data
        net_debt = df_balance['NetDebt'].iloc[-1] if 'NetDebt' in df_balance.columns else 0
        historic_fcf = column_to_list(df_cash, 'FreeCashFlow')
        
        print(f"Historical FCF: {[f'{x/1e9:.1f}B' for x in historic_fcf]}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Net Debt: ${net_debt/1e9:.2f}B")
        print(f"Shares Outstanding: {shares_outstanding/1e6:.0f}M")
        
        # Run DCF valuation
        print(f"\nRunning DCF Valuation...")
        dcf_results = dcf_valuation(
            historic_fcf, 
            net_debt, 
            shares_outstanding,
            discount_rate=0.08  # Adjust based on company risk
        )
        
        # Print results
        print_valuation_results(dcf_results, current_price, symbol)
        
        # Sensitivity analysis
        print(f"\nSENSITIVITY ANALYSIS")
        print("=" * 40)
        scenarios = sensitivity_analysis(historic_fcf, net_debt, shares_outstanding)
        
        print("\nIntrinsic Value per Share under different scenarios:")
        for scenario, results in scenarios.items():
            print(f"{scenario}: ${results['intrinsic_value_per_share']:.2f}")
            
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()



with open("/Users/jorismeert/Desktop/Python/ProjectDCF/data/ticker_DJ.json", "r", encoding="utf-8") as f:
    dow_jones = json.load(f)

for c in dow_jones["DowJones"]:
    print(f"{c['ticker']}: {c['company']}")


