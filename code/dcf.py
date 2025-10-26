import os
import pandas as pd
import time
import numpy as np
from yahooquery import Ticker 
from scipy.stats import gmean
import json
from yahoo_fin import stock_info

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
    
    # Get additional info for growth classification
    try:
        info = stock.summary_profile
        industry = info[symbol]['industry'] if symbol in info else ''
        sector = info[symbol]['sector'] if symbol in info else ''
    except:
        industry = ''
        sector = ''
    
    return df_cash, df_balance, current_price, shares_outstanding, industry, sector

def column_to_list(df, column_name):
    """Extract column data as cleaned list"""
    data_list = df[column_name].tolist()
    data_list = [x for x in data_list if pd.notnull(x)]
    return data_list

def is_growth_company(symbol, historic_fcf, current_price, shares_outstanding, industry):
    """Identify if company should use growth DCF model"""
    market_cap = current_price * shares_outstanding
    
    # Growth industry check
    growth_industries = ['Auto Manufacturers', 'Software', 'Internet', 'Semiconductors', 
                        'Biotechnology', 'Solar', 'Renewable Energy', 'Technology']
    
    # High market cap + volatile FCF pattern often indicates growth company
    if market_cap > 500e9:  # $500B+ market cap
        return True
    
    # Check for growth characteristics
    if len(historic_fcf) >= 3:
        # High volatility often indicates growth company
        volatility = np.std(historic_fcf) / np.mean(historic_fcf) if np.mean(historic_fcf) > 0 else 1.0
        if volatility > 0.6 and market_cap > 50e9:
            return True
    
    # Specific company overrides
    growth_company_tickers = ['TSLA', 'NVDA', 'META', 'AMZN', 'NET', 'SNOW', 'CRWD', 'DDOG', 'ZS']
    if symbol in growth_company_tickers:
        return True
        
    # Industry-based classification
    if any(growth_ind in industry for growth_ind in growth_industries):
        if market_cap > 100e9:  # Large cap in growth industry
            return True
    
    return False

def growth_company_dcf(historic_fcf, net_debt, shares_outstanding, symbol, current_price):
    """Multi-stage DCF for high-growth companies"""
    
    # Classify growth stage based on market cap and characteristics
    market_cap = current_price * shares_outstanding
    
    if market_cap > 500e9:  # Mega-cap growth (Tesla, Nvidia)
        scenarios = {
            'Conservative': {
                'stage1_years': 5,
                'stage1_growth': 0.20,
                'stage2_years': 5, 
                'stage2_growth': 0.12,
                'terminal_growth': 0.04,
                'discount_rate': 0.10
            },
            'Base Case': {
                'stage1_years': 5,
                'stage1_growth': 0.25,
                'stage2_years': 5,
                'stage2_growth': 0.15,
                'terminal_growth': 0.045,
                'discount_rate': 0.09
            },
            'Optimistic': {
                'stage1_years': 5,
                'stage1_growth': 0.30,
                'stage2_years': 5,
                'stage2_growth': 0.18,
                'terminal_growth': 0.05,
                'discount_rate': 0.08
            }
        }
    else:  # Mid/small cap growth
        scenarios = {
            'Conservative': {
                'stage1_years': 5,
                'stage1_growth': 0.25,
                'stage2_years': 5,
                'stage2_growth': 0.15,
                'terminal_growth': 0.04,
                'discount_rate': 0.11
            },
            'Base Case': {
                'stage1_years': 5,
                'stage1_growth': 0.35,
                'stage2_years': 5,
                'stage2_growth': 0.20,
                'terminal_growth': 0.045,
                'discount_rate': 0.10
            },
            'Optimistic': {
                'stage1_years': 5,
                'stage1_growth': 0.45,
                'stage2_years': 5,
                'stage2_growth': 0.25,
                'terminal_growth': 0.05,
                'discount_rate': 0.09
            }
        }
    
    results = {}
    current_fcf = historic_fcf[-1]
    
    for scenario_name, params in scenarios.items():
        try:
            # Stage 1: High Growth
            stage1_fcf = []
            stage1_discount_factors = []
            fcf_stage1 = current_fcf
            
            for year in range(1, params['stage1_years'] + 1):
                fcf_stage1 = fcf_stage1 * (1 + params['stage1_growth'])
                stage1_fcf.append(fcf_stage1)
                discount_factor = 1 / (1 + params['discount_rate']) ** year
                stage1_discount_factors.append(discount_factor)
            
            # Stage 2: Transition Growth
            stage2_fcf = []
            stage2_discount_factors = []
            fcf_stage2 = fcf_stage1
            
            for year in range(1, params['stage2_years'] + 1):
                fcf_stage2 = fcf_stage2 * (1 + params['stage2_growth'])
                stage2_fcf.append(fcf_stage2)
                discount_factor = 1 / (1 + params['discount_rate']) ** (params['stage1_years'] + year)
                stage2_discount_factors.append(discount_factor)
            
            # Terminal Value
            terminal_value = (fcf_stage2 * (1 + params['terminal_growth'])) / (params['discount_rate'] - params['terminal_growth'])
            terminal_discount_factor = 1 / (1 + params['discount_rate']) ** (params['stage1_years'] + params['stage2_years'])
            present_terminal_value = terminal_value * terminal_discount_factor
            
            # Calculate total present value
            present_value_stage1 = np.sum(np.array(stage1_fcf) * np.array(stage1_discount_factors))
            present_value_stage2 = np.sum(np.array(stage2_fcf) * np.array(stage2_discount_factors))
            
            enterprise_value = present_value_stage1 + present_value_stage2 + present_terminal_value
            equity_value = enterprise_value - net_debt
            intrinsic_value_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0
            
            results[scenario_name] = {
                'intrinsic_value_per_share': intrinsic_value_per_share,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'stage1_fcf': stage1_fcf,
                'stage2_fcf': stage2_fcf,
                'terminal_value': terminal_value,
                'discount_rate': params['discount_rate'],
                'stage1_growth': params['stage1_growth'],
                'stage2_growth': params['stage2_growth'],
                'terminal_growth': params['terminal_growth']
            }
            
        except Exception as e:
            print(f"Growth scenario failed: {scenario_name}, Error: {e}")
            continue
    
    return results

def quality_check_fcf(historic_fcf):
    """Check FCF data quality"""
    if len(historic_fcf) < 3:
        return "Low confidence - insufficient data"
    
    if len(historic_fcf) >= 2:
        volatility = np.std(historic_fcf) / np.mean(historic_fcf)
        if volatility > 1.0:
            return "High volatility - use caution"
    
    if any(fcf < 0 for fcf in historic_fcf):
        return "Negative FCF periods - conservative assumptions needed"
    
    return "Good quality data"

def calculate_fcf_growth_rate(historic_fcf, method='geometric'):
    """
    Calculate average FCF growth rate with robust error handling
    """
    if len(historic_fcf) < 2:
        return 0.03
    
    historic_fcf = np.array(historic_fcf, dtype=float)
    
    try:
        if method == 'arithmetic':
            growth_rates = (historic_fcf[1:] - historic_fcf[:-1]) / np.where(historic_fcf[:-1] != 0, historic_fcf[:-1], np.nan)
            valid_rates = growth_rates[np.isfinite(growth_rates)]
            return np.mean(valid_rates) if len(valid_rates) > 0 else 0.03
        
        elif method == 'geometric':
            growth_factors = historic_fcf[1:] / historic_fcf[:-1]
            valid_factors = growth_factors[np.isfinite(growth_factors) & (growth_factors > 0)]
            if len(valid_factors) > 0:
                geometric_growth = gmean(valid_factors) - 1
            else:
                geometric_growth = 0.03
            
            # Use linear regression for trend as additional signal
            if len(historic_fcf) >= 3:
                x = np.arange(len(historic_fcf))
                slope, intercept = np.polyfit(x, historic_fcf, 1)
                trend_growth = slope / np.mean(historic_fcf) if np.mean(historic_fcf) > 0 else 0
                
                if geometric_growth < 0:
                    return max(trend_growth, 0.02)
                else:
                    return (geometric_growth * 0.7 + trend_growth * 0.3)
            else:
                return geometric_growth
                
    except Exception as e:
        print(f"Error calculating growth rate: {e}")
        return 0.03

def get_industry_parameters(symbol, historic_fcf):
    """Get industry-specific DCF parameters"""
    avg_fcf = np.mean(historic_fcf) if len(historic_fcf) > 0 else 0
    fcf_volatility = np.std(historic_fcf) / avg_fcf if avg_fcf > 0 else 1.0
    
    if fcf_volatility > 0.8 and avg_fcf < 5e9:
        return {'dr': 0.10, 'tg': 0.03, 'max_growth': 0.20, 'growth_decay': 0.15}
    elif fcf_volatility < 0.5 and avg_fcf > 1e9:
        return {'dr': 0.09, 'tg': 0.025, 'max_growth': 0.15, 'growth_decay': 0.10}
    elif fcf_volatility > 0.5:
        return {'dr': 0.09, 'tg': 0.02, 'max_growth': 0.12, 'growth_decay': 0.12}
    else:
        return {'dr': 0.09, 'tg': 0.025, 'max_growth': 0.15, 'growth_decay': 0.10}

def traditional_dcf_valuation(historic_fcf, net_debt, shares_outstanding, discount_rate=None, 
                  projection_years=5, terminal_growth_rate=None, growth_decay=None):
    """
    Traditional DCF for mature companies
    """
    if not historic_fcf or len(historic_fcf) < 2:
        raise ValueError("Need at least 2 historical FCF values")
    
    industry_params = get_industry_parameters('', historic_fcf)
    
    discount_rate = discount_rate or industry_params['dr']
    terminal_growth_rate = terminal_growth_rate or industry_params['tg']
    growth_decay = growth_decay or industry_params['growth_decay']
    max_growth = industry_params['max_growth']
    
    if discount_rate <= terminal_growth_rate:
        raise ValueError("Discount rate must be greater than terminal growth rate")
    
    fcf_avg_growth_rate = calculate_fcf_growth_rate(historic_fcf, method='geometric')
    
    if len(historic_fcf) >= 4:
        fcf_std = np.std(historic_fcf)
        fcf_volatility = fcf_std / np.mean(historic_fcf) if np.mean(historic_fcf) > 0 else 1.0
        if fcf_volatility > 0.8:
            realistic_max_growth = min(max_growth, 0.15)
        elif fcf_volatility > 0.5:
            realistic_max_growth = min(max_growth, 0.20)
        else:
            realistic_max_growth = max_growth
    else:
        realistic_max_growth = 0.15
    
    realistic_base_growth = np.clip(fcf_avg_growth_rate, -0.10, realistic_max_growth)
    data_quality = quality_check_fcf(historic_fcf)
    
    projected_fcf = []
    discount_factors = []
    current_fcf = historic_fcf[-1]
    
    for year in range(1, projection_years + 1):
        decayed_growth = realistic_base_growth * (1 - growth_decay) ** (year - 1)
        final_growth_rate = max(decayed_growth, terminal_growth_rate - 0.005)
        current_fcf = current_fcf * (1 + final_growth_rate)
        projected_fcf.append(current_fcf)
        discount_factor = 1 / (1 + discount_rate) ** year
        discount_factors.append(discount_factor)
    
    terminal_value = (current_fcf * (1 + terminal_growth_rate)) / (discount_rate - terminal_growth_rate)
    terminal_discount_factor = 1 / (1 + discount_rate) ** projection_years
    present_terminal_value = terminal_value * terminal_discount_factor
    present_value_fcf = np.sum(np.array(projected_fcf) * np.array(discount_factors))
    enterprise_value = present_value_fcf + present_terminal_value
    equity_value = enterprise_value - net_debt
    intrinsic_value_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0
    
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
        'shares_outstanding': shares_outstanding,
        'discount_rate_used': discount_rate,
        'terminal_growth_used': terminal_growth_rate,
        'data_quality': data_quality,
        'max_growth_applied': realistic_max_growth,
        'valuation_model': 'Traditional DCF'
    }

def print_growth_valuation_results(results, current_price, symbol):
    """Print formatted growth company valuation results"""
    print("=" * 70)
    print(f"GROWTH COMPANY DCF VALUATION - {symbol}")
    print("=" * 70)
    
    for scenario_name, scenario in results.items():
        margin_of_safety = ((scenario['intrinsic_value_per_share'] - current_price) / current_price) * 100
        
        print(f"\n{scenario_name.upper()} SCENARIO:")
        print(f"  Stage 1 Growth: {scenario['stage1_growth']:.1%} for 5 years")
        print(f"  Stage 2 Growth: {scenario['stage2_growth']:.1%} for 5 years") 
        print(f"  Terminal Growth: {scenario['terminal_growth']:.1%}")
        print(f"  Discount Rate: {scenario['discount_rate']:.1%}")
        print(f"  Intrinsic Value: ${scenario['intrinsic_value_per_share']:.2f}")
        print(f"  Current Price: ${current_price:.2f}")
        print(f"  Margin of Safety: {margin_of_safety:+.1f}%")
        
        if margin_of_safety > 20:
            print(f"  RECOMMENDATION: STRONG BUY 🟢")
        elif margin_of_safety > 10:
            print(f"  RECOMMENDATION: BUY 🟢")
        elif margin_of_safety > 0:
            print(f"  RECOMMENDATION: HOLD 🟡")
        elif margin_of_safety > -10:
            print(f"  RECOMMENDATION: CAUTION 🟠")
        elif margin_of_safety > -20:
            print(f"  RECOMMENDATION: SELL 🔴")
        else:
            print(f"  RECOMMENDATION: STRONG SELL 🔴")

def print_valuation_results(results, current_price, symbol):
    """Print formatted valuation results for traditional DCF"""
    print("=" * 60)
    print(f"DCF VALUATION ANALYSIS - {symbol}")
    print("=" * 60)
    
    print(f"\nData Quality: {results['data_quality']}")
    print(f"Discount Rate Used: {results['discount_rate_used']:.1%}")
    print(f"Terminal Growth Used: {results['terminal_growth_used']:.1%}")
    
    print(f"\nHistorical Analysis:")
    print(f"Base FCF Growth Rate: {results['base_growth_rate']:.2%}")
    print(f"Used Growth Rate: {results['used_growth_rate']:.2%} (capped at {results['max_growth_applied']:.1%})")
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
    
    if results['intrinsic_value_per_share'] > current_price * 1.1:
        print("RECOMMENDATION: STRONGLY UNDERVALUED 🟢")
    elif results['intrinsic_value_per_share'] > current_price:
        print("RECOMMENDATION: MODERATELY UNDERVALUED 🟡")
    elif results['intrinsic_value_per_share'] > current_price * 0.9:
        print("RECOMMENDATION: FAIRLY VALUED ⚪")
    elif results['intrinsic_value_per_share'] > current_price * 0.8:
        print("RECOMMENDATION: MODERATELY OVERVALUED 🟠")
    else:
        print("RECOMMENDATION: STRONGLY OVERVALUED 🔴")

# Load the full JSON data for company name lookup
with open("/Users/jorismeert/Desktop/Python/ProjectDCF/data/ticker_DJ.json", "r", encoding="utf-8") as f:
    dow_data_full = json.load(f)

with open("/Users/jorismeert/Desktop/Python/ProjectDCF/data/nasdaq100.json", "r", encoding="utf-8") as g:
    nasdaq_data_full = json.load(g)

def find_company_by_ticker(ticker, nasdaq_data_full, dow_data_full):
    """Find company name by ticker symbol"""
    if "Nasdaq100" in nasdaq_data_full:
        for company in nasdaq_data_full["Nasdaq100"]:
            if company["ticker"] == ticker:
                return company["company"]
    
    if "DowJones" in dow_data_full:
        for company in dow_data_full["DowJones"]:
            if company["ticker"] == ticker:
                return company["company"]
    
    return ticker

def load_ticker_symbols():
    """Load ticker symbols from JSON files with proper error handling"""
    try:
        with open("/Users/jorismeert/Desktop/Python/ProjectDCF/data/ticker_DJ.json", "r", encoding="utf-8") as f:
            dow_data = json.load(f)
        
        with open("/Users/jorismeert/Desktop/Python/ProjectDCF/data/nasdaq100.json", "r", encoding="utf-8") as g:
            nasdaq_data = json.load(g)
        
        def extract_symbols(data):
            symbols = []
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict) and 'ticker' in item:
                                symbols.append(item['ticker'].upper())
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'ticker' in item:
                        symbols.append(item['ticker'].upper())
                    elif isinstance(item, str):
                        symbols.append(item.upper())
            return symbols
        
        dow_symbols = extract_symbols(dow_data)
        nasdaq_symbols = extract_symbols(nasdaq_data)
        
        print(f"Loaded {len(dow_symbols)} Dow Jones symbols: {dow_symbols[:5]}...")
        print(f"Loaded {len(nasdaq_symbols)} NASDAQ 100 symbols: {nasdaq_symbols[:5]}...")
        
        return dow_symbols, nasdaq_symbols
        
    except Exception as e:
        print(f"Error loading ticker symbols: {e}")
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'], ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Main execution
if __name__ == "__main__":
    dow_symbols, nasdaq_symbols = load_ticker_symbols()
    all_tickers = list(set(dow_symbols + nasdaq_symbols))
    
    print(f"Dow Jones stocks: {len(dow_symbols)}")
    print(f"NASDAQ 100 stocks: {len(nasdaq_symbols)}")
    print(f"Total unique tickers: {len(all_tickers)}")
    
    # Include Tesla and other growth companies in test
    test_tickers = all_tickers[:8]
    
    # Ensure Tesla is included for testing
    if 'TSLA' in all_tickers and 'TSLA' not in test_tickers:
        test_tickers = ['TSLA'] + [t for t in test_tickers if t != 'TSLA'][:7]
    
    print(f"\nAnalyzing {len(test_tickers)} stocks: {test_tickers}")
    
    results = []
    
    for symbol in test_tickers:
        try:
            company_name = find_company_by_ticker(symbol, nasdaq_data_full, dow_data_full)
            print(f"\n{'='*70}")
            print(f"Analyzing {symbol} - {company_name}...")
            print(f"{'='*70}")
            
            df_cash, df_balance, current_price, shares_outstanding, industry, sector = get_financial_data(symbol)
            
            net_debt = df_balance['NetDebt'].iloc[-1] if 'NetDebt' in df_balance.columns else 0
            historic_fcf = column_to_list(df_cash, 'FreeCashFlow')
            
            if len(historic_fcf) < 2:
                print(f"⚠️  Insufficient FCF data for {symbol}, skipping...")
                continue
                
            print(f"Historical FCF: {[f'{x/1e9:.1f}B' for x in historic_fcf]}")
            print(f"Current Price: ${current_price:.2f}")
            print(f"Net Debt: ${net_debt/1e9:.2f}B")
            print(f"Shares Outstanding: {shares_outstanding/1e6:.0f}M")
            print(f"Industry: {industry}")
            
            # Choose valuation model based on company type
            if is_growth_company(symbol, historic_fcf, current_price, shares_outstanding, industry):
                print(f"🔬 Using GROWTH COMPANY DCF Model")
                growth_results = growth_company_dcf(historic_fcf, net_debt, shares_outstanding, symbol, current_price)
                print_growth_valuation_results(growth_results, current_price, symbol)
                
                # Use base case for summary
                base_case_value = growth_results['Base Case']['intrinsic_value_per_share']
                margin_of_safety = ((base_case_value - current_price) / current_price) * 100
                
                result_data = {
                    'symbol': symbol,
                    'company_name': company_name,
                    'current_price': current_price,
                    'intrinsic_value': base_case_value,
                    'margin_of_safety': margin_of_safety,
                    'growth_rate': 'Multi-stage',
                    'used_growth_rate': 'Multi-stage',
                    'net_debt': net_debt,
                    'shares_outstanding': shares_outstanding,
                    'discount_rate': growth_results['Base Case']['discount_rate'],
                    'terminal_growth': growth_results['Base Case']['terminal_growth'],
                    'data_quality': 'Growth Company Model',
                    'valuation_model': 'Growth DCF'
                }
                
            else:
                print(f"📊 Using TRADITIONAL DCF Model")
                dcf_results = traditional_dcf_valuation(historic_fcf, net_debt, shares_outstanding)
                print_valuation_results(dcf_results, current_price, symbol)
                
                result_data = {
                    'symbol': symbol,
                    'company_name': company_name,
                    'current_price': current_price,
                    'intrinsic_value': dcf_results['intrinsic_value_per_share'],
                    'margin_of_safety': ((dcf_results['intrinsic_value_per_share'] - current_price) / current_price) * 100,
                    'growth_rate': dcf_results['base_growth_rate'],
                    'used_growth_rate': dcf_results['used_growth_rate'],
                    'net_debt': net_debt,
                    'shares_outstanding': shares_outstanding,
                    'discount_rate': dcf_results['discount_rate_used'],
                    'terminal_growth': dcf_results['terminal_growth_used'],
                    'data_quality': dcf_results['data_quality'],
                    'valuation_model': 'Traditional DCF'
                }
            
            results.append(result_data)
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ Analysis failed for {symbol}: {e}")
            continue
    
    # Summary of all analyses
    if results:
        print(f"\n{'='*80}")
        print("SUMMARY OF ALL ANALYSES")
        print(f"{'='*80}")
        
        df_results = pd.DataFrame(results)
        
        conditions = [
            df_results['margin_of_safety'] > 20,
            df_results['margin_of_safety'] > 10,
            df_results['margin_of_safety'] > 0,
            df_results['margin_of_safety'] > -10,
            df_results['margin_of_safety'] > -20,
            df_results['margin_of_safety'] <= -20
        ]
        choices = ['STRONG BUY', 'BUY', 'HOLD', 'CAUTION', 'SELL', 'STRONG SELL']
        df_results['recommendation'] = np.select(conditions, choices, default='HOLD')
        
        df_results = df_results.sort_values('margin_of_safety', ascending=False)
        
        print("\nRanked by Margin of Safety:")
        for _, row in df_results.iterrows():
            if row['margin_of_safety'] > 20:
                status = "🟢 STRONG BUY"
            elif row['margin_of_safety'] > 10:
                status = "🟢 BUY"
            elif row['margin_of_safety'] > 0:
                status = "🟡 HOLD"
            elif row['margin_of_safety'] > -10:
                status = "🟠 CAUTION"
            elif row['margin_of_safety'] > -20:
                status = "🔴 SELL"
            else:
                status = "🔴 STRONG SELL"
                
            model_indicator = "🚀" if row['valuation_model'] == 'Growth DCF' else "📊"
            print(f"{model_indicator} {row['symbol']} ({row['company_name']}): ${row['intrinsic_value']:.2f} vs ${row['current_price']:.2f} "
                  f"({row['margin_of_safety']:+.1f}%) - {status}")
        
        # Show top picks
        strong_buys = df_results[df_results['recommendation'] == 'STRONG BUY']
        if len(strong_buys) > 0:
            print(f"\n🎯 TOP PICKS (STRONG BUY):")
            for _, stock in strong_buys.iterrows():
                print(f"  {stock['symbol']}: +{stock['margin_of_safety']:.1f}% margin of safety")
                
    else:
        print("No successful analyses completed.")