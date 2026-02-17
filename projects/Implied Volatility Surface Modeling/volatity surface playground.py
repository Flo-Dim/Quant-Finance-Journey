import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import norm
import gradio as gr
from functools import lru_cache
import warnings
import google.generativeai as genai
warnings.filterwarnings('ignore')



# Gemini AI Configuration
# TODO: Add your Gemini API key here
GEMINI_API_KEY = ""  # Get from: https://makersuite.google.com/app/apikey

def configure_gemini():
    """Configure Gemini AI with API key."""
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        return True
    return False


def generate_ai_interpretation(options_df, spot, ticker_symbol):
    """
    Generate AI-powered interpretation of options data using Gemini.
    
    Parameters:
    options_df: DataFrame with options and Greeks data
    spot: current spot price
    ticker_symbol: stock ticker
    
    Returns:
    str: AI-generated interpretation or error message
    """
    if not configure_gemini():
        return """
        **AI Interpretation Not Available**
        
        To enable AI-powered analysis:
        1. Get a free Gemini API key from: https://makersuite.google.com/app/apikey
        2. Add your key to the `GEMINI_API_KEY` variable in the code
        3. Restart the application
        
        The AI will provide insights on:
        - Volatility smile interpretation
        - Term structure analysis
        - Greeks patterns
        - Market sentiment indicators
        - Trading implications
        """
    
    try:
        # Prepare summary statistics for AI
        iv_stats = {
            'mean': options_df['IV'].mean(),
            'median': options_df['IV'].median(),
            'min': options_df['IV'].min(),
            'max': options_df['IV'].max(),
            'std': options_df['IV'].std()
        }
        
        # ATM volatility by maturity
        atm_data = []
        for mat in sorted(options_df["maturity"].unique()):
            subset = options_df[options_df["maturity"] == mat].copy()
            subset["dist"] = np.abs(subset["strike"] - spot)
            if len(subset) > 0:
                atm_row = subset.loc[subset["dist"].idxmin()]
                atm_data.append({
                    'maturity': mat,
                    'T': atm_row['T'],
                    'ATM_IV': atm_row['IV']
                })
        
        # Volatility skew (put vs call IV at same strikes)
        skew_analysis = ""
        if len(options_df) > 0:
            calls = options_df[options_df['type'] == 'C']
            puts = options_df[options_df['type'] == 'P']
            if len(calls) > 0 and len(puts) > 0:
                avg_call_iv = calls['IV'].mean()
                avg_put_iv = puts['IV'].mean()
                skew_analysis = f"Average Call IV: {avg_call_iv:.2%}, Average Put IV: {avg_put_iv:.2%}"
        
        # Greeks summary
        greeks_summary = {
            'avg_delta': options_df['delta'].mean(),
            'avg_gamma': options_df['gamma'].mean(),
            'avg_vega': options_df['vega'].mean(),
            'avg_theta': options_df['theta'].mean()
        }
        
        # Term structure characterization
        term_structure = "flat"
        if len(atm_data) >= 2:
            first_iv = atm_data[0]['ATM_IV']
            last_iv = atm_data[-1]['ATM_IV']
            if last_iv > first_iv * 1.05:
                term_structure = "upward sloping (contango)"
            elif last_iv < first_iv * 0.95:
                term_structure = "downward sloping (backwardation)"
        
        # Construct prompt for Gemini
        prompt = f"""
You are an expert options trader and volatility analyst. Analyze the following options market data for {ticker_symbol} and provide a comprehensive interpretation.

**Market Data Summary:**
- Ticker: {ticker_symbol}
- Spot Price: ${spot:.2f}
- Total Options Analyzed: {len(options_df)}
- Number of Maturities: {len(options_df['maturity'].unique())}

**Implied Volatility Statistics:**
- Mean IV: {iv_stats['mean']:.2%}
- Median IV: {iv_stats['median']:.2%}
- Min IV: {iv_stats['min']:.2%}
- Max IV: {iv_stats['max']:.2%}
- Std Dev: {iv_stats['std']:.2%}

**Volatility Skew:**
{skew_analysis}

**Term Structure:**
The volatility term structure is {term_structure}.

ATM Volatility by Maturity:
{chr(10).join([f"- {d['maturity']} (T={d['T']:.2f}y): {d['ATM_IV']:.2%}" for d in atm_data])}

**Greeks Summary:**
- Average Delta: {greeks_summary['avg_delta']:.4f}
- Average Gamma: {greeks_summary['avg_gamma']:.4f}
- Average Vega: {greeks_summary['avg_vega']:.4f}
- Average Theta: {greeks_summary['avg_theta']:.4f}

**Please provide:**

1. **Volatility Smile Interpretation** (2-3 sentences)
   - What does the current IV pattern suggest about market expectations?
   - Are there signs of skew (puts more expensive than calls)?

2. **Term Structure Analysis** (2-3 sentences)
   - What does the term structure tell us about future volatility expectations?
   - Is the market pricing in any specific events?

3. **Greeks Insights** (2-3 sentences)
   - What do the Greeks patterns reveal about hedging activity?
   - Where is vega/gamma exposure concentrated?

4. **Market Sentiment** (2-3 sentences)
   - Is the market bullish, bearish, or neutral on {ticker_symbol}?
   - Any signs of elevated fear or complacency?

5. **Trading Implications** (2-3 sentences)
   - What strategies might be favored in this environment?
   - Key risks or opportunities?

Keep the analysis concise, practical, and actionable. Use proper markdown formatting with headers and bullet points.
"""
        
        # Call Gemini API
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        interpretation = f"""
## AI-Powered Market Analysis

{response.text}

---
*Analysis generated by Google Gemini AI based on computed options data*
"""
        
        return interpretation
    
    except Exception as e:
        return f"""
        **AI Interpretation Error**
        
        Failed to generate AI analysis: {str(e)}
        
        Common issues:
        - Invalid API key
        - API rate limit exceeded
        - Network connectivity issues
        
        Please check your Gemini API key and try again.
        """



# Black-Scholes Functions


def bsm(c, s, k, r, q, v, t):
    """Black-Scholes-Merton option pricing formula."""
    d1_val = (np.log(s / k) + (r - q + 0.5 * v * v) * t) / (v * np.sqrt(t))
    d2_val = d1_val - v * np.sqrt(t)
    
    cp = c.strip().upper()
    
    if cp == "C":
        price = (s * np.exp(-q * t) * norm.cdf(d1_val) - 
                 k * np.exp(-r * t) * norm.cdf(d2_val))
    elif cp == "P":
        price = (k * np.exp(-r * t) * norm.cdf(-d2_val) - 
                 s * np.exp(-q * t) * norm.cdf(-d1_val))
    else:
        raise ValueError("Invalid option type.")
    
    return price


def d1(s, k, r, q, v, t):
    """Calculate d1 parameter."""
    return (np.log(s / k) + (r - q + 0.5 * v * v) * t) / (v * np.sqrt(t))


def implied_vol(c, s, k, r, q, t, p_mkt, sigma0=0.2, max_iter=50):
    """Calculate implied volatility using Newton-Raphson (optimized)."""
    vol_old = sigma0
    
    for i in range(max_iter):
        price_diff = bsm(c, s, k, r, q, vol_old, t) - p_mkt
        d1_val = d1(s, k, r, q, vol_old, t)
        vega = s * np.exp(-q * t) * np.sqrt(t) * norm.pdf(d1_val)
        
        if vega < 1e-10:
            return np.nan
        
        vol_new = vol_old - price_diff / vega
        
        if abs(vol_new - vol_old) < 0.0001:
            return vol_new
        
        vol_old = vol_new
    
    return vol_new


def calculate_greeks(option_type, s, k, r, q, v, t):
    """Calculate option Greeks."""
    d1_val = d1(s, k, r, q, v, t)
    d2_val = d1_val - v * np.sqrt(t)
    
    sqrt_t = np.sqrt(t)
    exp_qt = np.exp(-q * t)
    exp_rt = np.exp(-r * t)
    pdf_d1 = norm.pdf(d1_val)
    
    opt_type = option_type.strip().upper()
    
    if opt_type == "C":
        delta = exp_qt * norm.cdf(d1_val)
    else:
        delta = -exp_qt * norm.cdf(-d1_val)
    
    gamma = (exp_qt * pdf_d1) / (s * v * sqrt_t)
    vega = (s * exp_qt * pdf_d1 * sqrt_t) / 100
    
    if opt_type == "C":
        theta = ((-s * pdf_d1 * v * exp_qt) / (2 * sqrt_t) - 
                 r * k * exp_rt * norm.cdf(d2_val) + 
                 q * s * exp_qt * norm.cdf(d1_val)) / 365
    else:
        theta = ((-s * pdf_d1 * v * exp_qt) / (2 * sqrt_t) + 
                 r * k * exp_rt * norm.cdf(-d2_val) - 
                 q * s * exp_qt * norm.cdf(-d1_val)) / 365
    
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta
    }


# Optimized Data Processing

@lru_cache(maxsize=10)
def get_ticker_data(ticker_symbol):
    """Cached ticker fetch to avoid repeated API calls."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        spot = ticker.history(period="1d")["Close"].iloc[-1]
        maturities = ticker.options
        return ticker, spot, maturities, None
    except Exception as e:
        return None, None, None, str(e)


def compute_time_to_maturity(maturity_date):
    """Convert maturity date to years."""
    maturity = datetime.strptime(maturity_date, "%Y-%m-%d")
    days_to_maturity = (maturity - datetime.today()).days
    return max(days_to_maturity / 365, 0.001)


def process_option_chain(ticker, maturity):
    """Fetch and process options for a single maturity with timeout protection."""
    try:
        chain = ticker.option_chain(maturity)
        
        calls = chain.calls.copy()
        calls["type"] = "C"
        
        puts = chain.puts.copy()
        puts["type"] = "P"
        
        options = pd.concat([calls, puts], ignore_index=True)
        options["maturity"] = maturity
        options["T"] = compute_time_to_maturity(maturity)
        options["mid"] = (options["bid"] + options["ask"]) / 2
        
        return options
    except Exception as e:
        print(f"Warning: Failed to fetch {maturity}: {e}")
        return pd.DataFrame()


def compute_iv_vectorized(options_subset, spot, r, q):
    """Optimized IV computation - processes in batches."""
    iv_list = []
    
    options_filtered = options_subset[
        (options_subset["mid"] > 0.01) &
        (options_subset["T"] > 0.001)
    ].copy()
    
    for idx, row in options_filtered.iterrows():
        try:
            iv = implied_vol(
                c=row["type"],
                s=spot,
                k=row["strike"],
                r=r,
                q=q,
                t=row["T"],
                p_mkt=row["mid"],
                sigma0=0.2
            )
            iv_list.append((idx, iv))
        except:
            iv_list.append((idx, np.nan))
    
    iv_series = pd.Series(dict(iv_list))
    return iv_series


def fetch_and_process_options(ticker_symbol, num_maturities, strike_lower, strike_upper, 
                               r, q, min_maturity, iv_min, iv_max, progress=gr.Progress()):
    """Main data processing function with progress tracking."""
    try:
        progress(0, desc="Fetching ticker data...")
        
        ticker, spot, all_maturities, error = get_ticker_data(ticker_symbol)
        
        if error:
            return None, None, f"Error fetching ticker: {error}"
        
        if len(all_maturities) == 0:
            return None, None, f"No options available for {ticker_symbol}"
        
        progress(0.2, desc="Fetching option chains...")
        
        maturities = all_maturities[:num_maturities]
        options_list = []
        
        for i, mat in enumerate(maturities):
            progress(0.2 + 0.2 * (i / len(maturities)), desc=f"Fetching maturity {i+1}/{len(maturities)}...")
            chain_data = process_option_chain(ticker, mat)
            if len(chain_data) > 0:
                options_list.append(chain_data)
        
        if len(options_list) == 0:
            return None, None, "No option data retrieved"
        
        progress(0.4, desc="Filtering options...")
        
        options_df = pd.concat(options_list, ignore_index=True)
        
        initial_count = len(options_df)
        options_df = options_df.dropna(subset=["mid"])
        options_df = options_df[options_df["mid"] > 0]
        options_df = options_df[(options_df["volume"] > 0) | (options_df["openInterest"] > 0)]
        options_df = options_df[
            (options_df["strike"] > strike_lower * spot) &
            (options_df["strike"] < strike_upper * spot)
        ]
        options_df = options_df[options_df["T"] > min_maturity]
        
        if len(options_df) == 0:
            return None, None, "No options remaining after filtering. Try widening parameters."
        
        if len(options_df) > 500:
            options_df = options_df.sample(n=500, random_state=42)
        
        progress(0.6, desc=f"Computing IV for {len(options_df)} options...")
        
        iv_series = compute_iv_vectorized(options_df, spot, r, q)
        options_df["IV"] = iv_series
        
        options_df = options_df.dropna(subset=["IV"])
        options_df = options_df[
            (options_df["IV"] > iv_min) &
            (options_df["IV"] < iv_max)
        ]
        
        if len(options_df) == 0:
            return None, None, "No valid IVs computed. Try adjusting IV bounds."
        
        progress(0.8, desc="Computing Greeks...")
        
        greeks_list = []
        for idx, row in options_df.iterrows():
            try:
                greeks = calculate_greeks(
                    option_type=row["type"],
                    s=spot,
                    k=row["strike"],
                    r=r,
                    q=q,
                    v=row["IV"],
                    t=row["T"]
                )
                greeks_list.append(greeks)
            except:
                greeks_list.append({
                    'delta': np.nan,
                    'gamma': np.nan,
                    'vega': np.nan,
                    'theta': np.nan
                })
        
        greeks_df = pd.DataFrame(greeks_list)
        options_complete = pd.concat([options_df.reset_index(drop=True), greeks_df], axis=1)
        
        options_complete = options_complete.dropna(subset=['delta', 'gamma', 'vega'])
        
        progress(1.0, desc="Complete!")
        
        return options_complete, spot, None
    
    except Exception as e:
        return None, None, f"Error: {str(e)}"

# Visualization Functions

def create_iv_smile_plot(options_df, spot, selected_maturity):
    """Create IV smile plot."""
    if options_df is None or len(options_df) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    maturities = sorted(options_df["maturity"].unique())
    
    if selected_maturity == "First" and len(maturities) > 0:
        maturity = maturities[0]
    elif selected_maturity == "Second" and len(maturities) > 1:
        maturity = maturities[1]
    elif selected_maturity == "Third" and len(maturities) > 2:
        maturity = maturities[2]
    else:
        maturity = maturities[0] if len(maturities) > 0 else None
    
    if maturity is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    subset = options_df[options_df["maturity"] == maturity]
    calls = subset[subset["type"] == "C"]
    puts = subset[subset["type"] == "P"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if len(calls) > 0:
        ax.scatter(calls["strike"], calls["IV"], alpha=0.6, s=60, 
                   label="Calls", color="blue")
    
    if len(puts) > 0:
        ax.scatter(puts["strike"], puts["IV"], alpha=0.6, s=60, 
                   label="Puts", color="red")
    
    ax.axvline(spot, color='green', linestyle='--', linewidth=2, label="Spot")
    ax.set_xlabel("Strike", fontsize=12)
    ax.set_ylabel("Implied Volatility", fontsize=12)
    ax.set_title(f"IV Smile - {maturity}", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_term_structure_plot(options_df, spot):
    """Create term structure plot."""
    if options_df is None or len(options_df) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    atm_iv = []
    
    for mat in options_df["maturity"].unique():
        subset = options_df[options_df["maturity"] == mat].copy()
        subset["dist"] = np.abs(subset["strike"] - spot)
        
        if len(subset) == 0:
            continue
            
        atm_row = subset.loc[subset["dist"].idxmin()]
        
        atm_iv.append({
            "maturity": mat,
            "T": atm_row["T"],
            "ATM_IV": atm_row["IV"]
        })
    
    if len(atm_iv) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No ATM data available', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    atm_df = pd.DataFrame(atm_iv)
    atm_df = atm_df.sort_values("T")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(atm_df["T"], atm_df["ATM_IV"], marker='o', linewidth=2, 
            markersize=10, color='darkblue')
    
    ax.set_xlabel("Time to Maturity (years)", fontsize=12)
    ax.set_ylabel("ATM Implied Volatility", fontsize=12)
    ax.set_title("Term Structure of ATM Volatility", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if len(atm_df) >= 2:
        first_iv = atm_df.iloc[0]["ATM_IV"]
        last_iv = atm_df.iloc[-1]["ATM_IV"]
        
        if last_iv > first_iv * 1.05:
            structure = "Upward Sloping (Contango)"
            interpretation = "Market expects higher future volatility"
        elif last_iv < first_iv * 0.95:
            structure = "Downward Sloping (Backwardation)"
            interpretation = "Market expects lower future volatility"
        else:
            structure = "Flat"
            interpretation = "Market expects stable volatility"
        
        ax.text(0.02, 0.98, f"{structure}\n{interpretation}", 
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
    
    plt.tight_layout()
    return fig


def create_greeks_plot(options_df, spot, selected_maturity, greek_type):
    """Create Greeks plot."""
    if options_df is None or len(options_df) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    maturities = sorted(options_df["maturity"].unique())
    
    if selected_maturity == "First" and len(maturities) > 0:
        maturity = maturities[0]
    elif selected_maturity == "Second" and len(maturities) > 1:
        maturity = maturities[1]
    elif selected_maturity == "Third" and len(maturities) > 2:
        maturity = maturities[2]
    else:
        maturity = maturities[0] if len(maturities) > 0 else None
    
    if maturity is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    subset = options_df[options_df["maturity"] == maturity]
    calls = subset[subset["type"] == "C"]
    puts = subset[subset["type"] == "P"]
    
    greek_col = greek_type.lower()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if len(calls) > 0:
        ax.scatter(calls["strike"], calls[greek_col], alpha=0.6, s=60, 
                   label="Calls", color="blue")
    
    if len(puts) > 0:
        ax.scatter(puts["strike"], puts[greek_col], alpha=0.6, s=60, 
                   label="Puts", color="red")
    
    ax.axvline(spot, color='green', linestyle='--', linewidth=2, label="Spot")
    ax.set_xlabel("Strike", fontsize=12)
    ax.set_ylabel(greek_type, fontsize=12)
    ax.set_title(f"{greek_type} vs Strike - {maturity}", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_summary_stats(options_df, spot):
    """Create summary statistics."""
    if options_df is None or len(options_df) == 0:
        return "No data available"
    
    summary = f"""
    📊 **Summary Statistics**
    
    **Spot Price:** ${spot:.2f}
    **Total Options:** {len(options_df)}
    **Calls:** {len(options_df[options_df['type'] == 'C'])}
    **Puts:** {len(options_df[options_df['type'] == 'P'])}
    **Maturities:** {len(options_df['maturity'].unique())}
    
    **IV Statistics:**
    - Mean: {options_df['IV'].mean():.2%}
    - Median: {options_df['IV'].median():.2%}
    - Min: {options_df['IV'].min():.2%}
    - Max: {options_df['IV'].max():.2%}
    - Std Dev: {options_df['IV'].std():.2%}
    
    **Greeks Averages:**
    - Delta: {options_df['delta'].mean():.4f}
    - Gamma: {options_df['gamma'].mean():.4f}
    - Vega: {options_df['vega'].mean():.4f}
    - Theta: {options_df['theta'].mean():.4f}
    """
    
    return summary

# Gradio Interface

def analyze_options(ticker, num_mat, strike_low, strike_high, risk_free, div_yield, 
                   min_mat, iv_low, iv_high, maturity_select, greek_select,
                   progress=gr.Progress()):
    """Main function for Gradio interface with progress tracking."""
    
    ticker = ticker.strip().upper()
    
    # Fetch and process data
    options_df, spot, error = fetch_and_process_options(
        ticker_symbol=ticker,
        num_maturities=int(num_mat),
        strike_lower=strike_low,
        strike_upper=strike_high,
        r=risk_free / 100,
        q=div_yield / 100,
        min_maturity=min_mat,
        iv_min=iv_low / 100,
        iv_max=iv_high / 100,
        progress=progress
    )
    
    if error:
        return None, None, None, error, None, ""
    
    # Create plots
    iv_smile_fig = create_iv_smile_plot(options_df, spot, maturity_select)
    term_struct_fig = create_term_structure_plot(options_df, spot)
    greeks_fig = create_greeks_plot(options_df, spot, maturity_select, greek_select)
    summary = create_summary_stats(options_df, spot)
    
    # Generate AI interpretation
    progress(0.95, desc="Generating AI interpretation...")
    ai_interpretation = generate_ai_interpretation(options_df, spot, ticker)
    
    # Create data table
    if options_df is not None and len(options_df) > 0:
        display_cols = ['type', 'strike', 'maturity', 'T', 'mid', 'IV', 
                       'delta', 'gamma', 'vega', 'theta']
        table_df = options_df[display_cols].head(20).copy()
        table_df['IV'] = table_df['IV'].apply(lambda x: f"{x:.2%}")
        table_df['delta'] = table_df['delta'].apply(lambda x: f"{x:.4f}")
        table_df['gamma'] = table_df['gamma'].apply(lambda x: f"{x:.4f}")
        table_df['vega'] = table_df['vega'].apply(lambda x: f"{x:.4f}")
        table_df['theta'] = table_df['theta'].apply(lambda x: f"{x:.4f}")
    else:
        table_df = None
    
    return iv_smile_fig, term_struct_fig, greeks_fig, summary, table_df, ai_interpretation

# Build Gradio App

with gr.Blocks(title="Options IV & Greeks Playground", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown(
    """
    # 📈 Options Implied Volatility & Greeks Playground
    
    Interactive tool to analyze options data, compute implied volatilities, and visualize Greeks with **AI-powered interpretation**.
    
    **Instructions:**
    1. Enter a ticker symbol (e.g., AAPL, TSLA, SPY)
    2. Adjust parameters as needed
    3. Click "Analyze Options" to generate plots and AI analysis
    4. Explore different maturities and Greeks using the dropdown menus
    
    **Note:** Analysis may take 10-30 seconds depending on the ticker and parameters.
    """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🔧 Parameters")
            
            ticker_input = gr.Textbox(
                label="Ticker Symbol",
                value="AAPL",
                placeholder="Enter ticker (e.g., AAPL, TSLA, SPY)"
            )
            
            num_mat_slider = gr.Slider(
                minimum=1,
                maximum=6,
                value=3,
                step=1,
                label="Number of Maturities (more = slower)"
            )
            
            with gr.Row():
                strike_low_slider = gr.Slider(
                    minimum=0.3,
                    maximum=0.9,
                    value=0.5,
                    step=0.05,
                    label="Strike Lower (% of Spot)"
                )
                
                strike_high_slider = gr.Slider(
                    minimum=1.1,
                    maximum=2.0,
                    value=1.5,
                    step=0.05,
                    label="Strike Upper (% of Spot)"
                )
            
            with gr.Row():
                risk_free_slider = gr.Slider(
                    minimum=0,
                    maximum=10,
                    value=4.5,
                    step=0.1,
                    label="Risk-Free Rate (%)"
                )
                
                div_yield_slider = gr.Slider(
                    minimum=0,
                    maximum=5,
                    value=0.5,
                    step=0.1,
                    label="Dividend Yield (%)"
                )
            
            min_mat_slider = gr.Slider(
                minimum=0.001,
                maximum=0.1,
                value=0.01,
                step=0.001,
                label="Min Maturity (years)"
            )
            
            with gr.Row():
                iv_low_slider = gr.Slider(
                    minimum=0,
                    maximum=50,
                    value=1,
                    step=1,
                    label="IV Lower Bound (%)"
                )
                
                iv_high_slider = gr.Slider(
                    minimum=50,
                    maximum=500,
                    value=300,
                    step=10,
                    label="IV Upper Bound (%)"
                )
            
            analyze_btn = gr.Button("🚀 Analyze Options", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Results")
            
            summary_output = gr.Markdown()
            
            with gr.Tabs():
                with gr.Tab("🤖 AI Analysis"):
                    ai_output = gr.Markdown()
                
                with gr.Tab("IV Smile"):
                    maturity_dropdown = gr.Dropdown(
                        choices=["First", "Second", "Third"],
                        value="First",
                        label="Select Maturity"
                    )
                    iv_smile_plot = gr.Plot()
                
                with gr.Tab("Term Structure"):
                    term_struct_plot = gr.Plot()
                
                with gr.Tab("Greeks"):
                    greek_dropdown = gr.Dropdown(
                        choices=["Delta", "Gamma", "Vega", "Theta"],
                        value="Vega",
                        label="Select Greek"
                    )
                    greeks_plot = gr.Plot()
                
                with gr.Tab("Data Table"):
                    data_table = gr.Dataframe()
    
    # Event handlers
    analyze_btn.click(
        fn=analyze_options,
        inputs=[
            ticker_input, num_mat_slider, strike_low_slider, strike_high_slider,
            risk_free_slider, div_yield_slider, min_mat_slider, 
            iv_low_slider, iv_high_slider, maturity_dropdown, greek_dropdown
        ],
        outputs=[iv_smile_plot, term_struct_plot, greeks_plot, summary_output, data_table, ai_output]
    )
    
    maturity_dropdown.change(
        fn=analyze_options,
        inputs=[
            ticker_input, num_mat_slider, strike_low_slider, strike_high_slider,
            risk_free_slider, div_yield_slider, min_mat_slider, 
            iv_low_slider, iv_high_slider, maturity_dropdown, greek_dropdown
        ],
        outputs=[iv_smile_plot, term_struct_plot, greeks_plot, summary_output, data_table, ai_output]
    )
    
    greek_dropdown.change(
        fn=analyze_options,
        inputs=[
            ticker_input, num_mat_slider, strike_low_slider, strike_high_slider,
            risk_free_slider, div_yield_slider, min_mat_slider, 
            iv_low_slider, iv_high_slider, maturity_dropdown, greek_dropdown
        ],
        outputs=[iv_smile_plot, term_struct_plot, greeks_plot, summary_output, data_table, ai_output]
    )

# Launch with queue for production
if __name__ == "__main__":
    demo.queue(
        concurrency_count=3,
        max_size=20
    ).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )