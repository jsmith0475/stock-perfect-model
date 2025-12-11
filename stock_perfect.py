"""
Stock Perfect Model - Algebraic Topology Trading Algorithm

This module implements a trading algorithm based on:
1. Graph Signal Processing (GSP) - Laplacian diffusion on correlation graphs
2. Algebraic Topology - Persistent Homology for regime detection
3. LLM Integration - AWS Bedrock for sentiment analysis and report generation

Author: Stock Perfect Model Team
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import expm
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class StockPerfectModel:
    """
    Main trading model combining topology, graph signal processing, and LLM analysis.
    
    Pipeline:
    1. Fetch stock data
    2. Compute returns
    3. Build correlation graph
    4. Compute Laplacian diffusion residuals (mispricings)
    5. Compute persistent homology (regime features)
    6. [Optional] LLM: Sentiment analysis
    7. [Optional] LLM: Regime interpretation
    8. [Optional] LLM: Report generation
    """
    
    def __init__(self, tickers: list[str], start_date: str, end_date: str):
        """
        Initialize the Stock Perfect Model.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date for data (YYYY-MM-DD)
            end_date: End date for data (YYYY-MM-DD)
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        
        # Data storage
        self.data: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.DataFrame] = None
        self.correlation_matrix: Optional[np.ndarray] = None
        self.adj_matrix: Optional[np.ndarray] = None
        self.residuals: Optional[np.ndarray] = None
        
        # Topology features
        self.diagrams = None
        self.h1_total_persistence: float = 0.0
        self.h1_max_persistence: float = 0.0
        self.h1_feature_count: int = 0
        self.h0_component_count: int = 0
        
        # LLM components (lazy loaded)
        self._bedrock_client = None
        self._sentiment_analyzer = None
        self._regime_interpreter = None
        self._report_generator = None
        
    # ==================== DATA LAYER ====================
        
    def fetch_data(self) -> pd.DataFrame:
        """Fetches historical data using yfinance."""
        print(f"Fetching data for {len(self.tickers)} tickers...")
        
        df = yf.download(
            self.tickers, 
            start=self.start_date, 
            end=self.end_date, 
            auto_adjust=True
        )
        
        # Handle column structure
        if 'Close' in df.columns:
            self.data = df['Close']
        elif 'Adj Close' in df.columns:
            self.data = df['Adj Close']
        else:
            self.data = df
            
        # Drop tickers with missing data
        self.data = self.data.dropna(axis=1, how='any')
        print(f"Data fetched. Shape: {self.data.shape}")
        
        return self.data
        
    def compute_returns(self) -> pd.DataFrame:
        """Computes log returns."""
        if self.data is None:
            raise ValueError("Data not fetched yet. Call fetch_data() first.")
            
        self.returns = np.log(self.data / self.data.shift(1)).dropna()
        print(f"Returns computed. Shape: {self.returns.shape}")
        
        return self.returns
        
    # ==================== GRAPH LAYER ====================
        
    def build_graph(self, sigma: Optional[float] = None) -> np.ndarray:
        """
        Builds a correlation graph with Gaussian kernel weights.
        
        Args:
            sigma: Bandwidth for Gaussian kernel (auto-computed if None)
            
        Returns:
            Adjacency matrix (N x N)
        """
        if self.returns is None:
            raise ValueError("Returns not computed. Call compute_returns() first.")
        
        # Correlation matrix
        self.correlation_matrix = self.returns.corr().values
        
        # Distance metric: d = sqrt(2 * (1 - rho))
        # This is a proper metric on correlations
        dist_matrix = np.sqrt(2 * (1 - self.correlation_matrix))
        
        # Gaussian Kernel for adjacency: W_ij = exp(-d^2 / (2*sigma^2))
        if sigma is None:
            sigma = np.mean(dist_matrix)  # Heuristic bandwidth
            
        self.adj_matrix = np.exp(-(dist_matrix ** 2) / (2 * sigma ** 2))
        np.fill_diagonal(self.adj_matrix, 0)  # No self-loops
        
        print(f"Graph built. Adjacency matrix shape: {self.adj_matrix.shape}")
        print(f"  Sigma (bandwidth): {sigma:.4f}")
        print(f"  Mean edge weight: {np.mean(self.adj_matrix):.4f}")
        
        return self.adj_matrix
        
    def compute_laplacian_residuals(self, t: float = 1.0) -> np.ndarray:
        """
        Computes residuals using heat kernel diffusion on the graph Laplacian.
        
        The heat kernel smooths signals according to graph structure:
        - Smooth component: Captures market-wide/sector movements
        - Residual component: Captures idiosyncratic (stock-specific) movements
        
        Args:
            t: Diffusion time (larger = more smoothing)
            
        Returns:
            Residuals matrix (N_stocks x T_time)
        """
        if self.adj_matrix is None:
            raise ValueError("Adjacency matrix not built. Call build_graph() first.")
        
        n = self.adj_matrix.shape[0]
        
        # Degree Matrix with numerical stability
        degrees = np.sum(self.adj_matrix, axis=1)
        
        # Add small epsilon to prevent division by zero
        eps = 1e-10
        degrees = np.maximum(degrees, eps)
        
        # Normalized Laplacian: L_norm = I - D^-1/2 W D^-1/2
        d_inv_sqrt = np.power(degrees, -0.5)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        
        L_norm = np.eye(n) - D_inv_sqrt @ self.adj_matrix @ D_inv_sqrt
        
        # Ensure L_norm is symmetric (numerical stability)
        L_norm = (L_norm + L_norm.T) / 2
        
        # Heat Kernel via eigendecomposition (more stable than matrix exponential)
        # H = exp(-t * L) = U * exp(-t * Lambda) * U^T
        print(f"Computing heat kernel via eigendecomposition (t={t})...")
        
        # Suppress expected numerical warnings during matrix operations
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            try:
                # Use symmetric eigendecomposition for better stability
                eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
                
                # Clip eigenvalues to valid range [0, 2] for normalized Laplacian
                eigenvalues = np.clip(eigenvalues, 0, 2)
                
                # Compute exp(-t * lambda) for each eigenvalue
                exp_eigenvalues = np.exp(-t * eigenvalues)
                
                # Reconstruct heat kernel: H = U * diag(exp(-t*lambda)) * U^T
                H = eigenvectors @ np.diag(exp_eigenvalues) @ eigenvectors.T
                
            except np.linalg.LinAlgError:
                print("  Warning: Eigendecomposition failed, using matrix exponential fallback")
                H = expm(-t * L_norm)
            
            # Ensure H is valid
            H = np.nan_to_num(H, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Apply diffusion to returns signal
            # Signal shape: (N_stocks, T_time)
            signal = self.returns.T.values.astype(np.float64)
            
            # Smoothed signal (market component)
            smoothed_signal = H @ signal
            
            # Handle any remaining numerical issues
            smoothed_signal = np.nan_to_num(smoothed_signal, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Residuals (idiosyncratic component)
        self.residuals = signal - smoothed_signal
        
        print(f"Residuals computed. Shape: {self.residuals.shape}")
        
        return self.residuals
    
    def get_residual_rankings(self, time_index: int = -1) -> pd.DataFrame:
        """
        Get stocks ranked by their residuals at a given time.
        
        Args:
            time_index: Which time point to use (-1 = most recent)
            
        Returns:
            DataFrame with ticker, residual, z_score, signal, sorted by absolute residual
        """
        if self.residuals is None:
            raise ValueError("Residuals not computed. Call compute_laplacian_residuals() first.")
            
        # Get residuals at specified time
        residuals_at_t = self.residuals[:, time_index]
        
        # Z-score normalization (Section 5.2 of technical paper)
        mu = np.mean(residuals_at_t)
        sigma = np.std(residuals_at_t)
        z_scores = (residuals_at_t - mu) / sigma if sigma > 0 else residuals_at_t
        
        # Create DataFrame
        df = pd.DataFrame({
            'ticker': self.returns.columns,
            'residual': residuals_at_t,
            'z_score': z_scores,
        })
        
        # Generate signals based on z-score thresholds (Section 5.2)
        def get_signal(z):
            if z > 2:
                return 'STRONG_BUY'
            elif z > 1:
                return 'BUY'
            elif z < -2:
                return 'STRONG_SELL'
            elif z < -1:
                return 'SELL'
            else:
                return 'HOLD'
        
        df['signal'] = df['z_score'].apply(get_signal)
        df['abs_residual'] = np.abs(df['residual'])
        df = df.sort_values('abs_residual', ascending=False)
        
        return df[['ticker', 'residual', 'z_score', 'signal', 'abs_residual']]
    
    def get_trading_signals(self, min_z_score: float = 1.0) -> dict:
        """
        Get actionable trading signals filtered by z-score threshold.
        
        Args:
            min_z_score: Minimum absolute z-score for a signal (default 1.0)
            
        Returns:
            Dictionary with 'buy' and 'sell' lists of tickers
        """
        rankings = self.get_residual_rankings()
        
        buys = rankings[rankings['z_score'] > min_z_score]['ticker'].tolist()
        sells = rankings[rankings['z_score'] < -min_z_score]['ticker'].tolist()
        
        return {
            'buy': buys,
            'sell': sells,
            'buy_count': len(buys),
            'sell_count': len(sells),
        }
    
    def get_position_size_multiplier(self) -> float:
        """
        Get position size multiplier based on regime (Section 5.3).
        
        Returns:
            Multiplier: 1.0 (stable), 0.5 (transitioning), 0.25 (fragmented)
        """
        # Thresholds calibrated for typical H1 persistence values
        # These should be tuned based on historical data
        H1_LOW = 0.1    # Below this = stable
        H1_HIGH = 0.5   # Above this = fragmented
        
        h1 = self.h1_total_persistence
        
        if h1 < H1_LOW:
            return 1.0   # Stable regime: full size
        elif h1 < H1_HIGH:
            return 0.5   # Transitioning: half size
        else:
            return 0.25  # Fragmented: quarter size
    
    def get_regime_status(self) -> dict:
        """
        Get current regime status with position sizing recommendation.
        
        Returns:
            Dictionary with regime info and sizing
        """
        multiplier = self.get_position_size_multiplier()
        
        if multiplier == 1.0:
            regime = "STABLE"
            description = "Low H1 persistence - simple market structure"
        elif multiplier == 0.5:
            regime = "TRANSITIONING"
            description = "Medium H1 persistence - market structure shifting"
        else:
            regime = "FRAGMENTED"
            description = "High H1 persistence - complex/unstable structure"
            
        return {
            'regime': regime,
            'description': description,
            'h1_persistence': self.h1_total_persistence,
            'h1_features': self.h1_feature_count,
            'position_multiplier': multiplier,
            'recommended_sizing': f"{int(multiplier * 100)}% of normal"
        }
        
    # ==================== TOPOLOGY LAYER ====================
        
    def compute_topology(self) -> dict:
        """
        Computes Persistent Homology (H0, H1) on the correlation structure.
        
        Returns:
            Dictionary with topology features
        """
        if self.returns is None:
            raise ValueError("Returns not computed. Call compute_returns() first.")
        
        from ripser import ripser
        
        print("Computing Persistent Homology...")
        
        # Distance matrix from correlations
        corr = self.returns.corr().values
        dist_matrix = np.sqrt(2 * (1 - corr))
        np.fill_diagonal(dist_matrix, 0)
        
        # Rips Filtration (maxdim=1 computes H0 and H1)
        result = ripser(dist_matrix, distance_matrix=True, maxdim=1)
        self.diagrams = result['dgms']
        
        # Extract features
        h0 = self.diagrams[0]  # Connected components
        h1 = self.diagrams[1]  # Loops
        
        # H0 features (filter out infinite)
        h0_finite = h0[np.isfinite(h0[:, 1])]
        self.h0_component_count = len(h0)
        
        # H1 features (loops are key for regime detection)
        h1_lifetimes = h1[:, 1] - h1[:, 0]
        h1_lifetimes = h1_lifetimes[np.isfinite(h1_lifetimes)]
        
        self.h1_total_persistence = float(np.sum(h1_lifetimes))
        self.h1_max_persistence = float(np.max(h1_lifetimes)) if len(h1_lifetimes) > 0 else 0.0
        self.h1_feature_count = len(h1)
        
        print(f"Topology computed:")
        print(f"  H0 Components: {self.h0_component_count}")
        print(f"  H1 Features (loops): {self.h1_feature_count}")
        print(f"  H1 Total Persistence: {self.h1_total_persistence:.4f}")
        print(f"  H1 Max Persistence: {self.h1_max_persistence:.4f}")
        
        return {
            'h0_count': self.h0_component_count,
            'h1_count': self.h1_feature_count,
            'h1_total': self.h1_total_persistence,
            'h1_max': self.h1_max_persistence,
            'diagrams': self.diagrams,
        }
    
    def plot_persistence_diagram(self, save_path: Optional[str] = None):
        """Plot the persistence diagram."""
        if self.diagrams is None:
            raise ValueError("Topology not computed. Call compute_topology() first.")
            
        from persim import plot_diagrams
        
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_diagrams(self.diagrams, ax=ax)
        ax.set_title("Persistence Diagram (Stock Correlation Structure)")
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Diagram saved to {save_path}")
        else:
            plt.show()
            
    # ==================== LLM LAYER ====================
    
    @property
    def sentiment_analyzer(self):
        """Lazy load sentiment analyzer."""
        if self._sentiment_analyzer is None:
            from news_sentiment import NewsSentimentAnalyzer
            self._sentiment_analyzer = NewsSentimentAnalyzer()
        return self._sentiment_analyzer
    
    @property
    def regime_interpreter(self):
        """Lazy load regime interpreter."""
        if self._regime_interpreter is None:
            from regime_interpreter import RegimeInterpreter
            self._regime_interpreter = RegimeInterpreter()
        return self._regime_interpreter
    
    @property
    def report_generator(self):
        """Lazy load report generator."""
        if self._report_generator is None:
            from report_generator import ReportGenerator
            self._report_generator = ReportGenerator()
        return self._report_generator
    
    def analyze_sentiment(self, top_n: int = 5) -> pd.DataFrame:
        """
        Analyze news sentiment for top residual stocks.
        
        Args:
            top_n: Number of top residual stocks to analyze
            
        Returns:
            DataFrame with sentiment results
        """
        if self.residuals is None:
            raise ValueError("Run the quantitative pipeline first.")
            
        # Get top residual stocks
        rankings = self.get_residual_rankings()
        top_tickers = rankings.head(top_n)['ticker'].tolist()
        
        # Get deviations for context
        deviations = dict(zip(rankings['ticker'], rankings['residual']))
        
        print(f"\nAnalyzing sentiment for top {top_n} residual stocks...")
        return self.sentiment_analyzer.analyze_batch(top_tickers, deviations)
    
    def interpret_regime(
        self,
        previous_h1: Optional[float] = None,
        previous_h0: Optional[int] = None,
        market_context: Optional[str] = None,
    ):
        """
        Interpret the current market regime using LLM.
        
        Args:
            previous_h1: Previous H1 total persistence for comparison
            previous_h0: Previous H0 count for comparison
            market_context: Optional market context string
            
        Returns:
            RegimeInterpretation object
        """
        if self.diagrams is None:
            raise ValueError("Run compute_topology() first.")
            
        print("\nInterpreting market regime...")
        return self.regime_interpreter.interpret(
            h1_total_persistence=self.h1_total_persistence,
            h1_max_persistence=self.h1_max_persistence,
            h1_feature_count=self.h1_feature_count,
            h0_component_count=self.h0_component_count,
            previous_h1_total=previous_h1,
            previous_h0_count=previous_h0,
            market_context=market_context,
        )
    
    def generate_report(
        self,
        regime_type: str = "Unknown",
        risk_level: str = "Medium",
        sentiment_data: Optional[pd.DataFrame] = None,
    ):
        """
        Generate a comprehensive trading report.
        
        Args:
            regime_type: Current regime type
            risk_level: Current risk level
            sentiment_data: Optional sentiment analysis results
            
        Returns:
            TradingReport object
        """
        if self.residuals is None:
            raise ValueError("Run the quantitative pipeline first.")
            
        residuals_df = self.get_residual_rankings()
        
        print("\nGenerating trading report...")
        return self.report_generator.generate_daily_report(
            h1_total_persistence=self.h1_total_persistence,
            h1_max_persistence=self.h1_max_persistence,
            h1_feature_count=self.h1_feature_count,
            regime_type=regime_type,
            risk_level=risk_level,
            top_residuals=residuals_df.head(10),
            sentiment_data=sentiment_data,
        )
        
    # ==================== PIPELINE ====================

    def run_quantitative_pipeline(self, diffusion_time: float = 1.0):
        """Run the core quantitative analysis pipeline."""
        print("="*60)
        print("STOCK PERFECT MODEL - Quantitative Pipeline")
        print("="*60)
        
        self.fetch_data()
        self.compute_returns()
        self.build_graph()
        self.compute_laplacian_residuals(t=diffusion_time)
        self.compute_topology()
        
        print("\n" + "="*60)
        print("Quantitative Pipeline Complete")
        print("="*60)
        
    def run_full_pipeline(
        self,
        diffusion_time: float = 1.0,
        analyze_sentiment: bool = True,
        interpret_regime: bool = True,
        generate_report: bool = True,
        save_report_path: Optional[str] = None,
    ):
        """
        Run the complete pipeline including LLM analysis.
        
        Args:
            diffusion_time: Heat kernel diffusion time
            analyze_sentiment: Whether to run sentiment analysis
            interpret_regime: Whether to interpret regime
            generate_report: Whether to generate report
            save_report_path: Optional path to save HTML report
        """
        # Run quantitative pipeline
        self.run_quantitative_pipeline(diffusion_time)
        
        # LLM Analysis
        sentiment_df = None
        regime_type = "Unknown"
        risk_level = "Medium"
        
        if interpret_regime:
            try:
                interpretation = self.interpret_regime()
                regime_type = interpretation.regime_type
                risk_level = interpretation.risk_level
                print(f"\nRegime: {regime_type} | Risk: {risk_level}")
                print(f"Description: {interpretation.description}")
            except Exception as e:
                print(f"\nRegime interpretation skipped: {e}")
                
        if analyze_sentiment:
            try:
                sentiment_df = self.analyze_sentiment(top_n=5)
                print(f"\nSentiment Analysis:\n{sentiment_df}")
            except Exception as e:
                print(f"\nSentiment analysis skipped: {e}")
                
        if generate_report:
            try:
                report = self.generate_report(
                    regime_type=regime_type,
                    risk_level=risk_level,
                    sentiment_data=sentiment_df,
                )
                
                print("\n" + "="*60)
                print(report.to_markdown())
                
                if save_report_path:
                    with open(save_report_path, 'w') as f:
                        f.write(report.to_html())
                    print(f"\nHTML report saved to: {save_report_path}")
                    
            except Exception as e:
                print(f"\nReport generation skipped: {e}")
                
        return self


if __name__ == "__main__":
    # Example usage
    tickers = [
        'AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 
        'TSLA', 'META', 'AMD', 'INTC', 'QCOM', 
        'SPY', 'QQQ'
    ]
    
    model = StockPerfectModel(
        tickers=tickers,
        start_date="2023-01-01",
        end_date="2024-01-01"
    )
    
    # Run the quantitative pipeline
    model.run_quantitative_pipeline()
    
    # Show residual rankings with z-scores and signals
    print("\n" + "="*70)
    print("RESIDUAL RANKINGS (with z-scores and signals)")
    print("="*70)
    print(model.get_residual_rankings().head(10).to_string(index=False))
    
    # Show trading signals
    print("\n" + "="*70)
    print("TRADING SIGNALS")
    print("="*70)
    signals = model.get_trading_signals(min_z_score=1.0)
    print(f"✅ BUY  (z > +1): {signals['buy']}")
    print(f"⚠️  SELL (z < -1): {signals['sell']}")
    
    # Show regime status
    print("\n" + "="*70)
    print("REGIME STATUS")
    print("="*70)
    regime = model.get_regime_status()
    print(f"Regime:              {regime['regime']}")
    print(f"Description:         {regime['description']}")
    print(f"H1 Persistence:      {regime['h1_persistence']:.4f}")
    print(f"H1 Features:         {regime['h1_features']}")
    print(f"Position Sizing:     {regime['recommended_sizing']}")
    
    # Option 2: Run full pipeline with LLM (requires AWS Bedrock credentials)
    # Uncomment to enable:
    # model.run_full_pipeline(
    #     analyze_sentiment=True,
    #     interpret_regime=True,
    #     generate_report=True,
    #     save_report_path="daily_report.html"
    # )
