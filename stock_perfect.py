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
        
        # Spectral Components
        self.eigenvalues: Optional[np.ndarray] = None
        self.eigenvectors: Optional[np.ndarray] = None
        self.spectral_features: dict = {}
        
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
                # Store as class attributes for Spectral Analysis
                self.eigenvalues, self.eigenvectors = np.linalg.eigh(L_norm)
                
                # Clip eigenvalues to valid range [0, 2] for normalized Laplacian
                self.eigenvalues = np.clip(self.eigenvalues, 0, 2)
                
                # Compute exp(-t * lambda) for each eigenvalue
                exp_eigenvalues = np.exp(-t * self.eigenvalues)
                
                # Reconstruct heat kernel: H = U * diag(exp(-t*lambda)) * U^T
                H = self.eigenvectors @ np.diag(exp_eigenvalues) @ self.eigenvectors.T
                
            except np.linalg.LinAlgError:
                print("  Warning: Eigendecomposition failed, using matrix exponential fallback")
                H = expm(-t * L_norm)
                # Cannot do spectral analysis if this fails
                self.eigenvalues, self.eigenvectors = None, None
            
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
        
        # Add Walsh Stability (Time-Series Quality)
        # We compute this for every ticker
        stability_metrics = self.compute_walsh_stability()
        if stability_metrics:
            df['walsh_score'] = df['ticker'].map(lambda t: stability_metrics.get(t, {}).get('score', 0))
            df['walsh_class'] = df['ticker'].map(lambda t: stability_metrics.get(t, {}).get('class', 'Unknown'))
        else:
            df['walsh_score'] = 0.0
            df['walsh_class'] = 'Unknown'
            
        df = df.sort_values('abs_residual', ascending=False)
        
        # Reorder columns
        cols = ['ticker', 'residual', 'z_score', 'signal', 'walsh_score', 'walsh_class', 'abs_residual']
        return df[cols]
    
    def compute_walsh_stability(self) -> dict:
        """
        Computes Walsh-like stability metrics (Sequency/Choppiness) for residuals.
        
        Analyzes the time-series of residuals to determine if they are:
        - Elastic (High Choppiness/Mean Reverting) -> GOOD SIGNAL
        - Drifting (Low Choppiness/Trending) -> RISKY SIGNAL
        
        Returns:
            Dictionary mapping ticker -> {score, class}
        """
        if self.residuals is None:
            return {}
            
        results = {}
        tickers = self.returns.columns
        n_days = self.residuals.shape[1]
        
        # Look at last 60 days or full history if shorter
        lookback = min(60, n_days)
        
        for i, ticker in enumerate(tickers):
            # Extract residual time series
            r_series = self.residuals[i, -lookback:]
            
            # center
            r_centered = r_series - np.mean(r_series)
            
            # Count Zero Crossings (Sequency Proxy)
            # Signal: +1 if positive, -1 if negative
            signs = np.sign(r_centered)
            # Remove zeros
            signs = signs[signs != 0]
            
            # Count flips
            flips = np.sum(np.abs(np.diff(signs))) / 2
            
            # Max possible flips is length - 1
            max_flips = len(signs) - 1 if len(signs) > 0 else 1
            
            # Sequency Score (0.0 to 1.0)
            # 1.0 = Flips every day (Perfectly Elastic)
            # 0.0 = Never flips (Perfect Trend/Drift)
            sequency_score = flips / max_flips if max_flips > 0 else 0
            
            # Classify
            if sequency_score > 0.3:
                classification = "Elastic"
            elif sequency_score > 0.15:
                classification = "Mixed"
            else:
                classification = "Drifting"
                
            results[ticker] = {
                'score': sequency_score,
                'class': classification
            }
            
        return results
    
    def get_trading_signals(self, min_z_score: float = 1.0) -> dict:
        """
        Get actionable trading signals filtered by Z-score and coupled with Regime Sizing.
        
        Args:
            min_z_score: Minimum absolute z-score for a signal (default 1.0)
            
        Returns:
            Dictionary with 'buy' and 'sell' lists and sizing info
        """
        rankings = self.get_residual_rankings()
        
        # Get Regime Sizing
        size_multiplier, reasons = self.get_position_size_multiplier()
        
        # If size is too small, filtering is stricter or we kill signals entirely
        if size_multiplier < 0.1:
            return {
                'buy': [],
                'sell': [],
                'buy_count': 0,
                'sell_count': 0,
                'regime_size_multiplier': 0.0,
                'regime_note': "NO TRADE: Regime too unstable"
            }
            
        # Filter raw signals
        raw_buys = rankings[rankings['z_score'] > min_z_score]
        raw_sells = rankings[rankings['z_score'] < -min_z_score]
        
        # Helper to format signal
        def format_signal(row):
            # Base size from global regime
            final_size = size_multiplier
            note = ""
            
            # Stock-specific Walsh adjustment
            w_class = row.get('walsh_class', 'N/A')
            w_score = row.get('walsh_score', 0.0)
            
            if w_class == 'Drifting':
                final_size = 0.0
                note = " (Drifting - Avoid)"
            elif w_class == 'Mixed':
                final_size *= 0.5
                note = " (Mixed Quality)"
                
            return {
                'ticker': row['ticker'],
                'z_score': round(row['z_score'], 2),
                'residual': round(row['residual'], 4),
                'walsh_class': w_class,
                'walsh_score': w_score,
                'recommended_size': f"{int(final_size * 100)}%{note}"
            }
            
        buy_signals = [format_signal(row) for _, row in raw_buys.iterrows()]
        sell_signals = [format_signal(row) for _, row in raw_sells.iterrows()]
        
        return {
            'buy': buy_signals,
            'sell': sell_signals,
            'buy_count': len(buy_signals),
            'sell_count': len(sell_signals),
            'regime_size_multiplier': size_multiplier,
            'regime_reasons': reasons
        }
    
    def get_position_size_multiplier(self) -> tuple[float, list[str]]:
        """
        Get position size multiplier based on Regime (Topology + Spectral).
        
        Returns:
            Tuple of (Multiplier, List of reasons)
            Multiplier: 0.0 to 1.0
        """
        multiplier = 1.0
        reasons = []
        
        # 1. Topology Check (H1 Loops)
        # Thresholds calibrated for typical H1 persistence
        H1_LOW = 0.1    # Below this = stable
        H1_HIGH = 0.5   # Above this = fragmented
        
        if self.h1_total_persistence > H1_HIGH:
            multiplier *= 0.25
            reasons.append("High Topological Complexity (Fragmented Structure)")
        elif self.h1_total_persistence > H1_LOW:
            multiplier *= 0.5
            reasons.append("Moderate Topological Complexity")
            
        # 2. Spectral Check (Coherence & Entropy)
        if self.spectral_features:
            coh = self.spectral_features.get('market_coherence', 0)
            ent = self.spectral_features.get('normalized_entropy', 0)
            
            # High Coherence = Market is moving as one block
            # Hard to pick distinct winners, increasing Z-threshold effectively
            # But here we just reduce size for caution against false positives
            if coh > 0.7: 
                multiplier *= 0.75
                reasons.append(f"High Market Coherence ({coh:.2f}) - Beta driving returns")
                
            # High Entropy = Chaotic energy distribution
            if ent > 0.8:
                multiplier *= 0.5
                reasons.append(f"High Spectral Entropy ({ent:.2f}) - Disordered Market")
                
        # Cap min/max
        multiplier = max(0.0, min(1.0, multiplier))
        
        if multiplier == 1.0:
            reasons.append("Stable Market Regime")
            
        return multiplier, reasons
    
    def get_regime_status(self) -> dict:
        """
        Get current regime status with position sizing recommendation.
        
        Returns:
            Dictionary with regime info and sizing
        """
        multiplier, reasons = self.get_position_size_multiplier()
        
        if multiplier >= 0.8:
            regime = "STABLE"
        elif multiplier >= 0.4:
            regime = "TRANSITIONING"
        else:
            regime = "DEFENSIVE"
            
        description = " | ".join(reasons)
        
        # Add spectral context if available
        spectral_desc = ""
        if self.spectral_features:
            coh = self.spectral_features.get('market_coherence', 0)
            ent = self.spectral_features.get('normalized_entropy', 0)
            
            if coh > 0.5:
                spectral_desc = " | High Market Coherence"
            elif ent > 0.8:
                spectral_desc = " | High Entropy"
        
        # Add Walsh context (Market Elasticity) if available
        walsh_desc = ""
        walsh_data = self.compute_walsh_stability()
        avg_elasticity = 0.0
        if walsh_data:
            scores = [d['score'] for d in walsh_data.values()]
            avg_elasticity = sum(scores) / len(scores) if scores else 0
            
            if avg_elasticity > 0.4:
                walsh_desc = " | High Elasticity (Mean Reverting)"
            elif avg_elasticity < 0.2:
                walsh_desc = " | Low Elasticity (Trending)"
                
        return {
            'regime': regime,
            'description': description + spectral_desc + walsh_desc,
            'h1_persistence': self.h1_total_persistence,
            'h1_features': self.h1_feature_count,
            'spectral_features': self.spectral_features,
            'market_elasticity': avg_elasticity,
            'position_multiplier': multiplier,
            'recommended_sizing': f"{int(multiplier * 100)}% of normal",
            'sizing_reasons': reasons
        }
            
    def compute_spectral_features(self) -> dict:
        """
        Computes Graph Fourier Transform (GFT) features.
        
        Analyzes the power spectrum of the returns on the graph to measure:
        - Market Coherence: How much energy is in the global market mode (lambda_0)
        - Spectral Entropy: How complex/disordered the market signal is
        
        Returns:
            Dictionary of spectral features
        """
        if self.eigenvectors is None:
            print("  Warning: Eigenvectors not available. Skipping spectral analysis.")
            return {}
            
        if self.returns is None:
            raise ValueError("Returns not computed.")
            
        print("Computing Spectral Analysis (GFT)...")
        
        # 1. Graph Fourier Transform (GFT)
        # Project signal (returns) onto eigenvectors (U^T * s)
        # Using the most recent time step
        recent_returns = self.returns.iloc[-1].values
        gft_coefficients = self.eigenvectors.T @ recent_returns
        
        # 2. Power Spectrum (Energy)
        power_spectrum = gft_coefficients ** 2
        total_energy = np.sum(power_spectrum)
        
        # Normalize to get probability distribution for specific modes
        p_spectrum = power_spectrum / total_energy
        
        # 3. Market Coherence (Energy in lowest frequency mode lambda_0)
        # The first eigenvector (lambda ~ 0) usually represents the global market trend
        market_coherence = p_spectrum[0]
        
        # 4. Spectral Entropy
        # H(s) = -sum(p * log(p))
        # Add epsilon for numerical stability
        p_spectrum_safe = p_spectrum[p_spectrum > 0]
        spectral_entropy = -np.sum(p_spectrum_safe * np.log(p_spectrum_safe))
        
        # Normalize entropy between 0 and 1 (divided by log(N))
        n_modes = len(p_spectrum)
        normalized_entropy = spectral_entropy / np.log(n_modes)
        
        self.spectral_features = {
            'market_coherence': market_coherence,
            'spectral_entropy': spectral_entropy,
            'normalized_entropy': normalized_entropy,
            'total_energy': total_energy
        }
        
        print(f"Spectral Analysis Computed:")
        print(f"  Market Coherence (Lambda_0 Energy): {market_coherence:.4f}")
        print(f"  Normalized Spectral Entropy: {normalized_entropy:.4f}")
        
        return self.spectral_features
        
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
        self.compute_spectral_features()
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
    print("Legend:")
    print("  residual = How much stock deviates from graph-diffusion expectation")
    print("  z_score  = Standard deviations from mean (>±1 = tradeable signal)")
    print("  walsh_score = Sequency (sign-flip rate): >0.3=Elastic, <0.15=Drifting")
    print("")
    print(model.get_residual_rankings().head(10).to_string(index=False))
    
    # Show trading signals
    print("\n" + "="*70)
    print("TRADING SIGNALS (Regime Adjusted)")
    print("="*70)
    signals = model.get_trading_signals(min_z_score=1.0)
    
    print(f"Regime Multiplier: {signals['regime_size_multiplier']:.2f}")
    print(f"  → This is the GLOBAL position sizing based on market topology/spectral risk")
    if isinstance(signals.get('regime_reasons'), list):
         print(f"  → Factors: {', '.join(signals['regime_reasons'])}")
    
    print(f"\n✅ BUY  (Count: {signals['buy_count']})")
    print(f"   Format: [Ticker] (Z-score) | Walsh Class (Score) | Final Size")
    print(f"   Legend: Z-score = How unusual (>1 = tradeable) | Walsh = Mean-reversion quality (>0.3 = Elastic)")
    for s in signals['buy']:
        print(f"   - {s['ticker']:<5} (z={s['z_score']:>4.1f}) | Walsh: {s['walsh_class']:<8} ({s['walsh_score']:.2f}) | Size: {s['recommended_size']}")
        
    print(f"\n⚠️  SELL (Count: {signals['sell_count']})")
    for s in signals['sell']:
        print(f"   - {s['ticker']:<5} (z={s['z_score']:>4.1f}) | Walsh: {s['walsh_class']:<8} ({s['walsh_score']:.2f}) | Size: {s['recommended_size']}")
    
    # Show regime status
    print("\n" + "="*70)
    print("REGIME STATUS")
    print("="*70)
    regime = model.get_regime_status()
    print(f"Regime:              {regime['regime']}")
    print(f"Description:         {regime['description']}")
    print(f"H1 Persistence:      {regime['h1_persistence']:.4f}  (Low < 0.1 = Stable | High > 0.5 = Fragmented)")
    
    if regime.get('spectral_features'):
        sf = regime['spectral_features']
        print(f"Market Coherence:    {sf['market_coherence']:.4f}  (High > 0.7 = Beta-driven | Low < 0.3 = Stock-picking)")
        print(f"Spectral Entropy:    {sf['normalized_entropy']:.4f}  (High > 0.8 = Chaotic | Low < 0.5 = Orderly)")
        
    print(f"Avg Elasticity:      {regime.get('market_elasticity', 0):.4f}  (High > 0.4 = Mean-reverting market)")
    print(f"Position Sizing:     {regime['recommended_sizing']}")
    
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print("Size = Regime Multiplier × Walsh Multiplier")
    print("  - 100% = Perfect conditions (Stable regime + Elastic stock)")
    print("  - 50%  = Moderate risk (Transitioning regime OR Mixed elasticity)")
    print("  - 0%   = DO NOT TRADE (Fragmented regime OR Drifting stock)")
    print("\nWalsh Score (Sequency):")
    print("  - >0.5  = Highly Elastic (stock bounces aggressively around mean)")
    print("  - 0.3-0.5 = Elastic (good mean-reversion behavior)")
    print("  - 0.15-0.3 = Mixed (indeterminate structure)")
    print("  - <0.15 = Drifting (trending away - AVOID falling knife)")
    
    # Option 2: Run full pipeline with LLM (requires AWS Bedrock credentials)
    # Uncomment to enable:
    # model.run_full_pipeline(
    #     analyze_sentiment=True,
    #     interpret_regime=True,
    #     generate_report=True,
    #     save_report_path="daily_report.html"
    # )
