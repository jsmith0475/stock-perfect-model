"""
Regime Interpretation Module.

Uses AWS Bedrock (Claude Sonnet) to interpret topology-based market regime changes.
When the persistent homology features (H0, H1) change significantly, this module
provides human-readable explanations of what the structural changes mean.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

from bedrock_client import BedrockClient, ClaudeModel


@dataclass
class RegimeInterpretation:
    """Structured regime change interpretation."""
    regime_type: str           # e.g., "Risk-On", "Risk-Off", "Rotation", "Fragmentation"
    confidence: float          # 0 to 1
    description: str           # Human-readable explanation
    trading_implications: str  # What this means for trading
    risk_level: str            # "Low", "Medium", "High", "Extreme"
    recommended_actions: list[str]
    timestamp: datetime


class RegimeInterpreter:
    """
    Interprets market regime changes using topological features and Claude Sonnet.
    
    Uses Sonnet for complex reasoning about market structure.
    """
    
    SYSTEM_PROMPT = """You are a quantitative analyst specializing in market microstructure and regime detection.

You understand:
- Persistent Homology: H0 (connected components), H1 (loops/cycles) in correlation networks
- Market regimes: Risk-on, risk-off, sector rotation, flight to quality, fragmentation
- How correlation structure changes reflect underlying market dynamics

When H1 persistence increases: More correlation loops = more complex, potentially unstable structure
When H1 persistence decreases: Simpler structure = market dominated by few factors (risk-on/off)
When H0 changes: Number of distinct market clusters is changing

Provide actionable, quantitative insights. Be specific about implications."""

    # Regime thresholds (calibrate these based on historical data)
    THRESHOLDS = {
        "h1_spike": 1.5,      # H1 increased by 50%+ = regime change
        "h1_drop": 0.5,       # H1 decreased by 50%+ = simplification
        "fragmentation": 2.0, # H1 > 2x historical mean = fragmented market
    }

    def __init__(self):
        self.client = BedrockClient()
        self.history: list[dict] = []
        
    def _compute_regime_metrics(
        self,
        h1_current: float,
        h1_previous: float,
        h0_current: int,
        h0_previous: int,
        h1_historical_mean: Optional[float] = None,
    ) -> dict:
        """Compute regime change metrics from topology features."""
        
        h1_change_ratio = h1_current / max(h1_previous, 0.001)
        h0_change = h0_current - h0_previous
        
        metrics = {
            "h1_current": h1_current,
            "h1_previous": h1_previous,
            "h1_change_ratio": h1_change_ratio,
            "h1_change_pct": (h1_change_ratio - 1) * 100,
            "h0_current": h0_current,
            "h0_previous": h0_previous,
            "h0_change": h0_change,
            "is_h1_spike": h1_change_ratio > self.THRESHOLDS["h1_spike"],
            "is_h1_drop": h1_change_ratio < self.THRESHOLDS["h1_drop"],
            "is_fragmenting": (
                h1_historical_mean and h1_current > self.THRESHOLDS["fragmentation"] * h1_historical_mean
            ),
        }
        
        return metrics
    
    def interpret(
        self,
        h1_total_persistence: float,
        h1_max_persistence: float,
        h1_feature_count: int,
        h0_component_count: int,
        previous_h1_total: Optional[float] = None,
        previous_h0_count: Optional[int] = None,
        h1_historical_mean: Optional[float] = None,
        market_context: Optional[str] = None,
    ) -> RegimeInterpretation:
        """
        Interpret the current market regime based on topology features.
        
        Args:
            h1_total_persistence: Sum of H1 feature lifetimes (loop complexity)
            h1_max_persistence: Maximum H1 lifetime (dominant loop strength)
            h1_feature_count: Number of H1 features (loops)
            h0_component_count: Number of H0 components (clusters)
            previous_h1_total: Previous period's H1 total for comparison
            previous_h0_count: Previous period's H0 count
            h1_historical_mean: Long-term H1 mean for context
            market_context: Optional recent market events for context
            
        Returns:
            RegimeInterpretation with analysis and recommendations
        """
        # Use defaults if no history
        previous_h1 = previous_h1_total or h1_total_persistence
        previous_h0 = previous_h0_count or h0_component_count
        
        metrics = self._compute_regime_metrics(
            h1_total_persistence, previous_h1,
            h0_component_count, previous_h0,
            h1_historical_mean
        )
        
        # Build the prompt
        context_section = f"\nRecent Market Context: {market_context}" if market_context else ""
        
        prompt = f"""Analyze this market regime change based on persistent homology features:

CURRENT TOPOLOGY METRICS:
- H1 Total Persistence: {h1_total_persistence:.4f} (previous: {previous_h1:.4f})
- H1 Change: {metrics['h1_change_pct']:+.1f}%
- H1 Max Persistence: {h1_max_persistence:.4f}
- H1 Feature Count (loops): {h1_feature_count}
- H0 Components (clusters): {h0_component_count} (previous: {previous_h0})

DETECTED FLAGS:
- H1 Spike Detected: {metrics['is_h1_spike']}
- H1 Drop Detected: {metrics['is_h1_drop']}
- Market Fragmenting: {metrics['is_fragmenting']}
{context_section}

Provide your analysis as JSON:
{{
    "regime_type": "<one of: Risk-On, Risk-Off, Sector-Rotation, Fragmentation, Consolidation, Crisis, Stable>",
    "confidence": <float 0-1>,
    "description": "<2-3 sentence explanation of what the topology changes mean for market structure>",
    "trading_implications": "<specific implications for trading - position sizing, sector exposure, hedging>",
    "risk_level": "<Low, Medium, High, or Extreme>",
    "recommended_actions": [<list of 2-4 specific actionable recommendations>]
}}"""

        try:
            result = self.client.invoke_for_json(
                prompt=prompt,
                model=ClaudeModel.SONNET,  # Use Sonnet for complex reasoning
                system_prompt=self.SYSTEM_PROMPT,
                max_tokens=1024,
            )
            
            interpretation = RegimeInterpretation(
                regime_type=result.get("regime_type", "Unknown"),
                confidence=float(result.get("confidence", 0.5)),
                description=result.get("description", ""),
                trading_implications=result.get("trading_implications", ""),
                risk_level=result.get("risk_level", "Medium"),
                recommended_actions=result.get("recommended_actions", []),
                timestamp=datetime.now(),
            )
            
            # Store in history
            self.history.append({
                "timestamp": interpretation.timestamp,
                "metrics": metrics,
                "interpretation": interpretation,
            })
            
            return interpretation
            
        except Exception as e:
            print(f"Regime interpretation error: {e}")
            return RegimeInterpretation(
                regime_type="Unknown",
                confidence=0.0,
                description=f"Analysis failed: {str(e)}",
                trading_implications="Manual review recommended",
                risk_level="High",
                recommended_actions=["Review topology metrics manually", "Check AWS credentials"],
                timestamp=datetime.now(),
            )
    
    def quick_regime_check(
        self,
        h1_total: float,
        h1_historical_mean: float,
    ) -> str:
        """
        Quick heuristic regime check without LLM call.
        
        Args:
            h1_total: Current H1 total persistence
            h1_historical_mean: Historical mean H1 persistence
            
        Returns:
            Quick regime classification string
        """
        ratio = h1_total / max(h1_historical_mean, 0.001)
        
        if ratio > 2.0:
            return "FRAGMENTED - High correlation complexity, reduce position sizes"
        elif ratio > 1.5:
            return "ELEVATED - Above-normal complexity, monitor closely"
        elif ratio < 0.5:
            return "SIMPLIFIED - Factor-dominated market, trend-following may work"
        elif ratio < 0.75:
            return "CONSOLIDATING - Below-normal complexity"
        else:
            return "STABLE - Normal market structure"


def interpret_regime(
    h1_total: float,
    h1_max: float,
    h1_count: int,
    h0_count: int,
    **kwargs
) -> RegimeInterpretation:
    """Convenience function for quick regime interpretation."""
    interpreter = RegimeInterpreter()
    return interpreter.interpret(
        h1_total_persistence=h1_total,
        h1_max_persistence=h1_max,
        h1_feature_count=h1_count,
        h0_component_count=h0_count,
        **kwargs
    )


if __name__ == "__main__":
    # Test the interpreter
    print("Testing Regime Interpreter...")
    
    interpreter = RegimeInterpreter()
    
    # Simulate a regime change scenario
    print("\n--- Scenario: H1 Spike (Market Fragmenting) ---")
    result = interpreter.interpret(
        h1_total_persistence=2.5,
        h1_max_persistence=0.8,
        h1_feature_count=12,
        h0_component_count=3,
        previous_h1_total=1.2,
        previous_h0_count=2,
        h1_historical_mean=1.0,
        market_context="Recent Fed rate decision, earnings season ongoing"
    )
    
    print(f"\nRegime Type: {result.regime_type}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Risk Level: {result.risk_level}")
    print(f"\nDescription:\n{result.description}")
    print(f"\nTrading Implications:\n{result.trading_implications}")
    print(f"\nRecommended Actions:")
    for action in result.recommended_actions:
        print(f"  • {action}")

