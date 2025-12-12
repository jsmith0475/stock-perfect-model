# Task Plan: Phase 3 - Walsh-Hadamard Transform (Time-Series Analysis)

## Objective
Implement **Walsh-Hadamard Transform (WHT)** to analyze the time-series properties of the Laplacian residuals. This will act as a signal quality filter, distinguishing between "elastic" mispricings (good for mean reversion) and "structural" drifts (dangerous).

## Tasks

### 1. Implement Fast Walsh-Hadamard Transform (FWHT)
- [ ] **Create `walsh_transform` utility**: Since standard SciPy doesn't always have a direct FWHT for arbitrary lengths, implement a robust Walsh transform or use a "Sequency" counter (Zero-Crossing Rate) as a proxy if a full transform is overkill for short windows.
    *   *Note*: For trading windows (T=252), a full power-of-2 Walsh might be truncated. A simpler "Sequency Energy Ratio" or direct Zero-Crossing/Turning-Point metric using Walsh logic is often more robust.
    *   *Decision*: We will implement a **Sequency Entropy** metric based on sign-flips, akin to a simplified Walsh analysis.

### 2. Compute Stability Scores
- [ ] **Method `compute_walsh_stability`**:
    - [ ] For each stock's residual history (past N days):
    - [ ] Calculate **Sequency**: How often does the residual flip signs?
    - [ ] Calculate **Walsh Energy**: Is the energy concentrated in high sequency (noise), low sequency (trend), or mid sequency (mean reversion)?
    - [ ] Output a **Stability Score**: High Score = Elastic (Snapper); Low Score = Drifting (Trend).

### 3. Integrate into Signals
- [ ] **Update `get_residual_rankings`**:
    - [ ] Include "Walsh Score" in the ranking DataFrame.
- [ ] **Update `get_trading_signals`**:
    - [ ] **Filter**: Downgrade or reject signals with Low Walsh Scores (Drifters).
    - [ ] **Annotate**: Add "Signal Quality" context to the output (e.g., "High Quality Mean Reversion").

### 4. Verification
- [ ] Run the pipeline and ensure "Drifter" stocks (stocks moving away from the pack without returning) are flagged or filtered out.
