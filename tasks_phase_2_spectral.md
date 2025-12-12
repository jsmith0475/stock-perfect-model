# Task Plan: Phase 2 - Spectral Analysis & Graph Fourier Transform

## Objective
Enhance the Stock Perfect Model by implementing **Graph Fourier Transform (GFT)**. This will allow us to decompose market returns into "modes" (Market, Sector, Idiosyncratic) and measure the "Spectral Energy" to detect market coherence and regime changes.

## Tasks

### 1. Refactor Laplacian Logic
- [ ] **Promote Eigendecomposition**: Modify `compute_laplacian_residuals` in `stock_perfect.py` to store `eigenvalues` and `eigenvectors` as class attributes (`self.eigenvalues`, `self.eigenvectors`) so they can be reused for GFT.

### 2. Implement Spectral Analysis
- [ ] **Create `compute_spectral_features` method**:
    - [ ] Project return signals onto the Laplacian eigenvectors (GFT).
    - [ ] Compute **Power Spectrum** (Energy per mode).
    - [ ] Calculate **Market Coherence**: The percentage of total energy concentrated in the lowest frequency eigenmodes (the "Market Mode").
    - [ ] Calculate **Spectral Entropy**: A measure of market complexity (Low entropy = ordered/one factor driving market; High entropy = chaotic/many factors).

### 3. Integrate & Visualize
- [ ] **Update Pipeline**: Add `compute_spectral_features` to `run_quantitative_pipeline`.
- [ ] **Update Regime Status**: Include "Market Coherence" and "Spectral Entropy" in the `get_regime_status` output.
- [ ] **Verification**: Run `stock_perfect.py` to verify the new metrics are calculated and displayed correctly.
