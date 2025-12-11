# Stock Perfect Model (Algebraic Topology Trading Algorithm)

## Context
The user has provided a transcript describing a trading algorithm based on **Algebraic Topology** and **Graph Signal Processing**. The goal is to separate specific stock movements from general market movements using graph diffusion and to characterize market regimes using persistent homology.

## Core Concepts (Inferred)

1.  **Market Graph**: Stocks are nodes, edges are correlations.
2.  **Laplacian Diffusion**: Used to separate "market" signal (smooth) from "idiosyncratic" signal (rough/residuals).
    - $Signal = Smooth + Residual$
    - Smooth part = Global market trend influencing the sector/cluster.
    - Residual part = Local deviation (alpha/mispricing).
3.  **Persistent Homology**: Analyzes the topology of the correlation matrix over time to detect structural changes (regimes).
    - Top logic features (Betti numbers, persistence diagrams) feed into the model to adjust risk or strategy based on market stability.

## Goal
Replicate the logic described:
- Construct stock graph.
- Compute Laplacian Diffusion residuals.
- Compute Topological features.
- Combine into a trading strategy.
