import os
import sys


def get_static_content():
    """
    Return the full LaTeX content representing the model formulation.
    
    This template is kept in sync with model.py manually.
    Last verified against model.py: 2026-02-20.
    """

    TEMPLATE = r"""
%===================================================
% Model Formulation
%===================================================

\section{Model Formulation}

%---------------------------------------------------
\subsection{Sets}
%---------------------------------------------------

\begin{flushleft}
\begin{tabular}{@{}ll@{}}
$r, i, j, e \in R$ & Set of regions (exporters, importers) \\
$t \in T$ & Set of time periods (e.g., 2025, 2030, 2035, 2040) \\
\end{tabular}
\end{flushleft}

%---------------------------------------------------
\subsection{Parameters}
%---------------------------------------------------

\begin{flushleft}
\begin{tabular}{@{}lp{10cm}@{}}
\toprule
\textbf{Symbol} & \textbf{Description} \\
\midrule
$a_{i,t} > 0$ & Inverse demand intercept in region $i$ at time $t$ \\
$b_{i,t} > 0$ & Inverse demand slope in region $i$ at time $t$ \\
$D^{max}_{i,t} > 0$ & Demand/installation capacity cap in region $i$ at time $t$ \\
$c^{man}_r$ & Manufacturing cost in region $r$ \\
$c^{ship}_{ri}$ & Shipping cost from region $r$ to region $i$ \\
$K^{cap,init}_r$ & Initial production capacity in region $r$ (at $t=2025$) \\
$\overline{\tau}^{imp}_{ir} \ge 0$ & Upper bound on import tariff (set by $i$ on imports from $r$) \\
$\overline{\tau}^{exp}_{ri} \ge 0$ & Upper bound on export tax (set by $r$ on exports to $i$) \\
$s^{ub}_r \ge 0$ & Upper bound on production subsidy in region $r$ \\
$f^{hold}_r \ge 0$ & Capacity holding cost in region $r$ \\
$c^{inv}_r \ge 0$ & Capacity investment cost in region $r$ \\
$\beta_t \in (0,1]$ & Discount factor for period $t$ \\
$y_t \ge 0$ & Duration (years) associated with period $t$ (e.g., 5.0) \\
$\rho^{imp}_r, \rho^{exp}_r \ge 0$ & Penalty weights on tariffs/taxes \\
$\kappa_r \ge 0$ & Penalty weight on offered capacity $Q^{offer}_{r,t}$ \\
$w_r \ge 0$ & Weight on consumer surplus in welfare objective \\
$\varepsilon_x \ge 0$ & Flow regularization parameter \\
$\varepsilon_{comp} \ge 0$ & Complementarity relaxation tolerance \\
$\rho_{prox} \ge 0$ & Proximal regularization weight (stabilization across Gauss--Seidel iterations) \\
$Q^{offer,last}_{r,t}$ & Last-iterate offered capacity \\
$\tau^{imp,last}_{ir,t}$ & Last-iterate import tariff \\
$\tau^{exp,last}_{ri,t}$ & Last-iterate export tax \\
\bottomrule
\end{tabular}
\end{flushleft}

%---------------------------------------------------
\subsection{Upper Level Problem (ULP) Variables}
%---------------------------------------------------

\begin{flushleft}
\begin{tabular}{@{}lp{10cm}@{}}
\toprule
\textbf{Symbol} & \textbf{Description} \\
\midrule
$Q^{offer}_{r,t} \ge 0$ & Offered production capacity in region $r$ at time $t$ \\
$\tau^{imp}_{ir,t} \in [0, \overline{\tau}^{imp}_{ir}]$ & Import tariff set by region $i$ on imports from $r$ at time $t$ \\
$\tau^{exp}_{ri,t} \in [0, \overline{\tau}^{exp}_{ri}]$ & Export tax set by region $r$ on exports to $i$ at time $t$ \\
$s_{r,t} \in [0, s^{ub}_r]$ & Production subsidy given by region $r$ at time $t$ \\
$K^{cap}_{r,t} \ge 0$ & Total available capacity in region $r$ at time $t$ \\
$I^{cap}_{r,t} \ge 0$ & Capacity investment in region $r$ at time $t$ \\
$D^{cap}_{r,t} \ge 0$ & Capacity decommissioning in region $r$ at time $t$ \\
\bottomrule
\end{tabular}
\end{flushleft}

\noindent\textit{Note:} Domestic tariffs are zero: $\tau^{imp}_{ii,t} = 0$ and $\tau^{exp}_{ii,t} = 0$ for all $i, t$.

%---------------------------------------------------
\subsection{Lower Level Problem (LLP) Variables}
%---------------------------------------------------

\paragraph{Primal Variables}

\begin{flushleft}
\begin{tabular}{@{}lp{10cm}@{}}
\toprule
\textbf{Symbol} & \textbf{Description} \\
\midrule
$x_{ri,t} \ge 0$ & Shipment from region $r$ to region $i$ at time $t$ \\
$x^{dem}_{i,t} \in [0, D^{max}_{i,t}]$ & Consumption (demand) in region $i$ at time $t$ \\
\bottomrule
\end{tabular}
\end{flushleft}

\paragraph{Dual Variables}

\begin{flushleft}
\begin{tabular}{@{}lp{10cm}@{}}
\toprule
\textbf{Symbol} & \textbf{Description} \\
\midrule
$\lambda_{i,t} \in \mathbb{R}$ & Dual of node balance constraint in region $i$ at time $t$ \\
$\mu_{r,t} \ge 0$ & Dual of exporter capacity constraint in region $r$ at time $t$ \\
$\gamma_{ri,t} \ge 0$ & Dual of non-negativity constraint $x_{ri,t} \ge 0$ \\
$\beta_{i,t} \ge 0$ & Dual of demand cap constraint $x^{dem}_{i,t} \le D^{max}_{i,t}$ \\
$\psi_{i,t} \ge 0$ & Dual of non-negativity constraint $x^{dem}_{i,t} \ge 0$ \\
\bottomrule
\end{tabular}
\end{flushleft}

%---------------------------------------------------
\subsection{Auxiliary Definitions}
%---------------------------------------------------

\begin{flushleft}
\textbf{Utility function:}
\end{flushleft}
\begin{equation}
U_{i,t}(x^{dem}_{i,t}) = a_{i,t} x^{dem}_{i,t} - \tfrac{1}{2} b_{i,t} (x^{dem}_{i,t})^2
\end{equation}

\begin{flushleft}
\textbf{Delivered cost wedge:}
\end{flushleft}
\begin{equation}
k_{ri,t} := c^{man}_r - s_{r,t} + c^{ship}_{ri} + \tau^{exp}_{ri,t} + \tau^{imp}_{ir,t}
\end{equation}

%===================================================
\section{Lower Level Problem (LLP)}
%===================================================

The Lower Level Problem represents the multi-period market clearing (system operator) problem:
\begin{equation}
\min_{x, x^{dem}} \quad \sum_{t \in T} \left[ \sum_{r,i \in R} k_{ri,t} \, x_{ri,t} + \frac{\varepsilon_x}{2} \sum_{r,i \in R} x_{ri,t}^2 - \sum_{i \in R} U_{i,t}(x^{dem}_{i,t}) \right]
\end{equation}

\noindent subject to:
\begin{align}
\sum_{r \in R} x_{ri,t} - x^{dem}_{i,t} &= 0 \quad (\lambda_{i,t}) && \forall i \in R, t \in T \\
Q^{offer}_{r,t} - \sum_{i \in R} x_{ri,t} &\ge 0 \quad (\mu_{r,t}) && \forall r \in R, t \in T \\
x_{ri,t} &\ge 0 \quad (\gamma_{ri,t}) && \forall r, i \in R, t \in T \\
D^{max}_{i,t} - x^{dem}_{i,t} &\ge 0 \quad (\beta_{i,t}) && \forall i \in R, t \in T \\
x^{dem}_{i,t} &\ge 0 \quad (\psi_{i,t}) && \forall i \in R, t \in T
\end{align}

%---------------------------------------------------
\subsection{KKT Stationarity Conditions}
%---------------------------------------------------

\begin{align}
k_{ri,t} + \varepsilon_x x_{ri,t} - \lambda_{i,t} + \mu_{r,t} - \gamma_{ri,t} &= 0 && \forall r, i \in R, t \in T \\
-(a_{i,t} - b_{i,t} x^{dem}_{i,t}) + \lambda_{i,t} + \beta_{i,t} - \psi_{i,t} &= 0 && \forall i \in R, t \in T
\end{align}

%---------------------------------------------------
\subsection{Complementarity Conditions (Relaxed)}
%---------------------------------------------------

\begin{align}
\mu_{r,t} \cdot \left( Q^{offer}_{r,t} - \sum_{i \in R} x_{ri,t} \right) &\le \varepsilon_{comp} && \forall r \in R, t \in T \\
\gamma_{ri,t} \cdot x_{ri,t} &\le \varepsilon_{comp} && \forall r, i \in R, t \in T \\
\beta_{i,t} \cdot (D^{max}_{i,t} - x^{dem}_{i,t}) &\le \varepsilon_{comp} && \forall i \in R, t \in T \\
\psi_{i,t} \cdot x^{dem}_{i,t} &\le \varepsilon_{comp} && \forall i \in R, t \in T
\end{align}

%===================================================
\section{Upper Level Problem (ULP)}
%===================================================

Each region $r$ solves a multi-period welfare maximization problem:
\begin{equation}
\max_{Q^{offer}_r, \tau^{imp}_{r \cdot}, \tau^{exp}_{r \cdot}, s_r, K^{cap}_r, I^{cap}_r, D^{cap}_r} \quad W_r
\end{equation}

\noindent where the welfare objective is:
\begin{align}
W_r = \; & \sum_{t \in T} \beta_t \Bigg\{ w_r \left[ U_{r,t}(x^{dem}) - \lambda_{r,t} x^{dem}_{r,t} \right] \nonumber \\
& + \underbrace{\sum_{j \in R} \tau^{imp}_{rj,t} x_{jr,t}}_{\text{tariff revenue}}
  + \underbrace{\sum_{j \in R} \tau^{exp}_{rj,t} x_{rj,t}}_{\text{export tax revenue}} \nonumber \\
& + \underbrace{\sum_{j \in R} \left( \lambda_{j,t} - c^{man}_r - c^{ship}_{rj} - \tau^{imp}_{jr,t} - \tau^{exp}_{rj,t} \right) x_{rj,t}}_{\text{producer surplus}} \nonumber \\
& - \underbrace{\sum_{j \in R} s_{r,t} x_{rj,t}}_{\text{subsidy cost}}
  - \underbrace{f^{hold}_r y_t K^{cap}_{r,t}}_{\text{holding cost}}
  - \underbrace{c^{inv}_r y_t I^{cap}_{r,t}}_{\text{investment cost}} \Bigg\} \nonumber \\
& + \sum_{t \in T} \text{Penalty}_{r,t}(\tau^{imp}, \tau^{exp}, Q^{offer}) + \sum_{t \in T} \text{Proximal}_{r,t}
\end{align}

%---------------------------------------------------
\subsection{Penalty Terms (Linear Mode, default)}
%---------------------------------------------------

When \texttt{use\_quad = False} (default):
\begin{align}
\text{Penalty}_{r,t}(\tau^{imp}, \tau^{exp}, Q^{offer}) = \;
& - \rho^{imp}_r \sum_{j \in R} \tau^{imp}_{rj,t}
  - \rho^{exp}_r \sum_{j \in R} \tau^{exp}_{rj,t}
  - \kappa_r Q^{offer}_{r,t}
\end{align}

%---------------------------------------------------
\subsection{Penalty Terms (Quadratic Mode)}
%---------------------------------------------------

When \texttt{use\_quad = True}:
\begin{align}
\text{Penalty}_{r,t}(\tau^{imp}, \tau^{exp}, Q^{offer}) = \;
& - \tfrac{1}{2} \rho^{imp}_r \sum_{j \in R} \left(\tau^{imp}_{rj,t}\right)^2
  - \tfrac{1}{2} \rho^{exp}_r \sum_{j \in R} \left(\tau^{exp}_{rj,t}\right)^2 \nonumber \\
& - \tfrac{1}{2} \kappa_r \left( Q^{offer}_{r,t} - \sum_{j \in R} x_{rj,t} \right)^2
\end{align}

%---------------------------------------------------
\subsection{Proximal Regularization}
%---------------------------------------------------

Applied in both modes to stabilize the Gauss--Seidel iterations:
\begin{align}
\text{Proximal}_{r,t} = \;
& - \tfrac{1}{2} \rho_{prox} \left( Q^{offer}_{r,t} - Q^{offer,last}_{r,t} \right)^2 \nonumber \\
& - \tfrac{1}{2} \rho_{prox} \sum_{j \in R} \left( \tau^{imp}_{rj,t} - \tau^{imp,last}_{rj,t} \right)^2 \nonumber \\
& - \tfrac{1}{2} \rho_{prox} \sum_{j \in R} \left( \tau^{exp}_{rj,t} - \tau^{exp,last}_{rj,t} \right)^2
\end{align}

%---------------------------------------------------
\subsection{ULP Constraints}
%---------------------------------------------------

\noindent subject to:
\begin{align}
0 &\le Q^{offer}_{r,t} \le K^{cap}_{r,t} && \forall t \in T \\
K^{cap}_{r,t+1} &= K^{cap}_{r,t} + I^{cap}_{r,t} - D^{cap}_{r,t} && \forall \text{ valid transitions } t \to t+1 \\
0 &\le \tau^{imp}_{re,t} \le \overline{\tau}^{imp}_{re} && \forall e \in R, t \in T \\
0 &\le \tau^{exp}_{ri,t} \le \overline{\tau}^{exp}_{ri} && \forall i \in R, t \in T \\
0 &\le s_{r,t} \le s^{ub}_r && \forall t \in T \\
& \text{LLP KKT conditions hold.} \nonumber
\end{align}

\noindent\textit{Note:} $K^{cap}_{r,t}$ at the initial period is fixed to $K^{cap,init}_r$. $I^{cap}_{r,t}$ and $D^{cap}_{r,t}$ are non-negative and fixed to zero at the terminal period.

%---------------------------------------------------
\subsection{Numerical Stabilization}
%---------------------------------------------------

\begin{flushleft}
\textbf{Implemented variable bounds:}
\end{flushleft}
\begin{align}
0 &\le \lambda_{i,t} \le \max_{t} (a_{i,t}) && \forall i \in R, t \in T \\
0 &\le \mu_{r,t} \le \mu^{ub}_r && \forall r \in R, t \in T \\
0 &\le \gamma_{ri,t} \le \gamma^{ub}_{ri} && \forall r, i \in R, t \in T \\
0 &\le \beta_{i,t} \le \max_{t} (a_{i,t}) && \forall i \in R, t \in T \\
0 &\le \psi_{i,t} \le \max_{t} (a_{i,t}) && \forall i \in R, t \in T
\end{align}

\noindent where $\mu^{ub}_r = \max_{i,t} \left( a_{i,t} - c^{man}_r - c^{ship}_{ri} \right)^+$ and $\gamma^{ub}_{ri} = c^{man}_r + c^{ship}_{ri} + \overline{\tau}^{imp}_{ir} + \overline{\tau}^{exp}_{ri} + \varepsilon_x K^{cap,init}_r + \mu^{ub}_r + s^{ub}_r$.
    """
    return TEMPLATE


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'outputs')
    overleaf_dir = os.path.join(base_dir, 'overleaf')
    os.makedirs(output_dir, exist_ok=True)

    content = get_static_content()

    out_file = os.path.join(output_dir, 'model_equations.tex')
    with open(out_file, 'w') as f:
        f.write(content)
    print(f"LaTeX source written to {out_file}")

    if os.path.exists(overleaf_dir):
        out_ol = os.path.join(overleaf_dir, 'model_equations.tex')
        with open(out_ol, 'w') as f:
            f.write(content)
        print(f"LaTeX source written to {out_ol}")

if __name__ == "__main__":
    main()
