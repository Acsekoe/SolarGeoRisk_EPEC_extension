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
$a_i > 0$ & Inverse demand intercept in region $i$: $P_i(x_i) = a_i - b_i x_i$ \\
$b_i > 0$ & Inverse demand slope in region $i$ \\
$D^{max}_i > 0$ & Demand/installation capacity cap in region $i$ \\
$c^{man}_r$ & Manufacturing cost in region $r$ \\
$c^{ship}_{ri}$ & Shipping cost from region $r$ to region $i$ \\
$Q^{cap}_r$ & Existing production capacity in region $r$ \\
$\overline{\tau}^{imp}_{ir} \ge 0$ & Upper bound on import tariff (set by $i$ on imports from $r$) \\
$\overline{\tau}^{exp}_{ri} \ge 0$ & Upper bound on export tax (set by $r$ on exports to $i$) \\
$\rho^{imp}_r, \rho^{exp}_r \ge 0$ & Penalty weights on tariffs/taxes (linear or quadratic, see below) \\
$\kappa_r \ge 0$ & Penalty on offered capacity $Q^{offer}_r$ (linear or quadratic) \\
$w_r \ge 0$ & Weight on consumer surplus in welfare objective \\
$\varepsilon_x \ge 0$ & Flow regularization parameter \\
$\varepsilon_{comp} \ge 0$ & Complementarity relaxation tolerance \\
$\rho_{prox} \ge 0$ & Proximal regularization weight (stabilization across Gauss--Seidel iterations) \\
$Q^{offer,last}_r$ & Last-iterate offered capacity (anchor for proximal term) \\
$\tau^{imp,last}_{ir}$ & Last-iterate import tariff (anchor for proximal term) \\
$\tau^{exp,last}_{ri}$ & Last-iterate export tax (anchor for proximal term) \\
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
$Q^{offer}_r \in [0, Q^{cap}_r]$ & Offered production capacity in region $r$ \\
$\tau^{imp}_{ir} \in [0, \overline{\tau}^{imp}_{ir}]$ & Import tariff set by region $i$ on imports from $r$ \\
$\tau^{exp}_{ri} \in [0, \overline{\tau}^{exp}_{ri}]$ & Export tax set by region $r$ on exports to $i$ \\
\bottomrule
\end{tabular}
\end{flushleft}

\noindent\textit{Note:} Domestic tariffs are zero: $\tau^{imp}_{ii} = 0$ and $\tau^{exp}_{ii} = 0$ for all $i$.

%---------------------------------------------------
\subsection{Lower Level Problem (LLP) Variables}
%---------------------------------------------------

\paragraph{Primal Variables}

\begin{flushleft}
\begin{tabular}{@{}lp{10cm}@{}}
\toprule
\textbf{Symbol} & \textbf{Description} \\
\midrule
$x_{ri} \ge 0$ & Shipment from region $r$ to region $i$ \\
$x^{dem}_i \in [0, D^{max}_i]$ & Consumption (demand) in region $i$ \\
\bottomrule
\end{tabular}
\end{flushleft}

\paragraph{Dual Variables}

\begin{flushleft}
\begin{tabular}{@{}lp{10cm}@{}}
\toprule
\textbf{Symbol} & \textbf{Description} \\
\midrule
$\lambda_i \in \mathbb{R}$ & Dual of node balance constraint in region $i$ \\
$\mu_r \ge 0$ & Dual of exporter capacity constraint in region $r$ \\
$\gamma_{ri} \ge 0$ & Dual of non-negativity constraint $x_{ri} \ge 0$ \\
$\beta_i \ge 0$ & Dual of demand cap constraint $x^{dem}_i \le D^{max}_i$ \\
$\psi_i \ge 0$ & Dual of non-negativity constraint $x^{dem}_i \ge 0$ \\
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
U_i(x) = a_i x^{dem}_i - \tfrac{1}{2} b_i (x^{dem}_i)^2
\end{equation}

\begin{flushleft}
\textbf{Delivered cost wedge:}
\end{flushleft}
\begin{equation}
k_{ri} := c^{man}_r + c^{ship}_{ri} + \tau^{exp}_{ri} + \tau^{imp}_{ir}
\end{equation}

%===================================================
\section{Lower Level Problem (LLP)}
%===================================================

The Lower Level Problem represents the market clearing (system operator) problem:
\begin{equation}
\min_{x, x^{dem}} \quad \sum_{r,i \in R} k_{ri} \, x_{ri} + \frac{\varepsilon_x}{2} \sum_{r,i \in R} x_{ri}^2 - \sum_{i \in R} U_i(x)
\end{equation}

\noindent subject to:
\begin{align}
\sum_{r \in R} x_{ri} - x^{dem}_i &= 0 \quad (\lambda_i) && \forall i \in R \\
Q^{offer}_r - \sum_{i \in R} x_{ri} &\ge 0 \quad (\mu_r) && \forall r \in R \\
x_{ri} &\ge 0 \quad (\gamma_{ri}) && \forall r, i \in R \\
D^{max}_i - x^{dem}_i &\ge 0 \quad (\beta_i) && \forall i \in R \\
x^{dem}_i &\ge 0 \quad (\psi_i) && \forall i \in R
\end{align}

%---------------------------------------------------
\subsection{KKT Stationarity Conditions}
%---------------------------------------------------

\begin{align}
k_{ri} + \varepsilon_x x_{ri} - \lambda_i + \mu_r - \gamma_{ri} &= 0 && \forall r, i \in R \\
-(a_i - b_i x^{dem}_i) + \lambda_i + \beta_i - \psi_i &= 0 && \forall i \in R
\end{align}

%---------------------------------------------------
\subsection{Complementarity Conditions (Relaxed)}
%---------------------------------------------------

\begin{align}
\mu_r \cdot \left( Q^{offer}_r - \sum_{i \in R} x_{ri} \right) &\le \varepsilon_{comp} && \forall r \in R \\
\gamma_{ri} \cdot x_{ri} &\le \varepsilon_{comp} && \forall r, i \in R \\
\beta_i \cdot (D^{max}_i - x^{dem}_i) &\le \varepsilon_{comp} && \forall i \in R \\
\psi_i \cdot x^{dem}_i &\le \varepsilon_{comp} && \forall i \in R
\end{align}

%===================================================
\section{Upper Level Problem (ULP)}
%===================================================

Each region $r$ solves a welfare maximization problem:
\begin{equation}
\max_{Q^{offer}_r, \tau^{imp}_{r \cdot}, \tau^{exp}_{r \cdot}} \quad W_r
\end{equation}

\noindent where the welfare objective is:
\begin{align}
W_r = \; & w_r \left[ U_r(x) - \lambda_r x^{dem}_r \right] \nonumber \\
& + \underbrace{\sum_{j \in R} \tau^{imp}_{rj} x_{jr}}_{\text{tariff revenue}}
  + \underbrace{\sum_{j \in R} \tau^{exp}_{rj} x_{rj}}_{\text{export tax revenue}} \nonumber \\
& + \underbrace{\sum_{j \in R} \left( \lambda_j - c^{man}_r - c^{ship}_{rj} - \tau^{imp}_{jr} - \tau^{exp}_{rj} \right) x_{rj}}_{\text{producer surplus}} \nonumber \\
& + \text{Penalty}(\tau^{imp}, \tau^{exp}, Q^{offer}) + \text{Proximal}(r)
\end{align}

%---------------------------------------------------
\subsection{Penalty Terms (Linear Mode, default)}
%---------------------------------------------------

When \texttt{use\_quad = False} (default):
\begin{align}
\text{Penalty}(\tau^{imp}, \tau^{exp}, Q^{offer}) = \;
& - \rho^{imp}_r \sum_{j \in R} \tau^{imp}_{rj}
  - \rho^{exp}_r \sum_{j \in R} \tau^{exp}_{rj}
  - \kappa_r Q^{offer}_r
\end{align}

%---------------------------------------------------
\subsection{Penalty Terms (Quadratic Mode)}
%---------------------------------------------------

When \texttt{use\_quad = True}:
\begin{align}
\text{Penalty}(\tau^{imp}, \tau^{exp}, Q^{offer}) = \;
& - \tfrac{1}{2} \rho^{imp}_r \sum_{j \in R} \left(\tau^{imp}_{rj}\right)^2
  - \tfrac{1}{2} \rho^{exp}_r \sum_{j \in R} \left(\tau^{exp}_{rj}\right)^2 \nonumber \\
& - \tfrac{1}{2} \kappa_r \left( Q^{offer}_r - \sum_{j \in R} x_{rj} \right)^2
\end{align}

%---------------------------------------------------
\subsection{Proximal Regularization}
%---------------------------------------------------

Applied in both modes to stabilize the Gauss--Seidel iterations:
\begin{align}
\text{Proximal}(r) = \;
& - \tfrac{1}{2} \rho_{prox} \left( Q^{offer}_r - Q^{offer,last}_r \right)^2 \nonumber \\
& - \tfrac{1}{2} \rho_{prox} \sum_{j \in R} \left( \tau^{imp}_{rj} - \tau^{imp,last}_{rj} \right)^2 \nonumber \\
& - \tfrac{1}{2} \rho_{prox} \sum_{j \in R} \left( \tau^{exp}_{rj} - \tau^{exp,last}_{rj} \right)^2
\end{align}

%---------------------------------------------------
\subsection{ULP Constraints}
%---------------------------------------------------

\noindent subject to:
\begin{align}
0 &\le Q^{offer}_r \le Q^{cap}_r \\
0 &\le \tau^{imp}_{re} \le \overline{\tau}^{imp}_{re} \\
0 &\le \tau^{exp}_{ri} \le \overline{\tau}^{exp}_{ri} \\
& \text{LLP KKT conditions hold.} \nonumber
\end{align}

%---------------------------------------------------
\subsection{Numerical Stabilization}
%---------------------------------------------------

\begin{flushleft}
\textbf{Implemented variable bounds:}
\end{flushleft}
\begin{align}
0 &\le \lambda_i \le a_i && \forall i \in R \\
0 &\le \mu_r \le \mu^{ub}_r && \forall r \in R \\
0 &\le \gamma_{ri} \le \gamma^{ub}_{ri} && \forall r, i \in R \\
0 &\le \beta_i \le a_i && \forall i \in R \\
0 &\le \psi_i \le a_i && \forall i \in R
\end{align}

\noindent where $\mu^{ub}_r = \max_{i} \left( a_i - c^{man}_r - c^{ship}_{ri} \right)^+$ and $\gamma^{ub}_{ri} = c^{man}_r + c^{ship}_{ri} + \overline{\tau}^{imp}_{ir} + \overline{\tau}^{exp}_{ri} + \varepsilon_x Q^{cap}_r + \mu^{ub}_r$.
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
