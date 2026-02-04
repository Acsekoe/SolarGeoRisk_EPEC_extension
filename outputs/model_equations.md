# Model Equations

## Upper Level Problem (ULP)

### Variables
$$ \Q\_offer \geq 0 \quad \forall  \in [R] $$
$$ \\tau_{imp} \geq 0 \quad \forall  \in [imp, exp] $$
$$ \\tau_{exp} \geq 0 \quad \forall  \in [exp, imp] $$

### Objectives
**Objective (Welfare)**:
$$ w_{r} \cdot  cons\_surplus + imp\_tariff\_rev + exp\_tax\_rev + producer\_term + pen\_imp\_quad + pen\_exp\_quad + pen\_q\_quad $$

**Objective (Welfare)**:
$$ w_{r} \cdot  cons\_surplus + imp\_tariff\_rev + exp\_tax\_rev + producer\_term + pen\_imp\_lin + pen\_exp\_lin + pen\_q\_lin $$


## Lower Level Problem (LLP)

### Variables
$$ \x \geq 0 \quad \forall  \in [exp, imp] $$
$$ \x\_dem \geq 0 \quad \forall  \in [R] $$
$$ \\mu \geq 0 \quad \forall  \in [R] $$
$$ \\gamma \geq 0 \quad \forall  \in [exp, imp] $$
$$ \\beta_{dem} \geq 0 \quad \forall  \in [R] $$
$$ \\psi_{dem} \geq 0 \quad \forall  \in [R] $$

### Equations
**eq obj llp** (, \forall Ellipsis):
$$ z_{LLP} = llp\_total\_cost - llp\_gross\_surplus $$

**eq bal** (, \forall imp):
$$ \sum_{exp} (x_{exp,imp}) - x\_dem_{imp} = 0 $$

**eq cap** (, \forall exp):
$$ Q\_offer_{exp} - \sum_{imp} (x_{exp,imp}) \geq 0 $$

**eq stat x** (, \forall exp, imp):
$$ c\_man_{exp} + c\_ship_{exp,imp} + \tau_{exp}_{exp,imp} + \tau_{imp}_{imp,exp} + \eps_{x} \cdot  x_{exp,imp} - \lam_{imp} + \mu_{exp} - \gamma_{exp,imp} = 0 $$

**eq stat dem** (, \forall imp):
$$ -a\_dem_{imp} - b\_dem_{imp} \cdot  x\_dem_{imp} + \lam_{imp} + \beta_{dem}_{imp} - \psi_{dem}_{imp} = 0 $$

**eq comp mu** (, \forall exp):
$$ \mu_{exp} \cdot  (Q\_offer_{exp} - \sum_{imp} (x_{exp,imp})) = 0 $$

**eq comp mu** (, \forall exp):
$$ \mu_{exp} \cdot  (Q\_offer_{exp} - \sum_{imp} (x_{exp,imp})) \leq \eps_{value} $$

**eq comp gamma** (, \forall exp, imp):
$$ \gamma_{exp,imp} \cdot  x_{exp,imp} = 0 $$

**eq comp gamma** (, \forall exp, imp):
$$ \gamma_{exp,imp} \cdot  x_{exp,imp} \leq \eps_{value} $$

**eq comp beta dem** (, \forall imp):
$$ \beta_{dem}_{imp} \cdot  (Dmax_{imp} - x\_dem_{imp}) = 0 $$

**eq comp beta dem** (, \forall imp):
$$ \beta_{dem}_{imp} \cdot  (Dmax_{imp} - x\_dem_{imp}) \leq \eps_{value} $$

**eq comp psi dem** (, \forall imp):
$$ \psi_{dem}_{imp} \cdot  x\_dem_{imp} = 0 $$

**eq comp psi dem** (, \forall imp):
$$ \psi_{dem}_{imp} \cdot  x\_dem_{imp} \leq \eps_{value} $$


## Numerical Stabilization

The following penalty terms are used in the ULP objective or LLP regularization:

**pen imp quad**:
$$ -0.5 \cdot  \rho_{imp}_{r} \cdot  \sum_{j} (\tau_{imp}_{r,j} \cdot  \tau_{imp}_{r,j}) $$

**pen exp quad**:
$$ -0.5 \cdot  \rho_{exp}_{r} \cdot  \sum_{j} (\tau_{exp}_{r,j} \cdot  \tau_{exp}_{r,j}) $$

**pen q quad**:
$$ -0.5 \cdot  kappa\_Q_{r} \cdot  Q\_offer_{r} \cdot  Q\_offer_{r} $$

**pen imp lin**:
$$ -\rho_{imp}_{r} \cdot  \sum_{j} (\tau_{imp}_{r,j}) $$

**pen exp lin**:
$$ -\rho_{exp}_{r} \cdot  \sum_{j} (\tau_{exp}_{r,j}) $$

**pen q lin**:
$$ -kappa\_Q_{r} \cdot  Q\_offer_{r} $$

