# AMaZoN OPTIMization for local delivery

> This is a Spring 25 Math 381 Course Project of University of Washington.

## User Guide

### (Recommended) Google Colab Demo

WIP

### Local Installation

1. Install `git` and optain AMZNOptim repository

```bash
# Use HTTPS
git clone https://github.com/KagurazakaHinagi/AMZNOptim_MAT381.git
# or SSH
git clone git@github.com:KagurazakaHinagi/AMZNOptim_MAT381.git
```

2. (Recommended) Set up Virtual Environment

```bash
# Use Conda
conda create -n amznoptim python=3.12
conda activate amznoptim
# or Python venv
python3 -m venv .venv
source .venv/bin/activate
```

3. Install `amznoptim` package

```bash
pip install .
# For development, use
pip install -e '.[dev]'
```

4. For example usage, please refer to [example folder](example)

### Google Maps Platform

To run this project, you need to have your own Google Maps Platform account that enables the following APIs:

- Routes API
- Address Validation API
- Maps Static API

You can acquire an account with monthly free trials at [here](https://developers.google.com/maps).

### Data Preparation and Intepretation

Please refer to [docs](docs).

## Mixed-Integer Programming Spec

#### Sets and Indices

- $D$: set of depots, $|D| = n_d$.
- $S$: set of customer stops, $|S| = n_s$.
- $N = D \cup S$: set of all nodes, $|N|=n=n_d+n_s$.
- $K$: set of vehicles, $|K| = m$.
- $O$: set of packages/orders, $|O| = p$

#### Decision Variables

- $x_{i, j, k} \in \\{0, 1\\}$: truck $k$ travels directly from node $i$ to node $j$.
- $y_{o, k} \in \\{0, 1\\}$: order $o$ is served by vehicle $k$.
- $z_{k} \in \\{0, 1\\}$: vehicle $k$ is used.
- $u_{i, k} \ge 0$: arrival time at node $i$ by vehicle $k$.
- $s_{i, k} \in $\\{0, n-1\\}$: visit order of node $i$ by vehicle $k$.

#### Parameters

- $t_{i, j}$: travel time from stop $i$ to stop $j$.
- $\hat{t}_j$: stopover/service time at stop $j$.
- $d_{i, j}$: distance from stop $i$ to stop $j$.
- $w_o, v_o$: weight and volume of package $o$.
- $W_k, V_k$: weight and volume capacity of vehicle $k$.
- $P_o$: priority score for order $o$, calculated by the order waiting time.
- $R_k$: maximum cruising range of vehicle $k$.
- $T$: maximum driver duty time.
- $s(o)$: stop index of order $o$.
- $d(k)$: depot index assigned to vehicle $k$.
- **Hyperparameters**:
  - $\alpha$: penalty weight of priority score.
  - $\beta$: penalty weight of extra vehicle usage.

#### Objective and constraints

```math
\begin{aligned}
    \text{minimize} &\space \sum_{k\in K}\sum_{i\in N}\sum_{j\in N} t_{i, j}x_{i, j, k}
                        -\alpha \sum_{k\in K}\sum_{o\in O} P_oy_{o, k}
                        +\beta \sum_{k\in K} z_k & \cr
    \text{s.t.}     &\space \sum_{j\in S} x_{d(k), j, k}
                        =\sum_{i \in S} x_{i, d(k), k} = z_k,
                        \quad \forall k\in K
                        & \text{(1a)} \cr
                    &\space x_{m,j,k} = x_{i,m,k} = 0,
                        \quad \forall k\in K,
                        \quad \forall i,j \in N, i \ne j,
                        \quad \forall m\in D \setminus \{d(k)\}
                        & \text{(1b)} \cr
                    &\space \sum_{i\in N,i\ne s(o)+n_d} x_{i, s(o)+n_d,k} \ge y_{o, k},
                        \quad \forall o\in O,
                        \quad \forall k\in K
                        & \text{(2a)} \cr
                    &\space \sum_{i\in N,i\ne h} x_{i, h, k} =\sum_{j\in N,j\ne h} x_{h, j, k},
                        \quad \forall k\in K,
                        \quad \forall h\in S
                        & \text{(2b)} \cr
                    &\space \sum_{o\in O} w_o y_{o,k} \le W_k,
                        \space \sum_{o=1}^{O} v_o y_{o,k} \le V_k,
                        \quad \forall k\in K
                        & \text{(3)} \cr
                    &\space z_k=\max \{y_{o, k}: o\in O\},
                        \quad \forall k\in K
                        & \text{(4)} \cr
                    &\space s_{d(k), k} = 0,
                        \quad \forall k\in K
                        & \text{(5a)} \cr
                    &\space s_{i,k} - s_{j,k} + n \cdot x_{i,j,k} \le n - 1,
                        \quad \forall k\in K,
                        \quad \forall i,j \in S, i \ne j
                        & \text{(5b)} \cr
                    &\space u_{d(k), k} = 0,
                        \quad \forall k\in K
                        & \text{(6a)} \cr
                    &\space u_{j,k} \ge u_{i,k} + \hat{t}(i) + t_{i,j}
                        - T\cdot (1 - x_{i,j,k}),
                        \quad \forall k\in K,
                        \quad \forall i,j \in S, i\ne j
                        & \text{(6b)} \cr
                    &\space u_{i,k} \le T,
                        \quad \forall k\in K,
                        \quad \forall i\in N
                        & \text{(6c)} \cr
                    &\space \sum_{i\in N} \sum_{j\in N, j\ne i}d_{i,j}x_{i,j,k} \le R_k,
                        \quad \forall k\in K
                        & \text{(7)} \cr
                    &\space \sum_{k\in K} y_{o,k} = 1,
                        \quad \forall o\in O
                        & \text{(8)}
\end{aligned}
```

##### Explanation of constraints

1. Depot Assignment:\
   (1a). The vehicle $k$ can only depart from and return to its assigned depot $d(k)$ when it's been chosen to use.\
   (1b). The vehicle $k$ do not commute to other depots except its assigned one within its planned route.
2. Package Assignment and Routing:\
   (2a). There's an incoming route to the stop $s(o)$ when order $o$ is assigned to be delivered by vehicle $k$.\
   (2b). Flow conservation: For each stop served by vehicle $k$, there's exactly one incoming route and one outgoing route.
3. Vehicle Capacity:\
   Each vehicle $k$ cannot load packages in such a way that the total exceeds the maximum capacity of weight or volume of that vehicle $k$.
4. Vehicle Usage:\
   The vehicle $k$ would be in use if there's any packages assigned to $k$.
5. Subtour Elimination (Miller-Tucker-Zemlin):\
   Prevent disconnected loops by constraining on the order of the visiting time.\
   See [Wikipedia page](https://en.wikipedia.org/wiki/Travelling_salesman_problem#Miller%E2%80%93Tucker%E2%80%93Zemlin_formulation).\
   (5a). The visit starts at the depot (order = $0$).\
   (5b). For each other node, the visit order must be greater or equal to $1$.
6. Time Progression:\
   (6a). The depot arrival time is $0 \space \text{s}$ after departure.\
   (6b). If vehicle $k$ travels directly from node $i$ to node $j$, then the arrival time at node $j$ must be at least the arrival time of node $i$, plus the stopover time at node $i$, plus the travel time from $i$ to $j$.\
   (6c). The arrival time of each node could not exceed the driver's maximum duty time.
7. Distance Constraint:\
   The total traveling distance could not exceed the vehicle $k$'s cruising range $R_k$.
8. Package Assignment Constraint:\
   Each package is assigned to exactly one vehicle.

### Depot VRP for Amazon Regular Orders

#### Priority Score $P_o$

For Amazon Regular delivery services, the priority score $P_o$ defined in the objective function is the order $o$'s waiting time, calculated by ($\text{departure time} - \text{order placing time}$) and with units in $\text{ns}$.

### Depot VRP for Amazon Same-Day Orders

> Same-Day orders including Amazon Fresh, Prime Now

Besides regular constraints, the same-day delivery services has a user-specified delivery window (either $1\text{h}$ or $2\text{h}$)

#### Additional Parameters

- $e_o$: Earliest delivery time for order $o$.
- $l_o$: Latest delivery time for order $o$.

#### Additional Constraints

```math
\begin{aligned}
    &\space l_o - T(1 - y_{o, k}) \ge u_{s(o)+n_d,k}
        \ge e_o - T(1 - y_{o,k}),
        \quad \forall o\in O,
        \quad \forall k\in K
        & \text{(9)}
\end{aligned}
```

##### Explanation of constraint

9. Ensure that the vehicle arrives within the delivery window.

#### Priority Score $P_o$

- **Amazon Fresh**: the priority score $P_o$ is defined by a boolean value that indicates whether order $o$ contain perishable items or not.
- **Prime Now**: the priority score is set to $0$ exclusively. (not in use)

## References

- WIP
