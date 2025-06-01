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

### Data Preparation

WIP

## Constraint Programming Spec

### Depot VRP for Amazon Regular Orders

#### Decision Variables

- $x_{i, j, k} \in \\{0, 1\\}$: truck $k$ travels directly from stop $i$ to stop $j$.
- $y_{o, k} \in \\{0, 1\\}$: order $o$ is served by truck $k$.
- $u_{i, k} \in \\{0, \dots, N+M-1\\}$: "timepoint" when truck $k$ arrives at stop $i$.
- $z_{k} \in \\{0, 1\\}$: truck $k$ is used.
- $\text{vis}_{j, k} \in \\{0, 1\\}$: (auxiliary variable) stop $j$ is visited by vehicle $k$.

#### Parameters

- $t_{i, j}$: travel time from stop $i$ to stop $j$.
- $\hat{t}_j$: stopover time at stop $j$.
- $d_{i, j}$: distance from stop $i$ to stop $j$.
- $w_o, v_o$: weight and volume of package $o$.
- $W_k, V_k$: weight and volume capacity of vehicle $k$.
- $P_o$: priority score for order $o$, calculated by the order waiting time.
- $D_k$: maximum cruising range of vehicle $k$.
- $T$: maximum driver duty time.
- $K$: total number of available trucks.
- $N$: total number of distinct addresses in the order list.
- $M$: total number of depots.
- $O$: total number of orders in the order.
- $s(o)$: stop index of order $o$.
- $d(k)$: depot index assigned to vehicle $k$.

#### Objective and constraints

```math
\begin{aligned}
    \text{minimize} &\space \sum_{k=1}^{K}\sum_{i=0}^{N+M-1}\sum_{j=0}^{N+M-1} t_{i, j}x_{i, j, k}
                        -\alpha \sum_{k=1}^K\sum_{o=1}^{O} P_oy_{o, k}
                        +\beta \sum_{k=1}^K z_k & \cr
    \text{s.t.}     &\space \sum_{j=M}^{N+M-1} x_{d(k), j, k} = z_k
                        =\sum_{i=M}^{N+M-1} x_{i, d(k), k},
                        \quad \forall k & \text{(1a)} \cr
                    &\space x_{m,j,k} = x_{i,m,k} = 0,
                        \quad \forall k, i, j, m \ne d(k) & \text{(1b)} \cr
                    &\space \sum_{i=0,i\ne s(o)+M-1}^{N+M-1} x_{i, s(o)+M-1,k} \ge y_{o, k},
                        \quad \forall o, k & \text{(2a)} \cr
                    &\space \sum_{i=0,i\ne h}^{N+M-1} x_{i, h, k} =\sum_{j=0,j\ne h}^{N+M-1} x_{h, j, k},
                        \quad \forall k, h\in \{M,\dots,N+M-1\} & \text{(2b)} \cr
                    &\space \sum_{o=1}^{O} w_o y_{o,k} \le W_k, \space \sum_{o=1}^{O} v_o y_{o,k} \le V_k,
                        \quad \forall k & \text{(3)} \cr
                    &\space z_k=\max \{y_{o, k}: o \in \{1, \dots, O\}\},
                        \quad \forall k & \text{(4)} \cr
                    &\space u_{i,k}+1 \le u_{j,k} + (N + M) \cdot (1 - x_{i, j, k}),
                        \quad \forall i, j \in \{M,\dots, N+M-1\},i \ne j,k & \text{(5)} \cr
                    &\space \sum_{i=0}^{N+M-1} \sum_{j=0, j\ne i}^{N+M-1} t_{i, j}x_{i, j, k}+\sum_{j=M}^{N+M-1} \hat{t}_j
                        \cdot \text{vis}_{j,k} \le T,
                        \quad \forall k & \text{(6a)} \cr
                    &\space \text{vis}_{j,k} \ge y_{o,k},
                        \quad \forall k,o:s(o)=j-M+1 & \text{(6b)} \cr
                    &\space \text{vis}_{j,k} \le \sum_{o:s(o)=j-M+1} y_{o,k},
                        \quad \forall j\in \{M,\dots,N+M-1\},k & \text{(6c)} \cr
                    &\space \sum_{i=0}^{N+M-1} \sum_{j=0, j\ne i}^{N+M-1} d_{i,j}x_{i,j,k}\le D_k,
                        \quad \forall k & \text{(7)} \cr
                    &\space \sum_{k=1}^K y_{o,k}=1,
                        \quad \forall o & \text{(8)} \cr
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
   See [Wikipedia page](https://en.wikipedia.org/wiki/Travelling_salesman_problem#Miller%E2%80%93Tucker%E2%80%93Zemlin_formulation).
6. Time Constraints:\
   (6a). The total travel time + stopover time does not exceeds the driver duty time $T$.\
   (6b-6c). A stop $s(o)$ is visited iff some package corresponding to that stop is assigned to the vehicle $k$.
7. Distance Constraint:\
   The total traveling distance does not exceed the vehicle's cruising range $D_k$.
8. Package Assignment Constraint:\
   Each package is assigned to exactly one vehicle.

### Depot VRP for Amazon Same-Day Orders

> Same-Day orders including Amazon Fresh, Prime Now

WIP

## References

- WIP
