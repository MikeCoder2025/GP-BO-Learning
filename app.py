# ============================================================
# GP-BO LEARNING LAB
# Simple educational software for:
# 1. Gaussian Processes
# 2. Bayesian Optimisation
# 3. Pareto / Multi-objective optimisation
# 4. Pharmaceutical continuous-flow optimisation
# ============================================================

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from scipy.stats import norm, qmc

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    ConstantKernel,
    WhiteKernel
)
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="GP-BO Learning Lab",
    page_icon="📈",
    layout="wide"
)

st.title("GP-BO Learning Lab")

st.write(
    """
    A simple interactive tool for learning **Gaussian Processes**
    and **Bayesian Optimisation**.
    """
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def hidden_1d_function(x):
    """
    Hidden objective used in the 1D teaching example.
    Students do not need to know this equation.
    """

    x = np.asarray(x)

    return (
        2.0 * np.sin(x)
        + 0.25 * x
        + 0.5 * np.cos(2 * x)
    )


# ============================================================
# PHARMACEUTICAL VIRTUAL REACTOR
# ============================================================

def run_pharma_experiment(
    X,
    noise=True,
    random_state=None
):

    X = np.atleast_2d(X)

    temperature = X[:, 0]
    residence_time = X[:, 1]
    equivalence = X[:, 2]

    # Synthetic product yield
    yield_true = (
        55
        + 34 * np.exp(
            -((temperature - 103) / 20) ** 2
            -((residence_time - 13) / 6) ** 2
            -((equivalence - 1.45) / 0.35) ** 2
        )
        + 6 * np.exp(
            -((temperature - 116) / 13) ** 2
            -((residence_time - 8) / 5) ** 2
        )
    )

    # Synthetic impurity
    impurity_true = (
        0.6
        + 0.018 * (temperature - 80)
        + 0.035 * np.maximum(
            residence_time - 10,
            0
        )
        + 1.6 * (equivalence - 1.35) ** 2
        + 1.8 * np.exp(
            -((temperature - 118) / 10) ** 2
        )
    )

    impurity_true = np.maximum(
        impurity_true,
        0.05
    )

    if noise:

        rng = np.random.default_rng(
            random_state
        )

        yield_true = (
            yield_true
            + rng.normal(
                0,
                0.8,
                size=len(X)
            )
        )

        impurity_true = (
            impurity_true
            + rng.normal(
                0,
                0.08,
                size=len(X)
            )
        )

    return (
        yield_true,
        impurity_true
    )


# ============================================================
# CREATE GP KERNEL
# ============================================================

def create_kernel(
    kernel_name,
    length_scale,
    noise
):

    if kernel_name == "RBF":

        base_kernel = RBF(
            length_scale=length_scale
        )

    else:

        base_kernel = Matern(
            length_scale=length_scale,
            nu=2.5
        )

    kernel = (
        ConstantKernel(1.0)
        * base_kernel
        + WhiteKernel(
            noise_level=noise
        )
    )

    return kernel


# ============================================================
# EXPECTED IMPROVEMENT
# ============================================================

def expected_improvement(
    mean,
    std,
    best,
    xi=0.01
):

    std = np.maximum(
        std,
        1e-12
    )

    improvement = (
        mean
        - best
        - xi
    )

    Z = improvement / std

    EI = (
        improvement * norm.cdf(Z)
        + std * norm.pdf(Z)
    )

    return np.maximum(
        EI,
        0
    )


# ============================================================
# PARETO FRONT
# ============================================================

def find_pareto_front(
    yield_values,
    impurity_values
):

    n = len(yield_values)

    pareto = np.ones(
        n,
        dtype=bool
    )

    for i in range(n):

        for j in range(n):

            if i == j:
                continue

            better_or_equal = (
                yield_values[j] >= yield_values[i]
                and
                impurity_values[j] <= impurity_values[i]
            )

            strictly_better = (
                yield_values[j] > yield_values[i]
                or
                impurity_values[j] < impurity_values[i]
            )

            if (
                better_or_equal
                and
                strictly_better
            ):

                pareto[i] = False
                break

    return pareto


# ============================================================
# INITIAL PHARMA DATA
# ============================================================

def create_initial_pharma_data():

    bounds = np.array([
        [60, 120],
        [2, 20],
        [1.0, 2.0]
    ])

    sampler = qmc.LatinHypercube(
        d=3,
        seed=42
    )

    X_unit = sampler.random(
        n=12
    )

    X = qmc.scale(
        X_unit,
        bounds[:, 0],
        bounds[:, 1]
    )

    y, impurity = run_pharma_experiment(
        X,
        noise=True,
        random_state=42
    )

    data = pd.DataFrame(
        X,
        columns=[
            "Temperature",
            "Residence_Time",
            "Equivalence"
        ]
    )

    data["Yield"] = y
    data["Impurity"] = impurity

    data["Experiment"] = np.arange(
        1,
        13
    )

    return data


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "1. GP Lab",
    "2. BO Lab",
    "3. Pareto Lab",
    "4. Pharmaceutical Demo"
])


# ============================================================
# TAB 1
# GAUSSIAN PROCESS LAB
# ============================================================

with tab1:

    st.header(
        "Gaussian Process Lab"
    )

    st.write(
        """
        Explore how the **kernel, length scale,
        noise and number of observations**
        affect a Gaussian Process.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        kernel_choice = st.selectbox(
            "Kernel",
            [
                "Matern",
                "RBF"
            ]
        )

    with col2:

        length_scale = st.slider(
            "Length scale",
            0.1,
            5.0,
            1.0,
            0.1
        )

    with col3:

        noise_level = st.slider(
            "Noise level",
            0.001,
            0.5,
            0.05,
            0.01
        )

    n_points = st.slider(
        "Number of training experiments",
        3,
        15,
        5
    )

    # Training data
    X_train = np.linspace(
        0.5,
        9.5,
        n_points
    )

    rng = np.random.default_rng(
        42
    )

    y_train = (
        hidden_1d_function(
            X_train
        )
        + rng.normal(
            0,
            0.15,
            n_points
        )
    )

    X_train_2d = X_train.reshape(
        -1,
        1
    )

    # GP
    kernel = create_kernel(
        kernel_choice,
        length_scale,
        noise_level
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        random_state=42
    )

    gp.fit(
        X_train_2d,
        y_train
    )

    # Prediction
    X_test = np.linspace(
        0,
        10,
        300
    ).reshape(-1, 1)

    mean, std = gp.predict(
    X_test,
    return_std=True
)

    true_y = hidden_1d_function(
        X_test[:, 0]
    )

    # Plot
    fig = go.Figure()

    # uncertainty upper
    fig.add_trace(
        go.Scatter(
            x=X_test[:, 0],
            y=mean + 1.96 * std,
            mode="lines",
            line=dict(width=0),
            showlegend=False
        )
    )

    # uncertainty lower
    fig.add_trace(
        go.Scatter(
            x=X_test[:, 0],
            y=mean - 1.96 * std,
            fill="tonexty",
            mode="lines",
            line=dict(width=0),
            name="95% predictive interval"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=X_test[:, 0],
            y=mean,
            mode="lines",
            name="GP mean"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=X_train,
            y=y_train,
            mode="markers",
            marker=dict(size=10),
            name="Experiments"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=X_test[:, 0],
            y=true_y,
            mode="lines",
            line=dict(dash="dash"),
            name="Hidden true function"
        )
    )

    fig.update_layout(
        title="Gaussian Process Posterior",
        xaxis_title="Input x",
        yaxis_title="Output y",
        height=550
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        """
        **Try this:** increase the length scale,
        change Matern to RBF, or reduce the
        number of experiments. Observe what
        happens to the GP mean and uncertainty.
        """
    )


# ============================================================
# TAB 2
# BAYESIAN OPTIMISATION LAB
# ============================================================

with tab2:

    st.header(
        "Bayesian Optimisation Lab"
    )

    st.write(
        """
        Bayesian optimisation uses the GP prediction
        and uncertainty to decide **where to perform
        the next experiment**.
        """
    )

    if "bo_x" not in st.session_state:

        st.session_state.bo_x = np.array([
            1.0,
            4.0,
            8.5
        ])

        st.session_state.bo_y = (
            hidden_1d_function(
                st.session_state.bo_x
            )
        )

    X_bo = st.session_state.bo_x
    y_bo = st.session_state.bo_y

    # Fit GP
    kernel = (
        ConstantKernel(1.0)
        * Matern(
            length_scale=1.0,
            nu=2.5
        )
        + WhiteKernel(
            noise_level=0.01
        )
    )

    gp_bo = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        random_state=42
    )

    gp_bo.fit(
        X_bo.reshape(-1, 1),
        y_bo
    )

    grid = np.linspace(
        0,
        10,
        500
    ).reshape(-1, 1)

    mean_bo, std_bo = gp_bo.predict(
        grid,
        return_std=True
    )

    best = np.max(
        y_bo
    )

    EI = expected_improvement(
        mean_bo,
        std_bo,
        best
    )

    next_index = np.argmax(
        EI
    )

    next_x = grid[
        next_index,
        0
    ]

    # --------------------------------------------------------
    # GP plot
    # --------------------------------------------------------

    fig_gp = go.Figure()

    fig_gp.add_trace(
        go.Scatter(
            x=grid[:, 0],
            y=mean_bo,
            mode="lines",
            name="GP mean"
        )
    )

    fig_gp.add_trace(
        go.Scatter(
            x=X_bo,
            y=y_bo,
            mode="markers",
            marker=dict(size=11),
            name="Experiments"
        )
    )

    fig_gp.add_vline(
        x=next_x,
        line_dash="dash",
        annotation_text="Next experiment"
    )

    fig_gp.update_layout(
        title="Current Gaussian Process",
        xaxis_title="x",
        yaxis_title="Objective"
    )

    st.plotly_chart(
        fig_gp,
        use_container_width=True
    )

    # --------------------------------------------------------
    # EI plot
    # --------------------------------------------------------

    fig_ei = go.Figure()

    fig_ei.add_trace(
        go.Scatter(
            x=grid[:, 0],
            y=EI,
            mode="lines",
            name="Expected Improvement"
        )
    )

    fig_ei.add_vline(
        x=next_x,
        line_dash="dash"
    )

    fig_ei.update_layout(
        title="Expected Improvement",
        xaxis_title="Candidate x",
        yaxis_title="EI"
    )

    st.plotly_chart(
        fig_ei,
        use_container_width=True
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Recommended x",
        f"{next_x:.2f}"
    )

    col2.metric(
        "Predicted value",
        f"{mean_bo[next_index]:.2f}"
    )

    col3.metric(
        "GP uncertainty",
        f"± {std_bo[next_index]:.2f}"
    )

    if st.button(
        "Run Next Experiment",
        key="run_bo"
    ):

        new_y = hidden_1d_function(
            next_x
        )

        st.session_state.bo_x = np.append(
            X_bo,
            next_x
        )

        st.session_state.bo_y = np.append(
            y_bo,
            new_y
        )

        st.rerun()

    if st.button(
        "Reset BO",
        key="reset_bo"
    ):

        del st.session_state.bo_x
        del st.session_state.bo_y

        st.rerun()


# ============================================================
# TAB 3
# PARETO / MULTI-OBJECTIVE LAB
# ============================================================

with tab3:

    st.header(
        "Pareto & Multi-Objective Lab"
    )

    st.write(
        """
        Here we consider two objectives:

        **Maximise Yield**

        **Minimise Impurity**
        """
    )

    if "pareto_data" not in st.session_state:

        st.session_state.pareto_data = (
            create_initial_pharma_data()
        )

    pdata = st.session_state.pareto_data.copy()

    pareto_mask = find_pareto_front(
        pdata["Yield"].values,
        pdata["Impurity"].values
    )

    pdata["Pareto"] = pareto_mask

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=pdata["Impurity"],
            y=pdata["Yield"],
            mode="markers+text",
            text=pdata["Experiment"],
            textposition="middle center",
            name="Experiments",
            marker=dict(size=22)
        )
    )

    pareto_data = pdata[
        pdata["Pareto"]
    ]

    fig.add_trace(
        go.Scatter(
            x=pareto_data["Impurity"],
            y=pareto_data["Yield"],
            mode="markers",
            marker=dict(
                size=30,
                symbol="star"
            ),
            name="Pareto optimal"
        )
    )

    fig.add_vline(
        x=2.0,
        line_dash="dash",
        annotation_text="Impurity = 2%"
    )

    fig.update_layout(
        title="Yield–Impurity Trade-off",
        xaxis_title="Impurity (%)",
        yaxis_title="Product Yield (%)",
        height=550
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Weighted-Sum Optimisation"
    )

    yield_weight = st.slider(
        "Weight assigned to Yield",
        0.0,
        1.0,
        0.7,
        0.05
    )

    impurity_weight = (
        1 - yield_weight
    )

    st.write(
        f"""
        Yield weight = **{yield_weight:.2f}**

        Impurity weight = **{impurity_weight:.2f}**
        """
    )

    yield_norm = (
        pdata["Yield"]
        - pdata["Yield"].min()
    ) / (
        pdata["Yield"].max()
        - pdata["Yield"].min()
    )

    impurity_norm = (
        pdata["Impurity"]
        - pdata["Impurity"].min()
    ) / (
        pdata["Impurity"].max()
        - pdata["Impurity"].min()
    )

    score = (
        yield_weight * yield_norm
        - impurity_weight * impurity_norm
    )

    best_index = score.idxmax()

    best_row = pdata.loc[
        best_index
    ]

    st.success(
        f"""
        Weighted-sum optimum:

        Experiment {int(best_row["Experiment"])}

        Yield = {best_row["Yield"]:.2f}%

        Impurity = {best_row["Impurity"]:.2f}%
        """
    )


# ============================================================
# TAB 4
# PHARMACEUTICAL DEMO
# ============================================================

with tab4:

    st.header(
        "Pharmaceutical Continuous-Flow Demo"
    )

    st.write(
        """
        Decision variables:

        - Temperature
        - Residence Time
        - Reagent Equivalence

        Objective:

        **Maximise Yield**

        Constraint:

        **Impurity ≤ 2%**
        """
    )

    if "pharma_data" not in st.session_state:

        st.session_state.pharma_data = (
            create_initial_pharma_data()
        )

    pharma_data = (
        st.session_state.pharma_data.copy()
    )

    # --------------------------------------------------------
    # FIT TWO GPS
    # --------------------------------------------------------

    variables = [
        "Temperature",
        "Residence_Time",
        "Equivalence"
    ]

    X = pharma_data[
        variables
    ].values

    y_yield = pharma_data[
        "Yield"
    ].values

    y_impurity = pharma_data[
        "Impurity"
    ].values

    scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    kernel_pharma = (
        ConstantKernel(1.0)
        * Matern(
            length_scale=[
                1,
                1,
                1
            ],
            nu=2.5
        )
        + WhiteKernel(
            noise_level=0.01
        )
    )

    gp_yield = GaussianProcessRegressor(
        kernel=kernel_pharma,
        normalize_y=True,
        random_state=42
    )

    gp_impurity = GaussianProcessRegressor(
        kernel=kernel_pharma,
        normalize_y=True,
        random_state=42
    )

    gp_yield.fit(
        X_scaled,
        y_yield
    )

    gp_impurity.fit(
        X_scaled,
        y_impurity
    )

    # --------------------------------------------------------
    # USER-DEFINED CONDITION
    # --------------------------------------------------------

    st.subheader(
        "Test Any Operating Condition"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        temperature = st.slider(
            "Temperature (°C)",
            60.0,
            120.0,
            100.0
        )

    with col2:

        residence = st.slider(
            "Residence Time (min)",
            2.0,
            20.0,
            12.0
        )

    with col3:

        equivalence = st.slider(
            "Reagent Equivalence",
            1.0,
            2.0,
            1.45
        )

    user_condition = np.array([[
        temperature,
        residence,
        equivalence
    ]])

    user_scaled = scaler.transform(
        user_condition
    )

    pred_yield,
    std_yield = gp_yield.predict(
        user_scaled,
        return_std=True
    )

    pred_impurity,
    std_impurity = gp_impurity.predict(
        user_scaled,
        return_std=True
    )

    probability_feasible = norm.cdf(
        (
            2.0
            - pred_impurity[0]
        )
        /
        (
            std_impurity[0]
            + 1e-12
        )
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Predicted Yield",
        f"{pred_yield[0]:.2f}%",
        f"± {std_yield[0]:.2f}%"
    )

    c2.metric(
        "Predicted Impurity",
        f"{pred_impurity[0]:.2f}%",
        f"± {std_impurity[0]:.2f}%"
    )

    c3.metric(
        "Probability Impurity ≤ 2%",
        f"{100 * probability_feasible:.1f}%"
    )

    # --------------------------------------------------------
    # CONSTRAINED BAYESIAN OPTIMISATION
    # --------------------------------------------------------

    st.subheader(
        "Constrained Bayesian Optimisation"
    )

    rng = np.random.default_rng(
        100
        + len(pharma_data)
    )

    n_candidates = 5000

    candidates = np.column_stack([
        rng.uniform(
            60,
            120,
            n_candidates
        ),
        rng.uniform(
            2,
            20,
            n_candidates
        ),
        rng.uniform(
            1,
            2,
            n_candidates
        )
    ])

    candidates_scaled = scaler.transform(
        candidates
    )

    mean_y, std_y = gp_yield.predict(
        candidates_scaled,
        return_std=True
    )

    mean_i, std_i = gp_impurity.predict(
        candidates_scaled,
        return_std=True
    )

    feasible_observed = (
        y_impurity <= 2.0
    )

    if np.any(
        feasible_observed
    ):

        best_yield = np.max(
            y_yield[
                feasible_observed
            ]
        )

    else:

        best_yield = np.max(
            y_yield
        )

    EI = expected_improvement(
        mean_y,
        std_y,
        best_yield
    )

    P_feasible = norm.cdf(
        (
            2.0
            - mean_i
        )
        /
        (
            std_i
            + 1e-12
        )
    )

    constrained_EI = (
        EI
        * P_feasible
    )

    best_candidate_index = np.argmax(
        constrained_EI
    )

    X_next = candidates[
        best_candidate_index
    ]

    st.write(
        "### Recommended Next Experiment"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Temperature",
        f"{X_next[0]:.2f} °C"
    )

    col2.metric(
        "Residence Time",
        f"{X_next[1]:.2f} min"
    )

    col3.metric(
        "Equivalence",
        f"{X_next[2]:.2f}"
    )

    st.write(
        f"""
        **GP predicted Yield:**  
        {mean_y[best_candidate_index]:.2f}
        ±
        {std_y[best_candidate_index]:.2f} %

        **GP predicted Impurity:**  
        {mean_i[best_candidate_index]:.2f}
        ±
        {std_i[best_candidate_index]:.2f} %

        **Probability of satisfying impurity constraint:**  
        {100 * P_feasible[best_candidate_index]:.1f} %
        """
    )

    if st.button(
        "Run Recommended Pharmaceutical Experiment"
    ):

        new_y,
        new_i = run_pharma_experiment(
            X_next,
            noise=True,
            random_state=(
                1000
                + len(pharma_data)
            )
        )

        new_row = pd.DataFrame({
            "Temperature": [
                X_next[0]
            ],
            "Residence_Time": [
                X_next[1]
            ],
            "Equivalence": [
                X_next[2]
            ],
            "Yield": [
                new_y[0]
            ],
            "Impurity": [
                new_i[0]
            ],
            "Experiment": [
                len(pharma_data)
                + 1
            ]
        })

        st.session_state.pharma_data = (
            pd.concat(
                [
                    pharma_data,
                    new_row
                ],
                ignore_index=True
            )
        )

        st.success(
            f"""
            Experiment completed.

            Observed Yield =
            {new_y[0]:.2f}%

            Observed Impurity =
            {new_i[0]:.2f}%
            """
        )

        st.rerun()

    # --------------------------------------------------------
    # EXPERIMENT TABLE
    # --------------------------------------------------------

    st.subheader(
        "Experimental History"
    )

    st.dataframe(
        pharma_data.round(2),
        hide_index=True,
        use_container_width=True
    )

    # --------------------------------------------------------
    # PARETO PLOT
    # --------------------------------------------------------

    pareto = find_pareto_front(
        pharma_data["Yield"].values,
        pharma_data["Impurity"].values
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=pharma_data["Impurity"],
            y=pharma_data["Yield"],
            mode="markers+text",
            text=pharma_data["Experiment"],
            textposition="top center",
            marker=dict(size=12),
            name="Experiments"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=pharma_data.loc[
                pareto,
                "Impurity"
            ],
            y=pharma_data.loc[
                pareto,
                "Yield"
            ],
            mode="markers",
            marker=dict(
                size=20,
                symbol="star"
            ),
            name="Pareto optimal"
        )
    )

    fig.add_vline(
        x=2.0,
        line_dash="dash",
        annotation_text="Impurity limit"
    )

    fig.update_layout(
        title=(
            "Yield–Impurity Trade-off"
        ),
        xaxis_title="Impurity (%)",
        yaxis_title="Yield (%)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    if st.button(
        "Reset Pharmaceutical Experiments"
    ):

        del st.session_state.pharma_data

        st.rerun()
