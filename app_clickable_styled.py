# ============================================================
# GP-BO LEARNING LAB
# Simple educational software for:
# 1. Gaussian Processes
# 2. Bayesian Optimisation
# 3. Pareto / Multi-objective optimisation
# 4. Pharmaceutical continuous-flow optimisation
# 5. Bring-your-own-data GP/BO workflow
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
    page_title="Optimisation Accelerator Programme",
    page_icon="🚀",
    layout="wide"
)


# ============================================================
# APP STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .hero {
        background: linear-gradient(120deg, #f8fbff 0%, #edf4ff 58%, #f8fbff 100%);
        border: 1px solid #d8e4f4;
        border-radius: 24px;
        padding: 2.2rem 2.2rem 1.6rem 2.2rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 30px rgba(23, 55, 94, 0.08);
    }

    .hero-title {
        font-size: 2.25rem;
        line-height: 1.15;
        font-weight: 800;
        color: #0b2447;
        margin-bottom: 0.25rem;
    }

    .hero-subtitle {
        font-size: 1.6rem;
        line-height: 1.2;
        font-weight: 700;
        color: #2457b8;
        margin-bottom: 1rem;
    }

    .hero-copy {
        font-size: 1.05rem;
        color: #24364b;
        max-width: 900px;
        margin-bottom: 1.1rem;
    }

    .hero-badges {
        display: flex;
        gap: 0.7rem;
        flex-wrap: wrap;
        margin-top: 0.5rem;
    }

    .hero-badge {
        background: white;
        border: 1px solid #d7e3f2;
        border-radius: 999px;
        padding: 0.45rem 0.8rem;
        font-size: 0.92rem;
        font-weight: 600;
        color: #17375e;
    }

    .lab-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 0.8rem;
        margin-top: 0.6rem;
        margin-bottom: 1.25rem;
    }

    .lab-card {
        background: white;
        border: 1px solid #dde6f2;
        border-radius: 18px;
        padding: 1rem;
        min-height: 145px;
        box-shadow: 0 6px 18px rgba(15, 43, 77, 0.05);
    }

    .lab-icon {
        font-size: 1.8rem;
        margin-bottom: 0.45rem;
    }

    .lab-title {
        font-weight: 800;
        color: #0d2d57;
        margin-bottom: 0.25rem;
    }

    .lab-copy {
        font-size: 0.9rem;
        color: #5b6b7f;
        line-height: 1.35;
    }

    @media (max-width: 1000px) {
        .lab-grid {
            grid-template-columns: 1fr 1fr;
        }
    }

    @media (max-width: 650px) {
        .lab-grid {
            grid-template-columns: 1fr;
        }
        .hero-title {
            font-size: 1.8rem;
        }
        .hero-subtitle {
            font-size: 1.3rem;
        }
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d2c54 0%, #09213f 100%);
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    div.stButton > button {
        border-radius: 12px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# NAVIGATION
# ============================================================

PAGES = [
    "Home",
    "GP Lab",
    "BO Lab",
    "Pareto Lab",
    "Pharmaceutical Demo",
    "My Own Data"
]

if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

with st.sidebar:

    st.markdown(
        "## 🚀 Optimisation Accelerator"
    )

    st.caption(
        "Introduction to Bayesian Optimisation"
    )

    for page_name in PAGES:

        button_label = {
            "Home": "🏠 Home",
            "GP Lab": "📈 GP Lab",
            "BO Lab": "🎯 BO Lab",
            "Pareto Lab": "🔗 Pareto Lab",
            "Pharmaceutical Demo": "⚗️ Pharmaceutical Demo",
            "My Own Data": "🗂️ My Own Data"
        }[page_name]

        if st.button(
            button_label,
            key=f"nav_{page_name}",
            use_container_width=True
        ):
            st.session_state.current_page = page_name
            st.rerun()

page = st.session_state.current_page

# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                Optimisation Accelerator Programme:
            </div>
            <div class="hero-subtitle">
                Introduction to Bayesian Optimisation
            </div>
            <div class="hero-copy">
                Learn Gaussian Processes, uncertainty, Bayesian Optimisation
                and multi-objective optimisation through interactive labs
                and real-world inspired examples.
            </div>
            <div class="hero-badges">
                <div class="hero-badge">✓ Intuitive & Interactive</div>
                <div class="hero-badge">✓ Learn by Doing</div>
                <div class="hero-badge">✓ Optimise Experiments</div>
                <div class="hero-badge">✓ Make Better Decisions</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="lab-grid">
            <div class="lab-card">
                <div class="lab-icon">📈</div>
                <div class="lab-title">1. GP Lab</div>
                <div class="lab-copy">Explore Gaussian Processes, kernels, prediction and uncertainty.</div>
            </div>
            <div class="lab-card">
                <div class="lab-icon">🎯</div>
                <div class="lab-title">2. BO Lab</div>
                <div class="lab-copy">See Expected Improvement and sequential optimisation in action.</div>
            </div>
            <div class="lab-card">
                <div class="lab-icon">🔗</div>
                <div class="lab-title">3. Pareto Lab</div>
                <div class="lab-copy">Explore trade-offs and multi-objective optimisation.</div>
            </div>
            <div class="lab-card">
                <div class="lab-icon">⚗️</div>
                <div class="lab-title">4. Pharmaceutical Demo</div>
                <div class="lab-copy">Optimise a continuous-flow reaction with Yield and Impurity.</div>
            </div>
            <div class="lab-card">
                <div class="lab-icon">🗂️</div>
                <div class="lab-title">5. My Own Data</div>
                <div class="lab-copy">Upload your own experimental data and build a GP/BO workflow.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "Tip: start with GP Lab, then move to BO Lab before trying the case studies."
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
    X, noise =True,
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
    best, xi =0.01
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
        n, dtype =bool
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
        X, noise =True,
        random_state=42
    )

    data = pd.DataFrame(
        X, columns =[
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

# ============================================================
# TAB 1
# GAUSSIAN PROCESS LAB
# ============================================================

if page == "GP Lab":

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
    X_test, return_std =True
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
        fig, use_container_width =True
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

if page == "BO Lab":

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
        grid, return_std =True
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
        fig_gp, use_container_width =True
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
        fig_ei, use_container_width =True
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

if page == "Pareto Lab":

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
        fig, use_container_width =True
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

if page == "Pharmaceutical Demo":

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

    pred_yield, std_yield = gp_yield.predict(
        user_scaled, return_std =True
    )

    pred_impurity, std_impurity = gp_impurity.predict(
        user_scaled, return_std =True
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
        candidates_scaled, return_std =True
    )

    mean_i, std_i = gp_impurity.predict(
        candidates_scaled, return_std =True
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

        new_y, new_i = run_pharma_experiment(
            X_next, noise =True,
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
        fig, use_container_width =True
    )

    if st.button(
        "Reset Pharmaceutical Experiments"
    ):

        del st.session_state.pharma_data

        st.rerun()



# ============================================================
# TAB 5
# MY OWN DATA
# ============================================================

if page == "My Own Data":

    st.header("My Own Experimental Data")

    st.write(
        """
        Upload a CSV dataset from your own case study.

        This tab can:
        - fit a Gaussian Process to your data;
        - predict the response and its uncertainty;
        - recommend the next experiment using Expected Improvement;
        - optionally include a second output as a constraint;
        - let you enter the new measured result and update the dataset.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        key="own_data_upload"
    )

    if uploaded_file is None:

        st.info(
            """
            Upload a CSV file to begin.

            Example columns:
            Experiment, Temp, Time, Concentration, Yield
            """
        )

    else:

        uploaded_name = uploaded_file.name

        if (
            "own_data_name" not in st.session_state
            or st.session_state.own_data_name != uploaded_name
        ):
            raw_data = pd.read_csv(uploaded_file)
            st.session_state.own_data = raw_data.copy()
            st.session_state.own_data_name = uploaded_name
            st.session_state.own_data_history = []

        own_data = st.session_state.own_data.copy()

        st.subheader("1. Uploaded Dataset")

        st.dataframe(
            own_data,
            hide_index=True,
            use_container_width=True
        )

        st.subheader("2. Identify the Experiment / ID Column")

        id_options = ["None"] + own_data.columns.tolist()

        default_id_index = 0
        for candidate_name in ["Experiment", "experiment", "ID", "Id", "id"]:
            if candidate_name in own_data.columns:
                default_id_index = id_options.index(candidate_name)
                break

        id_column_choice = st.selectbox(
            "Experiment / ID column",
            options=id_options,
            index=default_id_index,
            help=(
                "This column is used only as an experiment identifier. "
                "It is not used as a GP input or output."
            ),
            key="own_id_column"
        )

        id_column = None if id_column_choice == "None" else id_column_choice

        # Exclude the ID column from the GP modelling choices
        numeric_columns = own_data.select_dtypes(
            include=np.number
        ).columns.tolist()

        modelling_numeric_columns = [
            c for c in numeric_columns
            if c != id_column
        ]

        if len(modelling_numeric_columns) < 2:

            st.error(
                """
                After excluding the Experiment / ID column,
                the CSV must contain at least two numerical columns.
                """
            )

        else:

            st.subheader("3. Define the GP / BO Problem")

            input_columns = st.multiselect(
                "Select decision variables / inputs",
                options=modelling_numeric_columns,
                default=modelling_numeric_columns[:-1],
                key="own_inputs"
            )

            available_outputs = [
                c for c in modelling_numeric_columns
                if c not in input_columns
            ]

            if len(input_columns) == 0:

                st.warning(
                    "Select at least one decision variable."
                )

            elif len(available_outputs) == 0:

                st.warning(
                    "At least one numerical column must remain as an output."
                )

            else:

                objective_column = st.selectbox(
                    "Select objective / primary output",
                    options=available_outputs,
                    key="own_objective"
                )

                objective_direction = st.radio(
                    "Objective direction",
                    options=["Maximise", "Minimise"],
                    horizontal=True,
                    key="own_direction"
                )

                remaining_for_constraint = [
                    c for c in available_outputs
                    if c != objective_column
                ]

                use_constraint = st.checkbox(
                    "Use a second output as a constraint",
                    value=False,
                    key="own_use_constraint"
                )

                constraint_column = None
                constraint_operator = None
                constraint_limit = None

                if use_constraint:

                    if len(remaining_for_constraint) == 0:

                        st.warning(
                            "No additional numerical output is available."
                        )
                        use_constraint = False

                    else:

                        c1, c2, c3 = st.columns(3)

                        with c1:
                            constraint_column = st.selectbox(
                                "Constraint output",
                                options=remaining_for_constraint,
                                key="own_constraint_column"
                            )

                        with c2:
                            constraint_operator = st.selectbox(
                                "Constraint",
                                options=["≤", "≥"],
                                key="own_constraint_operator"
                            )

                        with c3:
                            default_limit = float(
                                own_data[
                                    constraint_column
                                ].median()
                            )

                            constraint_limit = st.number_input(
                                "Constraint limit",
                                value=default_limit,
                                key="own_constraint_limit"
                            )

                st.subheader("4. Gaussian Process Settings")

                g1, g2, g3 = st.columns(3)

                with g1:
                    own_kernel_choice = st.selectbox(
                        "Kernel",
                        options=["Matern", "RBF"],
                        key="own_kernel"
                    )

                with g2:
                    own_noise = st.number_input(
                        "Initial noise level",
                        min_value=1e-8,
                        value=0.01,
                        format="%.4f",
                        key="own_noise"
                    )

                with g3:
                    n_candidates_own = st.number_input(
                        "Candidate points",
                        min_value=500,
                        max_value=50000,
                        value=5000,
                        step=500,
                        key="own_candidates"
                    )

                model_columns = list(input_columns) + [objective_column]

                if use_constraint:
                    model_columns.append(constraint_column)

                model_data = own_data[
                    model_columns
                ].dropna().copy()

                if len(model_data) < 3:

                    st.error(
                        """
                        At least three complete experimental rows are required
                        to fit the Gaussian Process.
                        """
                    )

                else:

                    X_own = model_data[
                        input_columns
                    ].values.astype(float)

                    y_objective = model_data[
                        objective_column
                    ].values.astype(float)

                    own_scaler = MinMaxScaler()
                    X_own_scaled = own_scaler.fit_transform(
                        X_own
                    )

                    n_inputs = len(input_columns)

                    if own_kernel_choice == "RBF":

                        own_base_kernel = RBF(
                            length_scale=np.ones(n_inputs),
                            length_scale_bounds=(1e-2, 1e2)
                        )

                    else:

                        own_base_kernel = Matern(
                            length_scale=np.ones(n_inputs),
                            length_scale_bounds=(1e-2, 1e2),
                            nu=2.5
                        )

                    own_kernel_objective = (
                        ConstantKernel(
                            1.0,
                            (1e-3, 1e3)
                        )
                        * own_base_kernel
                        + WhiteKernel(
                            noise_level=own_noise,
                            noise_level_bounds=(1e-8, 1.0)
                        )
                    )

                    gp_objective = GaussianProcessRegressor(
                        kernel=own_kernel_objective,
                        normalize_y=True,
                        n_restarts_optimizer=3,
                        random_state=42
                    )

                    gp_objective.fit(
                        X_own_scaled,
                        y_objective
                    )

                    gp_constraint = None
                    y_constraint = None

                    if use_constraint:

                        y_constraint = model_data[
                            constraint_column
                        ].values.astype(float)

                        if own_kernel_choice == "RBF":

                            own_base_kernel_constraint = RBF(
                                length_scale=np.ones(n_inputs),
                                length_scale_bounds=(1e-2, 1e2)
                            )

                        else:

                            own_base_kernel_constraint = Matern(
                                length_scale=np.ones(n_inputs),
                                length_scale_bounds=(1e-2, 1e2),
                                nu=2.5
                            )

                        own_kernel_constraint = (
                            ConstantKernel(
                                1.0,
                                (1e-3, 1e3)
                            )
                            * own_base_kernel_constraint
                            + WhiteKernel(
                                noise_level=own_noise,
                                noise_level_bounds=(1e-8, 1.0)
                            )
                        )

                        gp_constraint = GaussianProcessRegressor(
                            kernel=own_kernel_constraint,
                            normalize_y=True,
                            n_restarts_optimizer=3,
                            random_state=42
                        )

                        gp_constraint.fit(
                            X_own_scaled,
                            y_constraint
                        )

                    st.success(
                        f"GP model fitted using {len(model_data)} experiments."
                    )

                    st.subheader("5. Predict at Any Conditions")

                    user_values = []

                    n_ui_cols = min(
                        3,
                        len(input_columns)
                    )

                    ui_cols = st.columns(
                        n_ui_cols
                    )

                    for j, variable in enumerate(input_columns):

                        v_min = float(
                            own_data[variable].min()
                        )

                        v_max = float(
                            own_data[variable].max()
                        )

                        v_default = float(
                            own_data[variable].median()
                        )

                        with ui_cols[
                            j % n_ui_cols
                        ]:

                            value = st.number_input(
                                variable,
                                min_value=v_min,
                                max_value=v_max,
                                value=v_default,
                                key=f"own_predict_{variable}"
                            )

                            user_values.append(
                                value
                            )

                    user_point = np.array(
                        [user_values],
                        dtype=float
                    )

                    user_point_scaled = own_scaler.transform(
                        user_point
                    )

                    pred_objective, std_objective = gp_objective.predict(
                        user_point_scaled,
                        return_std=True
                    )

                    p_feasible_user = None
                    pred_constraint = None
                    std_constraint = None

                    if use_constraint:

                        pred_constraint, std_constraint = gp_constraint.predict(
                            user_point_scaled,
                            return_std=True
                        )

                        if constraint_operator == "≤":

                            p_feasible_user = norm.cdf(
                                (
                                    constraint_limit
                                    - pred_constraint[0]
                                )
                                /
                                (
                                    std_constraint[0]
                                    + 1e-12
                                )
                            )

                        else:

                            p_feasible_user = 1.0 - norm.cdf(
                                (
                                    constraint_limit
                                    - pred_constraint[0]
                                )
                                /
                                (
                                    std_constraint[0]
                                    + 1e-12
                                )
                            )

                    metric_cols = st.columns(
                        3 if use_constraint else 2
                    )

                    metric_cols[0].metric(
                        f"Predicted {objective_column}",
                        f"{pred_objective[0]:.4g}"
                    )

                    metric_cols[1].metric(
                        "Objective uncertainty",
                        f"± {std_objective[0]:.4g}"
                    )

                    if use_constraint:

                        metric_cols[2].metric(
                            "Probability feasible",
                            f"{100 * p_feasible_user:.1f}%"
                        )

                        st.caption(
                            f"Predicted {constraint_column} = "
                            f"{pred_constraint[0]:.4g} ± "
                            f"{std_constraint[0]:.4g}"
                        )

                    st.subheader(
                        "6. Recommend the Next Experiment"
                    )

                    rng_own = np.random.default_rng(
                        2026
                        + len(model_data)
                    )

                    candidate_columns = []

                    for variable in input_columns:

                        low = float(
                            own_data[variable].min()
                        )

                        high = float(
                            own_data[variable].max()
                        )

                        if high <= low:

                            st.error(
                                f"Input '{variable}' has no variation."
                            )

                            candidate_columns = []
                            break

                        candidate_columns.append(
                            rng_own.uniform(
                                low,
                                high,
                                int(n_candidates_own)
                            )
                        )

                    if len(candidate_columns) == len(input_columns):

                        candidates_own = np.column_stack(
                            candidate_columns
                        )

                        candidates_own_scaled = own_scaler.transform(
                            candidates_own
                        )

                        mean_objective, std_objective_candidates = gp_objective.predict(
                            candidates_own_scaled,
                            return_std=True
                        )

                        if objective_direction == "Maximise":

                            best_observed = np.max(
                                y_objective
                            )

                            acquisition = expected_improvement(
                                mean_objective,
                                std_objective_candidates,
                                best_observed
                            )

                        else:

                            best_observed_internal = np.max(
                                -y_objective
                            )

                            acquisition = expected_improvement(
                                -mean_objective,
                                std_objective_candidates,
                                best_observed_internal
                            )

                        probability_candidates = None
                        mean_constraint_candidates = None
                        std_constraint_candidates = None

                        if use_constraint:

                            mean_constraint_candidates, std_constraint_candidates = gp_constraint.predict(
                                candidates_own_scaled,
                                return_std=True
                            )

                            if constraint_operator == "≤":

                                probability_candidates = norm.cdf(
                                    (
                                        constraint_limit
                                        - mean_constraint_candidates
                                    )
                                    /
                                    (
                                        std_constraint_candidates
                                        + 1e-12
                                    )
                                )

                            else:

                                probability_candidates = 1.0 - norm.cdf(
                                    (
                                        constraint_limit
                                        - mean_constraint_candidates
                                    )
                                    /
                                    (
                                        std_constraint_candidates
                                        + 1e-12
                                    )
                                )

                            acquisition = (
                                acquisition
                                * probability_candidates
                            )

                        next_own_index = np.argmax(
                            acquisition
                        )

                        next_own_point = candidates_own[
                            next_own_index
                        ]

                        st.write(
                            "#### Recommended Operating Conditions"
                        )

                        recommendation = pd.DataFrame({
                            "Decision Variable": input_columns,
                            "Recommended Value": next_own_point
                        })

                        st.dataframe(
                            recommendation.style.format(
                                {
                                    "Recommended Value":
                                        "{:.4g}"
                                }
                            ),
                            hide_index=True,
                            use_container_width=True
                        )

                        r1, r2, r3 = st.columns(3)

                        r1.metric(
                            f"Predicted {objective_column}",
                            f"{mean_objective[next_own_index]:.4g}"
                        )

                        r2.metric(
                            "GP uncertainty",
                            f"± {std_objective_candidates[next_own_index]:.4g}"
                        )

                        if use_constraint:

                            r3.metric(
                                "Probability feasible",
                                f"{100 * probability_candidates[next_own_index]:.1f}%"
                            )

                        st.subheader(
                            "7. Add the Measured Result"
                        )

                        st.write(
                            """
                            Perform the recommended experiment in your
                            laboratory, simulator, or digital twin, then
                            enter the measured result below.
                            """
                        )

                        measured_objective = st.number_input(
                            f"Measured {objective_column}",
                            value=float(
                                mean_objective[
                                    next_own_index
                                ]
                            ),
                            key="own_measured_objective"
                        )

                        measured_constraint = None

                        if use_constraint:

                            measured_constraint = st.number_input(
                                f"Measured {constraint_column}",
                                value=float(
                                    mean_constraint_candidates[
                                        next_own_index
                                    ]
                                ),
                                key="own_measured_constraint"
                            )

                        if st.button(
                            "Add New Experimental Result",
                            key="own_add_result"
                        ):

                            new_row_dict = {
                                variable:
                                    next_own_point[k]
                                for k, variable
                                in enumerate(input_columns)
                            }

                            new_row_dict[
                                objective_column
                            ] = measured_objective

                            if use_constraint:

                                new_row_dict[
                                    constraint_column
                                ] = measured_constraint

                            # Automatically assign the next Experiment / ID
                            if id_column is not None:

                                existing_ids = pd.to_numeric(
                                    own_data[id_column],
                                    errors="coerce"
                                )

                                if existing_ids.notna().any():

                                    next_id = int(
                                        existing_ids.max()
                                    ) + 1

                                else:

                                    next_id = (
                                        len(own_data)
                                        + 1
                                    )

                                new_row_dict[
                                    id_column
                                ] = next_id

                            # Preserve any other uploaded columns as blank
                            for column in own_data.columns:

                                if column not in new_row_dict:

                                    new_row_dict[
                                        column
                                    ] = np.nan

                            new_row = pd.DataFrame(
                                [new_row_dict]
                            )[
                                own_data.columns
                            ]

                            st.session_state.own_data = pd.concat(
                                [
                                    own_data,
                                    new_row
                                ],
                                ignore_index=True
                            )

                            history_row = {
                                "Experiment":
                                    (
                                        new_row_dict[id_column]
                                        if id_column is not None
                                        else len(
                                            st.session_state.own_data
                                        )
                                    ),

                                "Predicted_Objective":
                                    float(
                                        mean_objective[
                                            next_own_index
                                        ]
                                    ),

                                "Objective_Uncertainty":
                                    float(
                                        std_objective_candidates[
                                            next_own_index
                                        ]
                                    ),

                                "Measured_Objective":
                                    float(
                                        measured_objective
                                    )
                            }

                            st.session_state.own_data_history.append(
                                history_row
                            )

                            st.success(
                                """
                                New experimental result added.
                                The GP will be refitted automatically.
                                """
                            )

                            st.rerun()

                        st.subheader(
                            "8. Download Updated Dataset"
                        )

                        updated_csv = (
                            st.session_state.own_data
                            .to_csv(
                                index=False
                            )
                            .encode(
                                "utf-8"
                            )
                        )

                        st.download_button(
                            label="Download Updated CSV",
                            data=updated_csv,
                            file_name="updated_experimental_data.csv",
                            mime="text/csv",
                            key="own_download"
                        )

                        if len(
                            st.session_state.own_data_history
                        ) > 0:

                            st.subheader(
                                "BO Update History"
                            )

                            st.dataframe(
                                pd.DataFrame(
                                    st.session_state.own_data_history
                                ),
                                hide_index=True,
                                use_container_width=True
                            )
