import streamlit as st
import torch
import json
import numpy as np
from pinn import PINN
from solvers.heat_fd import solve_heat_fd
import plotly.graph_objects as go

st.set_page_config(
    page_title="SYNTHIA: PINN for the 1D Heat Equation",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

t_max = 1.0
# This is the universal end time t = 1

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700;9..144,900&family=JetBrains+Mono:wght@300;400;500;600&display=swap');

    :root {
        --bg: #0b0807;
        --surface: #151010;
        --hot: #f77f00;
        --peak: #ffc94a;
    }

    [data-testid="stHeader"] {
        background: var(--bg);
    }

    .stApp {
        background: var(--bg);
        color: #ece4d6;
    }

    h1, h2, h3, h4, h5 {
        font-family: 'Fraunces', serif !important;
        color: #f8f1e3 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid #2b2119;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #8a7f6f;
        background: transparent;
        padding: 10px 18px;
    }
    .stTabs [aria-selected="true"] {
        color: var(--peak) !important;
        border-bottom: 2px solid var(--hot) !important;
    }

    [data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid #2b2119;
    }
    [data-testid="stSidebar"] * { color: #ece4d6 !important; }

    p, li, .stMarkdown, label { color: #8a7f6f !important; }

    .block-container { padding-top: 2.5rem; max-width: 1100px; }

    [data-testid="stMetric"] {
        background: #1c1613;
        border: 1px solid #2b2119;
        border-left: 3px solid var(--hot);
        padding: 18px 22px;
        border-radius: 4px;
        min-height: 141px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 10px !important;
        letter-spacing: 2px;
        color: #8a7f6f !important;
    }
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
        color: #f8f1e3 !important;
        font-family: 'Fraunces', serif !important;
        font-size: 2.4rem !important;
        font-weight: 700 !important;
    }
    .stSlider [data-baseweb="slider"] > div > div { background: var(--hot) !important; }

    [data-testid="stExpander"] {
        background: #1c1613;
        border: 1px solid #2b2119 !important;
        border-radius: 4px;
    }

    [data-testid="stMetricDelta"] {
        color: var(--teal) !important;
    }
    [data-testid="stMetricDelta"] svg {
        display: none;
    }

    .stCaption, [data-testid="stCaptionContainer"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        color: #8a7f6f !important;
    }

    hr { border-color: #2b2119 !important; }

    .synthia-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 4px;
        color: var(--hot);
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .synthia-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        background: #1c1613;
        border: 1px solid #2b2119;
        color: #ece4d6 !important;
        border-radius: 6px;
        text-decoration: none;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
    }
    .synthia-badge:hover { border-color: var(--hot); }

    .synthia-card {
        background: #1c1613;
        border: 1px solid #2b2119;
        border-left: 3px solid #2ec4b6;
        padding: 16px 20px;
        border-radius: 4px;
        margin: 14px 0;
        font-size: 14px;
        line-height: 1.7;
        color: #8a7f6f;
    }
    .synthia-footer {
        margin-top: 48px;
        padding-top: 20px;
        border-top: 1px solid #2b2119;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10.5px;
        color: #8a7f6f;
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 8px;
    }
    .synthia-footer a { color: var(--peak); text-decoration: none; }

    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    model = PINN()
    model.load_state_dict(torch.load('models/heat_pinn.pth', map_location=torch.device('cpu')))
    # load_state_dict with map_location="cpu"
    model.eval()
    return model

@st.cache_data
def load_benchmarks():
    # open benchmark_phase1.json, return the dict
    with open('benchmarks/benchmark_phase1.json') as f:
        data = json.load(f)
    
    return data

@st.cache_data
def fd_solve(alpha, nx=100, t_max=t_max):
    dx = 1.0 / (nx - 1)
    dt = 0.4 * dx**2 / alpha        
    # must match r in heat_fd.py
    n_t = int(t_max / dt)
    return solve_heat_fd(alpha, nx, n_t)

@st.cache_data
def load_uncertainty():
    with open('benchmarks/benchmarks_uncertainty.json') as f:
        return json.load(f)['forward_uncertainty']


st.markdown('<div class="synthia-eyebrow">Physics-Informed Neural Network · 1D Heat Equation</div>', unsafe_allow_html=True)
st.markdown(
    """<h1 style="margin-bottom:0; font-size:56px;">SYNTHIA</h1>
    <p style="font-style:italic; color:#a89c8a; max-width:640px; margin-top:6px;
       border-left:2px solid var(--hot); padding-left:16px;">
       A neural network trained to obey the heat equation, not memorize its solution,
       forward-solved, benchmarked against a classical solver, and inverted to recover
       an unknown physical parameter from noisy data.
    </p>"""
    ,unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs(["Forward Solver", "Parameter Recovery", "Uncertainty Quantification"])

with tab1:
    st.subheader("Forward Solver")
    st.write(
        "A metal rod starts hot in the middle and cold at both ends, which are locked at 0°C. Heat spreads outward from the peak and the rod cools toward zero. The slider down below sets the thermal diffusivity α. It dictates how fast heat moves through the material. Higher the α, quicker the rod temperature settles."
    ) # explainer for non physicist viewers

    # --- PANEL A: slider-driven FD heatmap ---
    
    alpha = st.slider("Thermal diffusivity α", 0.01, 0.5, 0.1, step = 0.01 )
    # slider arguments: Label, min_val, max_val, initial val, and step
    x, t, u = fd_solve(alpha)
    # thin the rows: solver takes thousands of steps, we only need ~120 frames
    stride = max(1, len(t) // 120)
    u_plot, t_plot = u[::stride], t[::stride]

    fig = go.Figure(
    data=go.Heatmap(
        z=u_plot, x=x, y=t_plot,
        colorscale="Viridis", zmin=0, zmax=1,
        colorbar=dict(
        title=dict(text="temp u", font=dict(size=13, weight="bold", color="#ece4d6"), side="top"),
        thickness=40,
        len=1.054,
    ),
        hovertemplate="x: %{x:.3f}<br>t: %{y:.3f}<br>u: %{z:.3f}<extra></extra>"
    )
    )

    fig.update_layout(
        xaxis_title=dict(text="position x", font=dict(size=13, weight="bold")),
        yaxis_title=dict(text="time t", font=dict(size=13, weight="bold")),
        height=420,
        margin=dict(l=60, r=20, t=30, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ece4d6"),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("More information"):
        st.write("The forward solver integrates the 1D heat equation:")
        st.latex(r"\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}")
        st.write("with boundary and initial conditions:")
        st.latex(r"u(0,t) = u(1,t) = 0, \qquad u(x,0) = \sin(\pi x)")
        st.write("discretized with an explicit finite-difference scheme:")
        st.latex(
            r"u_i^{n+1} = u_i^n + \alpha \frac{\Delta t}{\Delta x^2}"
            r"\left(u_{i+1}^n - 2u_i^n + u_{i-1}^n\right)"
        )

    st.caption(
        f"α = {alpha} · {len(t)} timesteps to t = {t_max} · "
        f"solved with explicit finite differences"
    )

    # --- PANEL B: validation, PINNED at α = 0.1 ---

    x_05 = np.linspace(0, 1, 100)
    t_05 = np.full((100,), 0.5)
    x = torch.from_numpy(x_05).reshape(-1,1).float()
    t = torch.from_numpy(t_05).reshape(-1,1).float()

    model = load_model()
    with torch.no_grad():
        output = model(x, t)
        output_np = output.numpy().flatten()
        # We need the right shape to be (100, ) and flatten helps out with that
    
    print(output_np.shape)
    print(output_np[50])

    st.subheader("Validation: Is the PINN trustworthy?")
    st.write(
        "A PINN's accuracy isn't guaranteed by design, unlike the finite-difference solver, it never sees the governing equation solved directly, only learns to satisfy it through training. This panel is the proof: the network's own prediction, checked point-by-point against the classical solver (above) it was never shown. If the two curves overlap, the PINN has genuinely learned the physics, not just memorized an approximate shape."
    )

    # FD solution at the exact same setup the PINN was trained on
    x_fd, t_fd, u_fd = fd_solve(0.1)
    row_idx = np.argmin(np.abs(t_fd - 0.5))
    # t_fd - 0.5, subtracting 0.5 from each element in the array and then abs - absolute value of each elemt
    # argmin finds the 0 or closest to 0 in the array and that will be the closest to t = 0.5
    u_fd_at_05 = u_fd[row_idx]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x_fd, y=u_fd_at_05, mode="lines",
                           name="FD solver", line=dict(width=3, color="#2ec4b6")))
    fig2.add_trace(go.Scatter(x=x_05, y=output_np, mode="lines",
                           name="PINN", line=dict(width=2, dash="dot",color="#ffc94a")))
    fig2.update_layout(
        xaxis_title=dict(text="position x", font=dict(size=13, weight="bold")),
        yaxis_title=dict(text="u (temperature)", font=dict(size=13, weight="bold")),
        legend=dict(font=dict(size=14, weight="bold", color="#ece4d6")),
        height=380,
        margin=dict(l=60, r=20, t=30, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ece4d6"),
    )
    st.plotly_chart(fig2, width='stretch')

    benchmarks = load_benchmarks()

    with st.sidebar:
        st.markdown('<div class="synthia-eyebrow">Reference Library</div>', unsafe_allow_html=True)
        st.markdown("### SYNTHIA")
        st.caption("Physics-informed neural network")

        st.markdown("---")

        st.markdown(
            '<div class="synthia-card">'
            '<strong style="color:var(--text)">Forward Solver</strong><br>'
            'Network trained on a physics residual loss at random collocation points — '
            'no solution data given directly. Validated against an independently-built '
            'finite-difference solver.<br>'
            '<a href="https://arxiv.org/abs/1711.10561" target="_blank" style="color:var(--peak);">'
            'Raissi et al., 2019 ↗</a>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="synthia-card">'
            '<strong style="color:var(--text)">Parameter Recovery</strong><br>'
            'α made a learnable nn.Parameter, recovered from 150–500 sparse, noisy '
            'observations. Benchmarked against scipy.optimize.curve_fit across four '
            'noise levels and multiple seeds.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="synthia-card">'
            '<strong style="color:var(--text)">Uncertainty Quantification</strong><br>'
            'Dropout kept active at inference — 200 passes approximate a Bayesian '
            'posterior over predictions.<br>'
            '<a href="https://arxiv.org/abs/1506.02142" target="_blank" style="color:var(--peak);">'
            'Gal &amp; Ghahramani, 2016 ↗</a>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            '<a href="https://github.com/Abhineeer/synthia" target="_blank" class="synthia-badge">'
            '⚙ SYNTHIA on GitHub</a>',
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Relative L2 error vs FD", f"{benchmarks['rel_l2_vs_fd']['t_0.5']*100:.2f}%")
    with col2:
        st.metric("Inference speedup vs FD", f"{benchmarks['speedup_pinn_vs_fd']:.1f}×")
with tab2:
    st.subheader("Recovering hidden physics from noisy data")
    st.markdown("""
    Imagine you have a metal rod and a few scattered, imperfect temperature
    readings taken over time, like a handful of noisy sensor measurements,
    not a full picture. Can you work backward from those readings to figure
    out *how fast heat moves through the material*, a property called
    **thermal diffusivity (α)**? This tab compares two ways of solving that
    problem: a classical curve-fitting method, and a physics-informed neural
    network that uses the heat equation itself as a constraint. Adjust the
    noise slider below to see how each method holds up as the data gets messier.
    """)
    st.markdown("")
    st.markdown("")
    st.markdown("")

    with open("benchmarks/benchmark_inverse_final.json") as f:
        inverse_data = json.load(f)

    sigma_options = [0.01, 0.05, 0.1, 0.2]
    # fixed sigma (noise) values
    selected_sigma = st.select_slider(
        "Noise level (σ)",
        options=sigma_options,
        value=0.05
    )
    sigma_key = f"sigma_{selected_sigma}"

    pinn_result = inverse_data[sigma_key].get("inverse_pinn")
    curvefit_result = inverse_data[sigma_key].get("curve_fit")

    sigma_options_valid = []
    curvefit_errors = []
    pinn_errors = []

    for s in sigma_options:
        key = f"sigma_{s}"
        cf = inverse_data.get(key, {}).get("curve_fit")
        pn = inverse_data.get(key, {}).get("inverse_pinn")

        if cf is None:
            continue

        sigma_options_valid.append(s)
        curvefit_errors.append(cf["mean_error_pct"])

        if pn is not None and "mean_error_pct" in pn:
            pinn_errors.append(pn["mean_error_pct"])
        else:
            pinn_errors.append(None)

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
            x=sigma_options_valid,
            y=curvefit_errors,
            mode="lines+markers",
            name="curve_fit",
            line=dict(color="#7a5bc2")
        ))
    
    if any(e is not None for e in pinn_errors):
            fig.add_trace(go.Scatter(
                x=sigma_options_valid,
                y=pinn_errors,
                mode="lines+markers",
                name="PINN",
                line=dict(color="#ffc94a")
            ))
    
    fig.add_vline(x=selected_sigma, line_dash="dash", line_color="gray")
    
    fig.update_layout(
            xaxis_title="Noise level (σ)",
            yaxis_title="Mean recovery error (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,font=dict(color="#ece4d6")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ece4d6"),
        )
    
    st.plotly_chart(fig, use_container_width=True)

    pinn_ready = pinn_result is not None and "mean_alpha_recovered" in pinn_result

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("TRUE \u03b1", "0.100")

    with col2:
        if pinn_ready:
            st.metric(
                "PINN RECOVERED \u03b1",
                f"{pinn_result['mean_alpha_recovered']:.4f}",
                f"{pinn_result['mean_error_pct']:.1f}% error"
            )
        else:
            st.metric("PINN recovered \u03b1", "pending")

    with col3:
        st.metric(
            "CURVE_FIT RECOVERED \u03b1",
            f"{curvefit_result['mean_alpha_recovered']:.4f}",
            f"{curvefit_result['mean_error_pct']:.1f}% error"
        )

    if pinn_ready:
        st.caption(
            f"Across {pinn_result.get('n_seeds', len(pinn_result.get('alpha_values', [])))} independent training runs at this noise level, "
            f"recovered \u03b1 had a standard deviation of {pinn_result['std_alpha_recovered']:.4f} "
            f"({pinn_result['std_error_pct']:.1f}% of true α)."
        )

with tab3:
    st.subheader("Uncertainty Quantification")
    st.write(
        "Dropout stays active at inference instead of being switched off — 200 forward "
        "passes per point, and the spread across those passes approximates a Bayesian "
        "posterior over the network's predictions "
        "([Gal & Ghahramani, 2016](https://arxiv.org/abs/1506.02142))."
    )

    unc = load_uncertainty()

    st.image("figures/fig_uncertainty.png", use_container_width=True,
              caption="Standard deviation across 200 MC Dropout passes, full (x,t) domain. "
                      "White contour lines trace the mean predicted temperature.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean uncertainty", f"{unc['Mean uncertainty across the domain']:.4f}")
    with col2:
        st.metric("Max uncertainty", f"{unc['Max uncertainty across the domain']:.4f}")
    with col3:
        loc = unc['Max uncertainty co-ordinates']
        st.metric("Max at (x, t)", f"({loc['x']:.2f}, {loc['t']:.2f})")

    st.markdown(
        '<div class="synthia-card">'
        '<strong style="color:var(--text)">Why the interior, not the edges</strong><br>'
        'Boundary and initial conditions directly supervise the edges of the domain — '
        'the rod ends and t=0 are pinned by their own loss terms. Interior points have '
        'only the PDE residual to constrain them, so uncertainty is highest away from '
        'those directly-supervised regions, not at the steepest temperature gradient. '
        'Uncertainty here tracks proximity to constraints, not local gradient steepness.'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="synthia-footer">
        <span>SYNTHIA · Adii Singh · ASU Applied Physics + Computer Science</span>
        <a href="https://github.com/Abhineeer/synthia" target="_blank">github.com/Abhineeer/synthia</a>
    </div>
    """,
    unsafe_allow_html=True,
)