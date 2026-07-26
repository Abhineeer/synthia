import streamlit as st
import torch
import json
import numpy as np
from pinn import PINN
from solvers.heat_fd import solve_heat_fd
import plotly.graph_objects as go

t_max = 1.0
# This is the universal end time t = 1


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


st.markdown(
"""<h1 style="margin-bottom:0;">SYNTHIA</h1>
<h3 style="margin-top:-25px;">PINN for the 1D Heat Equation</h3>"""
,unsafe_allow_html=True
)

st.markdown(
    """
    <a href="https://github.com/Abhineeer/synthia" target="_blank"
       style="
           display: inline-flex;
           align-items: center;
           gap: 6px;
           padding: 6px 14px;
           background-color: #181717;
           color: #ffffff;
           border-radius: 6px;
           text-decoration: none;
           font-size: 13px;
           font-weight: 600;
           margin-top: 4px;
       ">
       ⚙ SYNTHIA on GitHub
    </a>
    """,
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(["Forward Solver", "Parameter Recovery"])

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
        title=dict(text="temp u", font=dict(size=13, weight="bold"), side="top"),
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
                           name="FD solver", line=dict(width=3, color="#41b9ff")))
    fig2.add_trace(go.Scatter(x=x_05, y=output_np, mode="lines",
                           name="PINN", line=dict(width=2, dash="dot",color="#000000")))
    fig2.update_layout(
        xaxis_title=dict(text="position x", font=dict(size=13, weight="bold")),
        yaxis_title=dict(text="u (temperature)", font=dict(size=13, weight="bold")),
        legend=dict(font=dict(size=14, weight="bold")),
        height=380,
        margin=dict(l=60, r=20, t=30, b=50),
    )
    st.plotly_chart(fig2, width='stretch')

    benchmarks = load_benchmarks()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"<div style='text-align:center;'>"
            f"<div style='font-size:13px;color:gray;'>Relative L2 error vs FD</div>"
            f"<div style='font-size:38px;font-weight:500;'>{benchmarks['rel_l2_vs_fd']['t_0.5']*100:.2f}%</div>"
            f"</div>", unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"<div style='text-align:center;'>"
            f"<div style='font-size:13px;color:gray;'>Inference speedup vs FD</div>"
            f"<div style='font-size:38px;font-weight:500;'>{benchmarks['speedup_pinn_vs_fd']:.1f}×</div>"
            f"</div>", unsafe_allow_html=True
        )
with tab2:
    st.info("Parameter recovery results coming soon.")
    # later: read benchmark_inverse.json, render method × noise × error
