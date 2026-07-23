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


st.title("SYNTHIA — PINN for the 1D Heat Equation")
tab1, tab2 = st.tabs(["Forward Solver", "Parameter Recovery"])

with tab1:
    st.subheader("Forward Solver")
    st.write(
        "A metal rod starts hot in the middle and cold at both ends, which are "
        "held at 0°C. Heat spreads outward and the rod cools toward zero. The "
        "slider sets the thermal diffusivity α — how fast heat moves through "
        "the material. Higher α, faster the rod settles."
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
            colorscale="Inferno", zmin=0, zmax=1,
            colorbar=dict(title="u (temp)"),
        )
    )
    fig.update_layout(
        xaxis_title="position x",
        yaxis_title="time t",
        height=420,
        margin=dict(l=60, r=20, t=30, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"α = {alpha} · {len(t)} timesteps to t = {t_max} · "
        f"solved with explicit finite differences"
    )


    # --- PANEL B: validation, PINNED at α = 0.1 ---
    # evaluate PINN on the x-grid at t = 0.5
    # overlay against FD solve at α = 0.1
    # display rel_l2_vs_fd["t_0.5"] and speedup from benchmarks
    # st.plotly_chart(fig2)

with tab2:
    st.info("Parameter recovery results coming soon.")
    # later: read benchmark_inverse.json, render method × noise × error
