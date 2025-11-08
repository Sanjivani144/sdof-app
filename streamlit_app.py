# streamlit_app.py (clean version for local/Streamlit Cloud)
import io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SDOF Damping Explorer", layout="wide")

def sdof_response(zeta, m, k, x0, v0, t):
    wn = np.sqrt(k / m)
    if zeta < 1:
        wd = wn * np.sqrt(1 - zeta**2)
        x = np.exp(-zeta * wn * t) * (x0 * np.cos(wd * t) +
                                      (v0 + zeta * wn * x0) / wd * np.sin(wd * t))
        label = "Underdamped"
    elif np.isclose(zeta, 1.0):
        x = (x0 + (v0 + wn * x0) * t) * np.exp(-wn * t)
        label = "Critically Damped"
    else:
        r1 = -wn * (zeta - np.sqrt(zeta**2 - 1))
        r2 = -wn * (zeta + np.sqrt(zeta**2 - 1))
        A = (v0 - r2 * x0) / (r1 - r2)
        B = x0 - A
        x = A * np.exp(r1 * t) + B * np.exp(r2 * t)
        label = "Overdamped"
    return x, label, wn

def create_figure(t, responses, title="SDOF Response"):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for resp in responses:
        ax.plot(t, resp["x"], linewidth=2, label=f"{resp['label']} (ζ={resp['zeta']})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (m)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")
    plt.tight_layout()
    return fig

# --- UI ---
st.title("SDOF Damping Explorer")

with st.sidebar:
    m = st.number_input("Mass (kg)", value=1.0)
    k = st.number_input("Stiffness (N/m)", value=100.0)
    x0 = st.number_input("Initial displacement (m)", value=1.0)
    v0 = st.number_input("Initial velocity (m/s)", value=0.0)
    t_max = st.number_input("Time span (s)", value=5.0)
    n_points = st.slider("Time resolution", 200, 5000, 1000)
    show_all = st.checkbox("Show standard cases (ζ=0.1,1,2)", True)
    if not show_all:
        zeta = st.slider("Damping ratio ζ", 0.0, 5.0, 0.1, 0.01)

t = np.linspace(0, t_max, int(n_points))
responses = []
if show_all:
    for z in (0.1, 1.0, 2.0):
        x, label, wn = sdof_response(z, m, k, x0, v0, t)
        responses.append({"zeta": z, "label": label, "x": x})
else:
    x, label, wn = sdof_response(zeta, m, k, x0, v0, t)
    responses.append({"zeta": zeta, "label": label, "x": x})

fig = create_figure(t, responses, title="SDOF System Response")
st.pyplot(fig)

df = pd.DataFrame({"time_s": t})
for resp in responses:
    df[f"x_zeta_{resp['zeta']}"] = resp["x"]
st.download_button("Download CSV", df.to_csv(index=False), "sdof_responses.csv", "text/csv")
