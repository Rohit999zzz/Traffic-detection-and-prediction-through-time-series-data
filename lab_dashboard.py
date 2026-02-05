
import streamlit as st
import time
import pandas as pd
import sys
from pathlib import Path

# Add src to path to import modules
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from junction_manager import JunctionManager

# Page Config
st.set_page_config(
    page_title="Traffic Lab Dashboard",
    page_icon="🚦",
    layout="wide"
)

# Initialize Session State for Simulation
if 'junction' not in st.session_state:
    st.session_state.junction = JunctionManager()
    st.session_state.simulation_time = 0.0

junction = st.session_state.junction

# Sidebar Controls
st.sidebar.header("🔬 Lab Controls")
st.sidebar.markdown("Stochastic Parameters")

# Allow adjusting arrival rates on the fly
for lane_id in junction.lanes:
    rate = st.sidebar.slider(
        f"Arrival Rate ({lane_id}) λ", 
        min_value=0.0, 
        max_value=30.0, 
        value=float(junction.lanes[lane_id].arrival_rate),
        step=0.5,
        help="Vehicles per minute (Poisson)"
    )
    junction.lanes[lane_id].arrival_rate = rate

simulation_speed = st.sidebar.slider("Simulation Speed", 0.5, 5.0, 1.0)
auto_refresh = st.sidebar.checkbox("Run Simulation", value=True)

# Main Dashboard
st.title("🚦 Discrete Event Traffic Management System")
st.markdown(f"**Grid ID:** G-YEL-001 | **Location:** NMIT Junction | **Algorithm:** Queue-Based Adaptive Control")

# 1. Top Row: Metrics
col1, col2, col3, col4 = st.columns(4)

active_lane = junction.active_lane_id
active_color = "normal" if junction.lanes[active_lane].queue_length < 20 else "inverse"

with col1:
    st.metric("Active Lane (Green)", active_lane, delta="Priority", delta_color=active_color)
with col2:
    st.metric("Green Timer", f"{time.time() - junction.last_switch_time:.1f}s")
with col3:
    total_q = sum(l.queue_length for l in junction.lanes.values())
    st.metric("Total Congestion", int(total_q), delta=f"{int(total_q/4)} avg/lane")
with col4:
    last_event = junction.events_log[-1]['event'] if junction.events_log else "None"
    st.metric("Last Discrete Event", last_event)

# 2. Main Visualization: Lane Queues
st.subheader("real-time Traffic Queue Status (Stochastic + Discrete Event)")

# Step Simulation (if running)
# Simulation logic moved to end of file

# Data for Chart
chart_data = []
for lane_id, lane in junction.lanes.items():
    chart_data.append({
        "Lane": lane_id,
        "Queue Length": lane.queue_length,
        "Status": "Green" if lane_id == active_lane else "Red",
        "Color": "#00CC00" if lane_id == active_lane else "#FF0000"
    })

df_chart = pd.DataFrame(chart_data)

# Custom Bar Chart using columns for custom styling
cols = st.columns(4)
for i, (index, row) in enumerate(df_chart.iterrows()):
    with cols[i]:
        st.markdown(f"### {row['Lane']}")
        if row['Status'] == 'Green':
            st.success(f"🟢 ACTIVE ({int(row['Queue Length'])})")
        else:
            st.error(f"🔴 WAITING ({int(row['Queue Length'])})")
        st.progress(min(row['Queue Length'] / 50.0, 1.0))

# 3. Discrete Event Log
st.subheader("📜 System Event Log (Discrete Event Logic)")
st.markdown("Logs state transitions triggered by queue thresholds.")

if junction.events_log:
    log_df = pd.DataFrame(junction.events_log)
    st.dataframe(log_df.iloc[::-1], use_container_width=True) # Show latest first
else:
    st.info("No switching events triggered yet.")

# 4. Stochastic Analysis View
st.subheader("📈 Stochastic Distribution Analysis")
st.markdown("Real-time view of Poisson arrival process.")
raw_json = {l: int(junction.lanes[l].vehicles_passed) for l in junction.lanes}
st.json(raw_json)

# Footer
st.markdown("---")
st.markdown("*Project: Grid-Based Intelligent Traffic Management | Dept: ISE & Planning, NMIT*")

# Step Simulation (if running) - Moved to end to ensure UI updates first
if auto_refresh:
    step_result = junction.step(dt=1.0 * simulation_speed)
    time.sleep(1.0 / simulation_speed) # Control visual update rate
    st.rerun()
