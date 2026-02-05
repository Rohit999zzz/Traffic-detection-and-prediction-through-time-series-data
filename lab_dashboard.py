
import streamlit as st
import time
import pandas as pd
import sys
from pathlib import Path
import altair as alt

# Add src to path to import modules
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from junction_manager import JunctionManager

# Page Config
st.set_page_config(
    page_title="Traffic Lab Dashboard",
    page_icon="🚦",
    layout="wide"
)

# Initialize Session State
if 'junction' not in st.session_state:
    st.session_state.junction = JunctionManager()

junction = st.session_state.junction

# ==============================================================================
# 1. Sidebar Controls (Configurability)
# ==============================================================================
st.sidebar.header("🔬 Lab Controls")

st.sidebar.subheader("Stochastic Parameters")
# Arrival Rates
for lane_id in junction.lanes:
    rate = st.sidebar.slider(
        f"Arrival Rate ({lane_id}) λ", 
        min_value=0.0, max_value=30.0, step=0.5,
        value=float(junction.lanes[lane_id].arrival_rate),
        help="Vehicles per minute (Poisson)"
    )
    junction.lanes[lane_id].arrival_rate = rate

st.sidebar.markdown("---")
st.sidebar.subheader("Logic Thresholds")
# Update Junction Config dynamically
min_green = st.sidebar.slider("Min Green Time (s)", 1.0, 15.0, junction.config["min_green_time"])
max_green = st.sidebar.slider("Max Green Time (s)", 10.0, 60.0, junction.config["max_green_time"])
queue_thresh = st.sidebar.slider("Queue Threshold", 5, 40, junction.config["queue_threshold"])
wait_thresh = st.sidebar.slider("Max Wait Time (s)", 20.0, 90.0, junction.config["max_wait_time"])

# Apply Config Updates
junction.config.update({
    "min_green_time": min_green,
    "max_green_time": max_green,
    "queue_threshold": queue_thresh,
    "max_wait_time": wait_thresh
})

st.sidebar.markdown("---")
simulation_speed = st.sidebar.slider("Simulation Speed", 0.5, 5.0, 1.0)
auto_refresh = st.sidebar.checkbox("Run Simulation", value=True)


# ==============================================================================
# 2. Main Dashboard Header
# ==============================================================================
st.title("🚦 Discrete Event Traffic Management System")
st.markdown(f"**Grid ID:** G-YEL-001 | **Algorithm:** Weighted Priority (Queue + Wait Time)")

# Metrics Row
col1, col2, col3, col4 = st.columns(4)

active_lane = junction.active_lane_id
active_color = "normal" if junction.lanes[active_lane].queue_length < 20 else "inverse"
green_duration = junction.sim_time - junction.last_switch_time

with col1:
    st.metric("Active Lane (Green)", active_lane, delta="Priority", delta_color=active_color)
with col2:
    st.metric("Green Timer", f"{green_duration:.1f}s")
with col3:
    total_q = sum(l.queue_length for l in junction.lanes.values())
    st.metric("Total Congestion", int(total_q), delta=f"{int(total_q/4)} avg/lane")
with col4:
    last_event = junction.events_log[-1]['event'] if junction.events_log else "None"
    st.metric("Last Discrete Event", last_event)


# ==============================================================================
# 3. Visualization (Charts & Stats)
# ==============================================================================

# Prepare Data
lane_data = []
for lid, lane in junction.lanes.items():
    stats = lane.process_traffic(0) # Logic hack: get stats without discharging? No, simulator handles discharge in step.
    # We access properties directly
    lane_data.append({
        "Lane": lid,
        "Queue": lane.queue_length,
        "Status": "Active" if lid == active_lane else "Waiting",
        "Wait Time": round(junction.sim_time - junction.lane_last_green_times[lid], 1) if lid != active_lane else 0,
        "Passed": lane.vehicles_passed
    })
df_lanes = pd.DataFrame(lane_data)

# Row 1: Queue Chart (Altair)
st.subheader("Real-time Queue Status")
chart = alt.Chart(df_lanes).mark_bar().encode(
    x=alt.X('Lane', axis=alt.Axis(labelAngle=0)),
    y='Queue',
    color=alt.Color('Status', scale=alt.Scale(domain=['Active', 'Waiting'], range=['#2ecc71', '#e74c3c'])),
    tooltip=['Lane', 'Queue', 'Wait Time']
).properties(height=300)

st.altair_chart(chart, use_container_width=True)

# Row 2: Throughput & Logic Log
c1, c2 = st.columns([1, 2])

with c1:
    st.subheader("Throughput Stats")
    st.dataframe(df_lanes[["Lane", "Passed", "Wait Time"]], hide_index=True)

with c2:
    st.subheader("System Event Log")
    if junction.events_log:
        log_df = pd.DataFrame(junction.events_log)
        st.dataframe(log_df.iloc[::-1].head(10), use_container_width=True) # Show latest 10
    else:
        st.info("No switching events triggered yet.")

# ==============================================================================
# 4. Simulation Loop (At End)
# ==============================================================================
if auto_refresh:
    junction.step(dt=1.0 * simulation_speed)
    time.sleep(1.0 / simulation_speed)
    st.rerun()
