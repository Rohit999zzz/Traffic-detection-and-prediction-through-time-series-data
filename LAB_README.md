# Grid-Based Intelligent Traffic Management System (Lab Manual)

A comprehensive system combining **Real-Time Computer Vision**, **Stochastic Simulation**, and **Discrete Event Control** for intelligent traffic management.

## 🚀 Lab Demo (Current Status)

The current lab deliverables focus on the **Simulation & Control** aspect, featuring a real-time dashboard with stochastic traffic generation.

### Quick Start
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Dashboard**:
   ```bash
   streamlit run lab_dashboard.py
   ```
3. **Control the System**:
   - Open your browser to the local URL (usually `http://localhost:8501`).
   - Use the **Sidebar** to adjust arrival rates ($\lambda$), signal timing limits, and starvation thresholds.
   - Watch the **Discrete Events** trigger real-time signal shifting.

---

## 🧠 Key Modules

### 1. Stochastic Traffic Simulator (`src/traffic_simulator.py`)
Generates realistic traffic data to test signal logic without needing 24/7 video feeds.
- **Poisson Distribution**: Vehicle arrivals follow a randomized Poisson process ($\lambda$) to mimic real-world unpredictability.
- **Rich Vehicle Objects**: Simulates specific vehicle types (Car, Bike, Bus, Truck) with variable discharge rates.
    - *Bikes*: Fast discharge (1.0s)
    - *Trucks*: Slow discharge (4.0s)
- **Deterministic Replay**: Uses a simulation clock (`sim_time`) for reproducible experiments.

### 2. Discrete Event Junction Manager (`src/junction_manager.py`)
Intelligent "Brain" controlling the traffic lights. It does **NOT** use simple timers.
- **Event-Driven**: Shifts signals based on logical events (e.g., `QUEUE_THRESHOLD_EXCEEDED`, `LANE_EMPTY`).
- **Weighted Priority Algorithm**:
  $$ \text{Score} = \text{Queue Length} + (\alpha \times \text{Wait Time}) $$
  - Balances **Efficiency** (clearing long queues) with **Fairness** (preventing starvation).
- **Starvation Protection**: Automatically forces Green if a lane waits > `max_wait_time` (e.g., 45s).

### 3. Real-Time Dashboard (`lab_dashboard.py`)
Interactive visualization built with **Streamlit**.
- **Real-time Queue Visualization**: Altair charts showing active/waiting queues.
- **Control Panel**: Adjust logical parameters (Thresholds, Timers) on the fly.
- **Event Log**: Live table of all state transitions and decision logic.

---

## 📂 Project Structure (Lab Components)

```text
traffic-analysis/
├── lab_dashboard.py            # [GUI] Real-time Simulation Dashboard
├── main.py                     # [CLI] Video Processing Pipeline
├── src/
│   ├── junction_manager.py     # [Logic] Discrete Event Control System
│   ├── traffic_simulator.py    # [Sim] Stochastic Data Generator
│   ├── vehicle_classifier.py   # [AI] YOLO Vehicle Detection
│   ├── density_calculator.py   # [Math] Weighted Density Logic
│   └── time_series_generator.py # [Data] Aggregation for LSTM
├── data/
│   ├── input/                  # Raw Video Files
│   └── output/                 # Processed CSVs/JSONs
└── requirements.txt            # Python Dependencies
```

## 🔬 Experiment Guide

### Experiment 1: Starvation Protection "Round Robin"
1. Set **Arrival Rate (North)** to `30.0` (Max).
2. Set **Arrival Rate (South)** to `2.0` (Low).
3. Set **Max Wait Time** to `10s`.
4. **Observation**: Even though North has a huge queue, South will force a switch every 10 seconds because its "Wait Score" becomes too high.

### Experiment 2: Empty Lane Optimization
1. Set **Arrival Rate** to `0.0` for the Active Lane.
2. Watch the queue drop to 0.
3. **Observation**: The system triggers `LANE_EMPTY` event and switches immediately, saving time.

### Experiment 3: Truck Congestion
1. Theoretically modify `traffic_simulator.py` to only spawn Trucks.
2. **Observation**: The queue decreases much slower because Trucks take 4.0s to clear vs 2.0s for Cars.
