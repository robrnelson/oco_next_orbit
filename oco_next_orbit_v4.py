import streamlit as st
import ephem
import numpy as np
import math
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- 1. Physics & TLE Handling ---
def get_altitude_from_tle(line1, line2):
    """
    Extracts the mean altitude from a TLE by calculating the semi-major axis 
    from the Mean Motion.
    """
    # Parse Mean Motion (revs per day) from columns 52-63 of Line 2
    mean_motion_revs_day = float(line2[52:63])
    
    # Constants
    mu = 398600.4418  # Earth Gravitational Parameter km^3/s^2
    Re = 6378.137     # Earth Radius km
    
    # Convert Mean Motion to rad/s
    n_rad_s = mean_motion_revs_day * (2 * np.pi / 86400.0)
    
    # Kepler's 3rd Law: n^2 = mu / a^3  =>  a = (mu / n^2)^(1/3)
    semi_major_axis = (mu / n_rad_s**2)**(1/3)
    
    # Altitude = Semi-Major Axis - Earth Radius
    altitude = semi_major_axis - Re
    return altitude

def propagate_orbit_tle(line1, line2, duration_hours=24, steps_per_orbit=60):
    """
    Uses PyEphem to propagate the orbit from a specific TLE.
    """
    sat = ephem.readtle("Sat", line1, line2)
    
    # Start time (Now)
    start_date = datetime.utcnow()
    
    lats = []
    lons = []
    
    # Calculate step size based on orbital period
    # Mean motion is in revs/day
    mm = float(line2[52:63])
    period_min = (24 * 60) / mm
    step_seconds = (period_min * 60) / steps_per_orbit
    
    total_steps = int((duration_hours * 3600) / step_seconds)
    
    for i in range(total_steps):
        t = start_date + timedelta(seconds=i*step_seconds)
        sat.compute(t)
        
        # Ephem returns radians
        lats.append(np.degrees(sat.sublat))
        lons.append(np.degrees(sat.sublong))
        
    return lats, lons

def calculate_swath_edges(lats, lons, swath_km):
    """
    Calculates Left/Right edge coordinates using spherical trigonometry.
    """
    R = 6378.137
    half_swath = swath_km / 2.0
    angular_dist = half_swath / R
    
    left_lats, left_lons = [], []
    right_lats, right_lons = [], []
    
    for i in range(len(lats)-1):
        lat1 = np.radians(lats[i])
        lon1 = np.radians(lons[i])
        lat2 = np.radians(lats[i+1])
        lon2 = np.radians(lons[i+1])
        
        # Bearing
        d_lon = lon2 - lon1
        y = np.sin(d_lon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lon)
        bearing = np.arctan2(y, x)
        
        for angle_offset, lat_list, lon_list in [(-np.pi/2, left_lats, left_lons), (np.pi/2, right_lats, right_lons)]:
            theta = bearing + angle_offset
            
            lat_new = np.arcsin(np.sin(lat1)*np.cos(angular_dist) + 
                                np.cos(lat1)*np.sin(angular_dist)*np.cos(theta))
            lon_new = lon1 + np.arctan2(np.sin(theta)*np.sin(angular_dist)*np.cos(lat1),
                                        np.cos(angular_dist)-np.sin(lat1)*np.sin(lat_new))
            
            lat_list.append(np.degrees(lat_new))
            lon_list.append(np.degrees(lon_new))
            
    return left_lats, left_lons, right_lats, right_lons


# --- 2. Session State ---
if 'swath_val' not in st.session_state:
    st.session_state.swath_val = 2826.0
if 'pixel_val' not in st.session_state:
    st.session_state.pixel_val = 1024

def reset_defaults():
    st.session_state.swath_val = 2826.0
    st.session_state.pixel_val = 1024

# --- 3. Calculation Core ---
def calculate_metrics(target_swath, num_pixels, altitude):
    h = altitude
    Re = 6378.137
    K = (Re + h) / Re
    gamma_max = np.arccos(1/K)
    max_swath = 2 * Re * gamma_max
    
    if target_swath >= max_swath:
        return None, f"Error: Horizon limit is {max_swath:.0f} km."

    gamma = (target_swath / 2.0) / Re
    num = np.sin(gamma)
    denom = K - np.cos(gamma)
    theta_rad = np.arctan(num / denom)
    fov_deg = np.degrees(2 * theta_rad)
    beta = (2 * theta_rad) / num_pixels
    w_nadir = h * beta
    
    num_edge = K * np.cos(theta_rad)
    sin_sq = (K * np.sin(theta_rad))**2
    if sin_sq >= 1.0: sin_sq = 0.999999
    denom_edge = np.sqrt(1 - sin_sq)
    ds_dtheta_edge = Re * ((num_edge / denom_edge) - 1)
    w_edge = ds_dtheta_edge * beta
    
    return {
        "fov": fov_deg,
        "w_nadir": w_nadir,
        "w_edge": w_edge,
        "beta": beta,
        "h": h,
        "theta_rad": theta_rad
    }, None

# --- 4. Streamlit UI ---
st.set_page_config(page_title="Orbit Designer", layout="wide")
st.title("🛰️ TLE Orbit & Swath Designer")

# --- Sidebar ---
st.sidebar.header("Mission Parameters")

# TLE INPUT
st.sidebar.subheader("Satellite TLE")
default_tle_1 = "1 43013U 17073A   22146.79629330  .00000059  00000-0  48737-4 0  9990"
default_tle_2 = "2 43013  98.7159  85.6898 0001514  97.0846 263.0503 14.19554052234151"

tle_line1 = st.sidebar.text_input("Line 1", value=default_tle_1)
tle_line2 = st.sidebar.text_input("Line 2", value=default_tle_2)

# Calculate Altitude from TLE
try:
    derived_altitude = get_altitude_from_tle(tle_line1, tle_line2)
    st.sidebar.success(f"Orbit Altitude: **{derived_altitude:.1f} km**")
except Exception as e:
    st.sidebar.error("Invalid TLE")
    derived_altitude = 828.0 # Fallback

# Other Sliders
target_swath = st.sidebar.slider("Swath Width (km)", 100.0, 3000.0, step=10.0, key='swath_val')
num_pixels = st.sidebar.slider("Sensor Pixels", 100, 5000, step=16, key='pixel_val')

st.sidebar.markdown("---")
st.sidebar.button("↺ Reset Defaults", on_click=reset_defaults)


# --- Run Calculations ---
data, error = calculate_metrics(target_swath, num_pixels, derived_altitude)

if error:
    st.error(error)
else:
    # --- Metrics Section ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Required FOV", f"{data['fov']:.2f}°")
    c2.metric("Nadir Pixel", f"{data['w_nadir']:.3f} km")
    c3.metric("Edge Pixel", f"{data['w_edge']:.2f} km")
    c4.metric("Distortion", f"{data['w_edge']/data['w_nadir']:.1f}x")
    
    st.markdown("---")
    
    # --- TABS ---
    tab1, tab2 = st.tabs(["📊 Pixel Analysis", "🌍 3D Orbit Visualizer"])
    
    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info("Analysis of pixel growth from Nadir to Swath Edge.")
            st.write(f"**Calculated Altitude:** {derived_altitude:.1f} km")
            st.write(f"**Target Swath:** {target_swath} km")
            st.caption("Green: Flat Earth Model\nRed: Real Curved Earth")
            
        with col2:
            angles = np.linspace(-data['theta_rad'], data['theta_rad'], 100)
            Re = 6378.137
            K = (Re + data['h']) / Re
            beta = data['beta']
            curved, flat = [], []
            for theta in angles:
                theta_abs = np.abs(theta)
                num = K * np.cos(theta_abs)
                denom = np.sqrt(1 - min((K * np.sin(theta_abs))**2, 0.9999))
                curved.append(Re * ((num/denom)-1) * beta)
                flat.append((data['h'] / np.cos(theta_abs)**2) * beta)
                
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(np.degrees(angles), curved, color='#FF4B4B', lw=2.5, label='Real Earth')
            ax.fill_between(np.degrees(angles), curved, color='#FF4B4B', alpha=0.1)
            ax.plot(np.degrees(angles), flat, color='#28a745', lw=2.5, ls='--', label='Flat Earth')
            ax.set_xlim(np.degrees(min(angles)), np.degrees(max(angles)))
            ax.set_ylim(0, max(curved)*1.15)
            ax.grid(True, ls='--', alpha=0.5)
            ax.legend()
            st.pyplot(fig)

    with tab2:
        st.subheader("Real-Time Ground Track (24h)")
        st.write("Visualizing orbit from provided TLE (NOAA-20 / JPSS-1).")
        
        with st.spinner("Propagating TLE..."):
            # 1. Propagate Center Track
            lats, lons = propagate_orbit_tle(tle_line1, tle_line2, duration_hours=24)
            
            # 2. Calculate Swath Edges
            lats_L, lons_L, lats_R, lons_R = calculate_swath_edges(lats, lons, target_swath)
            
            # 3. Plotly Globe
            fig3d = go.Figure()
            
            # Center Track
            fig3d.add_trace(go.Scattergeo(
                lon = lons, lat = lats,
                mode = 'lines',
                line = dict(width=1, color='white', dash='dot'),
                name = 'Nadir Track',
                opacity = 0.5
            ))
            
            # Left Edge
            fig3d.add_trace(go.Scattergeo(
                lon = lons_L, lat = lats_L,
                mode = 'lines',
                line = dict(width=1, color='cyan'),
                name = 'Left Edge',
                opacity = 0.8
            ))
            
            # Right Edge
            fig3d.add_trace(go.Scattergeo(
                lon = lons_R, lat = lats_R,
                mode = 'lines',
                line = dict(width=1, color='cyan'),
                name = 'Right Edge',
                opacity = 0.8
            ))

            fig3d.update_geos(
                projection_type="orthographic",
                showcoastlines=True, coastlinecolor="RebeccaPurple",
                showland=True, landcolor="rgb(20, 20, 40)",
                showocean=True, oceancolor="rgb(10, 10, 20)",
                projection_rotation=dict(lon=-100, lat=40, roll=0)
            )
            
            fig3d.update_layout(
                height=600,
                margin={"r":0,"t":0,"l":0,"b":0},
                paper_bgcolor="black",
                font_color="white",
                legend=dict(y=0.9, x=0.8)
            )
            
            st.plotly_chart(fig3d, use_container_width=True)

