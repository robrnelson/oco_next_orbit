import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Session State & Reset Logic ---
# Initialize session state for the sliders if they don't exist
if 'swath_val' not in st.session_state:
    st.session_state.swath_val = 2826.0
if 'pixel_val' not in st.session_state:
    st.session_state.pixel_val = 1024
if 'alt_val' not in st.session_state:
    st.session_state.alt_val = 828.0

def reset_defaults():
    st.session_state.swath_val = 2826.0
    st.session_state.pixel_val = 1024
    st.session_state.alt_val = 828.0

# --- 2. Calculation Core ---
def calculate_metrics(target_swath, num_pixels, altitude):
    h = altitude
    Re = 6378.137   # Earth Radius km
    K = (Re + h) / Re
    
    # Check Horizon Limit
    # The max swath visible from this altitude
    gamma_max = np.arccos(1/K)
    max_swath = 2 * Re * gamma_max
    
    if target_swath >= max_swath:
        return None, f"Error: At {h} km, the horizon is at {max_swath:.0f} km. You cannot see {target_swath} km."

    # Solve for FOV from Swath
    gamma = (target_swath / 2.0) / Re
    num = np.sin(gamma)
    denom = K - np.cos(gamma)
    theta_rad = np.arctan(num / denom) # Half-angle
    fov_rad = 2 * theta_rad
    fov_deg = np.degrees(fov_rad)
    
    # Resolutions
    beta = fov_rad / num_pixels # IFOV (radians per pixel)
    w_nadir = h * beta
    
    # Edge Resolution (Curved Earth)
    # ds/dtheta = Re * [ (K cos(theta)) / sqrt(1 - K^2 sin^2(theta)) - 1 ]
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

# --- 3. Streamlit UI ---
st.set_page_config(page_title="Sat Design Tool", layout="wide")
st.title("🛰️ Satellite Swath Designer")

# Sidebar
st.sidebar.header("Inputs")

# Sliders linked to session state
target_swath = st.sidebar.slider(
    "Target Swath Width (km)", 
    min_value=100.0, max_value=5000.0, step=10.0,
    key='swath_val' 
)

altitude = st.sidebar.slider(
    "Altitude (km)",
    min_value=300.0, max_value=2000.0, step=10.0,
    key='alt_val'
)

num_pixels = st.sidebar.slider(
    "Number of Pixels", 
    min_value=100, max_value=5000, step=16,
    key='pixel_val'
)

st.sidebar.markdown("---")
st.sidebar.button("↺ Reset Defaults", on_click=reset_defaults)

# Run Calc
data, error = calculate_metrics(target_swath, num_pixels, altitude)

if error:
    st.error(error)
else:
    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Required FOV", f"{data['fov']:.2f}°")
    c2.metric("Nadir Pixel", f"{data['w_nadir']:.3f} km")
    c3.metric("Edge Pixel", f"{data['w_edge']:.2f} km")
    c4.metric("Distortion", f"{data['w_edge']/data['w_nadir']:.1f}x", 
              delta="Bow-tie effect", delta_color="off")

    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Summary")
        st.write(f"**Altitude:** {data['h']} km")
        st.write(f"**Swath:** {target_swath} km")
        st.write(f"**Pixels:** {num_pixels}")
        st.write(f"**IFOV:** {data['beta']*1000:.4f} mrad")
        st.caption("Green line assumes Flat Earth. Red line accounts for Real Earth curvature.")

    with col2:
        st.subheader("Curved vs. Flat Earth Distortion")
        
        # Prepare Plot Data
        angles = np.linspace(-data['theta_rad'], data['theta_rad'], 100)
        Re = 6378.137
        h = data['h']
        K = (Re + h) / Re
        beta = data['beta']
        
        curved_sizes = []
        flat_sizes = []
        
        for theta in angles:
            theta_abs = np.abs(theta)
            
            # 1. Curved Earth Calc
            num = K * np.cos(theta_abs)
            sin_sq = (K * np.sin(theta_abs))**2
            if sin_sq >= 1: sin_sq = 0.9999
            denom = np.sqrt(1 - sin_sq)
            ds_curved = Re * ((num/denom) - 1)
            curved_sizes.append(ds_curved * beta)
            
            # 2. Flat Earth Calc (Slant Range / cos(theta))
            # Pixel = (h / cos^2(theta)) * beta
            ds_flat = h / (np.cos(theta_abs)**2)
            flat_sizes.append(ds_flat * beta)

        angles_deg = np.degrees(angles)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Curved Line
        ax.plot(angles_deg, curved_sizes, color='#FF4B4B', linewidth=2.5, label='Real Earth')
        ax.fill_between(angles_deg, curved_sizes, color='#FF4B4B', alpha=0.1)
        
        # Flat Line
        ax.plot(angles_deg, flat_sizes, color='#28a745', linewidth=2.5, linestyle='--', label='Flat Earth')
        
        ax.set_xlabel("Scan Angle (degrees)")
        ax.set_ylabel("Pixel Width (km)")
        ax.set_xlim(min(angles_deg), max(angles_deg))
        
        # Dynamic Y-limit handling
        max_y = max(curved_sizes)
        if max_y > 50: 
            ax.set_ylim(0, 50) # Cap it if it explodes to infinity near horizon
        else:
            ax.set_ylim(0, max_y*1.15)
            
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        
        st.pyplot(fig)
