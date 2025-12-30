import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path

# ================= 1. 核心算法 (完全复用) =================
class MesopicModel:
    def __init__(self, xp, yp, Lp, Ls, m):
        self.xp, self.yp = xp, yp
        self.Lp, self.Ls = Lp, Ls
        self.m = m

    def calculate(self):
        m, Lp, Ls, xp, yp = self.m, self.Lp, self.Ls, self.xp, self.yp
        safe_yp = max(yp, 0.0001)
        ratio_K = 683.0 / 1699.0
        
        num_L = m * Lp + (1 - m) * Ls * ratio_K
        den_L = m + (1 - m) * ratio_K
        if den_L == 0: den_L = 0.0001
        Lmes = num_L / den_L
        
        Lpa, Lsa = Lmes * m, Lmes * (1 - m)
        term_p, term_s = Lpa / safe_yp, Lsa / 0.3333
        denom = term_p + term_s
        if denom == 0: denom = 0.0001
        
        xm = (Lpa * xp / safe_yp + Lsa) / denom
        ym = (Lpa + Lsa) / denom
        return xm, ym, Lmes

    def xyY_to_rgb(self, x, y, Y):
        if y <= 1e-5: return '#000000'
        X, Z = (x * Y) / y, ((1 - x - y) * Y) / y
        r =  3.2406*X - 1.5372*Y - 0.4986*Z
        g = -0.9689*X + 1.8758*Y + 0.0415*Z
        b =  0.0557*X - 0.2040*Y + 1.0570*Z
        rgb = [max(0, c)**(1/2.2) for c in (r, g, b)]
        m_val = max(rgb)
        rgb = [c/m_val if m_val > 1 else min(1, c) for c in rgb]
        return '#{:02x}{:02x}{:02x}'.format(*(int(c*255) for c in rgb))

# ================= 2. CIE 背景生成 (缓存加速) =================
@st.cache_data # 关键：缓存背景生成结果，防止每次拖动滑块都重算，提升性能
def get_cie_background():
    spectral_locus = np.array([
        [0.1741, 0.0050], [0.1740, 0.0050], [0.1738, 0.0049], [0.1736, 0.0049], [0.1733, 0.0048],
        [0.1730, 0.0048], [0.1726, 0.0048], [0.1721, 0.0048], [0.1714, 0.0051], [0.1703, 0.0058],
        [0.1689, 0.0069], [0.1669, 0.0086], [0.1644, 0.0109], [0.1611, 0.0138], [0.1566, 0.0177],
        [0.1510, 0.0227], [0.1440, 0.0297], [0.1355, 0.0399], [0.1241, 0.0578], [0.1096, 0.0868],
        [0.0913, 0.1327], [0.0687, 0.2007], [0.0454, 0.2950], [0.0235, 0.4127], [0.0082, 0.5384],
        [0.0039, 0.6548], [0.0139, 0.7502], [0.0389, 0.8120], [0.0743, 0.8338], [0.1142, 0.8262],
        [0.1547, 0.8059], [0.1929, 0.7816], [0.2296, 0.7543], [0.2658, 0.7243], [0.3016, 0.6923],
        [0.3373, 0.6589], [0.3731, 0.6245], [0.4087, 0.5896], [0.4441, 0.5547], [0.4788, 0.5202],
        [0.5125, 0.4866], [0.5448, 0.4544], [0.5752, 0.4242], [0.6029, 0.3965], [0.6270, 0.3725],
        [0.6482, 0.3514], [0.6658, 0.3340], [0.6801, 0.3197], [0.6915, 0.3083], [0.7006, 0.2993],
        [0.7079, 0.2920], [0.7140, 0.2859], [0.7190, 0.2809], [0.7230, 0.2770], [0.7260, 0.2740],
        [0.7283, 0.2717], [0.7300, 0.2700], [0.7311, 0.2689], [0.7320, 0.2680], [0.7327, 0.2673],
        [0.7334, 0.2666], [0.7340, 0.2660], [0.7344, 0.2656], [0.7346, 0.2654], [0.7347, 0.2653]
    ])
    spectral_locus = np.vstack([spectral_locus, spectral_locus[0]])
    
    resolution = 800
    x = np.linspace(0, 0.8, resolution)
    y = np.linspace(0, 0.9, resolution)
    xx, yy = np.meshgrid(x, y)
    
    path = Path(spectral_locus)
    mask = path.contains_points(np.vstack((xx.flatten(), yy.flatten())).T).reshape(resolution, resolution)
    
    X, Y, Z = xx, yy, 1 - xx - yy
    R =  3.2406 * X - 1.5372 * Y - 0.4986 * Z
    G = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    B =  0.0557 * X - 0.2040 * Y + 1.0570 * Z
    
    RGB = np.clip(np.dstack((R, G, B)), 0, 1) ** (1/2.2)
    RGBA = np.zeros((resolution, resolution, 4))
    RGBA[..., 0:3] = RGB
    RGBA[..., 3] = mask.astype(float)
    
    return RGBA, spectral_locus

# ================= 3. 网页布局配置 =================
st.set_page_config(page_title="Mesopic Predictor", layout="wide")

# CSS 样式注入 (美化色块显示)
st.markdown("""
<style>
    .color-box {
        width: 100%;
        height: 80px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin-bottom: 10px;
    }
    .metric-label {
        font-size: 14px;
        color: #555;
        text-align: center;
    }
    .main-header {
        font-family: 'Segoe UI', sans-serif;
        font-weight: bold;
        color: #111827;
    }
</style>
""", unsafe_allow_html=True)

# ================= 4. 主程序逻辑 =================

st.title("Mesopic Vision Predictor")
st.markdown("Use the **sidebar** to adjust parameters.")

# --- 侧边栏 (Sidebar) ---
with st.sidebar:
    st.header("Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        xp = st.number_input("x (Photopic)", value=0.45, step=0.01, format="%.3f")
        Lp = st.number_input("Lp (cd/m²)", value=3.0, step=0.1)
    with col2:
        yp = st.number_input("y (Photopic)", value=0.40, step=0.01, format="%.3f")
        Ls = st.number_input("Ls (cd/m²)", value=1.0, step=0.1)
        
    st.divider()
    
    st.subheader("Adaptation State")
    m = st.slider("m Value", 0.0, 1.0, 1.0, 0.01)
    
    if m >= 0.8:
        st.caption("State: ☀️ Photopic (Day)")
    elif m <= 0.2:
        st.caption("State: 🌙 Scotopic (Night)")
    else:
        st.caption("State: 🌅 Mesopic (Dusk)")

# --- 计算逻辑 ---
model = MesopicModel(xp, yp, Lp, Ls, m)
xm, ym, Lmes = model.calculate()
hex_p = model.xyY_to_rgb(xp, yp, Lp)
hex_m = model.xyY_to_rgb(xm, ym, Lmes)

# --- 主界面布局 (两列) ---
col_charts, col_preview = st.columns([3, 1])

with col_charts:
    # 生成/获取背景
    cie_img, locus = get_cie_background()
    
    # Matplotlib 绘图
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 绘制背景
    ax.imshow(cie_img, origin='lower', extent=[0, 0.8, 0, 0.9], interpolation='bicubic', zorder=0)
    ax.plot(locus[:, 0], locus[:, 1], 'k-', linewidth=1.5, zorder=1)
    
    # 绘制连线
    ax.plot([xp, xm], [yp, ym], 'k--', lw=1.5, alpha=0.6, zorder=5)
    
    # 绘制点
    ax.plot(xp, yp, 'o', ms=12, mfc='#2563EB', mec='white', mew=2, label='Photopic', zorder=10)
    ax.plot(xm, ym, '^', ms=12, mfc='#F59E0B', mec='white', mew=2, label='Mesopic', zorder=10)
    
    # 样式
    ax.set_title(f"CIE 1931 Chromaticity Shift (m={m:.2f})", fontweight='bold')
    ax.set_xlim(-0.05, 0.8)
    ax.set_ylim(-0.05, 0.9)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(loc='upper right', frameon=False)
    
    # 隐藏边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    st.pyplot(fig)

with col_preview:
    st.subheader("Preview")
    
    # Photopic 预览
    st.markdown(f'<div class="color-box" style="background-color: {hex_p};"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-label">
        <b>Original</b><br>
        L = {Lp:.1f}<br>
        ({xp:.3f}, {yp:.3f})
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Mesopic 预览
    st.markdown(f'<div class="color-box" style="background-color: {hex_m};"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-label">
        <b>Predicted</b><br>
        L = {Lmes:.1f}<br>
        ({xm:.3f}, {ym:.3f})
    </div>
    """, unsafe_allow_html=True)

# --- 数据表格 ---
st.divider()
st.subheader("Data Table")
st.dataframe({
    "Parameter": ["x", "y", "Luminance (cd/m²)"],
    "Photopic (Input)": [f"{xp:.4f}", f"{yp:.4f}", f"{Lp:.2f}"],
    "Mesopic (Output)": [f"{xm:.4f}", f"{ym:.4f}", f"{Lmes:.2f}"]
}, use_container_width=True)