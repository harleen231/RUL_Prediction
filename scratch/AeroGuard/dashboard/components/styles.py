"""
AeroGuard Dashboard Aerospace Aesthetic Styling Module
Defines modern dark navy / charcoal CSS themes, responsive cards,
glow indicators, and telemetry typography.
"""

AEROGUARD_CSS = """
<style>
/* Main Dark Navy / Space-Age Theme */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #E2E8F0;
}

.stApp {
    background-color: #070B14;
    background-image: 
        radial-gradient(at 0% 0%, rgba(14, 165, 233, 0.08) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(6, 182, 212, 0.05) 0px, transparent 50%);
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #0C1222;
    border-right: 1px solid rgba(56, 189, 248, 0.15);
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    color: #F8FAFC !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}

/* Monospace for cycle & telemetry data */
.telemetry-val {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
}

/* Metric / KPI Card */
.aeroguard-kpi-card {
    background: linear-gradient(135deg, rgba(19, 28, 46, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 14px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.aeroguard-kpi-card:hover {
    border-color: rgba(56, 189, 248, 0.5);
    transform: translateY(-2px);
}

.kpi-title {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94A3B8;
    margin-bottom: 4px;
}

.kpi-value {
    font-size: 1.85rem;
    font-weight: 700;
    color: #38BDF8;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.2;
}

.kpi-subtitle {
    font-size: 0.75rem;
    color: #64748B;
    margin-top: 4px;
}

/* Status Badges */
.badge-healthy {
    background-color: rgba(16, 185, 129, 0.15);
    color: #10B981;
    border: 1px solid #10B981;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
    display: inline-block;
}

.badge-attention {
    background-color: rgba(245, 158, 11, 0.15);
    color: #F59E0B;
    border: 1px solid #F59E0B;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
    display: inline-block;
}

.badge-warning {
    background-color: rgba(249, 115, 22, 0.15);
    color: #F97316;
    border: 1px solid #F97316;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
    display: inline-block;
}

.badge-critical {
    background-color: rgba(239, 68, 68, 0.2);
    color: #EF4444;
    border: 1px solid #EF4444;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
    display: inline-block;
    animation: pulse-glow 2s infinite;
}

@keyframes pulse-glow {
    0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.5); }
    70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
    100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
}

/* Hazard Alert Box */
.hazard-box {
    background: linear-gradient(90deg, rgba(239, 68, 68, 0.15) 0%, rgba(15, 23, 42, 0.6) 100%);
    border-left: 4px solid #EF4444;
    border-top: 1px solid rgba(239, 68, 68, 0.3);
    border-right: 1px solid rgba(239, 68, 68, 0.3);
    border-bottom: 1px solid rgba(239, 68, 68, 0.3);
    padding: 12px 18px;
    border-radius: 0 8px 8px 0;
    margin: 12px 0;
}

/* Tabs styling */
button[data-baseweb="tab"] {
    background-color: transparent !important;
    color: #94A3B8 !important;
    font-weight: 600 !important;
    border-bottom: 2px solid transparent !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #38BDF8 !important;
    border-bottom: 2px solid #38BDF8 !important;
}

/* Table dark theme */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(56, 189, 248, 0.15);
    border-radius: 8px;
    overflow: hidden;
}
</style>
"""


def render_kpi_card(title: str, value: str, subtitle: str = "", color: str = "#38BDF8") -> str:
    """Returns HTML for an aerospace-styled KPI metric card."""
    return f"""
    <div class="aeroguard-kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value" style="color: {color};">{value}</div>
        <div class="kpi-subtitle">{subtitle}</div>
    </div>
    """


def render_status_badge(category: str) -> str:
    """Returns HTML for a glowing category pill badge."""
    cat_lower = category.lower()
    return f'<span class="badge-{cat_lower}">{category}</span>'


def render_hazard_alert(message: str) -> str:
    """Returns HTML for a high-risk overestimation alert banner."""
    return f"""
    <div class="hazard-box">
        <div style="font-weight: 700; color: #EF4444; font-size: 0.95rem; margin-bottom: 4px;">
            ⚠️ AEROSPACE SAFETY CAUTION: HIGH OVERESTIMATION RISK
        </div>
        <div style="color: #FCA5A5; font-size: 0.85rem;">
            {message}
        </div>
    </div>
    """
