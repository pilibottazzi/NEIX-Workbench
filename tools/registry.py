# tools/registry.py
import importlib

# -------------------------
# Tabs del Workbench
# -------------------------
TOOL_TABS = {
    "Mesa / Trading": [
        {"id": "ons", "label": "ONs — Screener"},
        {"id": "bonos", "label": "Bonos"},
        {"id": "vencimientos", "label": "Tenencias"},
        {"id": "cartera", "label": "Carteras comerciales"},
    ],
    "Middle Office": [
        {"id": "cheques", "label": "Cheques y Pagarés"},
        {"id": "cauciones_mae", "label": "Garantías MAE"},
        {"id": "cauciones_byma", "label": "Garantías BYMA"},
        {"id": "alquileres", "label": "Alquileres"},
    ],
    "Backoffice": [
        {"id": "bo_moc_tarde", "label": "MOC Tarde"},
        {"id": "bo_control_sliq", "label": "Control SLIQ"},
        {"id": "bo_ppt_manana", "label": "PPT Mañana"},
        # {"id": "bo_acreditacion_mav", "label": "Acreditación MAV"},
        {"id": "bo_cauciones", "label": "Cauciones (Backoffice)"},
    ],
}

# -------------------------
# Mapa REAL de módulos (por tu nueva estructura)
# -------------------------
TOOL_MODULES = {
    # Mesa
    "cartera": "tools.mesa.cartera",
    "ons": "tools.mesa.ons",
    "bonos": "tools.mesa.bonos",
    "vencimientos": "tools.mesa.vencimientos",

    # Comerciales / Middle
    "cheques": "tools.comerciales.cheques",
    "cauciones_mae": "tools.comerciales.cauciones_mae",
    "cauciones-mae": "tools.comerciales.cauciones_mae",
    "cauciones_byma": "tools.comerciales.cauciones_byma",
    "cauciones-byma": "tools.comerciales.cauciones_byma",
    "alquileres": "tools.comerciales.alquileres",

    # Backoffice
    "bo_moc_tarde": "tools.backoffice.moc_tarde",
    "bo_control_sliq": "tools.backoffice.control_sliq",
    "bo_ppt_manana": "tools.backoffice.ppt_manana",
    "bo_acreditacion_mav": "tools.backoffice.acreditacion_mav",
    "bo_cauciones": "tools.backoffice.cauciones",
}

# -------------------------
# Router
# -------------------------
def run_tool(tool_id: str, back_to_home=None) -> bool:
    """
    Carga el módulo correcto y ejecuta render().
    - Si render acepta back_to_home => render(back_to_home)
    - Si no => render()
    """
    module_name = TOOL_MODULES.get(tool_id)
    if not module_name:
        return False

    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return False

    render_fn = getattr(mod, "render", None)
    if not callable(render_fn):
        return False

    # Llamada tolerante a firmas distintas
    try:
        if back_to_home is None:
            render_fn()
        else:
            render_fn(back_to_home)
    except TypeError:
        render_fn()

    return True

