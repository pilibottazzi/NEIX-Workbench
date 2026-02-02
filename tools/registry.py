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
        {"id": "cauciones", "label": "Cauciones"},
        {"id": "control_sliq", "label": "Control SLIQ"},
        {"id": "moc_tarde", "label": "MOC Tarde"},
        {"id": "ppt_manana", "label": "PPT Mañana"},
        {"id": "acreditacion_mav", "label": "Acreditación MAV"},
    ],
}

# -------------------------
# Mapa tool_id -> módulo real
# -------------------------
TOOL_MODULES = {
    # Mesa
    "cartera": "tools.mesa.cartera",
    "ons": "tools.mesa.ons",
    "bonos": "tools.mesa.bonos",
    "vencimientos": "tools.mesa.vencimientos",

    # Comerciales
    "cheques": "tools.comerciales.cheques",
    "cauciones_mae": "tools.comerciales.cauciones_mae",
    "cauciones_byma": "tools.comerciales.cauciones_byma",
    "alquileres": "tools.comerciales.alquileres",

    # Backoffice
    "cauciones": "tools.backoffice.cauciones",
    "control_sliq": "tools.backoffice.control_sliq",
    "moc_tarde": "tools.backoffice.moc_tarde",
    "ppt_manana": "tools.backoffice.ppt_manana",
    "acreditacion_mav": "tools.backoffice.acreditacion_mav",
}

# -------------------------
# Router
# -------------------------
def run_tool(tool_id: str, back_to_home=None) -> bool:
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

    # tolerante a firmas distintas
    try:
        if back_to_home is None:
            render_fn()
        else:
            render_fn(lambda: back_to_home(tool_id))
    except TypeError:
        render_fn()

    return True
