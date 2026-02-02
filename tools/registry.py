# tools/registry.py
import importlib

# -------------------------
# Tabs del Workbench (UI)
# -------------------------
TOOL_TABS = {
    "Mesa / Trading": [
        {"id": "cartera", "label": "Carteras comerciales"},
        {"id": "bonos", "label": "Bonos"},
        {"id": "ons", "label": "ONs — Screener"},
        {"id": "vencimientos", "label": "Tenencias"},
    ],
    "Middle Office": [
        {"id": "cheques", "label": "Cheques y Pagarés"},
        {"id": "cauciones_mae", "label": "Cauciones MAE"},
        {"id": "cauciones_byma", "label": "Cauciones BYMA"},
        {"id": "alquileres", "label": "Alquileres"},
    ],
    "Backoffice": [
        {"id": "bo_cauciones", "label": "Backoffice — Cauciones"},
        {"id": "bo_control_sliq", "label": "Backoffice — Control SLIQ"},
        {"id": "bo_moc_tarde", "label": "Backoffice — MOC Tarde"},
        {"id": "bo_ppt_manana", "label": "Backoffice — PPT Mañana"},
        {"id": "bo_acreditacion_mav", "label": "Backoffice — Acreditación MAV"},
    ],
}

# -------------------------
# Mapa: tool_id -> módulo real
# (ACÁ se arregla el cambio de carpetas)
# -------------------------
TOOL_IMPORTS = {
    # Mesa
    "cartera": "tools.mesa.cartera",
    "bonos": "tools.mesa.bonos",
    "ons": "tools.mesa.ons",
    "vencimientos": "tools.mesa.vencimientos",

    # Comerciales / Middle
    "cheques": "tools.comerciales.cheques",
    "cauciones_mae": "tools.comerciales.cauciones_mae",
    "cauciones-mae": "tools.comerciales.cauciones_mae",   # alias por si quedó link viejo
    "cauciones_byma": "tools.comerciales.cauciones_byma",
    "cauciones-byma": "tools.comerciales.cauciones_byma", # alias por si quedó link viejo
    "alquileres": "tools.comerciales.alquileres",

    # Backoffice
    "bo_cauciones": "tools.backoffice.cauciones",
    "bo_control_sliq": "tools.backoffice.control_sliq",
    "bo_moc_tarde": "tools.backoffice.moc_tarde",
    "bo_ppt_manana": "tools.backoffice.ppt_manana",
    "bo_acreditacion_mav": "tools.backoffice.acreditacion_mav",
}


# -------------------------
# Router
# -------------------------
def run_tool(tool_id: str, back_to_home=None) -> bool:
    """
    Importa el módulo correcto según TOOL_IMPORTS y ejecuta render().

    back_to_home: callable opcional. Si el render acepta parámetro,
    lo llamamos como render(back_to_home=...), si no, render().
    """
    module_path = TOOL_IMPORTS.get(tool_id)
    if not module_path:
        return False

    try:
        mod = importlib.import_module(module_path)
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
            # preferimos keyword, pero si no existe cae al except TypeError
            render_fn(back_to_home=back_to_home)
    except TypeError:
        try:
            if back_to_home is None:
                render_fn()
            else:
                render_fn(back_to_home)
        except TypeError:
            render_fn()

    return True

