import importlib

# -------------------------
# Tabs del Workbench
# -------------------------
TOOL_TABS = {
    "Mesa / Trading": [
        {"id": "ons", "label": "ONs — Screener"},
    ],
    "Middle Office": [
        {"id": "cheques", "label": "Cheques y Pagarés"},
        {"id": "cauciones_mae", "label": "Garantías MAE"},
        {"id": "cauciones_byma", "label": "Garantías BYMA"},
    ],
    "Backoffice": [
        {"id": "backoffice", "label": "Backoffice"},
    ],
    "Admin": [
        {"id": "alquileres", "label": "Alquileres"},
    ],
}

# -------------------------
# Router
# -------------------------
def run_tool(tool_id: str, back_to_home=None) -> bool:
    """
    Carga tools.<tool_id> y ejecuta render().
    - Si render acepta back_to_home => render(back_to_home)
    - Si no => render()
    """
    module_name = f"tools.{tool_id}"
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return False

    render_fn = getattr(mod, "render", None)
    if not callable(render_fn):
        return False

    # ✅ Llamada tolerante a firmas distintas
    try:
        if back_to_home is None:
            render_fn()
        else:
            render_fn(lambda: back_to_home(tool_id))  # back button con key única por herramienta
    except TypeError:
        # render() sin parámetros
        render_fn()

    return True

