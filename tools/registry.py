from __future__ import annotations
from importlib import import_module
from typing import Callable, Dict, List

# Tabs + botones (UI)
TOOL_TABS: Dict[str, List[dict]] = {
    "Mesa": [
        {"id": "ons", "label": "ON’s"},
    ],
    "Backoffice": [
        {"id": "bo_ppt_manana", "label": "PPT Mañana"},
        {"id": "bo_moc_tarde", "label": "MOC Tarde"},
        {"id": "bo_control_sliq", "label": "Control SLIQ"},
        {"id": "bo_acreditacion_mav", "label": "Acreditación MAV"},
        {"id": "bo_cauciones", "label": "Cauciones"},
    ],
    "Comerciales": [
        {"id": "cheques", "label": "Cheques"},
        {"id": "cauciones_mae", "label": "Cauciones MAE"},
        {"id": "cauciones_byma", "label": "Cauciones BYMA"},
        {"id": "alquileres", "label": "Alquileres"},
    ],
}

# Router: tool_id -> (modulo, funcion)
ROUTES = {
    "ons": ("tools.ons", "render"),
    "cheques": ("tools.cheques", "render"),
    "cauciones_mae": ("tools.cauciones_mae", "render"),
    "cauciones-byma": ("tools.cauciones_byma", "render"),
    "cauciones_byma": ("tools.cauciones_byma", "render"),

    # Backoffice “sub-tools”
    "bo_ppt_manana": ("tools.backoffice", "render_ppt_manana"),
    "bo_moc_tarde": ("tools.backoffice", "render_moc_tarde"),
    "bo_control_sliq": ("tools.backoffice", "render_control_sliq"),
    "bo_acreditacion_mav": ("tools.backoffice", "render_acreditacion_mav"),
    "bo_cauciones": ("tools.backoffice", "render_cauciones"),

    # Placeholder por ahora
    "alquileres": ("tools.alquileres", "render"),
}

def run_tool(tool_id: str) -> bool:
    t = TOOL_MAP.get(tool_id)
    if not t:
        return False

    module = t["module"]
    # ✅ cada tool expone render() sin argumentos
    module.render()
    return True

