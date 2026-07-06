def build_app_context(app) -> dict:
    manifest = getattr(app, "manifest", None)
    return {
        "current_page": getattr(app, "current_page_name", "Library"),
        "project_loaded": bool(getattr(app, "project_dir", None)),
        "project_name": manifest.project_name if manifest else "",
        "workflow_completion": getattr(app, "workflow", {}),
        "selected_input_columns": manifest.selected_input_columns if manifest else [],
        "selected_output_columns": manifest.selected_output_columns if manifest else [],
        "sample_count": getattr(app, "sample_count", 0),
        "visible_warnings": getattr(app, "visible_warnings", []),
        "visible_metrics": getattr(app, "visible_metrics", {}),
        "app_version": getattr(app, "app_version", "1.0.0"),
    }
