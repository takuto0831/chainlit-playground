import os

from chainlit.utils import mount_chainlit
from fastapi import FastAPI


def get_app_path() -> str:
    import importlib  # noqa: PLC0415

    target = os.getenv("TARGET", "hello")
    targets = {
        "hello",
        "auth_demo",
        "history_demo",
        "demo",
        "step",
        "step_child",
        "step_child_base",
        "step_child_ui_markdown",
        "step_child_ui_tasklist",
        "step_child_ui_chart",
        "step_child_ui_trivia",
    }
    if target not in targets:
        msg = f"Unknown target: {target}"
        raise ValueError(msg)
    module = importlib.import_module(f"chainlit_playground.{target}.app")
    assert module.__file__ is not None
    return module.__file__


def app() -> FastAPI:
    app = FastAPI()
    target = get_app_path()
    mount_chainlit(app, target, path="/")
    return app
