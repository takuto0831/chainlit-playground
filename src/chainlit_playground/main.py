import os

from chainlit.utils import mount_chainlit
from fastapi import FastAPI


def get_app_path() -> str:
    target = os.getenv("TARGET", "hello")
    match target:
        case "hello":
            from chainlit_playground import hello  # noqa: PLC0415

            return hello.app.__file__
        case "auth_demo":
            from chainlit_playground import auth_demo  # noqa: PLC0415

            return auth_demo.app.__file__
        case "history_demo":
            from chainlit_playground import history_demo  # noqa: PLC0415

            return history_demo.app.__file__
        case "demo":
            from chainlit_playground import demo  # noqa: PLC0415

            return demo.app.__file__
        case _:
            msg = f"Unknown target: {target}"
            raise ValueError(msg)


def app() -> FastAPI:
    app = FastAPI()
    target = get_app_path()
    mount_chainlit(app, target, path="/")
    return app
