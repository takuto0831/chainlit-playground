import chainlit as cl


@cl.password_auth_callback
async def auth_callback(username: str, _password: str) -> cl.User | None:
    """A dummy authentication callback.

    This callback only allows "guest@example.com" and any password.
    """
    if username != "guest@example.com":
        return None
    return cl.User(identifier="guest", metadata={"role": "guest",  "provider": "credentials"})


@cl.on_chat_start
async def main() -> None:
    user = cl.user_session.get("user")
    await cl.Message(content=f"ようこそ {user.identifier} さん").send()
