import os
from typing import Any

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.input_widget import Select, Slider, Switch, Tags, TextInput
from chainlit.types import ThreadDict


@cl.password_auth_callback
async def auth_callback(username: str, _password: str) -> cl.User | None:
    """A dummy authentication callback.

    This callback only allows "guest@example.com" and any password.
    """
    if username != "guest@example.com":
        return None
    return cl.User(
        identifier="guest", metadata={"role": "guest", "provider": "credentials"}
    )


@cl.data_layer
def data_layer() -> SQLAlchemyDataLayer:
    return SQLAlchemyDataLayer(conninfo=os.environ["CHAINLIT_CONNINFO"])


@cl.on_chat_resume
async def on_chat_resume(_: ThreadDict) -> None:
    pass


@cl.set_chat_profiles  # type: ignore[no-matching-overload]
async def set_chat_profiles() -> list[cl.ChatProfile]:
    return [
        cl.ChatProfile(
            name="Assistant Alice",
            markdown_description="A friendly assistant.",
        ),
        cl.ChatProfile(
            name="Assistant Bob",
            markdown_description="A helpful assistant.",
        ),
    ]


@cl.on_settings_update
async def setup_agent(settings: dict[str, Any]) -> None:
    cl.user_session.set("chat_settings", settings)


@cl.action_callback("count_clicks")
async def count_clicks(action: cl.Action) -> None:
    await action.remove()
    count_key = f"count:{action.forId}"

    count = cl.user_session.get(count_key, 0)
    count += 1
    cl.user_session.set(count_key, count)

    action.payload["count"] = count
    action.label = f"Clicked {count} times!"
    await action.send(for_id=action.forId)  # type: ignore[invalid-argument-type]


@cl.set_starters  # type: ignore[no-matching-overload]
async def set_starters() -> list[cl.Starter]:
    return [
        cl.Starter(label="Message", message="message"),
        cl.Starter(label="Message Update", message="message update"),
        cl.Starter(label="Message Remove", message="message remove"),
        cl.Starter(label="Error Message", message="error message"),
        cl.Starter(label="Step", message="step"),
        cl.Starter(label="Action", message="action"),
        cl.Starter(label="Element", message="element"),
    ]


@cl.on_chat_start
async def on_chat_start() -> None:
    await cl.ChatSettings(
        [
            Select(
                id="Thinking mode",
                label="Thinking mode",
                values=["fast", "slow"],
                initial_index=0,
            ),
            Slider(
                id="Creativity",
                label="Creativity",
                initial=50,
                min=0,
                max=100,
            ),
            Switch(
                id="Enable feature X",
                label="Enable feature X",
                initial=False,
            ),
            Tags(
                id="Interests",
                label="Interests",
                values=["AI", "Machine Learning", "Data Science"],
                initial=["AI", "Data Science"],
            ),
            TextInput(id="Notes", label="Notes", placeholder="Enter your notes here"),
        ]
    ).send()
    await cl.context.emitter.set_commands(
        [
            {
                "id": "Meow",
                "icon": "cat",
                "description": "Sends a meow message",
                "button": True,
                "persistent": True,
            },
            {
                "id": "Word Count",
                "icon": "ruler",
                "description": "Counts the number of words in the message",
                "button": True,
                "persistent": True,
            },
        ]
    )


@cl.on_message
async def on_message(message: cl.Message) -> None:  # noqa: PLR0915, C901, PLR0912
    if command := message.command:
        match command:
            case "Meow":
                await cl.Message(content="Meow!").send()
            case "Word Count":
                word_count = len(message.content.split())
                await cl.Message(content=f"Word count: {word_count}").send()
        return

    match message.content.lower():
        case "message":
            await cl.Message(content="Message is displayed like this.").send()
            # you can also specify a language for code blocks,
            # and it will be rendered with syntax highlighting
            await cl.Message(content="print('Hello, World!')", language="python").send()
            await cl.Message(content="ls -l", language="bash").send()
            # content can also be a dict, list, etc. and it will be rendered as a JSON
            await cl.Message(content={"a": 1}).send()
        case "message update":
            msg = cl.Message(content="This message will be updated after 2 seconds.")
            await msg.send()
            await cl.sleep(2)
            msg.content = "Message is updated like this."
            await msg.update()
        case "message remove":
            msg = cl.Message(content="This message will be removed after 2 seconds.")
            await msg.send()
            await cl.sleep(2)
            await msg.remove()
        case "error message":
            await cl.ErrorMessage(content="Dummy error!").send()
            # you can also create an error message by setting the is_error flag to True
            msg = cl.Message(content="Dummy error!")
            msg.is_error = True
            await msg.send()
        case "step":
            async with cl.Step(
                name="Step started",
                default_open=True,
            ) as step:
                await cl.sleep(1)
                async with cl.Step(
                    name="Run LLM",
                    type="llm",
                    default_open=True,
                ) as first:
                    step.name = first.name

                    first.input = "input"
                    await step.update()
                    await first.update()
                    await cl.sleep(1)
                    first.output = "output"

                async with cl.Step(
                    name="Tool call",
                    type="tool",
                    default_open=True,
                    show_input="python",
                ) as second:
                    step.name = second.name

                    second.input = "add(1, 2)"
                    await step.update()
                    await second.update()
                    await cl.sleep(1)
                    second.output = {"output": 3}

                step.name = "Step completed"

        case "action":
            actions = [
                cl.Action(
                    name="count_clicks",
                    payload={"count": 0},
                    label="Click me!",
                    tooltip="This button will count the number of clicks.",
                    icon="mouse-pointer-click",
                )
            ]
            await cl.Message(
                content="Let's click some buttons!", actions=actions
            ).send()

        case "element":
            elements = [
                cl.Text(
                    name="text",
                    display="inline",
                    content="a text element with inline display.",
                ),
            ]
            await cl.Message(
                content="Here are some elements:", elements=elements
            ).send()
        case "chat profile":
            await cl.Message(
                content=f"Message from {cl.user_session.get('chat_profile')}"
            ).send()

        case "ask":
            res = await cl.AskUserMessage(
                content="What is your favorite color?",
            ).send()
            if res:
                await cl.Message(
                    content=f"Your favorite color is {res['output']}"
                ).send()

        case "chat settings":
            settings: dict[str, Any] = cl.user_session.get("chat_settings")
            await cl.Message(content=settings).send()
