"""Chainlit Step デモ：OpenAI API を使ったリサーチツール

UIの構造：
  ▼ 🔎 ウェブを調査しています        ← ルートstep
    ▼ 🔍「基本概念」を調査中          ← トピックstep（OpenAI が生成）
      ▼ 📄 ソース名                   ← サイトstep（OpenAI が調査）
      ...
    ...
  ▼ ✍️ 情報を集約中                  ← 集約step（OpenAI ストリーミング）
  [最終回答]
"""

import json

import chainlit as cl
from openai import AsyncOpenAI

client = AsyncOpenAI()


# ──────────────────────────────────────────────
# OpenAI ヘルパー
# ──────────────────────────────────────────────


async def generate_topics(query: str) -> list[str]:
    """クエリからリサーチトピックを3つ生成する。"""
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=256,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "あなたはリサーチアシスタントです。JSONのみを返してください。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"「{query}」を調査するための主要なトピックを3つ挙げてください。"
                    '{"topics": ["トピック1", "トピック2", "トピック3"]}'
                    " の形式で返してください。"
                ),
            },
        ],
    )
    raw = response.choices[0].message.content
    assert raw is not None
    data = json.loads(raw)
    return data["topics"]


MAX_SOURCES = 3


async def research_topic(query: str, topic: str) -> list[dict]:
    """web_search_preview で実際にウェブを検索し、ソース2〜3件を返す."""
    # Step 1: ウェブ検索
    search_response = await client.responses.create(
        model="gpt-4o",
        tools=[{"type": "web_search_preview"}],
        input=(
            f"「{query}」の「{topic}」について調査し、"
            "参考になる情報ソース2〜3件の内容を要約してください。"
        ),
    )

    # テキストと引用URLを抽出
    output_text = ""
    url_citations: list[dict] = []
    for item in search_response.output:
        if item.type == "message":
            for content in getattr(item, "content", []):
                if content.type == "output_text":
                    output_text = getattr(content, "text", "")
                    url_citations.extend(
                        {"url": ann.url, "title": getattr(ann, "title", ann.url)}
                        for ann in getattr(content, "annotations", [])
                        if ann.type == "url_citation"
                    )

    # 重複URLを除去して最大MAX_SOURCES件
    seen: set[str] = set()
    unique_citations = []
    for c in url_citations:
        if c["url"] not in seen:
            seen.add(c["url"])
            unique_citations.append(c)
        if len(unique_citations) == MAX_SOURCES:
            break

    citations_text = "\n".join(f"- {c['title']}: {c['url']}" for c in unique_citations)
    structure_response = await client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=512,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "あなたはリサーチアシスタントです。JSONのみを返してください。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"以下の調査結果と参考URL一覧をもとに情報ソースを最大3件まとめてください。\n\n"
                    f"調査結果:\n{output_text}\n\n"
                    f"参考URL:\n{citations_text}\n\n"
                    '{"sources": [{"name": "ソース名", "url": "実際のURL",'
                    ' "summary": "そのソースの要約"}]}'
                    " の形式で返してください。"
                    "urlは参考URL一覧にある実際のURLをそのまま使ってください。"
                ),
            },
        ],
    )
    raw = structure_response.choices[0].message.content
    assert raw is not None
    data = json.loads(raw)
    return data["sources"]


# ──────────────────────────────────────────────
# 集約・回答生成（ストリーミング）
# ──────────────────────────────────────────────


async def aggregate(
    query: str, all_findings: list[str], answer_msg: cl.Message
) -> None:
    """調査結果を集約し、answer_msg にストリーミング表示する."""
    findings_text = "\n".join(f"- {f}" for f in all_findings)

    async with cl.Step(name="✍️ 情報を集約中", type="llm", show_input=False) as agg_step:
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=4096,
            stream=True,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "あなたは優秀なリサーチアシスタントです。"
                        "調査結果を日本語でわかりやすくまとめてください。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"以下の調査結果をもとに「{query}」について"
                        f"詳しくまとめてください。\n\n{findings_text}"
                    ),
                },
            ],
        )
        full_response = ""
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_response += delta
                await answer_msg.stream_token(delta)
                await agg_step.stream_token(delta)

        agg_step.output = full_response


# ──────────────────────────────────────────────
# メインハンドラ
# ──────────────────────────────────────────────


@cl.on_message
async def main(message: cl.Message) -> None:
    query = message.content
    all_findings: list[str] = []

    async with cl.Step(name="🔎 ウェブを調査しています", type="tool") as root_step:
        root_step.input = query

        topics = await generate_topics(query)

        for topic in topics:
            async with cl.Step(
                name=f"🔍「{topic}」を調査中", type="tool"
            ) as topic_step:
                topic_step.input = f"「{topic}」の観点で調査"

                sites = await research_topic(query, topic)

                for site in sites:
                    async with cl.Step(
                        name=f"📄 {site['name']}", type="retrieval"
                    ) as site_step:
                        site_step.input = site["url"]
                        site_step.output = site["summary"]

                    all_findings.append(
                        f"**[{topic}｜{site['name']}]** {site['summary']}"
                    )

                topic_step.output = f"{len(sites)} 件のソースを確認しました"

        root_step.output = f"合計 {len(all_findings)} 件のソースを調査しました"

    answer_msg = cl.Message(content="")
    await answer_msg.send()
    await aggregate(query, all_findings, answer_msg)
    await answer_msg.update()
