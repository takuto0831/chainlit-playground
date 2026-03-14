"""Chainlit Step デモ：OpenAI API を使ったリサーチツール + Plotly 信頼度チャート

UIの構造：
  ▼ 🔎 ウェブを調査しています        <- ルートstep
    ▼ 🔍「基本概念」を調査中          <- トピックstep（OpenAI が生成）
      ▼ 📄 ソース名                   <- サイトstep（OpenAI が調査）
      ...
    ...
  [📊 信頼度チャート]                <- 全ソースの信頼度を棒グラフで表示
  ▼ 情報を集約中                     <- 集約step（OpenAI ストリーミング）
  [最終回答]
"""

import json
from urllib.parse import urlparse

import chainlit as cl
import plotly.graph_objects as go
from openai import AsyncOpenAI

client = AsyncOpenAI()

# ──────────────────────────────────────────────
# Markdown フォーマッター
# ──────────────────────────────────────────────


def fmt_topic_output(_topic: str, sites: list[dict]) -> str:
    """トピックstepのoutput：サイト名・ドメイン・信頼度のみ表示."""
    rows = "\n".join(
        f"| [{s['name']}]({s['url']}) | `{urlparse(s['url']).netloc}`"
        f" | {s['reliability']} |"
        for s in sites
    )
    return (
        f"| ソース | ドメイン | 信頼度 |\n"
        f"|--------|----------|--------|\n"
        f"{rows}\n\n"
        f"> 🔍 **{len(sites)} 件**のソースを確認しました"
    )


# ──────────────────────────────────────────────
# Plotly チャート
# ──────────────────────────────────────────────


def make_reliability_chart(all_sites: list[dict]) -> go.Figure:
    """トピック x ソース番号 のヒートマップで信頼度を返す."""
    topics = list(dict.fromkeys(s["topic"] for s in all_sites))

    # トピックごとにソースをグループ化
    topic_sources: dict[str, list[dict]] = {t: [] for t in topics}
    for s in all_sites:
        topic_sources[s["topic"]].append(s)

    max_sources = max(len(v) for v in topic_sources.values())
    x_labels = [f"ソース {i + 1}" for i in range(max_sources)]

    # z: 信頼度スコア行列、text: ソース名ラベル
    z: list[list[float]] = []
    text: list[list[str]] = []
    for topic in topics:
        sources = topic_sources[topic]
        row_z = [s["reliability"].count("⭐") for s in sources]
        row_text = [f"{s['name']}<br>{s['reliability']}" for s in sources]
        # 列数を揃えるために None で埋める
        pad = max_sources - len(sources)
        z.append(row_z + [None] * pad)
        text.append(row_text + [""] * pad)

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x_labels,
            y=topics,
            text=text,
            texttemplate="%{text}",
            colorscale="YlOrRd",
            zmin=1,
            zmax=5,
            colorbar={"title": "信頼度"},
        )
    )
    fig.update_layout(
        title="トピック別 情報ソース 信頼度ヒートマップ",
        xaxis_title="ソース",
        yaxis_title="トピック",
        height=80 + 100 * len(topics),
        margin={"t": 60, "b": 60},
    )
    return fig


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

    # Step 2: 構造化 (重複URLを除去して最大MAX_SOURCES件)
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
                    ' "summary": "そのソースの要約", "reliability": "⭐〜⭐⭐⭐⭐⭐"}]}'
                    " の形式で返してください。"
                    "urlは参考URL一覧にある実際のURLをそのまま使ってください。"
                    "reliabilityは情報源の信頼度を⭐1〜5個で表してください。"
                ),
            },
        ],
    )
    raw = structure_response.choices[0].message.content
    assert raw is not None
    data = json.loads(raw)
    return data["sources"]


# ──────────────────────────────────────────────
# 集約・回答生成
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
    """ユーザーメッセージを受け取り、リサーチを実行して回答を返す."""
    query = message.content
    all_findings: list[str] = []
    all_sites: list[dict] = []  # チャート用：topic 付きで全ソースを蓄積

    async with cl.Step(name="🔎 ウェブを調査しています", type="tool") as root_step:
        root_step.input = query

        topics = await generate_topics(query)

        for topic in topics:
            step_name = f"🔍「{topic}」を調査中"
            async with cl.Step(name=step_name, type="tool") as topic_step:
                topic_step.input = f"「{topic}」の観点で調査"

                sites = await research_topic(query, topic)

                for site in sites:
                    async with cl.Step(
                        name=f"📄 {site['name']}", type="retrieval"
                    ) as site_step:
                        site_step.input = site["url"]
                        site_step.output = site["summary"]

                    all_findings.append(
                        f"**[{topic}|{site['name']}]** {site['summary']}"
                    )
                    all_sites.append({**site, "topic": topic})

                topic_step.output = fmt_topic_output(topic, sites)

        root_step.output = f"合計 {len(all_findings)} 件のソースを調査しました"

    # 信頼度チャートを表示
    if all_sites:
        fig = make_reliability_chart(all_sites)
        await cl.Message(
            content="📊 **情報ソース 信頼度チャート**",
            elements=[
                cl.Plotly(name="reliability_chart", figure=fig, display="inline")
            ],
        ).send()

    answer_msg = cl.Message(content="")
    await answer_msg.send()
    await aggregate(query, all_findings, answer_msg)
    await answer_msg.update()
