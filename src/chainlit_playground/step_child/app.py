"""Chainlit Step デモ：複数トピックを親子stepで並列調査する

UIの構造：
  ▼ 🧭 調査計画を立てています
  ▼ 🔍「基本概念」を調査中        ← 親step（トピック単位）
    ▼ 📄 Wikipedia                  ← 子step（サイト単位）
    ▼ 📄 公式ドキュメント
  ▼ 🔍「最新動向」を調査中
    ▼ 📄 TechCrunch Japan
    ▼ 📄 ZDNet Japan
  ▼ ✍️ 情報を集約中（ストリーミング）
  [最終回答]
"""

import asyncio

import chainlit as cl

# ──────────────────────────────────────────────
# モック：トピックごとの調査サイト一覧
# ──────────────────────────────────────────────

TOPIC_SITES: dict[str, list[dict]] = {
    "基本概念": [
        {
            "name": "Wikipedia",
            "domain": "ja.wikipedia.org",
            "summary": "基本的な定義・歴史・関連概念が網羅されています。",
        },
        {
            "name": "公式ドキュメント",
            "domain": "docs.example.com",
            "summary": "技術仕様と公式ガイドラインが掲載されています。",
        },
    ],
    "最新動向": [
        {
            "name": "TechCrunch Japan",
            "domain": "jp.techcrunch.com",
            "summary": "最新リリースと業界トレンドが報告されています。",
        },
        {
            "name": "ZDNet Japan",
            "domain": "zdnet.com",
            "summary": "エンタープライズ向け活用事例が紹介されています。",
        },
        {
            "name": "Wired Japan",
            "domain": "wired.jp",
            "summary": "研究者・開発者のコメントと展望が掲載されています。",
        },
    ],
    "実用例": [
        {
            "name": "Qiita",
            "domain": "qiita.com",
            "summary": "実装例とサンプルコードが豊富に掲載されています。",
        },
        {
            "name": "Zenn",
            "domain": "zenn.dev",
            "summary": "実務での導入事例と注意点がまとめられています。",
        },
    ],
}


# ──────────────────────────────────────────────
# Step 1：調査計画（@cl.step デコレータ版）
# ──────────────────────────────────────────────


@cl.step(name="🧭 調査計画を立てています", type="tool", show_input=False)
async def plan_topics(_query: str) -> list[str]:
    """クエリを複数のトピックに分解する"""
    await asyncio.sleep(0.6)
    topics = list(TOPIC_SITES.keys())  # 実際はLLMでトピックを生成する
    cl.context.current_step.output = "調査トピック: " + " / ".join(topics)
    return topics


# ──────────────────────────────────────────────
# Step 2：トピック調査（親子 Step）
# ──────────────────────────────────────────────


async def research_topic(query: str, topic: str) -> list[str]:
    """1つのトピックを複数サイトで調査する。

    async with cl.Step() を2段ネストすることで
    「親：トピック調査」→「子：各サイト確認」の階層UIを作る。
    """
    sites = TOPIC_SITES.get(topic, [])
    findings: list[str] = []

    # 親 step：トピック単位
    async with cl.Step(name=f"🔍「{topic}」を調査中", type="tool") as parent_step:
        parent_step.input = f"{query} について「{topic}」の観点で調査"

        for site in sites:
            # 子 step：サイト単位（親の with ブロック内で作ると自動的に子になる）
            async with cl.Step(
                name=f"📄 {site['name']}", type="retrieval"
            ) as child_step:
                child_step.input = site["domain"]
                await asyncio.sleep(0.5)  # 実際はスクレイピング or RAG検索
                child_step.output = site["summary"]

            findings.append(f"**[{topic}｜{site['name']}]** {site['summary']}")

        parent_step.output = f"{len(sites)} 件のソースを確認しました"

    return findings


# ──────────────────────────────────────────────
# Step 3：集約・回答生成（@cl.step + ストリーミング）
# ──────────────────────────────────────────────


@cl.step(name="✍️ 情報を集約中", type="llm", show_input=False)
async def aggregate(query: str, all_findings: list[str]) -> str:
    """全トピックの調査結果を集約してストリーミング表示する。

    cl.context.current_step で実行中ステップを取得し
    stream_token() で1文字ずつUIに流す。
    """
    findings_text = "\n".join(f"- {f}" for f in all_findings)
    answer = (
        f"「{query}」について、"
        f"{len(all_findings)} 件のソースから情報を収集しました。\n\n"
        "## 調査結果のまとめ\n\n"
        f"{findings_text}\n\n"
        "---\n"
        "各トピックを横断して確認した結果、情報ソース間で大きな矛盾は見られませんでした。"
        "詳細は各リンク先をご参照ください。"
    )

    current_step = cl.context.current_step
    for char in answer:
        await asyncio.sleep(0.012)
        await current_step.stream_token(char)

    return answer


# ──────────────────────────────────────────────
# メインハンドラ
# ──────────────────────────────────────────────


@cl.on_message
async def main(message: cl.Message) -> None:
    query = message.content

    # Step 1：調査トピックを決定
    topics = await plan_topics(query)

    # Step 2：全トピックを並列調査
    # asyncio.gather で同時実行 → 複数の親stepがほぼ同時にUIに現れる
    results = await asyncio.gather(*[research_topic(query, topic) for topic in topics])

    # 全トピックの findings をフラットに結合
    all_findings = [finding for topic_findings in results for finding in topic_findings]

    # Step 3：集約してストリーミング回答
    answer = await aggregate(query, all_findings)

    await cl.Message(content=answer).send()
