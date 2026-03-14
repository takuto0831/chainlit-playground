# Chainlit の Step UI を段階的に改善する：4つのアプローチ

## 概要

Chainlit には `cl.Step` というコンポーネントがあり、LLM アプリケーションの処理過程をユーザーに可視化できる。本記事では、OpenAI の web_search_preview ツールを使ったリサーチツールを題材に、Step の基本実装から始め、4つの方向でUI/UXを段階的に改善する実装例を紹介する。

使用技術：
- Chainlit 2.9.6
- OpenAI API（gpt-4o / gpt-4o-mini / web_search_preview）
- Python 3.13

---

## ベース実装：`step_child_base`

まずベースとなる実装を示す。この実装が以降の比較の基準となる。

### アーキテクチャ

処理の流れは以下の通り：

```
ユーザー入力
  └─ generate_topics()     # gpt-4o-mini でトピック3つを生成
       └─ research_topic() # トピックごとに gpt-4o + web_search_preview で実Web検索
            └─ aggregate() # gpt-4o-mini で全調査結果をストリーミング集約
```

Step の入れ子構造：
```
▼ 🔎 ウェブを調査しています    (type="tool",      ルートstep)
  ▼ 🔍「基本概念」を調査中     (type="tool",      トピックstep × 3)
    ▼ 📄 ソース名              (type="retrieval", サイトstep × 2〜3)
    ...
  ▼ ✍️ 情報を集約中            (type="llm",       集約step)
[最終回答メッセージ]
```

### コード全文

```python
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
```

### ベース実装のポイント

**`cl.Step` の入れ子構造**

`async with cl.Step(...) as step:` のブロック内でさらに `cl.Step` を作ると、自動的に親子関係になる。Chainlit の UI では折りたたみ可能なツリー構造として表示される。

```python
async with cl.Step(name="親", type="tool") as parent:
    async with cl.Step(name="子", type="retrieval") as child:
        child.output = "子の結果"
    parent.output = "親の結果"
```

**`type` パラメータによる分類**

| type | 用途 |
|------|------|
| `"tool"` | ツール呼び出し（検索など） |
| `"retrieval"` | 情報取得（サイト参照など） |
| `"llm"` | LLM による生成処理 |

**ストリーミングの二重送信パターン**

`aggregate()` では `answer_msg`（チャット本文）と `agg_step`（Stepの出力）の両方に同時にトークンをストリームしている。これにより、Stepを折りたたんでいても展開していても、ユーザーはリアルタイムで生成中のテキストを確認できる。

```python
async for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        full_response += delta
        await answer_msg.stream_token(delta)  # チャット本文にストリーム
        await agg_step.stream_token(delta)    # Step内にもストリーム
```

**`answer_msg` を事前送信するパターン**

`aggregate()` は `cl.Step` のデコレータパターン（`@cl.step`）ではなく、`async with cl.Step(...)` を内部で使う関数として実装している。これは、デコレータパターンでは戻り値をチャット本文に直接流せないためである。

```python
answer_msg = cl.Message(content="")
await answer_msg.send()        # 空のメッセージを先に送信（ストリーム受信準備）
await aggregate(query, all_findings, answer_msg)
await answer_msg.update()      # 最終状態を確定
```

**web_search_preview による実Web検索**

`gpt-4o` + `web_search_preview` ツールを使うことで、実際のウェブ検索結果を取得できる。レスポンスの `annotations` フィールドに `url_citation` 型のアノテーションとして参照URLが含まれる。

```python
search_response = await client.responses.create(
    model="gpt-4o",
    tools=[{"type": "web_search_preview"}],
    input="...",
)

for item in search_response.output:
    if item.type == "message":
        for content in getattr(item, "content", []):
            if content.type == "output_text":
                url_citations.extend(
                    {"url": ann.url, "title": getattr(ann, "title", ann.url)}
                    for ann in getattr(content, "annotations", [])
                    if ann.type == "url_citation"
                )
```

**`step_child_base` の課題**

ベース実装で `topic_step.output` に設定しているのは単純なプレーンテキストのみ：

```python
topic_step.output = f"{len(sites)} 件のソースを確認しました"
```

この表示では「何件調べた」という数だけが分かり、どのサイトを調べたかは `site_step` を1つずつ展開しないと見えない。ユーザーが全体を把握するには多くのクリックが必要になる。

---

## 改善1：`step_child_ui_markdown` — Markdown による Step 出力の構造化

### 変更の概要

`step_child_base` からの変更点は1箇所：`topic_step.output` に設定するテキストを、プレーンテキストから Markdown テーブルに変更した。

### 追加したコード

```python
from urllib.parse import urlparse

def fmt_topic_output(_topic: str, sites: list[dict]) -> str:
    """トピックstepのoutput：サイト名・ドメイン・信頼度のみ表示

    After:
      | ソース | ドメイン | 信頼度 |
      |--------|----------|--------|
      | Wikipedia | ja.wikipedia.org | ⭐⭐⭐⭐ |
      ...
      > 🔍 **2 件**のソースを確認しました
    """
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
```

呼び出し側：
```python
# before (step_child_base)
topic_step.output = f"{len(sites)} 件のソースを確認しました"

# after (step_child_ui_markdown)
topic_step.output = fmt_topic_output(topic, sites)
```

また、`reliability` フィールドをOpenAI への構造化プロンプトに追加している：

```python
# step_child_base の JSON スキーマ（reliability なし）
'{"sources": [{"name": "ソース名", "url": "実際のURL", "summary": "要約"}]}'

# step_child_ui_markdown の JSON スキーマ（reliability あり）
'{"sources": [{"name": "ソース名", "url": "実際のURL", "summary": "要約", "reliability": "⭐〜⭐⭐⭐⭐⭐"}]}'
# + プロンプト: "reliabilityは情報源の信頼度を⭐1〜5個で表してください。"
```

### UIの変化

**before（step_child_base）**

`topic_step` を展開すると：
```
3 件のソースを確認しました
```

**after（step_child_ui_markdown）**

`topic_step` を展開すると：
```
| ソース         | ドメイン              | 信頼度     |
|----------------|----------------------|------------|
| Wikipedia      | ja.wikipedia.org     | ⭐⭐⭐⭐⭐  |
| NHK News       | www3.nhk.or.jp       | ⭐⭐⭐⭐    |
| Tech Blog      | example.com          | ⭐⭐⭐      |

> 🔍 **3 件**のソースを確認しました
```

### なぜ Markdown テーブルが有効か

- **1クリックで全ソースが見える**：`site_step` を個別に展開しなくても、トピックの `output` 一覧だけで調査先と信頼度が分かる
- **ソース名がリンク化**：クリックで実際のURLに飛べる
- **ドメインをコードブロックで表示**：`` `ja.wikipedia.org` `` の形式でソースの出所が一目で判断できる
- **信頼度が視覚的**：⭐ の数でスキャンしやすい
- **blockquote でサマリー**：`>` 記法でカウント情報を目立たせている

### コード全文

（step_child_base との差分：`fmt_topic_output` 関数の追加、`reliability` フィールドの追加、`topic_step.output` の変更のみ）

```python
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
from urllib.parse import urlparse

import chainlit as cl
from openai import AsyncOpenAI

client = AsyncOpenAI()

# ──────────────────────────────────────────────
# Markdown フォーマッター
# ──────────────────────────────────────────────


def fmt_topic_output(_topic: str, sites: list[dict]) -> str:
    """トピックstepのoutput：サイト名・ドメイン・信頼度のみ表示

    After:
      | ソース | ドメイン | 信頼度 |
      |--------|----------|--------|
      | Wikipedia | ja.wikipedia.org | ⭐⭐⭐⭐ |
      ...
      > 🔍 **2 件**のソースを確認しました
    """
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
# OpenAI ヘルパー（step_child_base と同一、reliability フィールドのみ追加）
# ──────────────────────────────────────────────

async def generate_topics(query: str) -> list[str]: ...  # （省略 - base と同一）

async def research_topic(query: str, topic: str) -> list[dict]: ...
# JSON スキーマに "reliability": "⭐〜⭐⭐⭐⭐⭐" を追加

async def aggregate(
    query: str, all_findings: list[str], answer_msg: cl.Message
) -> None: ...  # （省略 - base と同一）


# ──────────────────────────────────────────────
# メインハンドラ（topic_step.output のみ変更）
# ──────────────────────────────────────────────

@cl.on_message
async def main(message: cl.Message) -> None:
    query = message.content
    all_findings: list[str] = []

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

                # ★ ここだけ変更
                topic_step.output = fmt_topic_output(topic, sites)

        root_step.output = f"合計 {len(all_findings)} 件のソースを調査しました"

    answer_msg = cl.Message(content="")
    await answer_msg.send()
    await aggregate(query, all_findings, answer_msg)
    await answer_msg.update()
```

---

## 改善2：`step_child_ui_tasklist` — TaskList によるサイドバー進捗表示

### 変更の概要

`step_child_ui_markdown` をベースに、`cl.TaskList` と `cl.Task` を追加した。Step のツリー表示と並行して、チャット右側のサイドバーにタスクリストが表示され、各トピックの進捗状況（実行中 / 完了）をリアルタイムで確認できる。

### 追加・変更したコード

**`main()` 内：TaskList の初期化と per-topic タスク管理**

```python
@cl.on_message
async def main(message: cl.Message) -> None:
    query = message.content
    all_findings: list[str] = []

    # ★ TaskList を初期化してサイドバーに送信
    task_list = cl.TaskList()
    task_list.status = "調査中..."
    await task_list.send()

    async with cl.Step(name="🔎 ウェブを調査しています", type="tool") as root_step:
        root_step.input = query
        topics = await generate_topics(query)

        for topic in topics:
            # ★ トピックごとに Task を追加（RUNNING 状態で開始）
            task = cl.Task(title=f"🔍「{topic}」を調査中", status=cl.TaskStatus.RUNNING)
            await task_list.add_task(task)
            await task_list.update()

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

                topic_step.output = fmt_topic_output(topic, sites)

            # ★ トピック完了時に Task を DONE に更新
            task.status = cl.TaskStatus.DONE
            await task_list.update()

        root_step.output = f"合計 {len(all_findings)} 件のソースを調査しました"

    answer_msg = cl.Message(content="")
    await answer_msg.send()
    await aggregate(query, all_findings, answer_msg, task_list)  # ★ task_list を渡す
    await answer_msg.update()
```

**`aggregate()` 内：集約タスクの管理**

```python
async def aggregate(
    query: str,
    all_findings: list[str],
    answer_msg: cl.Message,
    task_list: cl.TaskList,   # ★ 引数追加
) -> None:
    findings_text = "\n".join(f"- {f}" for f in all_findings)

    # ★ 集約タスクを追加
    agg_task = cl.Task(title="✍️ 情報を集約中", status=cl.TaskStatus.RUNNING)
    await task_list.add_task(agg_task)
    await task_list.update()

    async with cl.Step(name="✍️ 情報を集約中", type="llm", show_input=False) as agg_step:
        # （ストリーミング処理は base と同一）
        ...

    # ★ 集約完了時にタスクリスト全体を「完了」に
    agg_task.status = cl.TaskStatus.DONE
    task_list.status = "完了"
    await task_list.update()
```

### UIの変化

**サイドバーに表示されるタスクリスト**

```
タスク                          状態
──────────────────────────────────
🔍「基本概念」を調査中           ✅ 完了
🔍「活用事例」を調査中           🔄 実行中  ← リアルタイムで変化
🔍「課題と展望」を調査中         （待機中）
✍️ 情報を集約中                  （待機中）

ステータス: 調査中...
```

**Step ツリーとの使い分け**

| 表示要素 | 役割 |
|---------|------|
| Step ツリー | 処理の詳細（入力・出力・階層）を展開して確認 |
| TaskList | 全体の進捗を一覧でリアルタイム把握 |

### `cl.TaskStatus` の状態遷移

```python
cl.TaskStatus.RUNNING  # 🔄 実行中（スピナーアニメーション）
cl.TaskStatus.DONE     # ✅ 完了
cl.TaskStatus.FAILED   # ❌ 失敗
```

### なぜ TaskList が有効か

Step ツリーは処理の「詳細」を見るものであり、全体進捗の把握には向かない。特にトピック数が増えたとき、どこまで終わっているかが一目で分からない。TaskList をサイドバーに置くことで：

- **スクロールせずに全体進捗が見える**（サイドバーは常時表示）
- **「今どのトピックを処理中か」が明確**
- **完了数 / 総数のカウントが分かりやすい**

### コード全文

（Markdown版からの差分：`task_list` の初期化・追加・更新処理のみ）

```python
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
from urllib.parse import urlparse

import chainlit as cl
from openai import AsyncOpenAI

client = AsyncOpenAI()

# ──────────────────────────────────────────────
# Markdown フォーマッター（ui_markdown と同一）
# ──────────────────────────────────────────────

def fmt_topic_output(_topic: str, sites: list[dict]) -> str: ...

# ──────────────────────────────────────────────
# OpenAI ヘルパー（ui_markdown と同一）
# ──────────────────────────────────────────────

async def generate_topics(query: str) -> list[str]: ...
async def research_topic(query: str, topic: str) -> list[dict]: ...

# ──────────────────────────────────────────────
# 集約・回答生成（task_list 引数を追加）
# ──────────────────────────────────────────────

async def aggregate(
    query: str,
    all_findings: list[str],
    answer_msg: cl.Message,
    task_list: cl.TaskList,
) -> None:
    findings_text = "\n".join(f"- {f}" for f in all_findings)

    agg_task = cl.Task(title="✍️ 情報を集約中", status=cl.TaskStatus.RUNNING)
    await task_list.add_task(agg_task)
    await task_list.update()

    async with cl.Step(name="✍️ 情報を集約中", type="llm", show_input=False) as agg_step:
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=4096,
            stream=True,
            messages=[...],
        )
        full_response = ""
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_response += delta
                await answer_msg.stream_token(delta)
                await agg_step.stream_token(delta)

        agg_step.output = full_response

    agg_task.status = cl.TaskStatus.DONE
    task_list.status = "完了"
    await task_list.update()


# ──────────────────────────────────────────────
# メインハンドラ
# ──────────────────────────────────────────────

@cl.on_message
async def main(message: cl.Message) -> None:
    query = message.content
    all_findings: list[str] = []

    task_list = cl.TaskList()
    task_list.status = "調査中..."
    await task_list.send()

    async with cl.Step(name="🔎 ウェブを調査しています", type="tool") as root_step:
        root_step.input = query
        topics = await generate_topics(query)

        for topic in topics:
            task = cl.Task(title=f"🔍「{topic}」を調査中", status=cl.TaskStatus.RUNNING)
            await task_list.add_task(task)
            await task_list.update()

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

                topic_step.output = fmt_topic_output(topic, sites)

            task.status = cl.TaskStatus.DONE
            await task_list.update()

        root_step.output = f"合計 {len(all_findings)} 件のソースを調査しました"

    answer_msg = cl.Message(content="")
    await answer_msg.send()
    await aggregate(query, all_findings, answer_msg, task_list)
    await answer_msg.update()
```

---

## 改善3：`step_child_ui_chart` — Plotly ヒートマップによる信頼度の可視化

### 変更の概要

`step_child_ui_markdown` をベースに、全トピックの調査完了後に Plotly のインタラクティブなヒートマップチャートを `cl.Plotly` で表示する。信頼度情報を数値化して色で表現することで、「どのトピックのどのソースが高品質か」を視覚的に把握できる。

### 追加したコード

**依存関係の追加**

```bash
uv add plotly
```

**インポート**

```python
import plotly.graph_objects as go
```

**ヒートマップ生成関数**

```python
def make_reliability_chart(all_sites: list[dict]) -> go.Figure:
    """トピック x ソース番号 のヒートマップで信頼度を返す."""
    topics = list(dict.fromkeys(s["topic"] for s in all_sites))

    # トピックごとにソースをグループ化
    topic_sources: dict[str, list[dict]] = {t: [] for t in topics}
    for s in all_sites:
        topic_sources[s["topic"]].append(s)

    max_sources = max(len(v) for v in topic_sources.values())
    x_labels = [f"ソース {i + 1}" for i in range(max_sources)]

    # z: 信頼度スコア行列（⭐ の数をカウント）、text: セルラベル
    z: list[list[float]] = []
    text: list[list[str]] = []
    for topic in topics:
        sources = topic_sources[topic]
        row_z = [s["reliability"].count("⭐") for s in sources]
        # ソース名を14文字ごとに改行して折り返す
        row_text = [
            "<br>".join(
                s["name"][i : i + 14] for i in range(0, len(s["name"]), 14)
            )
            + f"<br>{s['reliability']}"
            for s in sources
        ]
        # トピックごとのソース数が異なる場合、None で埋めて列数を揃える
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
            textfont={"size": 10},          # セル内フォントサイズを指定
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
        height=80 + 160 * len(topics),  # 折り返しテキスト分だけ行高を確保
        margin={"t": 60, "b": 60},
    )
    return fig
```

**`main()` 内：全ソースの蓄積とチャート表示**

```python
@cl.on_message
async def main(message: cl.Message) -> None:
    query = message.content
    all_findings: list[str] = []
    all_sites: list[dict] = []  # ★ チャート用：topic 付きで全ソースを蓄積

    async with cl.Step(name="🔎 ウェブを調査しています", type="tool") as root_step:
        ...
        for topic in topics:
            ...
            sites = await research_topic(query, topic)
            for site in sites:
                ...
                all_findings.append(...)
                all_sites.append({**site, "topic": topic})  # ★ topic を付与して蓄積
            ...

    # ★ 全トピック完了後にヒートマップを表示
    if all_sites:
        fig = make_reliability_chart(all_sites)
        await cl.Message(
            content="📊 **情報ソース 信頼度チャート**",
            elements=[cl.Plotly(name="reliability_chart", figure=fig, display="inline")],
        ).send()

    answer_msg = cl.Message(content="")
    await answer_msg.send()
    await aggregate(query, all_findings, answer_msg)
    await answer_msg.update()
```

### UIの変化

**チャート表示の位置**

リサーチ完了の Step ツリーの直後、最終要約の直前に表示される。

```
▼ 🔎 ウェブを調査しています（完了）
📊 情報ソース 信頼度チャート
  [ヒートマップ: インタラクティブなPlotlyチャート]
（最終要約が流れる...）
```

**ヒートマップの読み方**

```
        ソース 1    ソース 2    ソース 3
基本概念  ⭐⭐⭐⭐⭐   ⭐⭐⭐⭐    ⭐⭐⭐
活用事例  ⭐⭐⭐⭐    ⭐⭐⭐      ー（なし）
課題展望  ⭐⭐⭐⭐⭐   ⭐⭐⭐⭐    ⭐⭐
```

- Y軸：トピック名
- X軸：ソース番号（各トピック内の順序）
- セルの色（黄〜赤）：信頼度スコア（⭐1〜5）
- セルのテキスト：ソース名と信頼度絵文字

### 設計上の工夫

**棒グラフではなくヒートマップを選んだ理由**

最初の実装では棒グラフを使ったが、「信頼度のスコアを縦軸に数値化しているだけで情報量が増えていない」という課題があった。ヒートマップにすることで：

- **トピックという第2の次元**が加わり、「どのトピックのソースが全体的に信頼性が高いか」が分かる
- **欠損セル（None）**でソース数のばらつきも視覚化できる
- **インタラクティブ**：ホバーでソース名・信頼度の詳細が表示される

**`None` パディングによる不揃い行列の処理**

各トピックのソース数が異なる場合（例：2件 vs 3件）、Plotly の Heatmap は行列の列数が一致している必要があるため、`None` で埋めて対応している。

```python
pad = max_sources - len(sources)
z.append(row_z + [None] * pad)
text.append(row_text + [""] * pad)
```

**`dict.fromkeys()` でトピック順序を保持**

```python
topics = list(dict.fromkeys(s["topic"] for s in all_sites))
```

`set` を使うと順序が保証されないため、Python 3.7+ で挿入順序が保持される `dict.fromkeys()` でユニーク化している。

### コード全文

（Markdown版からの差分：`make_reliability_chart()` の追加、`all_sites` リストの追加、`cl.Plotly` の送信のみ。省略部分は `step_child_ui_markdown` と同一）

```python
"""Chainlit Step デモ：OpenAI API を使ったリサーチツール + Plotly 信頼度チャート

UIの構造：
  ▼ 🔎 ウェブを調査しています        <- ルートstep
    ▼ 🔍「基本概念」を調査中          <- トピックstep（OpenAI が生成）
      ▼ 📄 ソース名                   <- サイトstep（OpenAI が調査）
      ...
    ...
  [📊 信頼度チャート]                <- 全ソースの信頼度をヒートマップで表示
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
# Markdown フォーマッター（ui_markdown と同一）
# ──────────────────────────────────────────────

def fmt_topic_output(_topic: str, sites: list[dict]) -> str: ...

# ──────────────────────────────────────────────
# Plotly チャート（★ 追加）
# ──────────────────────────────────────────────

def make_reliability_chart(all_sites: list[dict]) -> go.Figure:
    """トピック x ソース番号 のヒートマップで信頼度を返す."""
    topics = list(dict.fromkeys(s["topic"] for s in all_sites))

    topic_sources: dict[str, list[dict]] = {t: [] for t in topics}
    for s in all_sites:
        topic_sources[s["topic"]].append(s)

    max_sources = max(len(v) for v in topic_sources.values())
    x_labels = [f"ソース {i + 1}" for i in range(max_sources)]

    z: list[list[float]] = []
    text: list[list[str]] = []
    for topic in topics:
        sources = topic_sources[topic]
        row_z = [s["reliability"].count("⭐") for s in sources]
        row_text = [
            "<br>".join(
                s["name"][i : i + 14] for i in range(0, len(s["name"]), 14)
            )
            + f"<br>{s['reliability']}"
            for s in sources
        ]
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
            textfont={"size": 10},
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
        height=80 + 160 * len(topics),
        margin={"t": 60, "b": 60},
    )
    return fig


# ──────────────────────────────────────────────
# OpenAI ヘルパー（ui_markdown と同一）
# ──────────────────────────────────────────────

async def generate_topics(query: str) -> list[str]: ...
async def research_topic(query: str, topic: str) -> list[dict]: ...
async def aggregate(query: str, all_findings: list[str], answer_msg: cl.Message) -> None: ...


# ──────────────────────────────────────────────
# メインハンドラ（all_sites と cl.Plotly 送信を追加）
# ──────────────────────────────────────────────

@cl.on_message
async def main(message: cl.Message) -> None:
    query = message.content
    all_findings: list[str] = []
    all_sites: list[dict] = []  # ★ 追加

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
                    all_sites.append({**site, "topic": topic})  # ★ 追加

                topic_step.output = fmt_topic_output(topic, sites)

        root_step.output = f"合計 {len(all_findings)} 件のソースを調査しました"

    # ★ チャートを送信
    if all_sites:
        fig = make_reliability_chart(all_sites)
        await cl.Message(
            content="📊 **情報ソース 信頼度チャート**",
            elements=[cl.Plotly(name="reliability_chart", figure=fig, display="inline")],
        ).send()

    answer_msg = cl.Message(content="")
    await answer_msg.send()
    await aggregate(query, all_findings, answer_msg)
    await answer_msg.update()
```

---

## 改善4：`step_child_ui_trivia` — `asyncio.gather` による豆知識の並行表示

### 変更の概要

`step_child_ui_markdown` をベースに、`asyncio.gather` を使って「トピック生成」と「豆知識生成」を並行実行し、リサーチ待機中にユーザーへ関連情報を表示する。また、Step 名を完了後に動的更新し、進捗カウンターを付与している。

### 追加したコード

**インポート**

```python
import asyncio
```

**豆知識生成関数**

```python
async def generate_trivia(query: str) -> str:
    """クエリに関連する面白い豆知識を1つ生成する。"""
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=256,
        messages=[
            {
                "role": "system",
                "content": (
                    "あなたは博識なアシスタントです。"
                    "与えられたテーマに関連する、知っていると少し得する面白い豆知識を1つ、"
                    "日本語で3〜5文程度で教えてください。"
                    "「💡 豆知識：」で始めてください。"
                ),
            },
            {
                "role": "user",
                "content": f"「{query}」に関連する豆知識を1つ教えてください。",
            },
        ],
    )
    return response.choices[0].message.content or ""
```

**`main()` 内：並行実行と豆知識の先行表示**

```python
@cl.on_message
async def main(message: cl.Message) -> None:
    query = message.content
    all_findings: list[str] = []

    # ★ トピック生成と豆知識生成を並行実行（asyncio.gather）
    topics, trivia = await asyncio.gather(
        generate_topics(query),
        generate_trivia(query),
    )

    # ★ 豆知識を即時表示（リサーチ開始の通知も兼ねる）
    await cl.Message(
        content=(
            f"{trivia}\n\n"
            f"---\n"
            f"*リサーチを開始しました。結果が出るまでしばらくお待ちください...*"
        )
    ).send()

    async with cl.Step(name="🔎 ウェブを調査しています", type="tool") as root_step:
        root_step.input = query

        for i, topic in enumerate(topics, 1):
            # ★ 進捗カウンターをStep名に含める
            step_name = f"🔍 [{i}/{len(topics)}]「{topic}」を調査中"
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

                topic_step.output = fmt_topic_output(topic, sites)

                # ★ 完了後にStep名を更新（✅ アイコン + 件数）
                topic_step.name = (
                    f"✅ [{i}/{len(topics)}]「{topic}」完了 ({len(sites)}件)"
                )

        root_step.output = f"合計 {len(all_findings)} 件のソースを調査しました"

    answer_msg = cl.Message(content="")
    await answer_msg.send()
    await aggregate(query, all_findings, answer_msg)
    await answer_msg.update()
```

### UIの変化

**表示順序**

```
[💡 豆知識メッセージ]                 ← ユーザーがすぐ読める
---
*リサーチを開始しました...*

▼ 🔎 ウェブを調査しています
  ▼ 🔍 [1/3]「基本概念」を調査中    ← 進捗カウンター付き
    ▼ 📄 ソース名
    ...
  ✅ [1/3]「基本概念」完了 (3件)     ← 完了後に名前が変化
  ▼ 🔍 [2/3]「活用事例」を調査中
  ...

（最終要約が流れる...）
```

**Step 名の動的変化**

| タイミング | Step名の表示 |
|-----------|-------------|
| 調査開始時 | `🔍 [1/3]「基本概念」を調査中` |
| 調査完了後 | `✅ [1/3]「基本概念」完了 (3件)` |

### `asyncio.gather` の効果

`generate_topics()` と `generate_trivia()` はどちらも独立した OpenAI API 呼び出しであり、互いに依存しない。`asyncio.gather` で並行実行することで：

- 豆知識のための「余分な待ち時間がほぼゼロ」になる
- トピック生成が完了した時点で豆知識も揃っており、即時表示できる

```python
# 直列実行（避けるべきパターン）
topics = await generate_topics(query)   # 例：0.8秒
trivia = await generate_trivia(query)   # 例：0.8秒
# 合計：約1.6秒

# asyncio.gather による並行実行（採用したパターン）
topics, trivia = await asyncio.gather(
    generate_topics(query),
    generate_trivia(query),
)
# 合計：約0.8秒（最長の処理時間のみ）
```

### Step 名の動的更新の実装

`cl.Step` はコンテキストマネージャを `__aexit__` するまでサーバーに確定されないが、`step.name` を変更して `await step.update()` を呼ぶことでリアルタイムに名前を更新できる。

```python
async with cl.Step(name="🔍 [1/3]「基本概念」を調査中") as topic_step:
    sites = await research_topic(...)
    topic_step.output = fmt_topic_output(...)
    # ★ with ブロックを抜ける前に名前を変更
    topic_step.name = "✅ [1/3]「基本概念」完了 (3件)"
    # with ブロックを抜けると自動的に update される
```

### コード全文

```python
"""Chainlit Step デモ：OpenAI API を使ったリサーチツール + 豆知識並行生成

UIの構造：
  [💡 豆知識カード]                  <- リサーチと並行して即時表示
  ▼ 🔎 ウェブを調査しています        <- ルートstep
    ▼ 🔍「基本概念」を調査中          <- トピックstep（OpenAI が生成）
      ▼ 📄 ソース名                   <- サイトstep（OpenAI が調査）
      ...
    ...
  ▼ ✍️ 情報を集約中                  <- 集約step（OpenAI ストリーミング）
  [最終回答]
"""

import asyncio
import json
from urllib.parse import urlparse

import chainlit as cl
from openai import AsyncOpenAI

client = AsyncOpenAI()

# ──────────────────────────────────────────────
# Markdown フォーマッター（ui_markdown と同一）
# ──────────────────────────────────────────────

def fmt_topic_output(_topic: str, sites: list[dict]) -> str:
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


async def generate_trivia(query: str) -> str:
    """クエリに関連する面白い豆知識を1つ生成する。"""
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=256,
        messages=[
            {
                "role": "system",
                "content": (
                    "あなたは博識なアシスタントです。"
                    "与えられたテーマに関連する、知っていると少し得する面白い豆知識を1つ、"
                    "日本語で3〜5文程度で教えてください。"
                    "「💡 豆知識：」で始めてください。"
                ),
            },
            {
                "role": "user",
                "content": f"「{query}」に関連する豆知識を1つ教えてください。",
            },
        ],
    )
    return response.choices[0].message.content or ""


MAX_SOURCES = 3


async def research_topic(query: str, topic: str) -> list[dict]:
    """web_search_preview で実際にウェブを検索し、ソース2〜3件を返す."""
    search_response = await client.responses.create(
        model="gpt-4o",
        tools=[{"type": "web_search_preview"}],
        input=(
            f"「{query}」の「{topic}」について調査し、"
            "参考になる情報ソース2〜3件の内容を要約してください。"
        ),
    )

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
# 集約・回答生成（ui_markdown と同一）
# ──────────────────────────────────────────────

async def aggregate(
    query: str, all_findings: list[str], answer_msg: cl.Message
) -> None:
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
    """ユーザーメッセージを受け取り、豆知識表示とリサーチを並行実行して回答を返す."""
    query = message.content
    all_findings: list[str] = []

    # トピック生成と豆知識生成を並行実行
    topics, trivia = await asyncio.gather(
        generate_topics(query),
        generate_trivia(query),
    )

    # 豆知識を即時表示
    await cl.Message(
        content=(
            f"{trivia}\n\n"
            f"---\n"
            f"*リサーチを開始しました。結果が出るまでしばらくお待ちください...*"
        )
    ).send()

    async with cl.Step(name="🔎 ウェブを調査しています", type="tool") as root_step:
        root_step.input = query

        for i, topic in enumerate(topics, 1):
            step_name = f"🔍 [{i}/{len(topics)}]「{topic}」を調査中"
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

                topic_step.output = fmt_topic_output(topic, sites)
                topic_step.name = (
                    f"✅ [{i}/{len(topics)}]「{topic}」完了 ({len(sites)}件)"
                )

        root_step.output = f"合計 {len(all_findings)} 件のソースを調査しました"

    answer_msg = cl.Message(content="")
    await answer_msg.send()
    await aggregate(query, all_findings, answer_msg)
    await answer_msg.update()
```

---

## 各実装の比較まとめ

| 実装 | 追加機能 | 変更箇所 | 主な効果 |
|------|---------|---------|---------|
| `step_child_base` | ベース実装 | — | Step ツリーで処理の入れ子を可視化 |
| `step_child_ui_markdown` | Markdown テーブル | `topic_step.output` のフォーマット | トピック単位でソース一覧・信頼度を一覧表示 |
| `step_child_ui_tasklist` | `cl.TaskList` | `task_list` の追加・ライフサイクル管理 | サイドバーで全体進捗をリアルタイム確認 |
| `step_child_ui_chart` | `cl.Plotly` ヒートマップ | `all_sites` 蓄積・`make_reliability_chart()` | 全ソースの信頼度を2次元で視覚的に比較 |
| `step_child_ui_trivia` | `asyncio.gather` + 豆知識 | `generate_trivia()` + Step名動的更新 | 待機時間に関連情報を提供、進捗が明確 |

### 各アプローチの使い所

- **Markdown**：最小コストで Step 出力を人間が読みやすくしたいとき
- **TaskList**：処理ステップが多く、全体進捗管理が重要なとき
- **Plotly**：数値・スコアデータを比較・可視化したいとき
- **Trivia（asyncio.gather）**：待機時間が長いアプリで、UX 上の「間」を埋めたいとき

### 実装上の共通パターン

これらの改善に共通する Chainlit の実装パターン：

1. **Step の `output` は Markdown が使える**：テーブル・blockquote・コードブロック等
2. **Step 名は `with` ブロック内で更新可能**：`step.name = "新しい名前"`
3. **`cl.Message` + `elements` で任意のUIを挿入**：`cl.Plotly`、`cl.Image` など
4. **ストリーミングは `cl.Message` と `cl.Step` に同時送信できる**
5. **`asyncio.gather` で独立した API 呼び出しは並行化できる**

---

## スクリーンショット

（ここに実際のUIスクリーンショットを挿入）

- [ ] `step_child_base` のUI
- [ ] `step_child_ui_markdown` のUI（Markdownテーブル展開状態）
- [ ] `step_child_ui_tasklist` のUI（TaskList サイドバー）
- [ ] `step_child_ui_chart` のUI（ヒートマップチャート）
- [ ] `step_child_ui_trivia` のUI（豆知識カード + Step名変化）
