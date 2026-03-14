# Chainlit Playground

このレポジトリは **Chainlit を試すためのサンドボックス** です。

## 開発者向けガイド

### アプリの構造

- 新しいアプリケーションは `src/chainlit_playground/` ディレクトリ以下に作成してください
- 作成したアプリは `src/chainlit_playground/main.py` の `get_app_path()` 関数に登録する必要があります

### アプリの切り替え

アプリケーションの切り替えは **`TARGET` 環境変数** で行います。詳細は [Makefile](Makefile) を参照してください。

```bash
make run TARGET=auth_demo   # auth_demo アプリを実行
make run TARGET=hello       # hello アプリを実行
```

### 環境変数の設定

- .env.example をコピーして .envを作成してください。
- OPENAI_API_KEY の値をご自身で取得した API Keyに変更してください。.envは github に commit/pushしないよう注意してください。

```
OPENAI_API_KEY=your_openai_api_key_here

```
