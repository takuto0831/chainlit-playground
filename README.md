# Chainlit Playground

このレポジトリは **Chainlit を試すためのサンドボックス** です。

## 開発者向けガイド

### アプリの構造

- 新しいアプリケーションは `src/chainlit_playground/` ディレクトリ以下に作成してください
- 作成したアプリは `src/chainlit_playground/main.py` の `APPS` に登録する必要があります

### アプリの切り替え

アプリケーションの切り替えは **`TARGET` 環境変数** で行います。詳細は [Makefile](Makefile) を参照してください。

```bash
make run TARGET=auth_demo   # auth_demo アプリを実行
make run TARGET=hello       # hello アプリを実行
```

### 環境変数の設定

環境変数の設定が必要な場合は：

1. `.env.example` をテンプレートとして参考にしてください
2. プロジェクトルートに `.env` ファイルを作成してください
3. 必要な環境変数を設定してください

例：認証機能を使用する場合は、以下のコマンドでシークレットを生成し、`.env` に設定してください。

```bash
chainlit create-secret
```

その後、生成されたシークレットを `.env` に設定：

```dotenv
CHAINLIT_AUTH_SECRET="生成されたシークレット"
```
