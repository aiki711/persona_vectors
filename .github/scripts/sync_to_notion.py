import os
import requests
from datetime import datetime, timedelta

# 環境変数の取得
TOKEN = os.environ["NOTION_TOKEN"]
DATABASE_ID = os.environ["NOTION_DATABASE_ID"]
REPO_NAME = os.environ["REPO_NAME"]

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28",
}

def get_weekly_commits():
    # 過去7日間のコミットを取得
    import subprocess
    cmd = ['git', 'log', '--since="1 week ago"', '--pretty=format:%s (%h)']
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()

def create_notion_page(content):
    url = "https://api.notion.com/v1/pages"
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    # データベースに新規ページを作成
    payload = {
        "parent": {"database_id": DATABASE_ID},
        "properties": {
            "Name": {"title": [{"text": {"content": f"{REPO_NAME} 進捗 ({date_str}週)"}}]},
            "Date": {"date": {"start": date_str}}
        },
        "children": [
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": [{"text": {"content": "今週のコミット履歴"}}]}
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"text": {"content": content if content else "今週の更新はありませんでした。"}}]
                }
            }
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()

if __name__ == "__main__":
    commits = get_weekly_commits()
    create_notion_page(commits)
