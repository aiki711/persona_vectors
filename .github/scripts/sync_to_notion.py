import os
import requests
from datetime import datetime
import re

TOKEN = os.environ["NOTION_TOKEN"]
DATABASE_ID = os.environ["NOTION_DATABASE_ID"]
REPO_NAME = os.environ["REPO_NAME"]

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28",
}

def get_weekly_commits():
    import subprocess
    # コミットメッセージとハッシュを取得
    cmd = ['git', 'log', '--since="1 week ago"', '--pretty=format:%s|%h']
    result = subprocess.run(cmd, capture_output=True, text=True)
    if not result.stdout.strip():
        return []
    return [line.split('|') for line in result.stdout.strip().split('\n')]

def create_notion_blocks(commits):
    if not commits:
        return [{"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"text": {"content": "今週の更新はありませんでした。"}}]}}]

    # カテゴリの定義
    categories = {
        "Features ✨": ["feat"],
        "Fixes 🛠️": ["fix"],
        "Refactoring ♻️": ["refactor"],
        "Chores/Docs 📝": ["chore", "docs", "test"],
        "Others 📄": []
    }
    
    grouped = {cat: [] for cat in categories}

    for msg, hash_id in commits:
        found = False
        for cat, keywords in categories.items():
            if any(msg.lower().startswith(k) for k in keywords):
                grouped[cat].append((msg, hash_id))
                found = True
                break
        if not found:
            grouped["Others 📄"].append((msg, hash_id))

    blocks = []
    for cat, items in grouped.items():
        if not items:
            continue
        
        # カテゴリの見出し
        blocks.append({
            "object": "block",
            "type": "heading_3",
            "heading_3": {"rich_text": [{"text": {"content": cat}}]}
        })
        
        # 各コミットを箇条書きで追加 (最大100ブロック制限を考慮しつつ)
        for msg, hash_id in items[:15]: # 各カテゴリ15件までに制限
            url = f"https://github.com/{os.environ.get('GITHUB_REPOSITORY')}/commit/{hash_id}"
            blocks.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [
                        {"text": {"content": f"{msg} "}},
                        {
                            "text": {"content": f"({hash_id})", "link": {"url": url}},
                            "annotations": {"code": True, "color": "gray"}
                        }
                    ]
                }
            })
    return blocks

def create_notion_page(blocks):
    url = "https://api.notion.com/v1/pages"
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    payload = {
        "parent": {"database_id": DATABASE_ID},
        "properties": {
            "Name": {"title": [{"text": {"content": f"{REPO_NAME} 進捗 ({date_str}週)"}}]},
            "Date": {"date": {"start": date_str}}
        },
        "children": blocks
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()

if __name__ == "__main__":
    commits = get_weekly_commits()
    blocks = create_notion_blocks(commits)
    response = create_notion_page(blocks)
    print(f"Notion APIからの返答: {response}")
