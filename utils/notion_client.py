# utils/notion_client.py

from notion_client import Client
from dotenv import load_dotenv
import os

load_dotenv()


def fetch_notion_data(token, database_id):
    notion = Client(auth=token)
    results = []
    has_more = True
    start_cursor = None

    while has_more:
        response = notion.databases.query(
            **{"database_id": database_id, "start_cursor": start_cursor}
        )
        results.extend(response.get("results", []))
        has_more = response.get("has_more", False)
        start_cursor = response.get("next_cursor", None)

    return results


if __name__ == "__main__":
    NOTION_TOKEN = os.getenv("NOTION_API_KEY")
    DATABASE_ID = "fffa17843b028105a10e000c206fcdd8"
    data = fetch_notion_data(NOTION_TOKEN, DATABASE_ID)
    print(f"Fetched {len(data)} records from Notion.")
