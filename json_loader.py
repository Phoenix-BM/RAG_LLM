import json
from langchain.schema import Document

class UnstructuredJSONLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            documents = []
            for entry in data:
                url = entry.get("url", "")
                title = entry.get("title", "")
                content = entry.get("content", "")
                full_text = f"{title}\n\n{url}\n\n{content}"
                documents.append(Document(page_content=full_text))
            return documents
