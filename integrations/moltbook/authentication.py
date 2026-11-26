from integrations.moltbook.client import MoltbookClient

import os
from dotenv import load_dotenv
load_dotenv()
API_KEY=os.getenv("MOLTBOOK_API_KEY") 

def register_agent(name: str, description: str):
    client = MoltbookClient(api_key=API_KEY)

    return client.register(
        name=name,
        description=description,
    )

def check_claim_status(client: MoltbookClient):
    return client.get("/agents/status")

def get_my_profile(client: MoltbookClient):
    return client.get("/agents/me")