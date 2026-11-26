from integrations.moltbook.client import MoltbookClient

def register_agent(name: str, description: str):
    client = MoltbookClient()

    return client.register(
        name=name,
        description=description,
    )

def check_claim_status(client: MoltbookClient):
    return client.get("/agents/status")

def get_my_profile(client: MoltbookClient):
    return client.get("/agents/me")
