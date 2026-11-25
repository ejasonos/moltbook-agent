from urllib.parse import quote

from integrations.moltbook.client import MoltbookClient


def get_agent_profile(
    client: MoltbookClient,
    agent_name: str,
):
    name = quote(agent_name, safe="")
    return client.get(f"/agents/profile?name={name}")


def follow_agent(
    client: MoltbookClient,
    agent_name: str,
):
    name = quote(agent_name, safe="")
    return client.post(f"/agents/{name}/follow")


def unfollow_agent(
    client: MoltbookClient,
    agent_name: str,
):
    name = quote(agent_name, safe="")
    return client.delete(f"/agents/{name}/follow")