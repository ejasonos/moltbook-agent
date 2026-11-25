from urllib.parse import quote

from integrations.moltbook.client import MoltbookClient


def get_submolt(
    client: MoltbookClient,
    submolt: str,
):
    name = quote(submolt, safe="")
    return client.get(f"/submolts/{name}")


def subscribe_submolt(
    client: MoltbookClient,
    submolt: str,
):
    name = quote(submolt, safe="")
    return client.post(f"/submolts/{name}/subscribe")


def unsubscribe_submolt(
    client: MoltbookClient,
    submolt: str,
):
    name = quote(submolt, safe="")
    return client.delete(f"/submolts/{name}/subscribe")