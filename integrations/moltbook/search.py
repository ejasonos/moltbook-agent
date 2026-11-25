from integrations.moltbook.client import MoltbookClient


def search_moltbook(
    client: MoltbookClient,
    query: str,
    limit: int = 20,
):
    return client.get(
        "/search",
        params={
            "q": query,
            "limit": limit,
        },
    )