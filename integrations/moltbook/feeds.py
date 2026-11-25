from integrations.moltbook.client import MoltbookClient


def get_feed(
    client: MoltbookClient,
    sort: str = "hot",
    limit: int = 25,
    cursor: str | None = None,
):
    params = {
        "sort": sort,
        "limit": limit,
    }

    if cursor:
        params["cursor"] = cursor

    return client.get("/feed", params=params)


def get_posts(
    client: MoltbookClient,
    sort: str = "new",
    limit: int = 25,
    submolt: str | None = None,
    cursor: str | None = None,
):
    params = {
        "sort": sort,
        "limit": limit,
    }

    if submolt:
        params["submolt"] = submolt

    if cursor:
        params["cursor"] = cursor

    return client.get("/posts", params=params)


def get_submolt_feed(
    client: MoltbookClient,
    submolt: str,
    sort: str = "new",
    limit: int = 25,
):
    return client.get(
        f"/submolts/{submolt}/feed",
        params={
            "sort": sort,
            "limit": limit,
        },
    )