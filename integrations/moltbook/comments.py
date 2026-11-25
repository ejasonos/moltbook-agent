from integrations.moltbook.client import MoltbookClient


def get_comments(
    client: MoltbookClient,
    post_id: str,
    sort: str = "best",
    limit: int = 35,
    cursor: str | None = None,
):
    params = {
        "sort": sort,
        "limit": limit,
    }

    if cursor:
        params["cursor"] = cursor

    return client.get(
        f"/posts/{post_id}/comments",
        params=params,
    )


def create_comment(
    client: MoltbookClient,
    post_id: str,
    content: str,
    parent_id: str | None = None,
):
    payload = {"content": content}

    if parent_id:
        payload["parent_id"] = parent_id

    return client.post(
        f"/posts/{post_id}/comments",
        payload,
    )


def reply_to_comment(
    client: MoltbookClient,
    post_id: str,
    comment_id: str,
    content: str,
):
    return create_comment(
        client=client,
        post_id=post_id,
        content=content,
        parent_id=comment_id,
    )