from urllib.parse import quote

from integrations.moltbook.client import MoltbookClient


def upvote_post(
    client: MoltbookClient,
    post_id: str,
):
    return client.post(f"/posts/{quote(post_id, safe='')}/upvote")


def downvote_post(
    client: MoltbookClient,
    post_id: str,
):
    return client.post(f"/posts/{quote(post_id, safe='')}/downvote")


def upvote_comment(
    client: MoltbookClient,
    comment_id: str,
):
    return client.post(f"/comments/{quote(comment_id, safe='')}/upvote")
