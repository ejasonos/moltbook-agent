from integrations.moltbook.client import MoltbookClient


def upvote_post(
    client: MoltbookClient,
    post_id: str,
):
    return client.post(f"/posts/{post_id}/upvote")


def downvote_post(
    client: MoltbookClient,
    post_id: str,
):
    return client.post(f"/posts/{post_id}/downvote")


def upvote_comment(
    client: MoltbookClient,
    comment_id: str,
):
    return client.post(f"/comments/{comment_id}/upvote")