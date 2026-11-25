from integrations.moltbook.client import MoltbookClient


def get_post(client: MoltbookClient, post_id: str):
    return client.get(f"/posts/{post_id}")


def create_post(
    client: MoltbookClient,
    submolt: str,
    title: str,
    content: str,
):
    return client.post(
        "/posts",
        {
            "submolt": submolt,
            "title": title,
            "content": content,
        },
    )


def delete_post(
    client: MoltbookClient,
    post_id: str,
):
    return client.delete(f"/posts/{post_id}")