CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agents (
    agent_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS posts (
    post_id TEXT PRIMARY KEY,
    author_id TEXT,
    submolt TEXT,
    title TEXT,
    content TEXT,
    created_at TEXT,
    discovered_at TEXT NOT NULL,
    last_seen_at TEXT,
    FOREIGN KEY (author_id) REFERENCES agents(agent_id)
);

CREATE TABLE IF NOT EXISTS comments (
    comment_id TEXT PRIMARY KEY,
    post_id TEXT NOT NULL,
    parent_comment_id TEXT,
    author_id TEXT,
    content TEXT NOT NULL,
    created_at TEXT,
    retrieved_at TEXT NOT NULL,
    FOREIGN KEY (author_id) REFERENCES agents(agent_id)
);

CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_key TEXT UNIQUE NOT NULL,
    interaction_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    content TEXT,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    updated_at TEXT,
    resource_id TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    content TEXT NOT NULL,
    topic TEXT,
    importance INTEGER NOT NULL DEFAULT 3,
    source_id TEXT,
    created_at TEXT NOT NULL,
    last_accessed TEXT
);

CREATE TABLE IF NOT EXISTS topics (
    topic_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    interest_score REAL DEFAULT 0,
    last_discussed TEXT
);

CREATE TABLE IF NOT EXISTS heartbeat_state (
    state_key TEXT PRIMARY KEY,
    state_value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS action_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action_type TEXT NOT NULL,
    target_id TEXT,
    result TEXT NOT NULL,
    details TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pending_followups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_id TEXT NOT NULL,
    description TEXT NOT NULL,
    due_at TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS post_topics (
    post_id TEXT NOT NULL,
    topic_id INTEGER NOT NULL,
    PRIMARY KEY (post_id, topic_id),
    FOREIGN KEY (post_id) REFERENCES posts(post_id),
    FOREIGN KEY (topic_id) REFERENCES topics(topic_id)
);

CREATE INDEX IF NOT EXISTS idx_memories_topic
ON memories(topic);

CREATE INDEX IF NOT EXISTS idx_memories_category
ON memories(category);

CREATE INDEX IF NOT EXISTS idx_posts_author
ON posts(author_id);

CREATE INDEX IF NOT EXISTS idx_comments_post
ON comments(post_id, created_at);

CREATE INDEX IF NOT EXISTS idx_interactions_target
ON interactions(target_id);

CREATE INDEX IF NOT EXISTS idx_actions_created
ON action_log(created_at);
