package com.mesh.client.data.local.db.entities

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Room entity for tracking connected nodes (roster).
 */
@Entity(tableName = "roster")
data class RosterEntry(
    @PrimaryKey
    @ColumnInfo(name = "node_id")
    val nodeId: String,

    @ColumnInfo(name = "nickname")
    val nickname: String,

    @ColumnInfo(name = "node_type")
    val nodeType: String, // "user" or agent type like "coder"

    @ColumnInfo(name = "description")
    val description: String = "",

    @ColumnInfo(name = "is_online")
    val isOnline: Boolean = true,

    @ColumnInfo(name = "last_seen")
    val lastSeen: String = "", // ISO timestamp

    @ColumnInfo(name = "llm_backend")
    val llmBackend: String = "", // LLM backend (e.g., "openai", "anthropic", "claude-code")

    @ColumnInfo(name = "llm_model")
    val llmModel: String = "", // LLM model (e.g., "gpt-5.1", "opus")

    @ColumnInfo(name = "hostname")
    val hostname: String = "", // Hostname where the node is running

    // Heartbeat-lite status fields (updated from LIST_NODES response)
    @ColumnInfo(name = "agent_state")
    val agentState: String = "", // "idle", "busy", "planning"

    @ColumnInfo(name = "context_tokens")
    val contextTokens: Int = 0, // Estimated total context length

    @ColumnInfo(name = "worker_elapsed_secs")
    val workerElapsedSecs: Float? = null, // Seconds since worker started (when busy)

    @ColumnInfo(name = "history_utilization_pct")
    val historyUtilizationPct: Float? = null, // History window usage %

    @ColumnInfo(name = "memory_pool_size")
    val memoryPoolSize: Int? = null, // Total memories in pool

    @ColumnInfo(name = "memory_active_size")
    val memoryActiveSize: Int? = null, // Active memories in prompt

    @ColumnInfo(name = "active_map")
    val activeMap: String = "" // Active project map name (e.g., "data-analysis")
) {
    /**
     * Whether this is a user node (vs agent).
     */
    val isUser: Boolean
        get() = nodeType == "user"

    /**
     * Whether this is an agent node.
     */
    val isAgent: Boolean
        get() = nodeType != "user"
}
