package com.mesh.client.data.local.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import com.mesh.client.data.local.db.entities.RosterEntry
import kotlinx.coroutines.flow.Flow

@Dao
interface RosterDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(entry: RosterEntry)

    @Update
    suspend fun update(entry: RosterEntry)

    @Query("SELECT * FROM roster WHERE node_id = :nodeId")
    suspend fun getByNodeId(nodeId: String): RosterEntry?

    @Query("SELECT * FROM roster WHERE is_online = 1 ORDER BY node_type, nickname")
    fun getOnlineNodes(): Flow<List<RosterEntry>>

    @Query("SELECT * FROM roster ORDER BY is_online DESC, node_type, nickname")
    fun getAllNodes(): Flow<List<RosterEntry>>

    @Query("SELECT * FROM roster")
    fun getAllRoster(): Flow<List<RosterEntry>>

    @Query("SELECT * FROM roster")
    suspend fun getAllNodesSnapshot(): List<RosterEntry>

    @Query("SELECT * FROM roster WHERE node_type != 'user' ORDER BY is_online DESC, nickname")
    fun getAgents(): Flow<List<RosterEntry>>

    @Query("SELECT * FROM roster WHERE node_type = 'user' ORDER BY is_online DESC, nickname")
    fun getUsers(): Flow<List<RosterEntry>>

    @Query("UPDATE roster SET is_online = 0 WHERE node_id = :nodeId")
    suspend fun markOffline(nodeId: String)

    @Query("UPDATE roster SET is_online = 1, last_seen = :timestamp WHERE node_id = :nodeId")
    suspend fun markOnline(nodeId: String, timestamp: String)

    @Query("UPDATE roster SET is_online = 0")
    suspend fun markAllOffline()

    @Query("SELECT * FROM roster WHERE nickname LIKE :query OR node_id LIKE :query")
    suspend fun search(query: String): List<RosterEntry>

    @Query("DELETE FROM roster WHERE node_id = :nodeId")
    suspend fun delete(nodeId: String)

    @Query("DELETE FROM roster")
    suspend fun deleteAll()

    @Query("""
        UPDATE roster SET
            agent_state = :state,
            context_tokens = :contextTokens,
            worker_elapsed_secs = :workerElapsed,
            history_utilization_pct = :historyPct,
            memory_pool_size = :memPool,
            memory_active_size = :memActive,
            active_map = :activeMap
        WHERE node_id = :nodeId
    """)
    suspend fun updateStatus(
        nodeId: String,
        state: String,
        contextTokens: Int,
        workerElapsed: Float?,
        historyPct: Float?,
        memPool: Int?,
        memActive: Int?,
        activeMap: String
    )
}
