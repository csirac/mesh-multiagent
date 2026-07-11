package com.mesh.client.data.local.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.mesh.client.data.local.db.entities.DraftEntity
import com.mesh.client.data.local.db.entities.MessageEntity
import com.mesh.client.data.local.db.entities.MessageStatus
import kotlinx.coroutines.flow.Flow

@Dao
interface MessageDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(message: MessageEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(messages: List<MessageEntity>)

    @Insert(onConflict = OnConflictStrategy.IGNORE)
    suspend fun insertIfNotExists(message: MessageEntity)

    @Insert(onConflict = OnConflictStrategy.IGNORE)
    suspend fun insertAllIfNotExists(messages: List<MessageEntity>)

    @Query("SELECT * FROM messages WHERE id = :id")
    suspend fun getById(id: String): MessageEntity?

    @Query("SELECT * FROM messages WHERE conversation_id = :conversationId ORDER BY timestamp ASC")
    fun getConversationMessages(conversationId: String): Flow<List<MessageEntity>>

    @Query("SELECT * FROM messages WHERE conversation_id = :conversationId ORDER BY timestamp DESC LIMIT :limit")
    suspend fun getRecentConversationMessages(conversationId: String, limit: Int): List<MessageEntity>

    @Query("""
        SELECT * FROM messages
        WHERE conversation_id IN (
            SELECT DISTINCT conversation_id FROM messages
        )
        GROUP BY conversation_id
        HAVING timestamp = MAX(timestamp)
        ORDER BY timestamp DESC
    """)
    fun getConversationPreviews(): Flow<List<MessageEntity>>

    @Query("SELECT * FROM messages WHERE is_read = 0 AND is_outgoing = 0")
    fun getUnreadMessages(): Flow<List<MessageEntity>>

    @Query("SELECT COUNT(*) FROM messages WHERE is_read = 0 AND is_outgoing = 0")
    fun getUnreadCount(): Flow<Int>

    @Query("UPDATE messages SET is_read = 1 WHERE conversation_id = :conversationId")
    suspend fun markConversationAsRead(conversationId: String)

    @Query("UPDATE messages SET is_read = 1 WHERE id = :messageId")
    suspend fun markAsRead(messageId: String)

    @Query("UPDATE messages SET status = :status WHERE id = :messageId")
    suspend fun updateStatus(messageId: String, status: MessageStatus)

    @Query("DELETE FROM messages WHERE conversation_id = :conversationId")
    suspend fun deleteConversation(conversationId: String)

    @Query("DELETE FROM messages WHERE id = :messageId")
    suspend fun deleteMessage(messageId: String)

    @Query("DELETE FROM messages")
    suspend fun deleteAll()

    @Query("SELECT * FROM messages ORDER BY timestamp DESC LIMIT :limit")
    fun getRecentMessages(limit: Int): Flow<List<MessageEntity>>

    /**
     * Get the current user's node ID from outgoing messages.
     * Returns the from_node from the most recent outgoing message, or null if none exist.
     */
    @Query("SELECT from_node FROM messages WHERE is_outgoing = 1 ORDER BY timestamp DESC LIMIT 1")
    suspend fun getCurrentUserNodeId(): String?

    // --- Drafts ---

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun saveDraft(draft: DraftEntity)

    @Query("SELECT text FROM drafts WHERE conversation_key = :conversationKey")
    suspend fun getDraft(conversationKey: String): String?

    @Query("DELETE FROM drafts WHERE conversation_key = :conversationKey")
    suspend fun deleteDraft(conversationKey: String)
}
