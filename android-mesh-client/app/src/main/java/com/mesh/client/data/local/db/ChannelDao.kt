package com.mesh.client.data.local.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Transaction
import com.mesh.client.data.local.db.entities.ChannelEntity
import com.mesh.client.data.local.db.entities.ChannelMemberEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface ChannelDao {

    // --- Channel operations ---

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertChannel(channel: ChannelEntity)

    @Query("SELECT * FROM channels WHERE name = :name")
    suspend fun getChannel(name: String): ChannelEntity?

    @Query("SELECT * FROM channels ORDER BY name")
    fun getAllChannels(): Flow<List<ChannelEntity>>

    @Query("SELECT EXISTS(SELECT 1 FROM channels WHERE name = :name)")
    suspend fun channelExists(name: String): Boolean

    @Query("DELETE FROM channels WHERE name = :name")
    suspend fun deleteChannel(name: String)

    @Query("UPDATE channels SET member_count = :count WHERE name = :name")
    suspend fun updateMemberCount(name: String, count: Int)

    // --- Membership operations ---

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertMember(member: ChannelMemberEntity)

    @Query("DELETE FROM channel_members WHERE channel_name = :channelName AND node_id = :nodeId")
    suspend fun removeMember(channelName: String, nodeId: String)

    @Query("SELECT * FROM channel_members WHERE channel_name = :channelName ORDER BY joined_at")
    fun getChannelMembers(channelName: String): Flow<List<ChannelMemberEntity>>

    @Query("SELECT * FROM channel_members WHERE channel_name = :channelName")
    suspend fun getChannelMembersList(channelName: String): List<ChannelMemberEntity>

    @Query("SELECT EXISTS(SELECT 1 FROM channel_members WHERE channel_name = :channelName AND node_id = :nodeId)")
    suspend fun isMember(channelName: String, nodeId: String): Boolean

    @Query("SELECT * FROM channel_members WHERE node_id = :nodeId")
    fun getNodeChannels(nodeId: String): Flow<List<ChannelMemberEntity>>

    @Query("SELECT c.* FROM channels c INNER JOIN channel_members m ON c.name = m.channel_name WHERE m.node_id = :nodeId ORDER BY c.name")
    fun getMyChannels(nodeId: String): Flow<List<ChannelEntity>>

    @Query("SELECT COUNT(*) FROM channel_members WHERE channel_name = :channelName")
    suspend fun getMemberCount(channelName: String): Int

    // --- Bulk operations ---

    @Query("DELETE FROM channel_members WHERE channel_name = :channelName")
    suspend fun deleteAllMembers(channelName: String)

    @Query("DELETE FROM channels")
    suspend fun deleteAllChannels()

    @Query("DELETE FROM channel_members")
    suspend fun deleteAllMemberships()

    @Transaction
    suspend fun deleteAll() {
        deleteAllMemberships()
        deleteAllChannels()
    }
}
