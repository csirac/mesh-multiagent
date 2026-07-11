package com.mesh.client.data.local.db.entities

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * Room entity for storing channels.
 */
@Entity(tableName = "channels")
data class ChannelEntity(
    @PrimaryKey
    @ColumnInfo(name = "name")
    val name: String,

    @ColumnInfo(name = "description")
    val description: String = "",

    @ColumnInfo(name = "created_at")
    val createdAt: String,

    @ColumnInfo(name = "created_by")
    val createdBy: String,

    @ColumnInfo(name = "member_count")
    val memberCount: Int = 0
)

/**
 * Room entity for channel membership.
 */
@Entity(
    tableName = "channel_members",
    primaryKeys = ["channel_name", "node_id"],
    foreignKeys = [
        ForeignKey(
            entity = ChannelEntity::class,
            parentColumns = ["name"],
            childColumns = ["channel_name"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [
        Index(value = ["channel_name"]),
        Index(value = ["node_id"])
    ]
)
data class ChannelMemberEntity(
    @ColumnInfo(name = "channel_name")
    val channelName: String,

    @ColumnInfo(name = "node_id")
    val nodeId: String,

    @ColumnInfo(name = "joined_at")
    val joinedAt: String
)
