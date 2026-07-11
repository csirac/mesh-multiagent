package com.mesh.client.data.local.db

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import com.mesh.client.data.local.db.entities.ChannelEntity
import com.mesh.client.data.local.db.entities.ChannelMemberEntity
import com.mesh.client.data.local.db.entities.DraftEntity
import com.mesh.client.data.local.db.entities.MessageEntity
import com.mesh.client.data.local.db.entities.RosterEntry
import com.mesh.client.data.local.db.entities.WorkspaceNote

@Database(
    entities = [
        MessageEntity::class,
        RosterEntry::class,
        ChannelEntity::class,
        ChannelMemberEntity::class,
        WorkspaceNote::class,
        DraftEntity::class
    ],
    version = 9,
    exportSchema = true
)
abstract class MeshDatabase : RoomDatabase() {

    abstract fun messageDao(): MessageDao
    abstract fun rosterDao(): RosterDao
    abstract fun channelDao(): ChannelDao
    abstract fun workspaceDao(): WorkspaceDao

    companion object {
        private const val DATABASE_NAME = "mesh_database"

        @Volatile
        private var INSTANCE: MeshDatabase? = null

        fun getInstance(context: Context): MeshDatabase {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: buildDatabase(context).also { INSTANCE = it }
            }
        }

        private fun buildDatabase(context: Context): MeshDatabase {
            return Room.databaseBuilder(
                context.applicationContext,
                MeshDatabase::class.java,
                DATABASE_NAME
            )
                .fallbackToDestructiveMigration()
                .build()
        }
    }
}
