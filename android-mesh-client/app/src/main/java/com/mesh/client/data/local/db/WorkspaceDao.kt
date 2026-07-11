package com.mesh.client.data.local.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import androidx.room.Update
import com.mesh.client.data.local.db.entities.WorkspaceNote
import kotlinx.coroutines.flow.Flow

@Dao
interface WorkspaceDao {
    @Query("SELECT * FROM workspace_notes ORDER BY updated_at DESC")
    fun getAllNotes(): Flow<List<WorkspaceNote>>

    @Query("SELECT * FROM workspace_notes ORDER BY updated_at DESC LIMIT 1")
    fun getLatestNote(): Flow<WorkspaceNote?>

    @Insert
    suspend fun insert(note: WorkspaceNote): Long

    @Update
    suspend fun update(note: WorkspaceNote)

    @Query("DELETE FROM workspace_notes WHERE id = :id")
    suspend fun delete(id: Long)

    @Query("DELETE FROM workspace_notes")
    suspend fun deleteAll()
}
