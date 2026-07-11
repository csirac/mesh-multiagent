package com.mesh.client.ui.workspace

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.mesh.client.data.local.db.MeshDatabase
import com.mesh.client.data.local.db.entities.WorkspaceNote
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import javax.inject.Inject

data class WorkspaceUiState(
    val content: String = "",
    val isSaving: Boolean = false,
    val lastSaved: Long? = null
)

@HiltViewModel
class WorkspaceViewModel @Inject constructor(
    application: Application
) : AndroidViewModel(application) {

    private val database = MeshDatabase.getInstance(application)
    private val workspaceDao = database.workspaceDao()

    private val _uiState = MutableStateFlow(WorkspaceUiState())
    val uiState: StateFlow<WorkspaceUiState> = _uiState.asStateFlow()

    private var currentNoteId: Long? = null

    init {
        loadLatestNote()
    }

    private fun loadLatestNote() {
        viewModelScope.launch {
            workspaceDao.getLatestNote().collect { note ->
                if (note != null) {
                    currentNoteId = note.id
                    _uiState.value = _uiState.value.copy(
                        content = note.content,
                        lastSaved = note.updatedAt
                    )
                }
            }
        }
    }

    fun updateContent(content: String) {
        _uiState.value = _uiState.value.copy(content = content)
    }

    fun save() {
        val content = _uiState.value.content
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isSaving = true)

            val now = System.currentTimeMillis()
            if (currentNoteId != null) {
                workspaceDao.update(
                    WorkspaceNote(
                        id = currentNoteId!!,
                        content = content,
                        updatedAt = now
                    )
                )
            } else {
                currentNoteId = workspaceDao.insert(
                    WorkspaceNote(content = content)
                )
            }

            _uiState.value = _uiState.value.copy(
                isSaving = false,
                lastSaved = now
            )
        }
    }

    fun clear() {
        viewModelScope.launch {
            currentNoteId?.let { workspaceDao.delete(it) }
            currentNoteId = null
            _uiState.value = WorkspaceUiState()
        }
    }
}
