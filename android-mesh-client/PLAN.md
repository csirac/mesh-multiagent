# Android Mesh Client - Implementation Plan

## Overview

Native Kotlin Android client for the mesh agent system. Provides a mobile interface for:
- Messaging agents (alice, bob, etc.)
- Receiving push notifications for incoming messages
- Viewing agent roster and status
- Handling confirmation requests from agents

## Architecture

### Layers

```
┌─────────────────────────────────────────────────────────────┐
│                     UI Layer (Jetpack Compose)               │
│  ConversationScreen │ RosterScreen │ SettingsScreen          │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                     ViewModel Layer                          │
│  ChatViewModel │ RosterViewModel │ ConnectionViewModel       │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                     Repository Layer                         │
│  MessageRepository │ RosterRepository │ ConnectionState      │
└────────────┬───────────────────────────────────┬────────────┘
             │                                   │
┌────────────▼────────────┐    ┌─────────────────▼────────────┐
│     Local Storage        │    │     Remote (WebSocket)       │
│  Room DB │ Preferences   │    │  MeshSocketClient            │
└──────────────────────────┘    └──────────────────────────────┘
```

### Foreground Service

The `MeshService` maintains the WebSocket connection in the foreground:
- Shows persistent notification with connection status
- Reconnects on network changes
- Delivers incoming messages as notifications
- Survives app backgrounding

## Project Structure

```
android-mesh-client/
├── app/
│   ├── src/main/
│   │   ├── java/com/mesh/client/
│   │   │   ├── MeshApplication.kt           # Application class, DI setup
│   │   │   │
│   │   │   ├── data/
│   │   │   │   ├── local/
│   │   │   │   │   ├── db/
│   │   │   │   │   │   ├── MeshDatabase.kt      # Room database
│   │   │   │   │   │   ├── MessageDao.kt        # Message queries
│   │   │   │   │   │   ├── RosterDao.kt         # Roster queries
│   │   │   │   │   │   └── entities/
│   │   │   │   │   │       ├── MessageEntity.kt
│   │   │   │   │   │       └── RosterEntry.kt
│   │   │   │   │   └── prefs/
│   │   │   │   │       └── UserPreferences.kt   # DataStore preferences
│   │   │   │   │
│   │   │   │   ├── remote/
│   │   │   │   │   ├── MeshSocketClient.kt      # WebSocket client
│   │   │   │   │   └── protocol/
│   │   │   │   │       ├── Message.kt           # Message data class
│   │   │   │   │       ├── MessageType.kt       # Enum
│   │   │   │   │       ├── ControlAction.kt     # Enum
│   │   │   │   │       └── MessageFactory.kt    # Convenience constructors
│   │   │   │   │
│   │   │   │   └── repository/
│   │   │   │       ├── MessageRepository.kt
│   │   │   │       ├── RosterRepository.kt
│   │   │   │       └── ConnectionRepository.kt
│   │   │   │
│   │   │   ├── service/
│   │   │   │   └── MeshService.kt               # Foreground service
│   │   │   │
│   │   │   ├── ui/
│   │   │   │   ├── MainActivity.kt              # Single activity
│   │   │   │   ├── navigation/
│   │   │   │   │   └── MeshNavigation.kt        # Nav graph
│   │   │   │   ├── chat/
│   │   │   │   │   ├── ChatScreen.kt
│   │   │   │   │   ├── ChatViewModel.kt
│   │   │   │   │   └── components/
│   │   │   │   │       ├── MessageBubble.kt
│   │   │   │   │       ├── InputBar.kt
│   │   │   │   │       └── ConfirmDialog.kt
│   │   │   │   ├── roster/
│   │   │   │   │   ├── RosterScreen.kt
│   │   │   │   │   └── RosterViewModel.kt
│   │   │   │   ├── settings/
│   │   │   │   │   ├── SettingsScreen.kt
│   │   │   │   │   └── SettingsViewModel.kt
│   │   │   │   └── theme/
│   │   │   │       ├── Theme.kt
│   │   │   │       ├── Color.kt
│   │   │   │       └── Type.kt
│   │   │   │
│   │   │   └── util/
│   │   │       ├── NetworkMonitor.kt
│   │   │       └── NotificationHelper.kt
│   │   │
│   │   ├── res/
│   │   │   ├── values/
│   │   │   │   ├── strings.xml
│   │   │   │   └── themes.xml
│   │   │   └── drawable/
│   │   │       └── ic_notification.xml
│   │   │
│   │   └── AndroidManifest.xml
│   │
│   └── build.gradle.kts
│
├── build.gradle.kts                             # Root build file
├── settings.gradle.kts
├── gradle.properties
└── gradle/
    └── libs.versions.toml                       # Version catalog
```

## Protocol Implementation

### Message Types (matching mesh/protocol.py)

```kotlin
enum class MessageType(val value: String) {
    MESSAGE("message"),
    TOOL_REQUEST("tool_request"),
    TOOL_RESULT("tool_result"),
    CONTROL("control"),
    CONFIRM_REQUEST("confirm_request"),
    CONFIRM_RESPONSE("confirm_response"),
    PRESENCE("presence"),
    STATUS_REQUEST("status_request"),
    STATUS_RESPONSE("status_response")
}
```

### Wire Format

The mesh uses length-prefixed JSON over TCP:
- 4-byte big-endian length prefix
- UTF-8 JSON payload

```kotlin
fun encodeForWire(msg: Message): ByteArray {
    val json = Json.encodeToString(msg)
    val payload = json.encodeToByteArray()
    val length = ByteBuffer.allocate(4).putInt(payload.size).array()
    return length + payload
}

fun decodeLengthPrefix(data: ByteArray): Int {
    return ByteBuffer.wrap(data.copyOfRange(0, 4)).int
}
```

### Message Data Class

```kotlin
@Serializable
data class Message(
    val id: String = generateMessageId(),
    @SerialName("from_node") val fromNode: String,
    @SerialName("to_node") val toNode: String,
    val type: MessageType,
    val content: JsonElement,  // String or Dict
    val timestamp: String = nowIso(),
    @SerialName("in_reply_to") val inReplyTo: String? = null,
    val metadata: Map<String, JsonElement> = emptyMap()
)
```

## Implementation Phases

### Phase 1: Core Infrastructure

1. **Project Setup**
   - Gradle configuration with version catalog
   - Android Manifest with permissions
   - Application class with Hilt DI

2. **Protocol Layer**
   - Message data classes with kotlinx.serialization
   - Wire encoding/decoding
   - Factory functions for common message types

3. **WebSocket Client**
   - OkHttp WebSocket connection
   - Length-prefixed message framing
   - Automatic reconnection
   - Auth token support

4. **Room Database**
   - Message entity and DAO
   - Roster entity and DAO
   - Migrations setup

### Phase 2: Service Layer

1. **Foreground Service**
   - Persistent notification
   - WebSocket lifecycle management
   - Network state monitoring

2. **Notifications**
   - Notification channels
   - Message notifications with reply action
   - Confirmation request notifications

### Phase 3: UI Layer

1. **Navigation**
   - Bottom nav: Chat | Roster | Settings
   - Compose Navigation setup

2. **Chat Screen**
   - Message list (LazyColumn)
   - Input bar with send button
   - Target selector
   - Confirmation dialogs

3. **Roster Screen**
   - List of connected agents
   - Status indicators (online/offline)
   - Tap to set as target

4. **Settings Screen**
   - Server URL/port
   - Auth token
   - Nickname
   - Notification preferences

### Phase 4: Polish

1. **Reconnection Logic**
   - Exponential backoff
   - Network change listener
   - Manual reconnect button

2. **Message Rendering**
   - Markdown support (via Markwon)
   - Code block highlighting
   - Timestamp formatting

3. **Error Handling**
   - Connection errors
   - Auth failures
   - Offline mode indicators

## Dependencies

```toml
[versions]
kotlin = "1.9.22"
compose-bom = "2024.01.00"
room = "2.6.1"
okhttp = "4.12.0"
hilt = "2.50"
serialization = "1.6.2"
datastore = "1.0.0"
markwon = "4.6.2"

[libraries]
# Compose
compose-bom = { group = "androidx.compose", name = "compose-bom", version.ref = "compose-bom" }
compose-material3 = { group = "androidx.compose.material3", name = "material3" }
compose-navigation = { group = "androidx.navigation", name = "navigation-compose", version = "2.7.6" }

# Room
room-runtime = { group = "androidx.room", name = "room-runtime", version.ref = "room" }
room-ktx = { group = "androidx.room", name = "room-ktx", version.ref = "room" }
room-compiler = { group = "androidx.room", name = "room-compiler", version.ref = "room" }

# Network
okhttp = { group = "com.squareup.okhttp3", name = "okhttp", version.ref = "okhttp" }

# DI
hilt-android = { group = "com.google.dagger", name = "hilt-android", version.ref = "hilt" }
hilt-compiler = { group = "com.google.dagger", name = "hilt-compiler", version.ref = "hilt" }

# Serialization
kotlinx-serialization = { group = "org.jetbrains.kotlinx", name = "kotlinx-serialization-json", version.ref = "serialization" }

# DataStore
datastore = { group = "androidx.datastore", name = "datastore-preferences", version.ref = "datastore" }

# Markdown
markwon-core = { group = "io.noties.markwon", name = "core", version.ref = "markwon" }
```

## Key Considerations

### Battery & Background

- Use `FOREGROUND_SERVICE` type `DATA_SYNC`
- Request `SCHEDULE_EXACT_ALARM` for reconnection
- Handle Doze mode by using high-priority FCM (future)

### Permissions

```xml
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
<uses-permission android:name="android.permission.FOREGROUND_SERVICE_DATA_SYNC" />
<uses-permission android:name="android.permission.POST_NOTIFICATIONS" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
```

### Security

- Auth token stored in EncryptedSharedPreferences
- TLS for production (configurable for local dev)
- No sensitive data in notifications

## Current Status (2026-01-24)

### Completed
- **Phase 1-3: COMPLETE** - Core infrastructure, service layer, and UI layer all functional
- **Phase 4: MOSTLY COMPLETE** - Reconnection, error handling working

### Recent Session Fixes
1. **Navigation bug fix** (`MeshNavigation.kt`) - Fixed MeshBottomBar state restoration issue when navigating from chat detail view. Root cause: `restoreState=true` caused Roster tab to restore its forward-navigation state (to Chat). Fix: Disable state restore when navigating from `chat/{nodeId}` route.

2. **Agent status query** (`StatusManager.kt`, `ChatViewModel.kt`, `ChatScreen.kt`) - Added info button in chat TopAppBar to query agent status. Shows loading dialog, then displays agent's recent context (messages, tool calls, tool results).

3. **Keyboard scroll fix** (`ChatScreen.kt`) - Added `LaunchedEffect` to scroll to latest message when IME keyboard becomes visible.

4. **Status dialog crash fix** (`ChatScreen.kt`) - Replaced `CircularProgressIndicator` with simple text to avoid Compose animation API version mismatch.

5. **Markdown + LaTeX rendering** (`MarkdownText.kt`, `ChatScreen.kt`) - Added Markwon integration with:
   - Standard markdown (bold, italic, headers, lists, code blocks)
   - Strikethrough and tables
   - JLatexMath for LaTeX math rendering (`$...$` inline, `$$...$$` block)
   - Auto-conversion of `\(...\)` → `$...$` and `\[...\]` → `$$...$$` for agent output

### Remaining Work
1. **Message search** - Full-text search across messages
2. **UI automation tests** - Comprehensive instrumentation tests for navigation and confirmation flows

### Completed Features
- **Channels** - Full channel support implemented (server + Android client):
  - `ChannelListScreen.kt` - Channel list UI with expandable menu (create/join/leave/delete/invite)
  - `ChannelListViewModel.kt` - Channel state management with roster integration
  - `ChannelEntity.kt`, `ChannelDao.kt` - Local persistence
  - Protocol support for `channel:` addressing in `MessageFactory.kt`
  - Channel message routing in `MeshService.kt`
  - Invite dialog with scrollable roster list showing online status

### Recently Completed (2026-01-24)
- **Conversation list view** (`ConversationListScreen.kt`, `ConversationListViewModel.kt`) - Groups messages by conversation, shows last message preview, unread badges, online status indicators. Set as home screen in navigation.

## Next Steps

1. ~~Create Gradle project structure~~ ✓
2. ~~Implement protocol data classes~~ ✓
3. ~~Build WebSocket client~~ ✓
4. ~~Create basic UI shell~~ ✓
5. ~~Implement foreground service~~ ✓
6. ~~Add notification handling~~ ✓
7. ~~Polish and test~~ (in progress)
8. ~~Add Markwon for markdown rendering~~ ✓
9. ~~Add conversation list view~~ ✓
10. ~~Add channel support~~ ✓
11. **Add message search**
12. **Write instrumentation tests**

## Timeline Estimate

Not providing time estimates per project guidelines. The phases are roughly ordered by dependency - each phase enables the next.
