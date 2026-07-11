package com.mesh.client.data.remote.protocol

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Types of messages in the mesh protocol.
 * Matches mesh/protocol.py MessageType enum.
 */
@Serializable
enum class MessageType(val value: String) {
    @SerialName("message")
    MESSAGE("message"),

    @SerialName("tool_request")
    TOOL_REQUEST("tool_request"),

    @SerialName("tool_result")
    TOOL_RESULT("tool_result"),

    @SerialName("control")
    CONTROL("control"),

    @SerialName("confirm_request")
    CONFIRM_REQUEST("confirm_request"),

    @SerialName("confirm_response")
    CONFIRM_RESPONSE("confirm_response"),

    @SerialName("presence")
    PRESENCE("presence"),

    @SerialName("status_request")
    STATUS_REQUEST("status_request"),

    @SerialName("status_response")
    STATUS_RESPONSE("status_response");

    companion object {
        fun fromValue(value: String): MessageType? =
            entries.find { it.value == value }
    }
}

/**
 * Control actions for managing nodes.
 * Matches mesh/protocol.py ControlAction enum.
 */
@Serializable
enum class ControlAction(val value: String) {
    @SerialName("spawn")
    SPAWN("spawn"),

    @SerialName("kill")
    KILL("kill"),

    @SerialName("status")
    STATUS("status"),

    @SerialName("pause")
    PAUSE("pause"),

    @SerialName("resume")
    RESUME("resume"),

    @SerialName("list_nodes")
    LIST_NODES("list_nodes"),

    @SerialName("register")
    REGISTER("register"),

    @SerialName("ack")
    ACK("ack"),

    @SerialName("register_push_token")
    REGISTER_PUSH_TOKEN("register_push_token"),

    // Channel operations
    @SerialName("channel_create")
    CHANNEL_CREATE("channel_create"),

    @SerialName("channel_delete")
    CHANNEL_DELETE("channel_delete"),

    @SerialName("channel_join")
    CHANNEL_JOIN("channel_join"),

    @SerialName("channel_leave")
    CHANNEL_LEAVE("channel_leave"),

    @SerialName("channel_list")
    CHANNEL_LIST("channel_list"),

    @SerialName("channel_members")
    CHANNEL_MEMBERS("channel_members"),

    @SerialName("channel_invite")
    CHANNEL_INVITE("channel_invite"),

    // Message sync operations
    @SerialName("history_sync")
    HISTORY_SYNC("history_sync"),

    @SerialName("history_response")
    HISTORY_RESPONSE("history_response"),

    @SerialName("mark_read")
    MARK_READ("mark_read"),

    // Message deletion
    @SerialName("delete_message")
    DELETE_MESSAGE("delete_message"),

    @SerialName("delete_conversation")
    DELETE_CONVERSATION("delete_conversation"),

    // CC usage
    @SerialName("cc_usage")
    CC_USAGE("cc_usage");

    companion object {
        fun fromValue(value: String): ControlAction? =
            entries.find { it.value == value }
    }
}
