package com.mesh.client.data.remote.protocol

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.int
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.nio.ByteBuffer
import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import java.util.UUID

/**
 * Represents image content within a message.
 */
data class ImageContent(
    val data: String,           // base64 encoded image data
    val mimeType: String,       // e.g., "image/jpeg"
    val thumbnail: String?,     // base64 encoded thumbnail
    val width: Int,
    val height: Int,
    val sizeBytes: Int,
    val caption: String? = null
)

/**
 * Core message structure for mesh communication.
 *
 * Every message has:
 * - id: Unique identifier
 * - fromNode: Sender node ID (e.g., "user:yourname", "agent:coder:alice")
 * - toNode: Recipient node ID or "router" for control messages
 * - type: MessageType indicating the purpose
 * - content: The actual payload (string for messages, dict for structured data)
 * - timestamp: When the message was created
 * - inReplyTo: Optional reference to a previous message ID
 * - metadata: Optional additional data (tool name, control action, etc.)
 */
@Serializable
data class Message(
    val id: String = generateMessageId(),
    @SerialName("from_node") val fromNode: String,
    @SerialName("to_node") val toNode: String,
    val type: MessageType,
    val content: JsonElement,
    val timestamp: String = nowIso(),
    @SerialName("in_reply_to") val inReplyTo: String? = null,
    val metadata: JsonObject = JsonObject(emptyMap())
) {
    /**
     * Get content as a string (for MESSAGE type).
     */
    fun contentAsString(): String? = when (content) {
        is JsonPrimitive -> content.jsonPrimitive.content
        else -> null
    }

    /**
     * Get content as a JSON object (for structured types).
     */
    fun contentAsObject(): JsonObject? = when (content) {
        is JsonObject -> content.jsonObject
        else -> null
    }

    /**
     * Create a reply to this message, swapping from/to.
     */
    fun reply(
        newContent: JsonElement,
        newType: MessageType = MessageType.MESSAGE,
        newMetadata: JsonObject = JsonObject(emptyMap())
    ): Message = Message(
        fromNode = toNode,
        toNode = fromNode,
        type = newType,
        content = newContent,
        inReplyTo = id,
        metadata = newMetadata
    )

    companion object {
        private val json = Json {
            ignoreUnknownKeys = true
            encodeDefaults = true
        }

        /**
         * Serialize to JSON string.
         */
        fun toJson(msg: Message): String = json.encodeToString(serializer(), msg)

        /**
         * Deserialize from JSON string.
         */
        fun fromJson(data: String): Message = json.decodeFromString(serializer(), data)

        /**
         * Encode a message for transmission over TCP.
         * Format: 4-byte big-endian length + JSON payload
         */
        fun encodeForWire(msg: Message): ByteArray {
            val payload = toJson(msg).toByteArray(Charsets.UTF_8)
            val buffer = ByteBuffer.allocate(4 + payload.size)
            buffer.putInt(payload.size)
            buffer.put(payload)
            return buffer.array()
        }

        /**
         * Decode the 4-byte length prefix.
         */
        fun decodeLengthPrefix(data: ByteArray): Int {
            return ByteBuffer.wrap(data.copyOfRange(0, 4)).int
        }
    }
}

/**
 * Generate a unique message ID.
 */
fun generateMessageId(): String = "msg-${UUID.randomUUID().toString().take(12)}"

/**
 * Current UTC timestamp in ISO format.
 */
fun nowIso(): String = DateTimeFormatter.ISO_INSTANT.format(Instant.now().atOffset(ZoneOffset.UTC))

/**
 * Normalize an ISO timestamp to UTC format.
 * Handles both offset formats (e.g., "2026-02-03T09:28:40-06:00")
 * and UTC Z format (e.g., "2026-02-03T15:28:37Z").
 * Returns the input unchanged if parsing fails.
 */
fun normalizeToUtc(timestamp: String): String {
    return try {
        // Parse any ISO-8601 timestamp and convert to UTC
        val instant = java.time.OffsetDateTime.parse(timestamp).toInstant()
        DateTimeFormatter.ISO_INSTANT.format(instant)
    } catch (e: Exception) {
        // If parsing fails, try ZonedDateTime format
        try {
            val instant = java.time.ZonedDateTime.parse(timestamp).toInstant()
            DateTimeFormatter.ISO_INSTANT.format(instant)
        } catch (e2: Exception) {
            // Return original if all parsing fails
            timestamp
        }
    }
}

/**
 * Check if this message contains an image.
 */
fun Message.isImageMessage(): Boolean {
    val contentObj = contentAsObject() ?: return false
    val typeElement = contentObj["type"] ?: return false
    return typeElement.jsonPrimitive.content == "image"
}

/**
 * Extract ImageContent from this message if it's an image message.
 */
fun Message.getImageContent(): ImageContent? {
    val contentObj = contentAsObject() ?: return null
    val typeElement = contentObj["type"] ?: return null
    if (typeElement.jsonPrimitive.content != "image") return null

    return try {
        ImageContent(
            data = contentObj["data"]?.jsonPrimitive?.content ?: return null,
            mimeType = contentObj["mime_type"]?.jsonPrimitive?.content ?: "image/jpeg",
            thumbnail = contentObj["thumbnail"]?.jsonPrimitive?.content,
            width = contentObj["width"]?.jsonPrimitive?.intOrNull ?: 0,
            height = contentObj["height"]?.jsonPrimitive?.intOrNull ?: 0,
            sizeBytes = contentObj["size_bytes"]?.jsonPrimitive?.intOrNull ?: 0,
            caption = contentObj["caption"]?.jsonPrimitive?.content
        )
    } catch (e: Exception) {
        null
    }
}

/**
 * Get the display text for this message (caption for images, content for text).
 */
fun Message.getDisplayText(): String? {
    return if (isImageMessage()) {
        getImageContent()?.caption ?: "[Image]"
    } else {
        contentAsString()
    }
}
