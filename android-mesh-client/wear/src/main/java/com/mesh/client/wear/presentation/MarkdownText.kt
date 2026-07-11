package com.mesh.client.wear.presentation

import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.wear.compose.material.MaterialTheme
import androidx.wear.compose.material.Text

/**
 * Simple markdown text renderer for Wear OS.
 * Supports: **bold**, *italic*, `code`
 */
@Composable
fun MarkdownText(
    text: String,
    modifier: Modifier = Modifier,
    color: Color = MaterialTheme.colors.onBackground
) {
    val codeColor = MaterialTheme.colors.secondary
    val codeBackground = MaterialTheme.colors.surface

    // Cache the parsed result to avoid re-parsing on every recomposition
    val annotatedString = remember(text, color, codeColor, codeBackground) {
        parseMarkdown(text, color, codeColor, codeBackground)
    }
    Text(
        text = annotatedString,
        style = MaterialTheme.typography.body2,
        modifier = modifier
    )
}

private fun parseMarkdown(
    text: String,
    baseColor: Color,
    codeColor: Color,
    codeBackground: Color
): AnnotatedString {
    return buildAnnotatedString {
        var i = 0
        while (i < text.length) {
            when {
                // Bold: **text**
                text.startsWith("**", i) -> {
                    val endIndex = text.indexOf("**", i + 2)
                    if (endIndex != -1) {
                        withStyle(SpanStyle(fontWeight = FontWeight.Bold)) {
                            append(text.substring(i + 2, endIndex))
                        }
                        i = endIndex + 2
                    } else {
                        append(text[i])
                        i++
                    }
                }
                // Italic: *text* (but not **)
                text.startsWith("*", i) && !text.startsWith("**", i) -> {
                    val endIndex = findClosingMark(text, i + 1, "*")
                    if (endIndex != -1) {
                        withStyle(SpanStyle(fontStyle = FontStyle.Italic)) {
                            append(text.substring(i + 1, endIndex))
                        }
                        i = endIndex + 1
                    } else {
                        append(text[i])
                        i++
                    }
                }
                // Inline code: `text`
                text.startsWith("`", i) && !text.startsWith("```", i) -> {
                    val endIndex = text.indexOf("`", i + 1)
                    if (endIndex != -1) {
                        withStyle(SpanStyle(
                            fontFamily = FontFamily.Monospace,
                            color = codeColor,
                            background = codeBackground.copy(alpha = 0.3f)
                        )) {
                            append(text.substring(i + 1, endIndex))
                        }
                        i = endIndex + 1
                    } else {
                        append(text[i])
                        i++
                    }
                }
                // Code block: ```text``` - render as plain code
                text.startsWith("```", i) -> {
                    val endIndex = text.indexOf("```", i + 3)
                    if (endIndex != -1) {
                        val codeContent = text.substring(i + 3, endIndex)
                            .trimStart('\n')  // Remove leading newline after ```
                            .trimEnd('\n')    // Remove trailing newline before ```
                        withStyle(SpanStyle(
                            fontFamily = FontFamily.Monospace,
                            color = codeColor,
                            background = codeBackground.copy(alpha = 0.3f)
                        )) {
                            append(codeContent)
                        }
                        i = endIndex + 3
                    } else {
                        append(text[i])
                        i++
                    }
                }
                else -> {
                    append(text[i])
                    i++
                }
            }
        }
    }
}

/**
 * Find closing mark that isn't escaped and isn't part of a double mark
 */
private fun findClosingMark(text: String, startIndex: Int, mark: String): Int {
    var i = startIndex
    while (i < text.length) {
        if (text.startsWith(mark, i)) {
            // Make sure it's not part of ** when looking for *
            if (mark == "*" && i + 1 < text.length && text[i + 1] == '*') {
                i += 2
                continue
            }
            return i
        }
        i++
    }
    return -1
}