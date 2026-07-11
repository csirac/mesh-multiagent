package com.mesh.client.ui.components

import android.widget.TextView
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import io.noties.markwon.Markwon
import androidx.compose.material3.MaterialTheme
import io.noties.markwon.ext.latex.JLatexMathPlugin
import io.noties.markwon.ext.strikethrough.StrikethroughPlugin
import io.noties.markwon.ext.tables.TablePlugin
import io.noties.markwon.inlineparser.MarkwonInlineParserPlugin

/**
 * Convert LaTeX delimiters from \(...\) and \[...\] to $...$ and $$...$$
 * for Markwon's JLatexMathPlugin.
 */
private fun convertLatexDelimiters(text: String): String {
    return text
        // Convert block math \[...\] to $$...$$
        .replace(Regex("""\\\[(.*?)\\\]""", RegexOption.DOT_MATCHES_ALL)) { match ->
            "$$${match.groupValues[1]}$$"
        }
        // Convert inline math \(...\) to $...$
        .replace(Regex("""\\\((.*?)\\\)""", RegexOption.DOT_MATCHES_ALL)) { match ->
            "$${match.groupValues[1]}$"
        }
}

/**
 * A Compose wrapper around Markwon for rendering Markdown with LaTeX math support.
 *
 * Supports:
 * - Standard Markdown (bold, italic, links, headers, lists, code blocks)
 * - Strikethrough (~~text~~)
 * - Tables
 * - LaTeX math (inline $...$ / \(...\) and block $$...$$ / \[...\])
 */
@Composable
fun MarkdownText(
    text: String,
    modifier: Modifier = Modifier,
    textColor: Color = Color.Unspecified,
    textSizeSp: Float = 16f,
    mathSizeSp: Float = 32f,  // Math renders smaller, so use larger size
    linkColor: Color = Color.Unspecified,
    onLongClick: (() -> Unit)? = null
) {
    val context = LocalContext.current
    val processedText = remember(text) { convertLatexDelimiters(text) }
    val resolvedTextColor = if (textColor == Color.Unspecified) MaterialTheme.colorScheme.onSurface else textColor
    val resolvedLinkColor = if (linkColor == Color.Unspecified) MaterialTheme.colorScheme.primary else linkColor
    val textColorArgb = resolvedTextColor.toArgb()
    val linkColorArgb = resolvedLinkColor.toArgb()

    // Create and remember the Markwon instance
    // Note: JLatexMathPlugin can crash on malformed LaTeX, so we use a fallback
    val markwon = remember(context, mathSizeSp) {
        try {
            Markwon.builder(context)
                .usePlugin(StrikethroughPlugin.create())
                .usePlugin(TablePlugin.create(context))
                // InlineParserPlugin is required by JLatexMathPlugin
                .usePlugin(MarkwonInlineParserPlugin.create())
                .usePlugin(JLatexMathPlugin.create(mathSizeSp) { builder ->
                    // Configure inline ($...$) and block ($$...$$) math
                    builder.inlinesEnabled(true)
                })
                .build()
        } catch (e: Exception) {
            // Fallback without LaTeX if plugin fails to initialize
            Markwon.builder(context)
                .usePlugin(StrikethroughPlugin.create())
                .usePlugin(TablePlugin.create(context))
                .build()
        }
    }

    AndroidView(
        modifier = modifier,
        factory = { ctx ->
            TextView(ctx).apply {
                setTextColor(textColorArgb)
                setLinkTextColor(linkColorArgb)
                textSize = textSizeSp
                // Handle long-click for context menu
                if (onLongClick != null) {
                    setOnLongClickListener {
                        onLongClick()
                        true
                    }
                }
            }
        },
        update = { textView ->
            try {
                markwon.setMarkdown(textView, processedText)
            } catch (e: Exception) {
                // If Markdown rendering fails (e.g., bad LaTeX), show plain text
                textView.text = text
            }
        }
    )
}
