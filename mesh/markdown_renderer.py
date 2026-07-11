
from pygments import highlight
from pygments.lexers import guess_lexer
from pygments.lexers.special import TextLexer
from pygments.formatters import Terminal256Formatter
from pygments.style import Style
from pygments.token import (
    Text, Comment, Keyword, Name,
    String, Number, Operator, Punctuation, Generic
)

import re
import sys

# Regexes used for alignment detection
_ALIGN_LEFT   = re.compile(r"^\s*:?-{3,}\s*$")          # :---   or  ---
_ALIGN_CENTER = re.compile(r"^\s*:?-{3,}:\s*$")        # :---:
_ALIGN_RIGHT  = re.compile(r"^\s*-{3,}:\s*$")          # ---:
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mK]")

class MyTerminalStyle(Style):
    default_style = ""
    background_color = None

    styles = {
        Text:                "#f0f0f0",
        Comment:             "italic #808080",
        Keyword:             "bold #ff5f5f",
        Name:                "#ffffff",
        Name.Function:        "#ffd75f",
        Name.Class:           "bold #ffd75f",
        String:              "#87ff5f",
        Number:              "#87afff",
        Operator:            "#ffaf00",
        Punctuation:         "#f0f0f0",
        Generic.Heading:     "bold #ffffff",
        Generic.Subheading:  "bold #ffd75f",
        Generic.Deleted:     "#ff5f5f",
        Generic.Inserted:    "#87ff5f",
        Generic.Error:       "bold #ff0000",
        Generic.Emph:        "italic",
        Generic.Strong:      "bold",
    }

class MarkdownRenderer:
    def __init__(self, math_renderer):
        self.math = math_renderer
        self.render_math = True        # when False, print raw LaTeX source instead of rendering
        self.in_code = False
        self.code_lines = []
        self.code_lang = None
        self.in_math_block = False
        self.math_block_lines = []
        self.in_table = False          # true while we are gathering table lines
        self.table_lines = []          # temporary storage for the raw lines

    def colorize(self, text: str) -> str:
        # Use SGR parameter numbers instead of full escape sequences
        COLOR_CODES = {
            "black":   "30",
            "red":     "31",
            "green":   "32",
            "yellow":  "33",
            "blue":    "34",
            "magenta": "35",
            "cyan":    "36",
            "white":   "37",
        }

        STYLE_CODES = {
            "bold":      "1",
            "dim":       "2",
            "italic":    "3",
            "underline": "4",
            "blink":     "5",
            "reverse":   "7",
        }

        RESET = "\x1b[0m"

        # Allow spaces inside the tag so [bold magenta] matches
        # Also match closing tags like [/green] or [/bold]
        TAG_PATTERN = re.compile(r"\[(/[a-zA-Z]*|[a-zA-Z ]+)\]")

        def replace_tag(match: re.Match) -> str:
            tag = match.group(1)

            # Handle closing tags: [/] or [/green] or [/bold]
            if tag.startswith("/"):
                return RESET

            parts = tag.lower().split()
            codes = []

            for part in parts:
                if part in COLOR_CODES:
                    codes.append(COLOR_CODES[part])
                elif part in STYLE_CODES:
                    codes.append(STYLE_CODES[part])
                else:
                    # Not a recognized style/color tag -> leave completely unchanged
                    return match.group(0)

            if not codes:
                return match.group(0)

            return f"\x1b[{';'.join(codes)}m"

        return TAG_PATTERN.sub(replace_tag, text)
        
    # =========================================================
    # Unified pipeline: styling → math rendering (safe)
    # =========================================================
    def render_text_with_math(self, text):
        # Apply ANSI styling to non-math segments
        styled = self._style_inline(text)

        if self.render_math:
            # Full math/text processing (handles backticks, math spans, etc.)
            self.math.render_tex(styled)
        else:
            # Skip math rendering — print styled text with raw LaTeX delimiters
            sys.stdout.write(styled)

    # =========================================================
    # Main render loop
    # =========================================================
    def render(self, text):
        text = self.colorize( text )
        
        for line in text.splitlines():
            stripped = line.strip()

            # ------------------------
            # Fenced code blocks (verbatim)
            # ------------------------
            if stripped.startswith("```"):
                if not self.in_code:
                    # Enter code mode
                    self.in_code = True
                    self.code_lines = []
                    self.code_lang = stripped[3:].strip() or None
                else:
                    # Exit code mode
                    self._render_code_block()
                    self.in_code = False
                continue

            # -------------------------------------------------
            # Table detection – pipe‑delimited markdown tables
            # -------------------------------------------------
            if stripped.startswith("|") or stripped.startswith("   |"):
                # A line that looks like a table row – start/continue collection
                self.in_table = True
                self.table_lines.append(line.rstrip("\n"))
                continue

            if self.in_table and not (stripped.startswith("|") or stripped.startswith("   |")):
                # Render the collected rows and reset state
                self._render_table(self.table_lines)
                self.in_table = False
                self.table_lines = []
                # Continue processing the current line normally

            if self.in_code:
                # Do NOT parse math or styling inside code fences
                self.code_lines.append(line.rstrip("\n"))
                continue

            # ------------------------
            # Multiline display math blocks: $$ ... $$
            # ------------------------
            if stripped in {"$$", "\\["}:
                if not self.in_math_block:
                    # Enter math block
                    self.in_math_block = True
                    self.math_block_lines = []
                else:
                    # Exit math block
                    self.math_block_lines.append( stripped );
                    content = "\n".join(self.math_block_lines)
                    if self.render_math:
                        self.math.render_tex(content)
                    else:
                        sys.stdout.write(content)
                    sys.stdout.write("\n" )   # newline after bullet item
                    self.in_math_block = False
                    self.math_block_lines = []
                    continue

            if self.in_math_block:
                if stripped != "\\]":
                    if self.in_math_block:
                        self.math_block_lines.append(line)
                        continue
                else:
                    # Exit math block
                    self.math_block_lines.append( stripped );
                    content = "\n".join(self.math_block_lines)
                    if self.render_math:
                        self.math.render_tex(content)
                    else:
                        sys.stdout.write(content)
                    sys.stdout.write("\n" )   # newline after bullet item
                    self.in_math_block = False
                    self.math_block_lines = []
                    continue
                    
            
            # ------------------------
            # Blank lines → output real newline
            # ------------------------
            if stripped == "":
                sys.stdout.write("\n" )   # newline after bullet item
                continue

            # ------------------------
            # Horizontal Rule
            # ------------------------
            if stripped == "---":
                self._render_hr()
                sys.stdout.write("\n" )   # newline after bullet item
                continue

            # ------------------------
            # Headings
            # ------------------------
            if stripped.startswith("#"):
                self._render_heading(stripped)
                continue

            # ------------------------
            # Blockquotes
            # ------------------------
            if stripped.startswith(">"):
                sys.stdout.write("\033[3;36m┃\033[0m ")
                self.render_text_with_math(stripped[1:].lstrip())
                sys.stdout.write("\n" )   # newline after bullet item
                continue


            # Bullets + Ordered Lists (with nesting)
            # ------------------------
            leading_spaces = len(line) - len(line.lstrip(" "))
            indent_level = leading_spaces // 2  # or //4 if you prefer
            stripped_l = line.lstrip()

            # Detect bullets: - item, * item
            is_bullet = stripped_l.startswith("- ") or stripped_l.startswith("* ")

            # Detect ordered lists: "1. item", "42. more text"
            m = re.match(r"(\d+)\.\s+(.*)", stripped_l)
            is_ordered = m is not None

            if is_bullet or is_ordered:
                indent = " " * (indent_level * 2)

                if is_bullet:
                    content = stripped_l[2:]
                    rendered = f"{indent}• {content}"

                else:  # ordered
                    number = m.group(1)
                    content = m.group(2)
                    rendered = f"{indent}{number}. {content}"

                self.render_text_with_math(rendered)
                sys.stdout.write("\n")
                continue

            # ------------------------
            # Normal paragraph text
            # ------------------------
            self.render_text_with_math(line)
            sys.stdout.write("\n")

            #print("")  # newline after paragrap
            
        self._finalize_open_blocks()
        sys.stdout.flush()


    def _finalize_open_blocks(self) -> None:
        """Render any block that is still open when we hit EOF."""
        # ---- Code block -------------------------------------------------
        if self.in_code:
            # Decide what you want to do with an un‑closed fence.
            # Option 1 – render it anyway (treat as a normal code block):
            self._render_code_block()
            # Option 2 – warn the user:
            # sys.stderr.write("Warning: file ended inside a fenced code block.\n")
            self.in_code = False
            self.code_lines = []

        # ---- Math block -------------------------------------------------
        if self.in_math_block:
            # Render whatever we have collected so far.
            content = "\n".join(self.math_block_lines)
            self.math.render_tex(content)
            sys.stdout.write("\n")                 # newline after the block
            self.in_math_block = False
            self.math_block_lines = []

        # ---- Table ------------------------------------------------------
        if self.in_table:
            self._render_table(self.table_lines)
            self.in_table = False
            self.table_lines = []
    # =========================================================
    # Code block rendering
    # =========================================================

    def _render_code_block(self):
        code_text = "\n".join(self.code_lines)

        # Heuristic: if it's very short, don't guess, just call it text.
        non_empty_lines = [ln for ln in code_text.splitlines() if ln.strip()]
        is_short = len(code_text) < 40 or len(non_empty_lines) <= 2

        # Supported languages for highlighting (using Pygments lexer names)
        supported_languages = ['TeX', 'Python', 'C', 'C++']

        if is_short:
            lexer = TextLexer()
            lang_name = "text"
        else:
            try:
                guessed_lexer = guess_lexer(code_text)
                guessed_name = getattr(guessed_lexer, "name", "text")
                if guessed_name in supported_languages:
                    lexer = guessed_lexer
                    lang_name = guessed_name
                else:
                    lexer = TextLexer()
                    lang_name = "text"
            except Exception:
                lexer = TextLexer()
                lang_name = "text"

        # Use a lighter / friendlier style to avoid very dark colors
        formatter = Terminal256Formatter(style=MyTerminalStyle)
        highlighted = highlight(code_text, lexer, formatter)    

        # ANSI colors
        bg_title  = "\033[48;5;238m"  # slightly different gray for title/footer bar
        fg_title  = "\033[38;5;254m"  # bright text for title/footer
        bg_body   = "\033[48;5;236m"  # gray for code area (optional)
        reset     = "\033[0m"

        # ----- Title -----
        title = f"{bg_title}{fg_title} {lang_name} block start {reset}"
        print(title)

        # ----- Code body -----
        for line in highlighted.rstrip("\n").splitlines():
            # If you want a solid gray background for the body:
            #print(f"{bg_body}{line}{reset}")
            # If you prefer terminal background instead, use:
            print(line)

        # ----- Footer -----
        footer = f"{bg_title}{fg_title} End {lang_name} block {reset}"
        print(footer)

        print(reset, end="")


    # =========================================================
    # Heading rendering
    # =========================================================
    def _render_heading(self, stripped):
        level = len(stripped) - len(stripped.lstrip("#"))
        title = stripped[level:].strip()

        if level == 1:
            print(f"\033[1;97m{title}\033[0m")
            print("\033[1;97m" + ("=" * len(title)) + "\033[0m")
        elif level == 2:
            print(f"\033[1;97m{title}\033[0m")
            print("\033[90m" + ("-" * len(title)) + "\033[0m")
        else:
            print(f"\033[1m{title}\033[0m")

    # =========================================================
    # Horizontal rule
    # =========================================================
    def _render_hr(self):
        print("━" * 60)

    # =========================================================
    # SAFE inline styling (callable replacements)
    # =========================================================
    def _style_inline(self, line):

        # Bold-italic
        def repl_bold_italic(m):
            return "\033[1;3m" + m.group(1) + "\033[0m"

        # Bold
        def repl_bold(m):
            return "\033[1m" + m.group(1) + "\033[0m"

        # Italic
        def repl_italic(m):
            return "\033[3m" + m.group(1) + "\033[0m"

        # MUST be in this order: *** → ** → *
        line = re.sub(r"\*\*\*(.+?)\*\*\*", repl_bold_italic, line)
        line = re.sub(r"\*\*(.+?)\*\*", repl_bold, line)
        line = re.sub(r"\*(.+?)\*", repl_italic, line)

        return line

    def _split_row(self, row: str) -> list[str]:
        """
        Split a raw markdown row like '| a | b |' into a list of cell strings.
        Leading/trailing pipes are ignored; surrounding whitespace is stripped.
        """
        # Remove the outer pipes but keep empty cells (e.g. '| a || c |')
        row = row.strip()
        if row.startswith("|"):
            row = row[1:]
        if row.endswith("|"):
            row = row[:-1]
        cells = [c.strip() for c in row.split("|")]
        return cells

    def _parse_table(self, raw_lines: list[str]) -> tuple[list[list[str]], list[str], list[list[str]]]:
        """
        Returns (header_cells, alignments, data_rows)
        * alignments = ["<", "^", ">"]  (left, centre, right)
        """
        # Minimum: header + alignment line
        if len(raw_lines) < 2:
            raise ValueError("Not a valid markdown table – missing alignment row")

        header_cells = self._split_row(raw_lines[0])
        align_cells  = self._split_row(raw_lines[1])

        # Determine alignment per column
        aligns = []
        for cell in align_cells:
            if _ALIGN_CENTER.match(cell):
                aligns.append("^")           # centre
            elif _ALIGN_RIGHT.match(cell):
                aligns.append(">")           # right
            else:
                aligns.append("<")           # left (default)
        # Normalise length – extra alignment markers are ignored
        aligns = aligns[:len(header_cells)]

        # Remaining rows
        data_rows = [self._split_row(r) for r in raw_lines[2:] if r.strip()]

        # Pad rows that are shorter than the header (Markdown permits it)
        ncols = len(header_cells)
        def pad(row):
            return row + [""] * (ncols - len(row))
        data_rows = [pad(r) for r in data_rows]

        return header_cells, aligns, data_rows

    def _visible_len(self, s: str) -> int:
        """Return the printable length of a string that may contain ANSI codes."""
        return len(_ANSI_RE.sub("", s))

    def _render_table(self, raw_lines: list[str]) -> None:
        """
        Takes the raw markdown lines that constitute a table and writes a pretty‑printed
        version directly to stdout using Unicode box‑drawing characters.
        """
        try:
            header, aligns, rows = self._parse_table(raw_lines)
        except Exception as exc:
            # If parsing fails we fall back to a raw dump – never break the whole renderer
            for l in raw_lines:
                self.render_text_with_math(l)
                sys.stdout.write("\n")
            return

        # Apply inline styling / math to each cell before measuring widths.
        # We reuse the existing pipeline so things like **bold**, [31m etc work.
        def style_cell(cell: str) -> str:
            # First run the custom colour tags
            cell = self.colorize(cell)
            # Then render possible markdown inline (bold/italic) and math
            # `_style_inline` does the markdown, `self.math.render_tex` does math.
            # We use a tiny wrapper that returns the (already printed) result.
            # Since `render_text_with_math` writes directly to stdout, we need a
            # version that returns a string.  For simplicity we reuse the same logic:
            #   - style inline markdown
            #   - let math render *into* a buffer we capture.
            # Here we cheat a bit and call the same helpers but capture output:
            from io import StringIO
            old_stdout = sys.stdout
            buf = StringIO()
            sys.stdout = buf
            self.render_text_with_math(cell)   # this will write to buf
            sys.stdout = old_stdout
            return buf.getvalue().rstrip("\n")

        # Style every cell once
        header = [style_cell(c) for c in header]
        rows   = [[style_cell(c) for c in r] for r in rows]

        # Normalize rows to have the same number of columns as the header
        ncols = len(header)
        rows = [r + [""] * (ncols - len(r)) if len(r) < ncols else r[:ncols] for r in rows]

        # Compute printable width per column (ignore ANSI escapes)
        col_widths = [0] * ncols
        for col in range(ncols):
            max_len = self._visible_len(header[col])
            for row in rows:
                max_len = max(max_len, self._visible_len(row[col]))
            col_widths[col] = max_len

        # Helpers to produce a formatted row string
        def format_row(cells, is_header=False):
            parts = []
            for idx, cell in enumerate(cells):
                width = col_widths[idx]
                align = aligns[idx] if idx < len(aligns) else "<"
                #fmt = f"{{:{align}{width}}}"
                if is_header:
                    cell = f"\033[1m{cell}\033[0m"
                txt = self._pad_ansi( cell, width, align ) 
                   # bold header
                parts.append(txt)
            return "│ " + " │ ".join(parts) + " │"

        # Box‑drawing characters
        horiz = "─"
        top    = "┌" + "┬".join([horiz * (w + 2) for w in col_widths]) + "┐"
        middle = "├" + "┼".join([horiz * (w + 2) for w in col_widths]) + "┤"
        bottom = "└" + "┴".join([horiz * (w + 2) for w in col_widths]) + "┘"

        # Print the table
        sys.stdout.write(top + "\n")
        sys.stdout.write(format_row(header, is_header=False) + "\n")
        sys.stdout.write(middle + "\n")
        for r in rows:
            sys.stdout.write(format_row(r) + "\n")
        sys.stdout.write(bottom + "\n")                

    def _pad_ansi(self, s: str, width: int, align: str = "<") -> str:
        """
        Pad a string to 'width' visible characters, ignoring ANSI codes.
        align: '<' left, '>' right, '^' center
        """
        vis = self._visible_len(s)
        pad = max(0, width - vis)
        if align == "<":
            return s + " " * pad
        elif align == ">":
            return " " * pad + s
        elif align == "^":
            left = pad // 2
            right = pad - left
            return " " * left + s + " " * right
        else:
            return s  # fallback
