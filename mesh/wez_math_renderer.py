#!/usr/bin/env python3
import subprocess
import tempfile
from pathlib import Path
from PIL import Image, ImageOps
import json
import base64
import sys
import re


class WezMathRenderer:
    """
    Full renderer with:
      • PNG → WezTerm image for inline/display math
      • Automatic or forced text fallback
      • Display math → braille (intensity-based)
      • Inline math → Unicode TeX approximation
    """

    # toggle from outside:
    #   renderer.force_text_fallback = True

    def __init__(self, force_text_fallback = False ):
        self.force_text_fallback = force_text_fallback
        
        self._fence = re.compile(r"```.*?```", re.DOTALL)
        self._inline_code = re.compile(r"`[^`]+`")

        # Display math patterns - using negative lookahead to avoid matching across list items
        # This prevents runaway matches when a closing delimiter is missing
        self._display_patterns = [
            re.compile(r"\$\$(.+?)\$\$", re.DOTALL),
            re.compile(r"\\\[(.+?)\\\]", re.DOTALL),
        ]
        # Pattern to detect if we've matched across a list item (indicates malformed input)
        self._list_item_pattern = re.compile(r"^\s*\d+\.\s", re.MULTILINE)

        self._inline_patterns = [
            # Inline $...$ — single-line, rejects currency ($5, $10.00) and
            # bare whitespace delimiters.  Mirrors web-client/index.html:4431.
            re.compile(r"(?<!\$)\$(?!\s|\d)([^\n$]+?)(?<!\s)\$(?!\d)"),
            re.compile(r"\\\((.+?)\\\)", re.DOTALL),
        ]

        # ... your existing init ...

        # Cached intrinsic LaTeX height of `$x$` in PNG pixels
        self._inline_ref_h_px = None
        # Cached scale factor for current terminal geometry
        self._inline_scale = None
        # Last seen terminal geometry
        self._last_geom = None

        #Same thing but for displaymath equations
        self._ref_line_height = None

    def _get_ref_line_height(self) -> int:
        """Render a simple one-line display equation and cache its height in pixels."""
        if self._ref_line_height is not None:
            return self._ref_line_height

        sample = r" \nabla \times \mathbf{F}) \cdot \mathbf{n}\, dS "  # or whatever, just a single display line
        png_path = self.tex_to_png(sample, is_display=True)
        with Image.open(png_path) as img:
            _, h = img.size

        self._ref_line_height = h
        return h

    def _ensure_inline_ref_height(self) -> int:
        """
        Ensure we know the LaTeX-rendered height of $x$ in pixels.
        This is independent of terminal size, so we do it once.
        """
        if self._inline_ref_h_px is not None:
            return self._inline_ref_h_px

        ref_png = self.tex_to_png("x", is_display=False)
        ref_img = Image.open(ref_png).convert("RGBA")
        self._inline_ref_h_px = ref_img.size[1]
        return self._inline_ref_h_px
        
    # ------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------

    def dbg(self, *a):
        print("[DBG]", *a, file=sys.stderr)

    # ------------------------------------------------------------
    # WezTerm image output
    # ------------------------------------------------------------

    def _display_image_raw(self, png_path, pixel_width, *, inline=True):
        with open(png_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")

        inline_flag = 1 if inline else 0
        sys.stdout.write(
            f"\033]1337;File=inline={inline_flag};width={pixel_width}px;"
            f"preserveAspectRatio=1:{data}\a"
        )
        sys.stdout.flush()

    # ------------------------------------------------------------
    # Strip align environments
    # ------------------------------------------------------------

    def strip_align_env(self, latex: str) -> str:
        pattern = re.compile(
            r"^\s*\\begin\{align\*?\}(.*?)\\end\{align\*?\}\s*$",
            re.DOTALL
        )
        m = pattern.match(latex)
        if m:
            return m.group(1).strip()
        return latex.strip()

    # ------------------------------------------------------------
    # LaTeX → PNG
    # ------------------------------------------------------------

    def tex_to_png(self, latex: str, *, is_display: bool) -> Path:
        tmpdir = Path(tempfile.mkdtemp())
        tex_file = tmpdir / "eq.tex"
        pdf_file = tmpdir / "eq.pdf"
        cropped_pdf = tmpdir / "eq-crop.pdf"
        png_file = tmpdir / "eq.png"

        bgcolor = "black"
        #latex = latex.replace("\\\\", "\\")

        if is_display:
            latex = latex.strip()
            if latex.startswith(r"\begin{"):
                latex = self.strip_align_env(latex)
                body = f"""
\\begin{{preview}}
\\begin{{tikzpicture}}[baseline]
  \\node[inner sep=6pt, anchor=west, fill={bgcolor}, text=white, outer sep=0pt, align=left] (eq) {{
       \\noindent
      $\\begin{{aligned}}
      {latex}
      \\end{{aligned}}$
  }};
\\end{{tikzpicture}}
\\end{{preview}}
"""
            else:
                body = rf"""
\begin{{preview}}
\[
\colorbox{{black}}{{\color{{white}}{{ $\displaystyle{{ {latex} }}$ }} }}
\]
\end{{preview}}
"""
        else:
            body = rf"""
\begin{{preview}}
\colorbox{{black}}{{\color{{white}}{{ $\textstyle{{ {latex} }}$ }} }}
\end{{preview}}
"""

        doc = rf"""
\documentclass[border=2pt]{{standalone}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage[active,tightpage]{{preview}}
\PreviewEnvironment{{align*}}
\PreviewEnvironment{{aligned}}
\PreviewEnvironment{{equation*}}
\PreviewEnvironment{{preview}}
\usepackage{{xcolor}}
\usepackage{{varwidth}}
\everymath{{\color{{white}}}}
\usepackage[pages=all]{{background}}
\backgroundsetup{{ scale=1, angle=0, color=blue!10, contents={{\rule{{\paperwidth}}{{\paperheight}}}} }}
\begin{{document}}
{body}
\end{{document}}
"""

        tex_file.write_text(doc)

        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_file.name],
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text = True
        )

        if result.returncode != 0 or not pdf_file.exists():
            raise RuntimeError(f"LaTeX compile error: {result.stdout}\n{result.stderr}")

        margin = "-1" if is_display else "-3"
        result = subprocess.run(
            ["pdfcrop", "--margin", margin, pdf_file.name, cropped_pdf.name],
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise RuntimeError("pdfcrop error")

        subprocess.run(
            ["pdftocairo", "-png", "-singlefile", "-r", "900",
             cropped_pdf.name, "eq"],
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

        return png_file

    # ------------------------------------------------------------
    # Terminal geometry
    # ------------------------------------------------------------

    def _get_term_geometry(self):
        if self.force_text_fallback:
            return None, None,None, None
        try:
            out = subprocess.check_output(
                ["wezterm", "cli", "list", "--format", "json"],
                text=True
            )
            pane = json.loads(out)[0]
            size = pane["size"]
            return size["pixel_width"], size["cols"], size["pixel_height"], size["rows"]
        except:
            return None, None, None, None

    # ------------------------------------------------------------
    # Unicode inline TeX fallback
    # ------------------------------------------------------------

    def _inline_tex_to_unicode(self, tex: str) -> str:
        t = tex.strip()

        symbols = {
            # Greek lower
            r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
            r"\epsilon": "ε", r"\zeta": "ζ", r"\eta": "η", r"\theta": "θ",
            r"\iota": "ι", r"\kappa": "κ", r"\lambda": "λ", r"\mu": "μ",
            r"\nu": "ν", r"\xi": "ξ", r"\pi": "π", r"\rho": "ρ",
            r"\sigma": "σ", r"\tau": "τ", r"\upsilon": "υ", r"\phi": "φ",
            r"\chi": "χ", r"\psi": "ψ", r"\omega": "ω",

            # Greek upper
            r"\Gamma": "Γ", r"\Delta": "Δ", r"\Theta": "Θ", r"\Lambda": "Λ",
            r"\Xi": "Ξ", r"\Pi": "Π", r"\Sigma": "Σ", r"\Upsilon": "Υ",
            r"\Phi": "Φ", r"\Psi": "Ψ", r"\Omega": "Ω",

            # Operators
            r"\times": "×", r"\cdot": "·", r"\le": "≤", r"\ge": "≥",
            r"\neq": "≠", r"\approx": "≈", r"\pm": "±",
            r"\to": "→", r"\leftarrow": "←", r"\infty": "∞",
            r"\sum": "∑", r"\int": "∫", r"\partial": "∂", r"\nabla": "∇",
        }

        for k, v in symbols.items():
            t = t.replace(k, v)

        # # Simple ^ and _ handling
        # t = re.sub(r"([A-Za-z0-9])\^([A-Za-z0-9])",
        #            lambda m: m.group(1) + {
        #                "0": "⁰","1": "¹","2": "²","3": "³","4": "⁴","5": "⁵",
        #                "6": "⁶","7": "⁷","8": "⁸","9": "⁹"
        #            }.get(m.group(2), "^" + m.group(2)),
        #            t)

        # t = re.sub(r"([A-Za-z0-9])_([A-Za-z0-9])",
        #            lambda m: m.group(1) + {
        #                "0": "₀","1": "₁","2": "₂","3": "₃","4": "₄","5": "₅",
        #                "6": "₆","7": "₇","8": "₈","9": "₉"
        #            }.get(m.group(2), "_" + m.group(2)),
        #            t)

        # # Simple \frac
        # t = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1/\2)", t)

        return t

    # ------------------------------------------------------------
    # Braille intensity renderer
    # ------------------------------------------------------------

    # def _img_to_braille(self, img):
    #     # convert to grayscale
    #     img = img.convert("L")

    #     pixels = img.load()
    #     w, h = img.size

    #     # pad to even 2×4 grid
    #     if w % 2 != 0:
    #         w += 1
    #     if h % 4 != 0:
    #         h += 4 - (h % 4)

    #     # resize BEFORE invert (important!)
    #     img = img.resize((w, h))
    #     img = ImageOps.invert(img)   # invert *after* resize
    #     # Simple threshold for crisp math
    #     img = img.point(lambda x: 0 if x < 210 else 255, "1")
    #     pixels = img.load()

    #     ###dot_order = [0, 3, 1, 4, 2, 5, 6, 7]
    #     dot_order = [0, 1,2,3,4,5,6,7]
    #     # invert first: your math is white-on-black
    #     output_lines = []
    #     for y in range(0, h, 4):
    #         line = []
    #         for x in range(0, w, 2):
    #             dots = 0
    #             # braille dot positions
    #             if pixels[x,   y] == 0: dots |= 1 << 0
    #             if pixels[x, y+1] == 0: dots |= 1 << 1
    #             if pixels[x, y+2] == 0: dots |= 1 << 2
    #             if pixels[x+1,   y] == 0: dots |= 1 << 3
    #             if pixels[x+1, y+1] == 0: dots |= 1 << 4
    #             if pixels[x+1, y+2] == 0: dots |= 1 << 5
    #             if pixels[x,   y+3] == 0: dots |= 1 << 6
    #             if pixels[x+1, y+3] == 0: dots |= 1 << 7

    #             char = chr(0x2800 + dots)
    #             line.append(char)
    #         output_lines.append("".join(line))

    #     return "\n".join(output_lines)

    def _img_to_braille(self, img, target_braille_cols=100):
        """
        Convert a PIL image of LaTeX math into braille blocks.
        Automatically tiles horizontally if equation is wider than the
        requested braille width.

        target_braille_cols = width of each braille block in characters
        """

        # ------------------------------------------------------------
        # Step 1: grayscale
        # ------------------------------------------------------------
        img = img.convert("L")
        orig_w, orig_h = img.size
        ratio = orig_h / orig_w
        is_multiline = ratio > 0.5
        
        # ------------------------------------------------------------
        # Step 2: vertical scaling ONLY (preserves stroke clarity)
        # ------------------------------------------------------------
        # Pick a consistent output height that makes math readable.
        # 160px is a good default for braille (tweak if needed).
        if not is_multiline:
            target_height = 40
            scale = target_height / orig_h
            new_w = int(orig_w * scale)
            img = img.resize((new_w, target_height), Image.LANCZOS)
        else:
            target_width = 200
            scale = target_width / orig_w
            new_h = int(orig_h * scale)
            img = img.resize((target_width, new_h), Image.LANCZOS)

        # ------------------------------------------------------------
        # Step 3: compute chunk width in pixels
        # ------------------------------------------------------------
        # One braille cell = 2 horizontal pixels
        chunk_pixel_width = target_braille_cols * 2

        full_w, full_h = img.size

        # ------------------------------------------------------------
        # Step 4: tile horizontally
        # ------------------------------------------------------------
        tiles = []
        for x0 in range(0, full_w, chunk_pixel_width):
            x1 = min(x0 + chunk_pixel_width, full_w)
            tile = img.crop((x0, 0, x1, full_h))
            tiles.append(tile)

        # ------------------------------------------------------------
        # Step 5: braille‑encode each tile
        # ------------------------------------------------------------
        rendered_tiles = []
        first_tile = True;
        for tile in tiles:
            w, h = tile.size

            # pad width to even
            if w % 2 != 0:
                w += 1
            # pad height to multiple of 4
            if h % 4 != 0:
                h += 4 - (h % 4)

            tile = tile.resize((w, h))

            # invert AFTER resizing
            tile = ImageOps.invert(tile)

            # crisp threshold (yours)
            tile = tile.point(lambda x: 0 if x < 210 else 255, "1")
            pixels = tile.load()

            # compact dot order
            dot_order = [0,1,2,3,4,5,6,7]

            lines = []
            for y in range(0, h, 4):
                row = []
                for x in range(0, w, 2):
                    dots = 0
                    if pixels[x,   y] == 0: dots |= 1 << 0
                    if pixels[x, y+1] == 0: dots |= 1 << 1
                    if pixels[x, y+2] == 0: dots |= 1 << 2
                    if pixels[x+1,   y] == 0: dots |= 1 << 3
                    if pixels[x+1, y+1] == 0: dots |= 1 << 4
                    if pixels[x+1, y+2] == 0: dots |= 1 << 5
                    if pixels[x,   y+3] == 0: dots |= 1 << 6
                    if pixels[x+1, y+3] == 0: dots |= 1 << 7

                    row.append(chr(0x2800 + dots))
                lines.append("".join(row))
            if first_tile:
                rendered_tiles.append("\n".join(lines))
                first_tile = False
            else:
                rendered_tiles.append("\n\t".join(lines))
                            

        # ------------------------------------------------------------
        # Step 6: join tiles with a blank line between each segment
        # ------------------------------------------------------------
        return "\n".join(rendered_tiles)

    def _img_to_blocks(self, img):
        """
        Convert a grayscale image to block characters (1 pixel per cell).
        Uses 4‑level grayscale: ░▒▓█
        """
        img = img.convert("L")
        w, h = img.size
        img = ImageOps.invert(img)   # invert *after* resize
        pixels = img.load()

        shades = [
            (230, " "),   # almost white → space
            (180, "X"),
            (100, "X"),
            (40,  "X"),
            (0,   "X"),   # full black
        ]

        lines = []
        for y in range(h):
            row = []
            for x in range(w):
                v = pixels[x, y]
                for thresh, char in shades:
                    if v >= thresh:
                        row.append(char)
                        break
            lines.append("".join(row))
        return "\n".join(lines)

    
    # ------------------------------------------------------------
    # Display math
    # ------------------------------------------------------------

    def render_display_math(self, latex: str, *, width: int = 20):
        png_path = self.tex_to_png(latex, is_display=True)
        pil_img = Image.open(png_path)

        pixel_width, cols, pixel_height, rows = self._get_term_geometry()

        if not pixel_width or not cols:
            # fallback: just braille
            braille = self._img_to_braille(pil_img)
            sys.stdout.write("\n" + braille + "\n")
            return

        orig_w, orig_h = pil_img.size
        px_per_col = pixel_width / cols
        px_per_row = pixel_height / rows

        # --- estimate number of math lines using reference equation height ---
        ref_h = self._get_ref_line_height()
        est_lines = max(1, int(round(orig_h / ref_h)))

        # --- terminal budget -------------------------------------------------

        # How many terminal rows should we give this?
        #   - base_rows: padding / baseline
        #   - rows_per_line: extra rows per extra math line
        base_rows = 2
        rows_per_line = 2
        ideal_rows = base_rows + rows_per_line * (est_lines - 1)

        # Don't exceed some fraction of the terminal height, or all rows
        max_rows_for_math = int(rows * 0.6)  # use at most 60% of screen height
        target_rows = min(ideal_rows, max_rows_for_math)
        max_height_px = target_rows * px_per_row

        # Width budget: either respect a 'width' in cols or a global fraction
        max_cols = int(cols * 0.8)

        
        max_width_px = max_cols * px_per_col
        # --- compute scale based on both width and height --------------------

        scale_h = max_height_px / orig_h
        scale_w = max_width_px / orig_w

        
        # final scale: obey both constraints, and cap at 1.0 so we don't blow up tiny images
        scale = min(scale_h, scale_w, 1.0)

        # if scale is *very* small, you might want to bump it a bit or warn; up to you
        target_w = max(1, int(orig_w * scale))
        target_h = max(1, int(orig_h * scale))

        resized = pil_img.resize((target_w, target_h), Image.LANCZOS)
        out_path = png_path.parent / "scaled.png"
        resized.save(out_path)

        self._display_image_raw(out_path, target_w, inline=True)
    
    # def render_display_math(self, latex: str, *, width: int = 20):
    #     png_path = self.tex_to_png(latex, is_display=True)
    #     pil_img = Image.open(png_path)

    #     pixel_width, cols, pixel_height, rows = self._get_term_geometry()
        
    #     if not pixel_width or not cols:
    #         braille = self._img_to_braille(pil_img)

    #         sys.stdout.write("\n" + braille + "\n")
    #         return

    #     orig_w, orig_h = pil_img.size
    #     px_per_col = pixel_width / cols
    #     px_per_row = pixel_height / rows

    #     ratio = orig_h / orig_w;
    #     is_multiline = ratio > 0.5
    #     if not is_multiline:
    #         target_h = int(px_per_row * 2)
    #         scale = target_h / orig_h
    #         target_w = int(orig_w * scale)
    #     else:
    #         #target_w = int( px_per_col * 40 );
    #         target_w = int( pixel_width * 0.7 );
    #         scale = target_w / orig_w;
    #         target_h = int( orig_h * scale );


    #     resized = pil_img.resize((target_w, target_h), Image.LANCZOS)
    #     out_path = png_path.parent / "scaled.png"
    #     resized.save(out_path)

    #     self._display_image_raw(out_path, target_w, inline=True)

    # ------------------------------------------------------------
    # Inline math
    # ------------------------------------------------------------
    def render_inline_math(self, latex: str):
        pixel_width, cols, pixel_height, rows = self._get_term_geometry()

        if not pixel_width or not cols:
            # Fallback: no geometry → Unicode approximation
            sys.stdout.write(self._inline_tex_to_unicode(latex))
            return

        # Current geometry identifier
        geom = (pixel_width, cols)
        px_per_col = pixel_width / cols;
        px_per_row = pixel_height / rows;

        # Normal rendering for this particular expression
        png_path = self.tex_to_png(latex, is_display=False)
        pil_img = Image.open(png_path).convert("RGBA")
        orig_w, orig_h = pil_img.size
        target_w, target_h = (0,0)
        
        dynamic_scale = False
        if dynamic_scale:
            # Make sure we have the intrinsic LaTeX height of $x$
            ref_h_px = self._ensure_inline_ref_height()

            # Recompute scale if geometry changed or scale not set yet
            if self._inline_scale is None or geom != self._last_geom:
                # Decide how tall you want $x$ in this terminal, in pixels
                target_ref_h = int(px_per_row * 0.8)  # "2 rows" worth of height, adjust to taste

                self._inline_scale = target_ref_h / ref_h_px 

                self._last_geom = geom
            target_h = int(orig_h * self._inline_scale)
            target_w = int(orig_w * self._inline_scale)
        else:
            target_h = int(px_per_row * 1);
            scale = target_h / orig_h
            target_w = int(orig_w * scale)
            
        resized = pil_img.resize((target_w, target_h), Image.LANCZOS)
        out_path = png_path.parent / "inline.png"
        resized.save(out_path)

        self._display_image_raw(out_path, target_w, inline=True)

    # ------------------------------------------------------------
    # Math rendering in text
    # ------------------------------------------------------------

    def _render_math_in_text(self, text):
        # DISPLAY math
        while True:
            # Find the earliest valid match across all display patterns
            best_match = None
            best_start = len(text) + 1

            for pat in self._display_patterns:
                pos = 0
                while pos < len(text):
                    m = pat.search(text, pos)
                    if not m:
                        break

                    content = m.group(1)
                    # Check if this match spans a list item (malformed)
                    if self._list_item_pattern.search(content):
                        # Skip past the opening delimiter and try again
                        pos = m.start() + 2
                        continue

                    # Valid match found
                    if m.start() < best_start:
                        best_match = m
                        best_start = m.start()
                    break

            if not best_match:
                break

            m = best_match
            before = text[:m.start()]
            self._render_math_in_text(before)
            sys.stdout.write("\n\t")
            try:
                self.render_display_math(m.group(1))
            except Exception as e:
                sys.stdout.write(f"render error in displaymath: {m.group(1)[:50]}...\n  Error: {e}")

            sys.stdout.write("\n")
            text = text[m.end():]

        # INLINE math
        def inline_recurse(s):
            for pat in self._inline_patterns:
                m = pat.search(s)
                if m:
                    before = s[:m.start()]
                    mid = m.group(1)
                    after = s[m.end():]
                    sys.stdout.write(before)
                    try:
                        self.render_inline_math(mid)
                    except Exception as e:
                        sys.stdout.write( "render error in inline:" + mid + f" [{type(e).__name__}: {e}]" );
                    
                    return inline_recurse(after)
            sys.stdout.write(s)
            return s

        inline_recurse(text)

    # ------------------------------------------------------------
    # Top-level
    # ------------------------------------------------------------

    def render_tex(self, text: str):
        pos = 0
        length = len(text)

        while pos < length:
            fence_match = self._fence.match(text, pos)
            if fence_match:
                sys.stdout.write(fence_match.group(0) + "\n")
                pos = fence_match.end()
                continue

            inline_match = self._inline_code.match(text, pos)
            if inline_match:
                sys.stdout.write(inline_match.group(0))
                pos = inline_match.end()
                continue

            next_fence = self._fence.search(text, pos)
            next_inline = self._inline_code.search(text, pos)

            boundaries = [m.start() for m in (next_fence, next_inline) if m]
            boundary = min(boundaries) if boundaries else length

            segment = text[pos:boundary]
            self._render_math_in_text(segment)

            pos = boundary
