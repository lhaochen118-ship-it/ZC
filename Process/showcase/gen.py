#!/usr/bin/env python3
"""README figures for the public showcase repo (light + dark), v2.

Writes <out>/<figure>-<theme>.html, one SVG each; render.js turns them into transparent 2x PNGs.
Public figures must not show the model name, the training recipe or what the verification module checks.
"""
import html
import os
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
os.makedirs(OUT, exist_ok=True)

THEMES = {
    "light": dict(
        page="#ffffff", ink="#1f2328", ink2="#59636e", muted="#818b98",
        card="#f6f8fa", stroke="#d1d9e0", hair="#e5e9ee",
        blue="#2a78d6", blue_t="#eaf2fc",
        violet="#6250d6", violet_t="#efedfb",
        aqua="#14946a", aqua_t="#e5f5ef",
        orange="#d9541f", orange_t="#fdeee6",
        good="#0ca30c", bad="#d03b3b", abst="#c9ced6",
        code_bg="#0f1720", code_ink="#e6edf3", code_dim="#7d8590",
        code_str="#a5d6ff", code_fn="#d2a8ff", code_num="#79c0ff",
        mess="#c77700", mess_t="#fff1d6", sheet_hdr="#eef1f4", strike="#d03b3b",
    ),
    "dark": dict(
        page="#0d1117", ink="#e6edf3", ink2="#9198a1", muted="#6e7681",
        card="#151b23", stroke="#3d444d", hair="#262c34",
        blue="#4a93ea", blue_t="#14243a",
        violet="#9085e9", violet_t="#1f1c3a",
        aqua="#2fbf8a", aqua_t="#0f2a22",
        orange="#ef7a45", orange_t="#2e1a10",
        good="#0ca30c", bad="#e05252", abst="#3a414b",
        code_bg="#010409", code_ink="#e6edf3", code_dim="#7d8590",
        code_str="#a5d6ff", code_fn="#d2a8ff", code_num="#79c0ff",
        mess="#e0a040", mess_t="#2b2010", sheet_hdr="#1c232c", strike="#e05252",
    ),
}

SANS = "Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
GOOD_OP = {"#ffffff": 0.55, "#0d1117": 0.75}
PRE = 'xml:space="preserve" style="font-variant-ligatures:none"'


def esc(s):
    return html.escape(s, quote=True)


def text(x, y, s, size=13, fill="ink", weight=400, anchor="start", font=SANS, extra="", t=None):
    col = t[fill] if t and fill in t else fill
    return (f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" font-weight="{weight}" '
            f'fill="{col}" text-anchor="{anchor}" {extra}>{esc(s)}</text>')


def rect(x, y, w, h, fill, stroke="none", r=10, sw=1, extra=""):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}" {extra}/>')


ICONS = {
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/>',
    "db": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    "chat": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "filter": '<path d="M22 3H2l8 9.46V19l4 2v-8.54L22 3z"/>',
    "table": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M3 15h18M9 3v18"/>',
    "cpu": ('<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6" rx="1"/>'
            '<path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/>'),
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="M9 12l2 2 4-4"/>',
    "term": '<rect x="2" y="3" width="20" height="18" rx="2"/><path d="M6 9l4 3-4 3M12 16h6"/>',
    "send": '<path d="M22 2L11 13"/><path d="M22 2l-7 20-4-9-9-4 20-7z"/>',
    "calendar": '<rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4M8 2v4M3 10h18"/>',
}


def icon(name, x, y, size, color, sw=2):
    s = size / 24
    return (f'<g transform="translate({x},{y}) scale({s})" fill="none" stroke="{color}" stroke-width="{sw}" '
            f'stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</g>')


def defs(t):
    out = ['<defs>']
    for key in ("muted", "blue", "violet", "aqua", "orange", "ink2"):
        out.append(f'<marker id="ah-{key}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" '
                   f'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{t[key]}"/></marker>')
    out.append(f'<linearGradient id="slm" x1="0" y1="0" x2="0" y2="1">'
               f'<stop offset="0" stop-color="{t["violet_t"]}"/><stop offset="1" stop-color="{t["blue_t"]}"/></linearGradient>')
    out.append(f'<filter id="sh" x="-10%" y="-10%" width="120%" height="130%"><feDropShadow dx="0" dy="2" stdDeviation="3" '
               f'flood-color="#000" flood-opacity="{0.10 if t["page"] == "#ffffff" else 0.45}"/></filter>')
    out.append('</defs>')
    return "".join(out)


def line(x1, y1, x2, y2, key, t, sw=1.8, dash=None, marker=True):
    d = f'stroke-dasharray="{dash}"' if dash else ""
    m = f'marker-end="url(#ah-{key})"' if marker else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{t[key]}" stroke-width="{sw}" '
            f'stroke-linecap="round" {d} {m}/>')


def path(d, key, t, sw=1.8, dash=None, marker=True):
    da = f'stroke-dasharray="{dash}"' if dash else ""
    m = f'marker-end="url(#ah-{key})"' if marker else ""
    return (f'<path d="{d}" fill="none" stroke="{t[key]}" stroke-width="{sw}" stroke-linecap="round" '
            f'stroke-linejoin="round" {da} {m}/>')


def badge(cx, cy, n, color):
    return (f'<circle cx="{cx}" cy="{cy}" r="11" fill="{color}"/>'
            + text(cx, cy + 4.3, str(n), 12, "#ffffff", 700, "middle"))


def card(x, y, w, h, t, fill=None, stroke=None, sw=1, r=12):
    return rect(x, y, w, h, fill or t["card"], stroke or t["stroke"], r, sw, 'filter="url(#sh)"')


def stage(x, y, w, h, n, title, lines, accent, tint, ic, t):
    o = [card(x, y, w, h, t)]
    o.append(f'<rect x="{x}" y="{y + 12}" width="3.5" height="{h - 24}" rx="1.75" fill="{accent}"/>')
    o.append(rect(x + 16, y + 16, 36, 36, tint, "none", 9))
    o.append(icon(ic, x + 24, y + 24, 20, accent))
    o.append(text(x + 64, y + 31, title, 14.5, "ink", 650, t=t))
    for i, s in enumerate(lines):
        o.append(text(x + 64, y + 50 + 16 * i, s, 12, "ink2", 400, t=t))
    o.append(badge(x + w - 4, y + 4, n, accent))
    return "".join(o)


def header(x, w, label, color):
    return (text(x, 26, label, 11.5, color, 700, extra='letter-spacing="1.4"')
            + f'<line x1="{x}" y1="36" x2="{x + w}" y2="36" stroke="{color}" stroke-width="2" stroke-linecap="round" opacity="0.55"/>')


def window_dots(x, y, r=3.4, step=11.5):
    return "".join(f'<circle cx="{x + i * step}" cy="{y}" r="{r}" fill="{c}"/>'
                   for i, c in enumerate(("#ff5f57", "#febc2e", "#28c840")))


def question_bubble(x, y, w, t):
    o = [f'<path d="M{x + 10} {y}h{w - 20}a10 10 0 0 1 10 10v38a10 10 0 0 1-10 10H{x + 30}l-12 10v-10h-8a10 10 0 0 1-10-10V{y + 10}a10 10 0 0 1 10-10z" '
         f'fill="{t["page"]}" stroke="{t["stroke"]}"/>']
    o.append(text(x + 12, y + 23, "“sp bán dc nhiều nhất", 12.5, "ink", 600, t=t))
    o.append(text(x + 12, y + 41, "tuần rồi là gì?”", 12.5, "ink", 600, t=t))
    return "".join(o)


def asof_pill(x, y, t):
    return (rect(x, y, 154, 26, t["page"], t["stroke"], 13) + icon("calendar", x + 9, y + 6, 14, t["ink2"])
            + text(x + 29, y + 17.5, "as-of 2026-03-04", 11.5, "ink", 600, font=MONO, t=t))


def code_skeleton(x, y, w, t):
    """abstract program: coloured bars, no real code"""
    o = [rect(x, y, w, 118, t["code_bg"], "none", 9), window_dots(x + 12, y + 12)]
    bars = [(0, 70, "code_fn"), (14, 104, "code_ink"), (14, 86, "code_str"), (14, 116, "code_ink"), (28, 64, "code_num"),
            (14, 92, "code_fn"), (0, 52, "code_ink")]
    yy = y + 30
    for ind, bw, key in bars:
        o.append(rect(x + 14 + ind, yy, min(bw, w - 28 - ind), 6, t[key], "none", 3, extra='opacity="0.8"'))
        yy += 11.5
    return "".join(o)


# ---------------------------------------------------------------- Figure 1: pipeline
def fig_pipeline(t):
    W, H = 1100, 470
    o = [defs(t)]
    o.append(header(20, 200, "INPUT", t["muted"]))
    o.append(header(268, 222, "PREPARE", t["blue"]))
    o.append(header(540, 200, "GENERATE", t["violet"]))
    o.append(header(800, 280, "VERIFY & ANSWER", t["aqua"]))

    # input: business data
    x, y, w, h = 20, 56, 200, 182
    o.append(card(x, y, w, h, t))
    o.append(icon("db", x + 16, y + 16, 20, t["ink2"]))
    o.append(text(x + 44, y + 31, "Business data", 14.5, "ink", 650, t=t))
    chips = [(".xlsx", "#1f8a4c"), (".csv", "#2a78d6"), (".sqlite", "#8a5cd6"), ("README.md", "#818b98")]
    for i, (lab, col) in enumerate(chips):
        cx, cy = x + 14 + (i % 2) * 88, y + 50 + (i // 2) * 42
        o.append(rect(cx, cy, 84, 32, t["page"], t["stroke"], 8))
        o.append(icon("file", cx + 7, cy + 8, 16, col))
        o.append(text(cx + 27, cy + 20.5, lab, 11.5 if lab != "README.md" else 10, "ink", 600, font=MONO, t=t))
    o.append(text(x + 16, y + 152, "multi-sheet, inconsistent names,", 11.5, "ink2", t=t))
    o.append(text(x + 16, y + 168, "mixed number and date formats", 11.5, "ink2", t=t))

    # input: question
    x, y, w, h = 20, 262, 200, 190
    o.append(card(x, y, w, h, t))
    o.append(icon("chat", x + 16, y + 16, 20, t["ink2"]))
    o.append(text(x + 44, y + 31, "Question", 14.5, "ink", 650, t=t))
    o.append(question_bubble(x + 14, y + 48, 172, t))
    o.append(text(x + 16, y + 135, "Which product sold the most", 11, "ink2", extra='font-style="italic"', t=t))
    o.append(text(x + 16, y + 150, "last week?", 11, "ink2", extra='font-style="italic"', t=t))
    o.append(rect(x + 14, y + 160, 154, 22, t["page"], t["stroke"], 11))
    o.append(icon("calendar", x + 23, y + 164, 13, t["ink2"]))
    o.append(text(x + 42, y + 175, "as-of 2026-03-04", 11, "ink", 600, font=MONO, t=t))

    # prepare
    o.append(stage(268, 56, 222, 86, 1, "Clean & filter", ["normalizes formats,", "removes non-data rows"],
                   t["blue"], t["blue_t"], "filter", t))
    o.append(stage(268, 166, 222, 86, 2, "Schema generation", ["builds a unified schema", "for each business"],
                   t["blue"], t["blue_t"], "table", t))

    # generate: small language model
    x, y, w, h = 540, 96, 200, 356
    o.append(rect(x, y, w, h, "url(#slm)", t["violet"], 14, 1.5, 'filter="url(#sh)"'))
    o.append(badge(x + w - 4, y + 4, 3, t["violet"]))
    o.append(rect(x + w / 2 - 26, y + 20, 52, 52, t["page"], t["violet"], 12, 1.2))
    o.append(icon("cpu", x + w / 2 - 16, y + 30, 32, t["violet"]))
    o.append(text(x + w / 2, y + 100, "Small language model", 15, "ink", 700, "middle", t=t))
    o.append(text(x + w / 2, y + 122, "reads the schema and", 12, "ink2", 400, "middle", t=t))
    o.append(text(x + w / 2, y + 138, "the question, then", 12, "ink2", 400, "middle", t=t))
    o.append(text(x + w / 2, y + 154, "writes a program", 12, "ink2", 400, "middle", t=t))
    o.append(code_skeleton(x + 16, y + 190, w - 32, t))
    o.append(text(x + w / 2, y + 334, "generated program", 11, "muted", 600, "middle", t=t))

    # verify & answer
    o.append(stage(800, 96, 280, 100, 4, "Verification module", ["checks and executes each", "program in a sandbox"],
                   t["aqua"], t["aqua_t"], "shield", t))
    x, y, w, h = 800, 262, 280, 190
    o.append(card(x, y, w, h, t, t["orange_t"], t["orange"], 1.3))
    o.append(icon("send", x + 16, y + 16, 18, t["orange"]))
    o.append(text(x + 42, y + 30, "Answer", 14.5, "ink", 650, t=t))
    o.append(rect(x + 16, y + 46, w - 32, 72, t["page"], t["stroke"], 8))
    o.append(text(x + 28, y + 68, '{"type": "string",', 11, "ink", 500, font=MONO, extra=PRE, t=t))
    o.append(text(x + 28, y + 86, ' "answer": "Xi măng Hà Tiên', 11, "ink", 500, font=MONO, extra=PRE, t=t))
    o.append(text(x + 28, y + 104, '            PCB40 bao 50kg"}', 11, "ink", 500, font=MONO, extra=PRE, t=t))
    o.append(text(x + 16, y + 146, "Other outcomes:", 11.5, "ink2", 600, t=t))
    o.append(text(x + 16, y + 164, "unanswerable, or abstain when uncertain", 11.5, "ink2", 400, t=t))

    # arrows
    o.append(line(220, 99, 265, 99, "blue", t))                              # data -> clean
    o.append(line(379, 142, 379, 163, "blue", t))                            # clean -> schema
    o.append(line(490, 209, 537, 209, "violet", t))                          # schema -> SLM
    o.append(line(220, 357, 537, 357, "violet", t))                          # question -> SLM
    o.append(text(379, 349, "question", 10.5, "muted", 600, "middle", t=t))
    o.append(text(513, 201, "schema", 10.5, "muted", 600, "middle", t=t))
    o.append(line(740, 128, 797, 128, "aqua", t))                            # SLM -> verification
    o.append(text(769, 120, "program", 10.5, "muted", 600, "middle", t=t))
    o.append(line(800, 170, 743, 170, "orange", t, sw=1.6, dash="5 4"))       # feedback
    o.append(text(771, 186, "feedback", 10.5, "orange", 600, "middle", t=t))
    o.append(line(940, 196, 940, 259, "aqua", t))                            # verification -> answer
    o.append(text(950, 232, "verified result", 10.5, "muted", 600, t=t))
    return W, H, "".join(o)


# ---------------------------------------------------------------- Figure 2: model input / output example
def fig_example(t):
    W, H = 1100, 404
    o = [defs(t)]
    o.append(header(20, 440, "RAW INPUT", t["muted"]))
    o.append(header(505, 235, "MODEL INPUT", t["blue"]))
    o.append(header(790, 290, "MODEL OUTPUT", t["violet"]))

    # spreadsheet
    x, y, w = 20, 52, 440
    cols = [("Ngày", 84), ("Sản phẩm", 136), ("SL", 50), ("Thành tiền", 80), ("TT", 74)]
    rows = [
        ("23/02", "XM Hà Tiên PCB40", "12 bao", "1tr440", "Xong"),
        ("46078", "Thép D10", "30 cây", "3.150.000đ", "hủy"),
        ("24/2/2026", "xi mang ha tien pcb40", "5", "600k", "ok"),
        ("26-02-2026", "Cát vàng", "3 khối", "1,05tr", "Hoàn thành"),
        ("27/02", "Thép D10", "8 cây", "840k", "Xong"),
        ("Tổng tháng", "", "", "7.08tr", ""),
    ]
    mess = {(0, 1), (0, 3), (1, 0), (1, 3), (2, 0), (2, 1), (2, 3), (3, 0), (3, 3), (4, 3), (5, 0)}
    rh = 30
    h = 34 + 34 + rh * (len(rows) + 1) + 36
    o.append(card(x, y, w, h, t))
    o.append(icon("file", x + 14, y + 9, 16, "#1f8a4c"))
    o.append(text(x + 36, y + 22, "ban_hang_T2.xlsx", 12, "ink", 600, font=MONO, t=t))
    o.append(text(x + w - 14, y + 22, "sheet “T2” · 1 of 4 files", 11, "muted", 500, "end", t=t))
    o.append(f'<line x1="{x}" y1="{y + 34}" x2="{x + w}" y2="{y + 34}" stroke="{t["stroke"]}"/>')
    gx, gy = x + 8, y + 42
    gw = sum(c[1] for c in cols)
    o.append(text(gx + 6, gy + 17, "BÁO CÁO BÁN HÀNG THÁNG 2/2026", 11.5, "ink", 700, t=t))
    o.append(rect(gx + 240, gy + 4, 92, 19, t["mess_t"], t["mess"], 5, 1, 'stroke-dasharray="3 2"'))
    o.append(text(gx + 286, gy + 17.5, "title row", 10.5, t["mess"], 600, "middle"))
    gy += 30
    o.append(rect(gx, gy, gw, rh, t["sheet_hdr"], "none", 0))
    cx = gx
    for name, cw in cols:
        o.append(text(cx + 6, gy + 19.5, name, 12, "ink", 700, t=t))
        cx += cw
    for r, row in enumerate(rows):
        ry = gy + rh * (r + 1)
        cx = gx
        for c, ((_, cw), val) in enumerate(zip(cols, row)):
            if (r, c) in mess:
                o.append(rect(cx + 2, ry + 4, cw - 4, rh - 8, t["mess_t"], "none", 4))
            o.append(text(cx + 6, ry + 19.5, val, 12, "ink2" if r == 1 else "ink", 700 if r == 5 else 450,
                          font=MONO if c in (0, 2, 3) else SANS, t=t))
            cx += cw
        o.append(f'<line x1="{gx}" y1="{ry}" x2="{gx + gw}" y2="{ry}" stroke="{t["hair"]}"/>')
    ry = gy + rh * 2
    o.append(f'<line x1="{gx + 4}" y1="{ry + 15}" x2="{gx + gw - 4}" y2="{ry + 15}" stroke="{t["strike"]}" '
             f'stroke-width="1.2" opacity="0.6"/>')
    ly = y + h - 14
    o.append(rect(x + 14, ly - 9, 12, 12, t["mess_t"], t["mess"], 3, 1))
    o.append(text(x + 32, ly + 1, "irregular cells: serial dates, mixed formats, unaccented names", 10.5, "ink2", 500, t=t))
    sheet_mid = y + 120

    # model input: schema
    x, w = 505, 235
    sy, sh = 52, 214
    o.append(card(x, sy, w, sh, t))
    o.append(rect(x + 14, sy + 12, 28, 28, t["blue_t"], "none", 7))
    o.append(icon("table", x + 20, sy + 18, 16, t["blue"]))
    o.append(text(x + 52, sy + 31, "Generated schema", 14, "ink", 650, t=t))
    sch = [("sales", True), ("date", False), ("product → products", False), ("qty, unit", False),
           ("amount (VND)", False), ("status: valid | cancelled", False), ("products", True), ("name, aliases", False)]
    yy = sy + 60
    for lab, is_tab in sch:
        if is_tab:
            o.append(text(x + 18, yy, lab, 11.5, t["blue"], 700, font=MONO))
            yy += 19
        else:
            o.append(f'<circle cx="{x + 24}" cy="{yy - 4}" r="2" fill="{t["muted"]}"/>')
            o.append(text(x + 32, yy, lab, 11, "ink", 450, font=MONO, extra=PRE, t=t))
            yy += 18.5
    # model input: question
    qy, qh = 282, 112
    o.append(card(x, qy, w, qh, t))
    o.append(icon("chat", x + 16, qy + 14, 18, t["ink2"]))
    o.append(text(x + 42, qy + 28, "Question", 14, "ink", 650, t=t))
    o.append(text(x + 14, qy + 52, "“sp bán dc nhiều nhất tuần rồi là gì?”", 11.5, "ink", 600, t=t))
    o.append(text(x + 16, qy + 69, "Which product sold the most last week?", 11, "ink2", 400, extra='font-style="italic"', t=t))
    o.append(rect(x + 16, qy + 78, 150, 22, t["page"], t["stroke"], 11))
    o.append(icon("calendar", x + 25, qy + 82, 13, t["ink2"]))
    o.append(text(x + 44, qy + 93, "as-of 2026-03-04", 11, "ink", 600, font=MONO, t=t))

    # model output: program
    x, w = 790, 290
    py, ph = 52, 172
    o.append(rect(x, py, w, ph, t["code_bg"], t["stroke"], 12, 1, 'filter="url(#sh)"'))
    o.append(window_dots(x + 16, py + 15, 4, 13))
    o.append(text(x + w - 14, py + 19, "program (simplified)", 10.5, t["code_dim"], 600, "end", font=MONO))
    code = [
        [("week = (", "code_ink"), ('"2026-02-23"', "code_str"), (", ", "code_ink"), ('"2026-03-01"', "code_str"), (")", "code_ink")],
        [("s = sales[sales.status == ", "code_ink"), ('"valid"', "code_str"), ("]", "code_ink")],
        [("s = s[s.date.", "code_ink"), ("between", "code_fn"), ("(*week)]", "code_ink")],
        [("answer = (s.", "code_ink"), ("groupby", "code_fn"), ("(", "code_ink"), ('"product"', "code_str"), (")", "code_ink")],
        [("          .qty.", "code_ink"), ("sum", "code_fn"), ("().", "code_ink"), ("idxmax", "code_fn"), ("())", "code_ink")],
    ]
    yy = py + 56
    for parts in code:
        spans = "".join(f'<tspan fill="{t[k]}">{esc(s)}</tspan>' for s, k in parts)
        o.append(f'<text x="{x + 16}" y="{yy}" font-family="{MONO}" font-size="11" font-weight="500" {PRE}>{spans}</text>')
        yy += 19
    # verification chips
    cy = py + ph + 16
    for i, (lab, ic) in enumerate((("verified", "shield"), ("executed", "term"))):
        cx = x + i * 116
        o.append(rect(cx, cy, 104, 28, t["aqua_t"], t["aqua"], 14, 1))
        o.append(icon(ic, cx + 11, cy + 6, 16, t["aqua"]))
        o.append(text(cx + 33, cy + 18.5, lab, 11.5, t["aqua"], 650))
    o.append(line(x + 105, cy + 14, x + 115, cy + 14, "aqua", t, sw=1.4, marker=False))
    # answer
    ay = cy + 44
    o.append(card(x, ay, w, 394 - ay, t, t["orange_t"], t["orange"], 1.3))
    o.append(icon("send", x + 14, ay + 14, 18, t["orange"]))
    o.append(text(x + 40, ay + 28, "Answer", 14, "ink", 650, t=t))
    o.append(rect(x + w - 74, ay + 13, 60, 22, t["page"], t["orange"], 11, 1))
    o.append(text(x + w - 44, ay + 28, "string", 11, t["orange"], 600, "middle", font=MONO))
    o.append(text(x + 16, ay + 70, "Xi măng Hà Tiên PCB40 bao 50kg", 15, "ink", 700, t=t))

    # arrows
    o.append(line(460, 159, 502, 159, "blue", t))
    o.append(text(481, 150, "prepare", 10, "muted", 600, "middle", t=t))
    o.append(line(740, 130, 787, 130, "violet", t))
    o.append(path(f"M740 338 H754 Q764 338 764 328 V140 Q764 130 774 130", "violet", t, marker=False))
    o.append(text(764, 122, "SLM", 10.5, t["violet"], 700, "middle"))
    return W, H, "".join(o)


# ---------------------------------------------------------------- Figure 3: public-test outcome
def fig_results(t):
    W, H = 1000, 290
    o = [defs(t)]
    right, abst, wrong = 995, 4, 1
    cell, gap, ncol, nrow = 9, 2, 50, 20
    x0, y0 = 24, 24
    for i in range(ncol * nrow):
        r, c = divmod(i, ncol)
        col = t["good"] if i < right else (t["abst"] if i < right + abst else t["bad"])
        op = f' opacity="{GOOD_OP[t["page"]]}"' if i < right else ''
        o.append(f'<rect x="{x0 + c * (cell + gap)}" y="{y0 + r * (cell + gap)}" width="{cell}" height="{cell}" rx="1.8" fill="{col}"{op}/>')
    gw = ncol * (cell + gap) - gap
    gh = nrow * (cell + gap) - gap
    o.append(text(x0, y0 + gh + 26, "Public test: 1,000 questions from 14 businesses, one square per question", 11.5, "muted", 500, t=t))
    # right panel
    px = x0 + gw + 56
    o.append(text(px, 92, "0.994", 58, "ink", 750, t=t))
    o.append(text(px + 2, 118, "PENALIZED ACCURACY", 11.5, "muted", 700, extra='letter-spacing="1.2"', t=t))
    o.append(f'<line x1="{px}" y1="140" x2="{W - 24}" y2="140" stroke="{t["hair"]}"/>')
    stats = [(f"{right}", "correct", "good"), (f"{abst}", "abstained", "abst"), (f"{wrong}", "wrong", "bad")]
    for i, (val, lab, key) in enumerate(stats):
        yy = 174 + 30 * i
        o.append(rect(px, yy - 12, 13, 13, t[key], "none", 3, extra=f'opacity="{GOOD_OP[t["page"]]}"' if key == "good" else ""))
        o.append(f'<text x="{px + 24}" y="{yy}" font-family="{SANS}" font-size="16" fill="{t["ink"]}">'
                 f'<tspan font-weight="700">{val}</tspan><tspan dx="7" font-size="14" fill="{t["ink2"]}">{lab}</tspan></text>')
    return W, H, "".join(o)


def page(W, H, body):
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">{body}</svg>'
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>html,body{{margin:0;background:transparent}}'
            f'svg{{display:block}}</style></head><body>{svg}</body></html>')


for name, fn in (("pipeline", fig_pipeline), ("example", fig_example), ("results", fig_results)):
    for theme, t in THEMES.items():
        W, H, body = fn(t)
        with open(os.path.join(OUT, f"{name}-{theme}.html"), "w") as f:
            f.write(page(W, H, body))
print("ok")
