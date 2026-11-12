#!/usr/bin/env python3
"""Generate README figures (light + dark) as standalone HTML pages holding one SVG each."""
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
        code_kw="#ff7b72", code_str="#a5d6ff", code_fn="#d2a8ff", code_tag="#7ee787",
        mess="#f2a33a", mess_t="#fff4e0", sheet_hdr="#eef1f4", strike="#d03b3b",
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
        code_kw="#ff7b72", code_str="#a5d6ff", code_fn="#d2a8ff", code_tag="#7ee787",
        mess="#e0952e", mess_t="#2b2010", sheet_hdr="#1c232c", strike="#e05252",
    ),
}

SANS = "Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
PRE = 'xml:space="preserve" style="font-variant-ligatures:none"'
MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def esc(s):
    return html.escape(s, quote=True)


def text(x, y, s, size=13, fill="ink", weight=400, anchor="start", font=SANS, extra="", t=None):
    col = t[fill] if t and fill in t else fill
    return (f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" font-weight="{weight}" '
            f'fill="{col}" text-anchor="{anchor}" {extra}>{esc(s)}</text>')


def rect(x, y, w, h, fill, stroke="none", r=10, sw=1, extra=""):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}" {extra}/>')


# ---- icons: 24x24 stroke paths (drawn at scale s, top-left x, y) ----
ICONS = {
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/>',
    "db": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    "chat": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "filter": '<path d="M22 3H2l8 9.46V19l4 2v-8.54L22 3z"/>',
    "table": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M3 15h18M9 3v18"/>',
    "type": '<path d="M4 7V4h16v3M9 20h6M12 4v16"/>',
    "cpu": ('<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6" rx="1"/>'
            '<path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/>'),
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="M9 12l2 2 4-4"/>',
    "term": '<rect x="2" y="3" width="20" height="18" rx="2"/><path d="M6 9l4 3-4 3M12 16h6"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "send": '<path d="M22 2L11 13"/><path d="M22 2l-7 20-4-9-9-4 20-7z"/>',
    "check": '<path d="M20 6L9 17l-5-5"/>',
    "alert": '<path d="M10.3 3.9L1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4M12 17h.01"/>',
    "calendar": '<rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4M8 2v4M3 10h18"/>',
}


def icon(name, x, y, size, color, sw=2):
    s = size / 24
    return (f'<g transform="translate({x},{y}) scale({s})" fill="none" stroke="{color}" stroke-width="{sw / s * s:.2f}" '
            f'stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</g>')


def arrow_defs(t):
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


def line(x1, y1, x2, y2, color_key, t, sw=1.8, dash=None, marker=True):
    d = f'stroke-dasharray="{dash}"' if dash else ""
    m = f'marker-end="url(#ah-{color_key})"' if marker else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{t[color_key]}" stroke-width="{sw}" '
            f'stroke-linecap="round" {d} {m}/>')


def path(d, color_key, t, sw=1.8, dash=None, marker=True):
    da = f'stroke-dasharray="{dash}"' if dash else ""
    m = f'marker-end="url(#ah-{color_key})"' if marker else ""
    return (f'<path d="{d}" fill="none" stroke="{t[color_key]}" stroke-width="{sw}" stroke-linecap="round" '
            f'stroke-linejoin="round" {da} {m}/>')


def num_badge(cx, cy, n, color, t):
    return (f'<circle cx="{cx}" cy="{cy}" r="11" fill="{color}"/>'
            + text(cx, cy + 4.3, str(n), 12, "#ffffff", 700, "middle"))


def stage_box(x, y, w, h, n, title, sub, accent, tint, ic, t, sub2=None):
    out = [rect(x, y, w, h, t["card"], t["stroke"], 12, 1, 'filter="url(#sh)"')]
    out.append(f'<rect x="{x}" y="{y + 12}" width="3.5" height="{h - 24}" rx="1.75" fill="{accent}"/>')
    out.append(rect(x + 16, y + 16, 36, 36, tint, "none", 9))
    out.append(icon(ic, x + 24, y + 24, 20, accent))
    out.append(text(x + 64, y + 31, title, 14.5, "ink", 650, t=t))
    out.append(text(x + 64, y + 50, sub, 12, "ink2", 400, t=t))
    if sub2:
        out.append(text(x + 64, y + 66, sub2, 12, "ink2", 400, t=t))
    out.append(num_badge(x + w - 4, y + 4, n, accent, t))
    return "".join(out)


def col_header(x, w, label, color, t):
    return (text(x, 26, label, 11.5, color, 700, extra='letter-spacing="1.4"')
            + f'<line x1="{x}" y1="36" x2="{x + w}" y2="36" stroke="{color}" stroke-width="2" stroke-linecap="round" opacity="0.55"/>')


# ---------------------------------------------------------------- Figure 1: pipeline
def fig_pipeline(t):
    W, H = 1100, 500
    o = [arrow_defs(t)]
    # column headers
    o.append(col_header(20, 210, "INPUT", t["muted"], t))
    o.append(col_header(282, 236, "PREPARE", t["blue"], t))
    o.append(col_header(565, 195, "GENERATE", t["violet"], t))
    o.append(col_header(840, 240, "VERIFY & ANSWER", t["aqua"], t))

    # --- input: business data
    x, y, w, h = 20, 56, 210, 196
    o.append(rect(x, y, w, h, t["card"], t["stroke"], 12, 1, 'filter="url(#sh)"'))
    o.append(icon("db", x + 16, y + 16, 20, t["ink2"]))
    o.append(text(x + 44, y + 31, "Business data", 14.5, "ink", 650, t=t))
    chips = [(".xlsx", "#1f8a4c"), (".csv", "#2a78d6"), (".sqlite", "#8a5cd6"), ("README.md", "#818b98")]
    for i, (lab, col) in enumerate(chips):
        cx = x + 16 + (i % 2) * 92
        cy = y + 50 + (i // 2) * 44
        cw = 86 if i != 3 else 86
        o.append(rect(cx, cy, cw, 34, t["page"], t["stroke"], 8))
        o.append(icon("file", cx + 7, cy + 9, 16, col))
        o.append(text(cx + 27, cy + 21.5, lab, 11.5 if lab != "README.md" else 10, "ink", 600, font=MONO, t=t))
    o.append(text(x + 16, y + 158, "multi-sheet · inconsistent names", 11.5, "ink2", t=t))
    o.append(text(x + 16, y + 175, "mixed number & date formats", 11.5, "ink2", t=t))

    # --- input: question
    x, y, w, h = 20, 282, 210, 196
    o.append(rect(x, y, w, h, t["card"], t["stroke"], 12, 1, 'filter="url(#sh)"'))
    o.append(icon("chat", x + 16, y + 16, 20, t["ink2"]))
    o.append(text(x + 44, y + 31, "Question", 14.5, "ink", 650, t=t))
    # speech bubble
    bx, by = x + 16, y + 50
    o.append(f'<path d="M{bx + 10} {by}h158a10 10 0 0 1 10 10v38a10 10 0 0 1-10 10H{bx + 30}l-12 10v-10h-8a10 10 0 0 1-10-10V{by + 10}a10 10 0 0 1 10-10z" '
             f'fill="{t["page"]}" stroke="{t["stroke"]}"/>')
    o.append(text(bx + 12, by + 23, "“sp bán dc nhiều nhất", 12.5, "ink", 600, t=t))
    o.append(text(bx + 12, by + 41, "tuần rồi là gì?”", 12.5, "ink", 600, t=t))
    o.append(text(x + 16, y + 140, "≈ best-selling product last week?", 11.5, "ink2", extra='font-style="italic"', t=t))
    o.append(rect(x + 16, y + 152, 154, 26, t["page"], t["stroke"], 13))
    o.append(icon("calendar", x + 25, y + 158, 14, t["ink2"]))
    o.append(text(x + 45, y + 169.5, "as-of 2026-03-04", 11.5, "ink", 600, font=MONO, t=t))

    # --- prepare
    o.append(stage_box(282, 56, 236, 86, 1, "Clean & filter", "headers, units, dates,", t["blue"], t["blue_t"], "filter", t,
                       "status values, noise rows"))
    o.append(stage_box(282, 166, 236, 86, 2, "Schema generation", "one canonical schema", t["blue"], t["blue_t"], "table", t,
                       "for every business"))
    o.append(stage_box(282, 330, 236, 86, 3, "Query understanding", "colloquial, no-accent text", t["blue"], t["blue_t"], "type", t,
                       "relative dates → periods"))

    # --- generate: SLM
    x, y, w, h = 565, 110, 195, 306
    o.append(rect(x, y, w, h, "url(#slm)", t["violet"], 14, 1.5, 'filter="url(#sh)"'))
    o.append(num_badge(x + w - 4, y + 4, 4, t["violet"], t))
    o.append(rect(x + w / 2 - 26, y + 18, 52, 52, t["page"], t["violet"], 12, 1.2))
    o.append(icon("cpu", x + w / 2 - 16, y + 28, 32, t["violet"]))
    o.append(text(x + w / 2, y + 96, "Small Language", 15, "ink", 700, "middle", t=t))
    o.append(text(x + w / 2, y + 115, "Model", 15, "ink", 700, "middle", t=t))
    o.append(text(x + w / 2, y + 137, "reads schema + question,", 11.5, "ink2", 400, "middle", t=t))
    o.append(text(x + w / 2, y + 152, "writes a short program", 11.5, "ink2", 400, "middle", t=t))
    # mini code card
    cx, cy, cw, ch = x + 14, y + 168, w - 28, 122
    o.append(rect(cx, cy, cw, ch, t["code_bg"], "none", 9))
    for i, c in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        o.append(f'<circle cx="{cx + 12 + i * 11}" cy="{cy + 11}" r="3.2" fill="{c}"/>')
    rows = [("<plan>", "code_tag", 0, None), (None, "code_dim", 12, 92), ("</plan>", "code_tag", 0, None),
            ("<code>", "code_tag", 0, None), (None, "code_fn", 12, 70), (None, "code_str", 12, 104), ("</code>", "code_tag", 0, None)]
    yy = cy + 30
    for lab, col, ind, bar in rows:
        if lab:
            o.append(text(cx + 12 + ind, yy, lab, 10.5, t[col], 600, font=MONO))
        else:
            o.append(rect(cx + 12 + ind, yy - 7, bar, 6, t[col], "none", 3, extra='opacity="0.75"'))
        yy += 13.5

    # --- verify & answer
    o.append(stage_box(840, 56, 240, 70, 5, "Verification module", "checks every program", t["aqua"], t["aqua_t"], "shield", t))
    o.append(stage_box(840, 150, 240, 70, 6, "Sandboxed execution", "isolated · fully offline", t["aqua"], t["aqua_t"], "term", t))
    o.append(stage_box(840, 244, 240, 86, 7, "Answer selection", "confident → answer", t["aqua"], t["aqua_t"], "target", t,
                       "uncertain → abstain"))
    # output card
    x, y, w, h = 840, 356, 240, 122
    o.append(rect(x, y, w, h, t["orange_t"], t["orange"], 12, 1.3, 'filter="url(#sh)"'))
    o.append(icon("send", x + 16, y + 15, 18, t["orange"]))
    o.append(text(x + 42, y + 29, "Typed answer", 14.5, "ink", 650, t=t))
    o.append(text(x + 16, y + 55, '{"type": "string",', 11, "ink", 500, font=MONO, extra=PRE, t=t))
    o.append(text(x + 16, y + 72, ' "answer": "Xi măng Hà Tiên', 11, "ink", 500, font=MONO, extra=PRE, t=t))
    o.append(text(x + 16, y + 89, '            PCB40 bao 50kg"}', 11, "ink", 500, font=MONO, extra=PRE, t=t))
    o.append(text(x + 16, y + 110, "or unanswerable / abstain", 11.5, "ink2", 400, t=t))

    # --- arrows
    o.append(line(230, 99, 279, 99, "blue", t))                      # data -> clean
    o.append(line(400, 142, 400, 163, "blue", t))                    # clean -> schema
    o.append(line(230, 373, 279, 373, "blue", t))                    # question -> query
    o.append(path("M518 209 H533 Q543 209 543 219 V232 Q543 240 551 240 H562", "violet", t))   # schema -> SLM
    o.append(path("M518 373 H533 Q543 373 543 363 V318 Q543 310 551 310 H562", "violet", t))   # query -> SLM
    o.append(path("M760 150 H790 Q800 150 800 140 V101 Q800 91 810 91 H837", "aqua", t))      # SLM -> verify
    o.append(line(960, 126, 960, 147, "aqua", t))                    # verify -> exec
    o.append(line(960, 220, 960, 241, "aqua", t))                    # exec -> select
    o.append(line(960, 330, 960, 353, "orange", t))                  # select -> output
    # feedback loop (verify/exec -> SLM)
    o.append(path("M840 185 H818 Q808 185 808 195 V235 Q808 245 798 245 H763", "orange", t, sw=1.6, dash="5 4"))
    o.append(text(814, 268, "feedback", 10.5, "orange", 600, "middle", t=t))
    o.append(text(814, 281, "& repair", 10.5, "orange", 600, "middle", t=t))
    return W, H, "".join(o)


# ---------------------------------------------------------------- Figure 2: example
def fig_example(t):
    W, H = 1100, 470
    o = [arrow_defs(t)]
    o.append(col_header(20, 450, "INPUT  ·  raw business files + question", t["muted"], t))
    o.append(col_header(500, 255, "PREPARED CONTEXT", t["blue"], t))
    o.append(col_header(795, 285, "MODEL OUTPUT", t["violet"], t))

    # ---- spreadsheet window
    x, y, w = 20, 52, 450
    cols = [("Ngày", 78), ("Sản phẩm", 140), ("SL", 52), ("Thành tiền", 84), ("TT", 80)]
    rows = [
        ("23/02", "XM Hà Tiên PCB40", "12 bao", "1tr440", "Xong"),
        ("46078", "Thép D10", "30 cây", "3.150.000đ", "hủy"),
        ("24/2/2026", "xi mang ha tien pcb40", "5", "600k", "ok"),
        ("26-02-2026", "Cát vàng", "3 khối", "1,05tr", "Hoàn thành"),
        ("Tổng tháng", "", "", "6.24tr", ""),
    ]
    mess = {(0, 1), (0, 3), (1, 0), (1, 3), (2, 0), (2, 1), (2, 3), (3, 0), (3, 3), (4, 0)}
    rh = 27
    h = 34 + 36 + rh * (len(rows) + 1) + 34
    o.append(rect(x, y, w, h, t["card"], t["stroke"], 12, 1, 'filter="url(#sh)"'))
    # tab
    o.append(icon("file", x + 14, y + 9, 16, "#1f8a4c"))
    o.append(text(x + 36, y + 22, "ban_hang_T2.xlsx", 12, "ink", 600, font=MONO, t=t))
    o.append(text(x + w - 14, y + 22, "sheet “T2”  ·  + 3 more files", 11, "muted", 500, "end", t=t))
    o.append(f'<line x1="{x}" y1="{y + 34}" x2="{x + w}" y2="{y + 34}" stroke="{t["stroke"]}"/>')
    gx, gy = x + 8, y + 42
    gw = sum(c[1] for c in cols)
    # title row (junk header above the real header)
    o.append(text(gx + 6, gy + 17, "BÁO CÁO BÁN HÀNG THÁNG 2/2026", 11.5, "ink", 700, t=t))
    o.append(rect(gx + 238, gy + 4, 170, 19, t["mess_t"], t["mess"], 5, 1, 'stroke-dasharray="3 2"'))
    o.append(text(gx + 323, gy + 17.5, "title row above header", 10.5, t["mess"], 600, "middle"))
    gy += 28
    o.append(rect(gx, gy, gw, rh, t["sheet_hdr"], "none", 0))
    cx = gx
    for name, cw in cols:
        o.append(text(cx + 6, gy + 18, name, 11.5, "ink", 700, t=t))
        cx += cw
    for r, row in enumerate(rows):
        ry = gy + rh * (r + 1)
        cancelled = r == 1
        cx = gx
        for c, ((_, cw), val) in enumerate(zip(cols, row)):
            if (r, c) in mess:
                o.append(rect(cx + 2, ry + 4, cw - 4, rh - 8, t["mess_t"], "none", 4))
            weight = 700 if r == 4 else 450
            o.append(text(cx + 6, ry + 18, val, 11.5, "ink2" if cancelled else "ink", weight,
                          font=MONO if c in (0, 2, 3) else SANS, t=t))
            cx += cw
        o.append(f'<line x1="{gx}" y1="{ry}" x2="{gx + gw}" y2="{ry}" stroke="{t["hair"]}"/>')
    # cancelled-order callout
    ry = gy + rh * 2
    o.append(f'<line x1="{gx + 4}" y1="{ry + 14}" x2="{gx + gw - 4}" y2="{ry + 14}" stroke="{t["strike"]}" stroke-width="1.2" opacity="0.55"/>')
    # legend
    ly = y + h - 14
    o.append(rect(x + 14, ly - 9, 12, 12, t["mess_t"], t["mess"], 3, 1))
    o.append(text(x + 32, ly + 1, "messy cells: Excel serial dates, “1tr440”, “600k”, unaccented names", 10.5, "ink2", 500, t=t))

    # ---- question bubble
    qy = y + h + 18
    global QC
    QC = qy + 31
    o.append(rect(x, qy, w, 62, t["card"], t["stroke"], 12, 1, 'filter="url(#sh)"'))
    o.append(icon("chat", x + 14, qy + 21, 20, t["ink2"]))
    o.append(text(x + 44, qy + 27, "“sp bán dc nhiều nhất tuần rồi là gì?”", 14, "ink", 650, t=t))
    o.append(text(x + 44, qy + 47, "which product sold the most last week?", 11.5, "ink2", 400, extra='font-style="italic"', t=t))
    o.append(rect(x + w - 128, qy + 18, 114, 26, t["page"], t["stroke"], 13))
    o.append(text(x + w - 71, qy + 35, "as-of 03-04 (Wed)", 10.5, "ink", 600, "middle", font=MONO, t=t))

    # ---- prepared context
    x, w = 500, 255
    o.append(rect(x, 52, w, 236, t["card"], t["stroke"], 12, 1, 'filter="url(#sh)"'))
    o.append(rect(x + 14, 64, 28, 28, t["blue_t"], "none", 7))
    o.append(icon("table", x + 20, 70, 16, t["blue"]))
    o.append(text(x + 52, 83, "Generated schema", 14, "ink", 650, t=t))
    sch = [
        ("sales", True), ("date", False), ("product → products", False), ("qty, unit", False),
        ("amount  (VND)", False), ("status  valid | cancelled", False),
        ("products", True), ("name, aliases[ ]", False),
    ]
    yy = 112
    for lab, is_tab in sch:
        if is_tab:
            o.append(text(x + 18, yy, lab, 11.5, t["blue"], 700, font=MONO))
        else:
            o.append(f'<circle cx="{x + 24}" cy="{yy - 4}" r="2" fill="{t["muted"]}"/>')
            o.append(text(x + 32, yy, lab, 11, "ink", 450, font=MONO, t=t))
        yy += 21 if not is_tab else 20
    # normalized query card
    o.append(rect(x, 304, w, 132, t["card"], t["stroke"], 12, 1, 'filter="url(#sh)"'))
    o.append(rect(x + 14, 316, 28, 28, t["blue_t"], "none", 7))
    o.append(icon("type", x + 20, 322, 16, t["blue"]))
    o.append(text(x + 52, 335, "Normalized query", 14, "ink", 650, t=t))
    o.append(text(x + 16, 365, "“sản phẩm bán được nhiều", 12, "ink", 500, t=t))
    o.append(text(x + 16, 382, "nhất tuần rồi là gì?”", 12, "ink", 500, t=t))
    o.append(rect(x + 16, 396, w - 32, 26, t["blue_t"], "none", 6))
    o.append(text(x + 26, 413, "tuần rồi → 02-23 … 03-01", 11, t["blue"], 600, font=MONO))

    # ---- model output
    x, w = 795, 285
    ch = 214
    o.append(rect(x, 52, w, ch, t["code_bg"], t["stroke"], 12, 1, 'filter="url(#sh)"'))
    for i, c in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        o.append(f'<circle cx="{x + 16 + i * 13}" cy="{66}" r="4" fill="{c}"/>')
    o.append(text(x + w - 14, 70, "SLM", 11, t["code_dim"], 600, "end", font=MONO))
    code = [
        [("<plan>", "code_tag")],
        [("week = 02-23 … 03-01", "code_dim")],
        [("valid orders only", "code_dim")],
        [("units per product → top-1", "code_dim")],
        [("</plan>", "code_tag")],
        [("<code>", "code_tag")],
        [("s = sales[sales.status == ", "code_ink"), ('"valid"', "code_str"), ("]", "code_ink")],
        [("s = s[s.date.", "code_ink"), ("between", "code_fn"), ("(W0, W1)]", "code_ink")],
        [("ans = s.", "code_ink"), ("groupby", "code_fn"), ("(", "code_ink"), ('"product"', "code_str"), (")", "code_ink")],
        [("       .qty.", "code_ink"), ("sum", "code_fn"), ("().", "code_ink"), ("idxmax", "code_fn"), ("()", "code_ink")],
        [("</code>", "code_tag")],
    ]
    yy = 96
    for parts in code:
        spans = "".join(f'<tspan fill="{t[k]}">{esc(s)}</tspan>' for s, k in parts)
        ind = 12 if parts[0][1] == "code_dim" else 0
        o.append(f'<text x="{x + 16 + ind}" y="{yy}" font-family="{MONO}" font-size="11" font-weight="500" '
                 f'{PRE}>{spans}</text>')
        yy += 15.5
    # verify / execute chips
    cy = 52 + ch + 14
    chips = [("verified", "shield"), ("executed", "term")]
    for i, (lab, ic) in enumerate(chips):
        cx = x + i * 112
        o.append(rect(cx, cy, 102, 28, t["aqua_t"], t["aqua"], 14, 1))
        o.append(icon(ic, cx + 10, cy + 6, 16, t["aqua"]))
        o.append(text(cx + 32, cy + 18.5, lab, 11.5, t["aqua"], 650))
    o.append(line(x + 102 + 1, cy + 14, x + 112 - 1, cy + 14, "aqua", t, sw=1.4, marker=False))
    # answer card
    ay = cy + 42
    o.append(rect(x, ay, w, 104, t["orange_t"], t["orange"], 12, 1.3, 'filter="url(#sh)"'))
    o.append(icon("send", x + 14, ay + 14, 18, t["orange"]))
    o.append(text(x + 40, ay + 28, "Answer", 14, "ink", 650, t=t))
    o.append(rect(x + w - 74, ay + 13, 60, 22, t["page"], t["orange"], 11, 1))
    o.append(text(x + w - 44, ay + 28, "string", 11, t["orange"], 600, "middle", font=MONO))
    o.append(text(x + 16, ay + 58, "Xi măng Hà Tiên PCB40 bao 50kg", 14.5, "ink", 700, t=t))
    o.append(text(x + 16, ay + 82, "17 bags — the 30 × Thép D10 order", 11.5, "ink2", 400, t=t))
    o.append(text(x + 16, ay + 97, "was cancelled, so it does not count", 11.5, "ink2", 400, t=t))

    # arrows between panels
    o.append(line(470, 170, 497, 170, "blue", t))
    o.append(line(470, QC, 497, QC, "blue", t))
    o.append(line(755, 170, 792, 170, "violet", t))
    o.append(path(f"M755 {QC} H765 Q774 {QC} 774 {QC - 9} V179 Q774 170 783 170", "violet", t, marker=False))
    return W, H, "".join(o)


# ---------------------------------------------------------------- Figure 3: results waffle
def fig_results(t):
    W, H = 1100, 372
    o = [arrow_defs(t)]
    sets = [
        ("Model only", "without the rule-based layer", 967, 17, 16, "0.951", 25),
        ("Full system", "submitted", 992, 2, 6, "0.986", 575),
    ]
    cell, gap, ncol, nrow = 8, 2, 50, 20
    for title, sub, right, abst, wrong, pa, x0 in sets:
        o.append(text(x0, 30, title, 17, "ink", 700, t=t))
        o.append(text(x0, 50, sub, 12.5, "ink2", 400, t=t))
        o.append(text(x0 + 498, 44, pa, 34, "ink", 750, "end", t=t))
        o.append(text(x0 + 498, 62, "penalized accuracy", 11, "muted", 600, "end", extra='letter-spacing="0.6"', t=t))
        y0 = 80
        for i in range(ncol * nrow):
            r, c = divmod(i, ncol)
            col = t["good"] if i < right else (t["abst"] if i < right + abst else t["bad"])
            op = ' opacity="0.55"' if i < right else ''
            o.append(f'<rect x="{x0 + c * (cell + gap)}" y="{y0 + r * (cell + gap)}" width="{cell}" height="{cell}" rx="1.6" fill="{col}"{op}/>')
        yb = y0 + nrow * (cell + gap) + 22
        stats = [(f"{right}", "correct", "good", "check"), (f"{abst}", "abstain", "abst", None), (f"{wrong}", "wrong", "bad", None)]
        xx = x0
        for val, lab, key, _ in stats:
            o.append(rect(xx, yb - 10, 11, 11, t[key], "none", 2.5, extra='opacity="0.55"' if key == "good" else ""))
            o.append(f'<text x="{xx + 17}" y="{yb}" font-family="{SANS}" font-size="13.5" fill="{t["ink"]}">'
                     f'<tspan font-weight="700">{val}</tspan><tspan dx="5" font-size="12.5" fill="{t["ink2"]}">{lab}</tspan></text>')
            xx += 130
    # footer legend
    fy = H - 22
    o.append(text(W / 2, fy, "Public test: 1,000 questions over 14 businesses · one square = one question · "
                  "PA = (correct − wrong) / N", 12, "muted", 500, "middle", t=t))
    o.append(f'<line x1="{W / 2 - 330}" y1="{fy - 24}" x2="{W / 2 + 330}" y2="{fy - 24}" stroke="{t["hair"]}"/>')
    return W, H, "".join(o)


def page(W, H, body, t):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">{body}</svg>')
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>html,body{{margin:0;background:transparent}}'
            f'svg{{display:block}}</style></head><body>{svg}</body></html>'), svg


for name, fn in (("pipeline", fig_pipeline), ("example", fig_example), ("results", fig_results)):
    for theme, t in THEMES.items():
        W, H, body = fn(t)
        html_doc, svg = page(W, H, body, t)
        with open(os.path.join(OUT, f"{name}-{theme}.html"), "w") as f:
            f.write(html_doc)
print("ok")
