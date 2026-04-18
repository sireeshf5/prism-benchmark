"""Generate a PDF benchmark report from benchmark-results/data.json."""
import json
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.platypus import PageBreak

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
DARK    = colors.HexColor("#1e293b")
BLUE    = colors.HexColor("#2563eb")
GREEN   = colors.HexColor("#16a34a")
RED     = colors.HexColor("#dc2626")
AMBER   = colors.HexColor("#d97706")
GRAY    = colors.HexColor("#6b7280")
LGRAY   = colors.HexColor("#f3f4f6")
MGRAY   = colors.HexColor("#e5e7eb")
WHITE   = colors.white
BLUE_BG = colors.HexColor("#eff6ff")
GREEN_BG= colors.HexColor("#f0fdf4")
RED_BG  = colors.HexColor("#fef2f2")
AMBER_BG= colors.HexColor("#fffbeb")

W, H = A4
MARGIN = 18*mm

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
data_path = Path("benchmark-results/data.json")
d = json.loads(data_path.read_text(encoding="utf-8"))

judge_scores_path = Path("benchmark-results/accuracy/judge_scores.md")
judge_md = judge_scores_path.read_text(encoding="utf-8") if judge_scores_path.exists() else ""

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
base = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, **kw)

sTitle    = S("sTitle",    fontName="Helvetica-Bold", fontSize=22, textColor=WHITE,    leading=28, spaceAfter=4)
sSubtitle = S("sSubtitle", fontName="Helvetica",      fontSize=12, textColor=colors.HexColor("#cbd5e1"), leading=16)
sMeta     = S("sMeta",     fontName="Helvetica",      fontSize=9,  textColor=colors.HexColor("#94a3b8"), leading=12)

sH1   = S("sH1",   fontName="Helvetica-Bold", fontSize=14, textColor=DARK,  spaceBefore=14, spaceAfter=6,  leading=18)
sH2   = S("sH2",   fontName="Helvetica-Bold", fontSize=11, textColor=DARK,  spaceBefore=10, spaceAfter=4,  leading=15)
sBody = S("sBody", fontName="Helvetica",      fontSize=9,  textColor=DARK,  spaceAfter=4,   leading=13)
sBold = S("sBold", fontName="Helvetica-Bold", fontSize=9,  textColor=DARK,  spaceAfter=4,   leading=13)
sSmall= S("sSmall",fontName="Helvetica",      fontSize=8,  textColor=GRAY,  spaceAfter=2,   leading=11)
sItalic=S("sItalic",fontName="Helvetica-Oblique",fontSize=8,textColor=GRAY, spaceAfter=2,   leading=11)
sGreen= S("sGreen",fontName="Helvetica-Bold", fontSize=9,  textColor=GREEN, spaceAfter=2,   leading=13)
sRed  = S("sRed",  fontName="Helvetica-Bold", fontSize=9,  textColor=RED,   spaceAfter=2,   leading=13)
sAmber= S("sAmber",fontName="Helvetica-Bold", fontSize=9,  textColor=AMBER, spaceAfter=2,   leading=13)
sBlue = S("sBlue", fontName="Helvetica-Bold", fontSize=9,  textColor=BLUE,  spaceAfter=2,   leading=13)
sCenter=S("sCenter",fontName="Helvetica",     fontSize=9,  textColor=DARK,  alignment=TA_CENTER, leading=13)
sNum  = S("sNum",  fontName="Helvetica",      fontSize=9,  textColor=DARK,  alignment=TA_RIGHT,  leading=13)
sNumB = S("sNumB", fontName="Helvetica-Bold", fontSize=9,  textColor=BLUE,  alignment=TA_RIGHT,  leading=13)
sNumR = S("sNumR", fontName="Helvetica-Bold", fontSize=9,  textColor=RED,   alignment=TA_RIGHT,  leading=13)
sNumG = S("sNumG", fontName="Helvetica-Bold", fontSize=9,  textColor=GREEN, alignment=TA_RIGHT,  leading=13)
sNumA = S("sNumA", fontName="Helvetica-Bold", fontSize=9,  textColor=AMBER, alignment=TA_RIGHT,  leading=13)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def HR():
    return HRFlowable(width="100%", thickness=0.5, color=MGRAY, spaceAfter=6, spaceBefore=6)

def SP(n=4):
    return Spacer(1, n)

def badge_text(status):
    return {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL", "SKIP": "SKIP"}.get(status, status)

def badge_color(status):
    return {"PASS": GREEN, "WARN": AMBER, "FAIL": RED, "SKIP": GRAY}.get(status, GRAY)

def badge_bg(status):
    return {"PASS": GREEN_BG, "WARN": AMBER_BG, "FAIL": RED_BG, "SKIP": LGRAY}.get(status, LGRAY)

def section_header(title, status=None):
    cells = [Paragraph(title, sH1)]
    row = [cells[0]]
    tdata = [row]
    if status:
        badge = Paragraph(f'<b>{badge_text(status)}</b>', ParagraphStyle(
            "badge", fontName="Helvetica-Bold", fontSize=8,
            textColor=badge_color(status), alignment=TA_RIGHT
        ))
        tdata = [[cells[0], badge]]
        t = Table(tdata, colWidths=[W - 2*MARGIN - 30, 30])
        t.setStyle(TableStyle([
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        return t
    return Paragraph(title, sH1)

def alert_box(text, kind="blue"):
    colour = {"green": GREEN, "red": RED, "amber": AMBER, "blue": BLUE}[kind]
    bg     = {"green": GREEN_BG, "red": RED_BG, "amber": AMBER_BG, "blue": BLUE_BG}[kind]
    style  = ParagraphStyle("alert", fontName="Helvetica", fontSize=9,
                            textColor=colour, leading=13)
    t = Table([[Paragraph(text, style)]], colWidths=[W - 2*MARGIN])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), bg),
        ("BOX",          (0,0), (-1,-1), 0.5, colour),
        ("LEFTPADDING",  (0,0), (-1,-1), 10),
        ("RIGHTPADDING", (0,0), (-1,-1), 10),
        ("TOPPADDING",   (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[bg]),
    ]))
    return t

def data_table(headers, rows, col_widths=None):
    hrow = [Paragraph(h, ParagraphStyle("th", fontName="Helvetica-Bold", fontSize=8,
                                         textColor=GRAY, leading=11)) for h in headers]
    tdata = [hrow] + rows
    if col_widths is None:
        col_widths = [(W - 2*MARGIN) / len(headers)] * len(headers)
    t = Table(tdata, colWidths=col_widths)
    ts = TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  LGRAY),
        ("LINEBELOW",     (0,0), (-1,0),  0.5, MGRAY),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),  [WHITE, colors.HexColor("#fafafa")]),
        ("LINEBELOW",     (0,1), (-1,-2), 0.3, MGRAY),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("BOX",           (0,0), (-1,-1), 0.5, MGRAY),
    ])
    t.setStyle(ts)
    return t

def score_box(label, score, tokens, colour, bg):
    inner = Table([
        [Paragraph(f"<b>{label}</b>", ParagraphStyle("sl", fontName="Helvetica-Bold",
                   fontSize=8, textColor=colour, leading=10))],
        [Paragraph(f"<b>{score}/15</b>", ParagraphStyle("sv", fontName="Helvetica-Bold",
                   fontSize=20, textColor=colour, leading=24))],
        [Paragraph(f"{tokens:,} input tokens", sSmall)],
    ], colWidths=[(W - 2*MARGIN)/2 - 8])
    inner.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), bg),
        ("BOX",          (0,0),(-1,-1), 0.5, colour),
        ("LEFTPADDING",  (0,0),(-1,-1), 10),
        ("RIGHTPADDING", (0,0),(-1,-1), 10),
        ("TOPPADDING",   (0,0),(-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1), 6),
    ]))
    return inner

# ---------------------------------------------------------------------------
# Build story
# ---------------------------------------------------------------------------
story = []

# ── Cover / Header ──────────────────────────────────────────────────────────
cover = Table([[
    Paragraph("Graphify Token Reduction Benchmark", sTitle),
    SP(2),
    Paragraph("Does graphify reduce tokens without compromising accuracy?", sSubtitle),
    SP(4),
    Paragraph("Generated: 2026-04-18  |  Model: claude-sonnet-4-20250514  |  Budget: 2,000 tokens/query  |  Encoder: cl100k_base", sMeta),
]], colWidths=[W - 2*MARGIN])
cover.setStyle(TableStyle([
    ("BACKGROUND",   (0,0),(-1,-1), DARK),
    ("LEFTPADDING",  (0,0),(-1,-1), 20),
    ("RIGHTPADDING", (0,0),(-1,-1), 20),
    ("TOPPADDING",   (0,0),(-1,-1), 22),
    ("BOTTOMPADDING",(0,0),(-1,-1), 22),
    ("ROUNDEDCORNERS", (0,0),(-1,-1), [6,6,6,6]),
]))
story.append(cover)
story.append(SP(12))

# ── Verdict ─────────────────────────────────────────────────────────────────
story.append(section_header("Overall Verdict"))
story.append(alert_box(
    "<b>Significant cost savings. Accuracy cost is real.</b><br/>"
    "Graphify delivers genuine token reduction (up to <b>37.8x vs naive</b>, <b>15.9x cheaper per query</b> on "
    "large corpora) and breaks even in under 5 queries. However the LLM judge found <b>raw files win 5/5 on "
    "answer quality</b> — graph answers lack code excerpts and specific values. Worth using when cost matters "
    "and slightly-less-precise answers are acceptable, or for architectural/structural queries. "
    "<b>Not suitable for doc-heavy corpora</b> (0% extraction coverage).",
    "amber"
))
story.append(SP(8))

# ── Key metrics ─────────────────────────────────────────────────────────────
story.append(section_header("Key Metrics at a Glance"))
metrics = [
    ("Max token reduction", "37.8x", "Large corpus vs naive read", GREEN),
    ("Cost savings (large)",  "15.9x", "$0.013 vs $0.205 per query", GREEN),
    ("Break-even queries",  "2.7–4.4", "Before graph pays off", AMBER),
    ("Accuracy — LLM judge", "0 / 5", "Graph wins vs raw files", RED),
    ("Cache hit rate",       "19.1%",  "WARN: expected ~89%", AMBER),
    ("Doc corpus coverage",  "0%",     "15 .md files → 0 graph nodes", RED),
]
metric_rows = []
for label, val, sub, col in metrics:
    metric_rows.append([
        Paragraph(label, sSmall),
        Paragraph(f"<b>{val}</b>", ParagraphStyle("mv", fontName="Helvetica-Bold",
                  fontSize=16, textColor=col, alignment=TA_CENTER, leading=20)),
        Paragraph(sub, sSmall),
    ])

mt = Table(metric_rows, colWidths=[80, 60, (W - 2*MARGIN - 145)])
mt.setStyle(TableStyle([
    ("GRID",          (0,0),(-1,-1), 0.5, MGRAY),
    ("BACKGROUND",    (0,0),(-1,-1), LGRAY),
    ("LEFTPADDING",   (0,0),(-1,-1), 8),
    ("RIGHTPADDING",  (0,0),(-1,-1), 8),
    ("TOPPADDING",    (0,0),(-1,-1), 6),
    ("BOTTOMPADDING", (0,0),(-1,-1), 6),
    ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    ("ROWBACKGROUNDS",(0,0),(-1,-1), [WHITE, colors.HexColor("#fafafa")]),
]))

# arrange in 2 columns
left_metrics = metric_rows[:3]
right_metrics = metric_rows[3:]

def mini_metric_table(rows):
    t = Table(rows, colWidths=[75, 55, None])
    t.setStyle(TableStyle([
        ("GRID",          (0,0),(-1,-1), 0.5, MGRAY),
        ("BACKGROUND",    (0,0),(-1,-1), LGRAY),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("RIGHTPADDING",  (0,0),(-1,-1), 8),
        ("TOPPADDING",    (0,0),(-1,-1), 6),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0,0),(-1,-1), [WHITE, colors.HexColor("#fafafa")]),
    ]))
    return t

story.append(mt)
story.append(SP(4))
story.append(alert_box(
    "<b>Claimed 71.5x reduction not validated.</b> Best observed is 37.8x on large corpus vs naive full-file read. "
    "Against a realistic grep-top-5 baseline, ratios are 2.3x–9.8x.", "amber"))
story.append(SP(10))

# ── Token Reduction ──────────────────────────────────────────────────────────
story.append(section_header("Tests 1–3 — Token Reduction vs Naive Full Read", "PASS"))
cw = [(W-2*MARGIN)/7]*7
rows = [
    [Paragraph("Small",  sBold), Paragraph("6",       sNum), Paragraph("3,138",  sNum),
     Paragraph("336",    sNum), Paragraph("9.3x",  sNumB), Paragraph("$0.0171", sNum), Paragraph("$0.0087", sNumG)],
    [Paragraph("Medium", sBold), Paragraph("47",      sNum), Paragraph("21,189", sNum),
     Paragraph("1,287",  sNum), Paragraph("16.5x", sNumB), Paragraph("$0.0712", sNum), Paragraph("$0.0115", sNumG)],
    [Paragraph("Large",  sBold), Paragraph("145",     sNum), Paragraph("65,854", sNum),
     Paragraph("1,741",  sNum), Paragraph("37.8x", sNumB), Paragraph("$0.2052", sNum), Paragraph("$0.0129", sNumG)],
]
story.append(data_table(
    ["Corpus", "Files", "Naive Tokens", "Graph Tokens", "Reduction", "$/q Naive", "$/q Graph"],
    rows, col_widths=cw
))
story.append(SP(6))
story.append(alert_box(
    "Graph queries are hard-capped at <b>--budget 2000 tokens</b>. As corpus size grows, the ratio improves "
    "mechanically because the numerator grows while the denominator stays bounded.", "blue"))
story.append(SP(10))

# ── Realistic baseline ───────────────────────────────────────────────────────
story.append(section_header("Test 4 — Realistic Baseline: Graphify vs Grep-Top-5-Files", "PASS"))
cw4 = [50, (W-2*MARGIN)*0.45, 70, 70, 50]
rows4 = [
    [Paragraph("Small", sBold),  Paragraph("main training loop?", sBody),         Paragraph("806",    sNum), Paragraph("336",   sNum), Paragraph("2.4x", sNumG)],
    [Paragraph("",      sBody),  Paragraph("attention mechanism?", sBody),         Paragraph("585",    sNum), Paragraph("646",   sNum), Paragraph("0.9x", sNumR)],
    [Paragraph("",      sBody),  Paragraph("model initialized?",   sBody),         Paragraph("2,261",  sNum), Paragraph("1,141", sNum), Paragraph("2.0x", sNumB)],
    [Paragraph("Medium",sBold),  Paragraph("main training loop?",  sBody),         Paragraph("12,598", sNum), Paragraph("1,287", sNum), Paragraph("9.8x", sNumG)],
    [Paragraph("",      sBody),  Paragraph("attention mechanism?", sBody),         Paragraph("10,046", sNum), Paragraph("1,280", sNum), Paragraph("7.8x", sNumG)],
    [Paragraph("",      sBody),  Paragraph("model initialized?",   sBody),         Paragraph("12,663", sNum), Paragraph("1,733", sNum), Paragraph("7.3x", sNumG)],
    [Paragraph("Large", sBold),  Paragraph("main training loop?",  sBody),         Paragraph("13,290", sNum), Paragraph("1,740", sNum), Paragraph("7.6x", sNumG)],
    [Paragraph("",      sBody),  Paragraph("attention mechanism?", sBody),         Paragraph("5,511",  sNum), Paragraph("1,499", sNum), Paragraph("3.7x", sNumB)],
    [Paragraph("",      sBody),  Paragraph("model initialized?",   sBody),         Paragraph("4,107",  sNum), Paragraph("1,778", sNum), Paragraph("2.3x", sNumB)],
]
story.append(data_table(["Corpus", "Question", "Grep-top-5 tokens", "Graph tokens", "Ratio"], rows4, col_widths=cw4))
story.append(SP(4))
story.append(alert_box(
    "On small corpora graphify can be <b>worse</b> than a simple grep (0.9x on Q2). "
    "The benefit becomes meaningful at medium+ sizes.", "amber"))
story.append(SP(10))

# ── Cost model ───────────────────────────────────────────────────────────────
story.append(section_header("Tests 6 & 10 — Build Cost, Break-Even & $/query", "PASS"))
cw6 = [(W-2*MARGIN)/4]*4
rows6 = [
    [Paragraph("Small",  sBold), Paragraph("7,800",   sNum), Paragraph("2,802",  sNum), Paragraph("2.8 queries", sNumB)],
    [Paragraph("Medium", sBold), Paragraph("88,400",  sNum), Paragraph("19,902", sNum), Paragraph("4.4 queries", sNumB)],
    [Paragraph("Large",  sBold), Paragraph("171,600", sNum), Paragraph("64,114", sNum), Paragraph("2.7 queries", sNumB)],
]
story.append(data_table(["Corpus", "Build cost (tokens)", "Savings/query", "Break-even"], rows6, col_widths=cw6))
story.append(SP(6))

cw10 = [55, 90, 75, 65, None]
rows10 = [
    [Paragraph("Small", sBold),  Paragraph("Naive (all files)", sBody),    Paragraph("3,138",  sNum), Paragraph("$0.0171", sNum), Paragraph("",      sSmall)],
    [Paragraph("",      sBody),  Paragraph("Graphify query",    sBody),    Paragraph("336",    sNum), Paragraph("$0.0087", sNumG),Paragraph("2.0x cheaper", sGreen)],
    [Paragraph("",      sBody),  Paragraph("Embed + top-5",     sBody),    Paragraph("1,217",  sNum), Paragraph("$0.0113", sNum), Paragraph("+$0.000063 index",sSmall)],
    [Paragraph("Medium",sBold),  Paragraph("Naive (all files)", sBody),    Paragraph("21,189", sNum), Paragraph("$0.0712", sNum), Paragraph("",      sSmall)],
    [Paragraph("",      sBody),  Paragraph("Graphify query",    sBody),    Paragraph("1,287",  sNum), Paragraph("$0.0115", sNumG),Paragraph("6.2x cheaper", sGreen)],
    [Paragraph("",      sBody),  Paragraph("Embed + top-5",     sBody),    Paragraph("11,769", sNum), Paragraph("$0.0430", sNum), Paragraph("+$0.000424 index",sSmall)],
    [Paragraph("Large", sBold),  Paragraph("Naive (all files)", sBody),    Paragraph("65,854", sNum), Paragraph("$0.2052", sNum), Paragraph("",      sSmall)],
    [Paragraph("",      sBody),  Paragraph("Graphify query",    sBody),    Paragraph("1,741",  sNum), Paragraph("$0.0129", sNumG),Paragraph("15.9x cheaper", sGreen)],
    [Paragraph("",      sBody),  Paragraph("Embed + top-5",     sBody),    Paragraph("7,636",  sNum), Paragraph("$0.0306", sNum), Paragraph("+$0.001317 index",sSmall)],
    [Paragraph("Docs",  sBold),  Paragraph("Naive (all files)", sBody),    Paragraph("63,874", sNum), Paragraph("$0.1993", sNum), Paragraph("",      sSmall)],
    [Paragraph("",      sBody),  Paragraph("Graphify query",    sBody),    Paragraph("5",      sNum), Paragraph("$0.0077", sNumR),Paragraph("EMPTY — useless", sRed)],
    [Paragraph("",      sBody),  Paragraph("Embed + top-5",     sBody),    Paragraph("22,836", sNum), Paragraph("$0.0762", sNumB),Paragraph("Best option for docs",sBlue)],
]
story.append(data_table(["Corpus", "Approach", "Input tokens", "$/query", "Notes"], rows10, col_widths=cw10))
story.append(SP(10))

# ── Accuracy ─────────────────────────────────────────────────────────────────
story.append(section_header("Tests 5 & 11 — Answer Quality: LLM-as-Judge (Claude)", "FAIL"))
story.append(alert_box(
    "<b>Raw files won every single question (5/5).</b> Graph answers describe structure but lack actual code, "
    "specific parameter values, and concrete examples. Average scores: Graphify 10.2/15 vs Raw 13.2/15.",
    "red"))
story.append(SP(8))

questions = d.get("test11", {}).get("questions", [])
judge_comments = [
    "Provides structural outline only; raw answer shows actual loop code.",
    "Graph makes assumptions about missing implementation; raw is more accurate.",
    "Raw gives concrete config values (n_layer=12, n_head=12, n_embd=768); graph gives generic description.",
    "Raw surfaces config parameters (always_save_checkpoint, out_dir); graph finds nothing.",
    "Raw infers AdamW from beta params; graph only says 'configure_optimizers exists'.",
]

for i, q in enumerate(questions):
    comment = judge_comments[i] if i < len(judge_comments) else q.get("comment","")
    gs = q["graph_score"]; rs = q["raw_score"]
    g_tok = d["test5"]["questions"][i].get("graph_input_tokens", 0)
    r_tok = d["test5"]["questions"][i].get("raw_input_tokens", 0)

    row_content = [
        score_box("GRAPHIFY", gs, g_tok, BLUE, BLUE_BG),
        score_box("RAW FILES", rs, r_tok, GREEN, GREEN_BG),
    ]
    pair = Table([row_content], colWidths=[(W-2*MARGIN)/2 - 4, (W-2*MARGIN)/2 - 4])
    pair.setStyle(TableStyle([("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
                               ("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),0),
                               ("COLWIDTH",(0,0),(0,0),(W-2*MARGIN)/2-4)]))

    block = KeepTogether([
        Paragraph(f"<b>Q{i+1}: {q['question']}</b>", sH2),
        pair,
        SP(3),
        Paragraph(f"<i>{comment}</i>", sItalic),
        SP(8),
    ])
    story.append(block)

story.append(SP(4))

# ── Doc corpus ───────────────────────────────────────────────────────────────
story.append(section_header("Test 9 — Doc Corpus: Complete Extraction Failure", "FAIL"))
story.append(alert_box(
    "<b>0 of 15 markdown files produced any graph nodes.</b> collect_files() excludes .md/.txt/.rst entirely. "
    "The AST extractor has no handler for documentation files. Every doc query returns 5 tokens (empty shell).",
    "red"))
story.append(SP(6))
cwd = [(W-2*MARGIN)*0.55, (W-2*MARGIN)*0.45]
rows_doc = [
    [Paragraph("Doc files in corpus", sBody),          Paragraph("15", sNum)],
    [Paragraph("Total naive tokens", sBody),            Paragraph("63,874", sNum)],
    [Paragraph("Graph nodes extracted", sBody),         Paragraph("0", sNumR)],
    [Paragraph("Graph edges extracted", sBody),         Paragraph("0", sNumR)],
    [Paragraph("AST extraction coverage", sBody),       Paragraph("0.0%", sNumR)],
    [Paragraph("Doc queries with useful answers", sBody),Paragraph("0 / 5", sNumR)],
]
story.append(data_table(["Metric", "Value"], rows_doc, col_widths=cwd))
story.append(SP(4))
story.append(alert_box(
    "<b>Fix:</b> Run the full /graphify skill (Claude subagents per doc file) for LLM-based extraction. "
    "Alternative: text-embedding-3-small ($0.02/1M tokens) + vector search + read top-5 files.", "blue"))
story.append(SP(10))

# ── Cache ────────────────────────────────────────────────────────────────────
story.append(section_header("Test 7 — SHA256 Cache Hit Rate", "WARN"))
story.append(alert_box(
    "<b>Cache over-invalidated.</b> Modifying 5 files caused 38 cache entries to be reprocessed (expected ~5). "
    "Hit rate: 19.1% — likely due to community re-clustering cascading rebuilds across the graph.",
    "amber"))
story.append(SP(6))
cwc = [(W-2*MARGIN)*0.55, (W-2*MARGIN)*0.45]
rows_cache = [
    [Paragraph("Files modified", sBody),             Paragraph("5", sNum)],
    [Paragraph("Cache entries reprocessed", sBody),  Paragraph("38  (expected ~5)", sNumR)],
    [Paragraph("Total files in corpus", sBody),      Paragraph("47", sNum)],
    [Paragraph("Cache hit rate", sBody),             Paragraph("19.1%", sNumR)],
    [Paragraph("Cache entries before", sBody),       Paragraph("68", sNum)],
    [Paragraph("Cache entries after", sBody),        Paragraph("106", sNum)],
]
story.append(data_table(["Metric", "Value"], rows_cache, col_widths=cwc))
story.append(SP(10))

# ── Decision framework ───────────────────────────────────────────────────────
story.append(section_header("Test 12 — Decision Framework"))

use_items = [
    "Code-heavy corpus (>80% .py/.ts/.go/.rs etc) — tree-sitter AST is free",
    "More than ~4–5 queries on the same corpus (break-even: 2.7–4.4 queries)",
    "Corpus has >50 files / >20k tokens (ratio grows with size)",
    "Architectural or relationship queries: 'what calls X?', 'how does A connect to B?'",
    "Cost per query matters — 15.9x cheaper than naive on large corpus",
]
avoid_items = [
    "Doc-heavy corpus (.md/.txt/.rst dominate) — 0% AST extraction coverage",
    "Need exact implementation details, specific values, or code excerpts",
    "Small corpus (<10 files) — grep-top-5 is as good or better",
    "One-off queries — build cost won't amortise in time",
    "Accuracy is non-negotiable — raw files win by 3 points/question on average",
]

use_content = [[Paragraph("USE Graphify when:", ParagraphStyle("uh", fontName="Helvetica-Bold",
                fontSize=10, textColor=GREEN, leading=14, spaceAfter=4))]]
for item in use_items:
    use_content.append([Paragraph(f"<b>+</b>  {item}", ParagraphStyle("ui", fontName="Helvetica",
                fontSize=9, textColor=colors.HexColor("#166534"), leading=13))])

avoid_content = [[Paragraph("AVOID Graphify when:", ParagraphStyle("ah", fontName="Helvetica-Bold",
                  fontSize=10, textColor=RED, leading=14, spaceAfter=4))]]
for item in avoid_items:
    avoid_content.append([Paragraph(f"<b>✗</b>  {item}", ParagraphStyle("ai", fontName="Helvetica",
                fontSize=9, textColor=colors.HexColor("#991b1b"), leading=13))])

half = (W - 2*MARGIN - 8) / 2
def box_table(rows, bg, border):
    t = Table(rows, colWidths=[half])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), bg),
        ("BOX",          (0,0),(-1,-1), 0.5, border),
        ("LEFTPADDING",  (0,0),(-1,-1), 10),
        ("RIGHTPADDING", (0,0),(-1,-1), 10),
        ("TOPPADDING",   (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
    ]))
    return t

decision_row = Table(
    [[box_table(use_content, GREEN_BG, GREEN), box_table(avoid_content, RED_BG, RED)]],
    colWidths=[half, half]
)
decision_row.setStyle(TableStyle([
    ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
    ("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),0),
    ("COLPADDING",(0,0),(0,0),4),
]))
story.append(decision_row)
story.append(SP(12))

# Scenario table
story.append(Paragraph("Scenario Summary", sH2))
scenarios = [
    (">50 code files, >20 queries",               "USE graphify",            GREEN),
    ("Mixed code+doc, code-focused questions",     "USE graphify",            GREEN),
    ("Exploring a large unfamiliar codebase",      "USE graphify",            GREEN),
    ("<10 code files, <5 queries",                 "SKIP — just read files",  RED),
    ("Doc-heavy corpus, doc questions",            "SKIP — use embeddings",   RED),
    ("Need exact implementation details",          "SKIP — raw files win",    RED),
    ("Writing a chatbot over documentation",       "SKIP — needs embeddings", RED),
]
scenario_rows = []
for scenario, verdict, col in scenarios:
    vs = ParagraphStyle("vs", fontName="Helvetica-Bold", fontSize=9,
                        textColor=col, alignment=TA_RIGHT, leading=13)
    scenario_rows.append([Paragraph(scenario, sBody), Paragraph(verdict, vs)])

story.append(data_table(["Scenario", "Verdict"], scenario_rows,
             col_widths=[(W-2*MARGIN)*0.60, (W-2*MARGIN)*0.40]))
story.append(SP(10))

# ── Test status ──────────────────────────────────────────────────────────────
story.append(section_header("Full Test Suite Status"))
test_rows = [
    ("1",  "Baseline Naive Token Count",                    "PASS"),
    ("2",  "Graph Query Output Size vs Naive",              "PASS"),
    ("3",  "Compression Ratio vs Corpus Size",              "PASS"),
    ("4",  "Realistic Naive Baseline (grep top-5)",         "PASS"),
    ("5",  "Query Accuracy Spot Check",                     "PASS"),
    ("6",  "Amortised Cost & Break-Even Analysis",          "PASS"),
    ("7",  "SHA256 Cache Hit Rate — 38/47 reprocessed",     "WARN"),
    ("8",  "Code-only vs Mixed Corpus Extraction Cost",     "PASS"),
    ("9",  "Doc Corpus: 0% AST extraction coverage",        "PASS"),
    ("10", "Dollar Cost Model ($/query)",                   "PASS"),
    ("11", "LLM-as-Judge Accuracy — Raw wins 5/5",          "PASS"),
    ("12", "Decision Framework",                            "PASS"),
]
status_rows = []
for num, name, status in test_rows:
    sc = ParagraphStyle("sc", fontName="Helvetica-Bold", fontSize=8,
                        textColor=badge_color(status), alignment=TA_RIGHT, leading=11)
    status_rows.append([
        Paragraph(f"Test {num}", sBold),
        Paragraph(name, sBody),
        Paragraph(badge_text(status), sc),
    ])
story.append(data_table(["#", "Test Name", "Status"], status_rows,
             col_widths=[40, W-2*MARGIN-100, 60]))
story.append(SP(12))

# ── Footer ───────────────────────────────────────────────────────────────────
story.append(HR())
story.append(Paragraph(
    "Graphify Token Reduction Benchmark  |  2026-04-18  |  Raw data: benchmark-results/data.json",
    ParagraphStyle("foot", fontName="Helvetica", fontSize=8, textColor=GRAY,
                   alignment=TA_CENTER, leading=11)
))

# ---------------------------------------------------------------------------
# Build PDF
# ---------------------------------------------------------------------------
out_path = Path("benchmark-results/graphify-benchmark-report.pdf")
doc = SimpleDocTemplate(
    str(out_path),
    pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=MARGIN,  bottomMargin=MARGIN,
    title="Graphify Token Reduction Benchmark Report",
    author="Benchmark Suite",
)

def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GRAY)
    canvas.drawRightString(W - MARGIN, 10*mm, f"Page {doc.page}")
    canvas.restoreState()

doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
print(f"PDF saved to {out_path}")
