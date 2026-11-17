"""Build submissions/CuleCode.pdf from docs/CuleCode.md."""

from html import escape
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
NAVY = colors.HexColor("#153B61")
BLUE = colors.HexColor("#28658A")
PALE = colors.HexColor("#EAF1F6")
INK = colors.HexColor("#263445")


def inline(text):
    text = escape(text.strip())
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"`(.+?)`", r'<font name="Courier">\1</font>', text)


class Pipeline(Flowable):
    """Small vector diagram that stays sharp in print and in the PDF."""

    def __init__(self):
        super().__init__()
        self.width = 170 * mm
        self.height = 32 * mm

    def draw(self):
        labels = [
            ("ATD tracking", "players + ball"),
            ("108 features", "geometry + time"),
            ("LightGBM + GRU", "H / A / D scores"),
            ("Viterbi", "per-half decode"),
            ("Submission", "H / A / D / X"),
        ]
        gap = 4 * mm
        box_w = (self.width - gap * 4) / 5
        box_h = 22 * mm
        y = 5 * mm
        for i, (title, subtitle) in enumerate(labels):
            x = i * (box_w + gap)
            self.canv.setFillColor(PALE if i % 2 == 0 else colors.white)
            self.canv.setStrokeColor(colors.HexColor("#AFC3D3"))
            self.canv.roundRect(x, y, box_w, box_h, 2 * mm, fill=1, stroke=1)
            self.canv.setFillColor(NAVY)
            self.canv.setFont("Helvetica-Bold", 7.2)
            self.canv.drawCentredString(x + box_w / 2, y + 13 * mm, title)
            self.canv.setFillColor(INK)
            self.canv.setFont("Helvetica", 6.6)
            self.canv.drawCentredString(x + box_w / 2, y + 7 * mm, subtitle)
            if i < len(labels) - 1:
                ax = x + box_w + 0.5 * mm
                self.canv.setStrokeColor(BLUE)
                self.canv.setFillColor(BLUE)
                self.canv.setLineWidth(1)
                self.canv.line(ax, y + box_h / 2, ax + gap - 1.5 * mm, y + box_h / 2)
                self.canv.line(ax + gap - 1.5 * mm, y + box_h / 2,
                               ax + gap - 2.8 * mm, y + box_h / 2 + 1.2 * mm)
                self.canv.line(ax + gap - 1.5 * mm, y + box_h / 2,
                               ax + gap - 2.8 * mm, y + box_h / 2 - 1.2 * mm)


def footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D6DFE8"))
    canvas.line(20 * mm, 19 * mm, 190 * mm, 19 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#536174"))
    canvas.drawString(20 * mm, 14 * mm, "CuleCode | More Than a Hack 2027 | Initial Challenge")
    canvas.drawRightString(190 * mm, 14 * mm, str(document.page))
    canvas.restoreState()


def markdown_table(block, styles):
    rows = []
    for line in block.strip().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", c or "-") for c in cells):
            continue
        rows.append([Paragraph(inline(c), styles["TableHeader" if not rows else "TableCell"])
                     for c in cells])
    if not rows:
        return None
    ncols = max(map(len, rows))
    widths = [170 * mm / ncols] * ncols
    if ncols == 3:
        widths = [39 * mm, 43 * mm, 88 * mm]
    elif ncols == 2:
        widths = [48 * mm, 122 * mm]
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C9D5DF")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def main():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("BodyFinal", fontName="Helvetica", fontSize=9.3,
                              leading=13, spaceAfter=7, alignment=TA_JUSTIFY,
                              textColor=INK, splitLongWords=True))
    styles.add(ParagraphStyle("SectionFinal", fontName="Helvetica-Bold", fontSize=12,
                              leading=15, spaceBefore=9, spaceAfter=6,
                              textColor=NAVY, keepWithNext=True))
    styles.add(ParagraphStyle("TitleFinal", fontName="Helvetica-Bold", fontSize=27,
                              leading=32, spaceAfter=6, textColor=NAVY, keepWithNext=True))
    styles.add(ParagraphStyle("SubtitleFinal", fontName="Helvetica", fontSize=16,
                              leading=20, spaceAfter=10, textColor=BLUE, keepWithNext=True))
    styles.add(ParagraphStyle("TableHeader", fontName="Helvetica-Bold", fontSize=7.7,
                              leading=9.5, textColor=colors.white))
    styles.add(ParagraphStyle("TableCell", fontName="Helvetica", fontSize=7.7,
                              leading=9.5, textColor=INK))
    styles.add(ParagraphStyle("Meta", fontName="Helvetica", fontSize=8.5,
                              leading=11, spaceAfter=5, textColor=INK))

    source = (ROOT / "docs" / "CuleCode.md").read_text(encoding="utf-8")
    story = []
    for block in re.split(r"\n\s*\n", source.strip()):
        block = block.strip()
        if block == "<!-- pagebreak -->":
            story.append(PageBreak())
        elif block == "<!-- pipeline -->":
            story.extend([Pipeline(), Spacer(1, 4 * mm)])
        elif block.startswith("|"):
            table = markdown_table(block, styles)
            if table:
                story.extend([table, Spacer(1, 4 * mm)])
        elif block.startswith("# "):
            for line in block.splitlines():
                if line.startswith("# "):
                    story.append(Paragraph(inline(line[2:]), styles["TitleFinal"]))
                elif line.startswith("## "):
                    story.append(Paragraph(inline(line[3:]), styles["SubtitleFinal"]))
                else:
                    story.append(Paragraph(inline(line), styles["Meta"]))
        elif block.startswith("### "):
            story.append(Paragraph(inline(block[4:]), styles["SectionFinal"]))
        else:
            story.append(Paragraph(inline(" ".join(block.splitlines())), styles["BodyFinal"]))

    destination = ROOT / "submissions" / "CuleCode.pdf"
    SimpleDocTemplate(
        str(destination), pagesize=A4, rightMargin=20 * mm, leftMargin=20 * mm,
        topMargin=17 * mm, bottomMargin=25 * mm,
        title="CuleCode - Possession from automatic tracking",
        author="CuleCode", pageCompression=1,
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    print(destination)


if __name__ == "__main__":
    main()
