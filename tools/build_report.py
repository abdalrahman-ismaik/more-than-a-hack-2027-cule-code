"""Build docs/CuleCode.pdf from its editable Markdown source.

Install requirements-docs.txt first. No match data or model packages are needed.
"""

from html import escape
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak, Paragraph, SimpleDocTemplate,
)

ROOT = Path(__file__).resolve().parents[1]


def inline(text):
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"`(.+?)`", r'<font name="Courier">\1</font>', text)


def footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D6DFE8"))
    canvas.line(20 * mm, 19 * mm, 190 * mm, 19 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#536174"))
    canvas.drawString(20 * mm, 14 * mm, "CuleCode | More Than a Hack 2027 | Initial Challenge")
    canvas.drawRightString(190 * mm, 14 * mm, str(document.page))
    canvas.restoreState()


def main():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("BodyFinal", fontName="Helvetica", fontSize=10,
                              leading=14, spaceAfter=9, alignment=TA_LEFT))
    styles.add(ParagraphStyle("SectionFinal", fontName="Helvetica-Bold", fontSize=12,
                              leading=16, spaceBefore=10, spaceAfter=9,
                              textColor=colors.HexColor("#153B61"), keepWithNext=True))
    styles.add(ParagraphStyle("TitleFinal", fontName="Helvetica-Bold", fontSize=27,
                              leading=32, spaceAfter=10,
                              textColor=colors.HexColor("#153B61"), keepWithNext=True))
    styles.add(ParagraphStyle("SubtitleFinal", fontName="Helvetica", fontSize=17,
                              leading=22, spaceAfter=14, keepWithNext=True))
    source = (ROOT / "docs" / "CuleCode.md").read_text(encoding="utf-8")
    story = []
    for block in re.split(r"\n\s*\n", source.strip()):
        block = block.strip()
        if block == "<!-- pagebreak -->":
            story.append(PageBreak())
            continue
        # Title and subtitle can share a Markdown block.
        if block.startswith("# "):
            for line in block.splitlines():
                if line.startswith("# "):
                    story.append(Paragraph(inline(line[2:]), styles["TitleFinal"]))
                elif line.startswith("## "):
                    story.append(Paragraph(inline(line[3:]), styles["SubtitleFinal"]))
                else:
                    story.append(Paragraph(inline(line), styles["BodyFinal"]))
        elif block.startswith("### "):
            story.append(Paragraph(inline(block[4:]), styles["SectionFinal"]))
        else:
            story.append(Paragraph(inline(" ".join(block.splitlines())), styles["BodyFinal"]))
    destination = ROOT / "docs" / "CuleCode.pdf"
    SimpleDocTemplate(
        str(destination), pagesize=A4, rightMargin=20 * mm, leftMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=25 * mm,
        title="CuleCode - Possession from automatic tracking",
        author="CuleCode", pageCompression=1,
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    print(destination)


if __name__ == "__main__":
    main()
