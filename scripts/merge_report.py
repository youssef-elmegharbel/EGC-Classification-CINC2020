"""Merge every member's section file in report/sections/ into report/phase1_report.docx (order by filename).

Section files: M1_sections.docx, M2_*.docx, M3_*.docx. M2/M3 files are only read, never modified.
Note: sections are concatenated in filename order (M1 holds sections 1-3, 7, 8; M2 and M3 hold 4-6), so
if the final order must follow the cover sheet, split M1 into M1_a (1-3) and M1_b (7-8) first, or edit in Word.
"""
from pathlib import Path

from docx import Document
from docxcompose.composer import Composer

root = Path(__file__).resolve().parents[1] / "report"
files = sorted((root / "sections").glob("*.docx"))
files = [f for f in files if not f.name.startswith("~$")]
assert files, "no section files found"
master = Document(files[0])
comp = Composer(master)
for f in files[1:]:
    master.add_page_break()
    comp.append(Document(f))
out = root / "phase1_report.docx"
comp.save(str(out))
print("merged", [f.name for f in files], "->", out)
