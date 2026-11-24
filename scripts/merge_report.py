"""Merge every member's section file in report/sections/ into report/phase1_report.docx, in section-number order.

Section files: M1_sections.docx (sections 1-3, 7, 8), M2_*.docx (4-5), M3_*.docx (6). M2/M3 files are only read.
Each file is split at its numbered Heading 1s ("4. Linear ..."); contiguous runs of sections become slices, and the
slices from all files are ordered by section number, so M1's 7-8 land after M3's 6. Missing sections are reported.
"""
import re
from pathlib import Path

from docx import Document
from docxcompose.composer import Composer

EXPECTED = range(1, 9)
root = Path(__file__).resolve().parents[1] / "report"
files = [f for f in sorted((root / "sections").glob("*.docx")) if not f.name.startswith("~$")]
assert files, "no section files found"


W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def h1_number(el):
    """Section number if the body element is a numbered Heading 1, else None."""
    style = el.find(f".//{W}pStyle")
    if el.tag != W + "p" or style is None or style.get(W + "val") != "Heading1":
        return None
    m = re.match(r"\s*(\d+)\.", "".join(t.text or "" for t in el.iter(W + "t")))
    return int(m.group(1)) if m else None


def slices(path):
    """Yield (first_section_number, section_numbers, Document) for each contiguous run of sections in a file."""
    marks = [(i, n) for i, el in enumerate(Document(path).element.body) if (n := h1_number(el)) is not None]
    assert marks, f"{path.name}: no numbered Heading 1 found"
    runs = [[marks[0]]]
    for m in marks[1:]:
        if m[1] == runs[-1][-1][1] + 1:
            runs[-1].append(m)
        else:
            runs.append([m])
    for k, run in enumerate(runs):
        start = run[0][0]
        end = runs[k + 1][0][0] if k + 1 < len(runs) else None
        doc = Document(path)
        for i, el in enumerate(list(doc.element.body)):
            if not el.tag.endswith("}sectPr") and (i < start or (end is not None and i >= end)):
                doc.element.body.remove(el)
        yield run[0][1], [n for _, n in run], doc


parts = sorted((p for f in files for p in slices(f)), key=lambda t: t[0])
found = sorted(n for _, ns, _ in parts for n in ns)
missing = [n for n in EXPECTED if n not in found]
master = parts[0][2]
comp = Composer(master)
for _, _, d in parts[1:]:
    master.add_page_break()
    comp.append(d)
out = root / "phase1_report.docx"
comp.save(str(out))
print("merged sections", found, "->", out)
if missing:
    print("WARNING: missing sections", missing, "- add their section files and re-run")
