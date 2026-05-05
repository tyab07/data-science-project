---
name: dir-to-pptx
description: "Create professional PowerPoint presentations by analyzing files in a directory. Use this skill whenever a user wants to generate a presentation from existing files, documents, data, or content in a folder. Triggers include: 'create a presentation from my files', 'make a deck from this directory', 'turn these documents into a slide presentation', 'generate slides from folder contents', 'analyze and present files', or any request to build a polished .pptx from multiple source files. Use this skill even if the user just says 'make a presentation' and points to a folder — automatically scan, analyze, and create professional slides."
compatibility: "Requires: python-pptx, python-docx (docx reading), pypdf (PDF reading), pillow. Node.js + pptxgenjs optional for advanced styling."
---

# Directory to PowerPoint Skill

Generate professional, visually polished presentations by analyzing files in a directory.

## Workflow Overview

1. **Scan & Analyze**: Read all files in the target directory (PDFs, Word docs, images, CSVs, text files)
2. **Extract Content**: Pull structured data, text, key points, and insights from each file
3. **Organize**: Structure findings into logical presentation sections
4. **Design & Build**: Create slides using a professional color palette, layout system, and visual elements
5. **Polish**: Add images, charts, formatted text, and visual consistency
6. **Verify**: Check for overflow, alignment, and content completeness

## How to Use This Skill

When the user points to a directory and asks for a presentation, follow these steps:

### Step 1: Scan the Directory

```bash
ls -lh /path/to/directory/
file /path/to/directory/*
```

Identify:
- Document types (PDF, DOCX, CSV, TXT, PNG/JPG, etc.)
- File count and rough sizes
- Potential structure (are files numbered? Named by topic?)

### Step 2: Extract & Analyze Content

**For each file type:**

**PDF**: Use `pypdf` to extract text, or `pdfplumber` for tables
```python
from pypdf import PdfReader
reader = PdfReader("file.pdf")
text = "\n".join([page.extract_text() for page in reader.pages])
```

**Word (.docx)**: Use `python-docx`
```python
from docx import Document
doc = Document("file.docx")
text = "\n".join([p.text for p in doc.paragraphs])
```

**CSV/TSV**: Use `csv` module or `pandas`
```python
import pandas as pd
df = pd.read_csv("data.csv")
# Extract summaries, create tables
```

**Text files**: Read directly
```python
with open("file.txt") as f:
    text = f.read()
```

**Images**: Use `Pillow` to check validity; reference in slides

### Step 3: Structure the Presentation

Map extracted content to slide types:

| Content Type | Slide Type | Example |
|---|---|---|
| Document title, date, author | Title slide | "Q3 Financial Report" |
| Chapter/section headers | Section divider | "Executive Summary" |
| Bullet points, lists | Content slide | Goals, findings, recommendations |
| Tables, numerical data | Data slide | Tables, metrics, comparisons |
| Key statistics | Callout slide | Large numbers (24pt+) with labels |
| Images, diagrams | Image slide | Charts, photos, mockups |
| Process, timeline | Flow slide | Steps with arrows or numbered items |
| Conclusion, next steps | Close slide | Summary, calls to action |

### Step 4: Design Decisions

**Pick ONE professional color palette** (see PPTX skill for palettes):

- **Midnight Executive**: Navy + ice blue + white → formal, trustworthy
- **Forest & Moss**: Green tones → organic, growth-focused
- **Coral Energy**: Coral + gold + navy → energetic, modern
- **Ocean Gradient**: Deep blue tones → professional, calm
- **Charcoal Minimal**: Minimal, dark → premium, clean

**Layout rules:**
- Title slide: Dark background, large white text, centered
- Content slides: Light background (white), dark text, left-aligned body
- Data slides: Light background with table or chart centered
- Closing slide: Dark background matching title slide

**Visual consistency:**
- All titles: Bold, 36-40pt, one color
- All body text: 14-16pt, left-aligned
- All data labels: 12pt, muted color
- Margins: minimum 0.5" on all sides
- Element spacing: 0.3-0.5" between blocks (consistent throughout)

### Step 5: Build with Python + pptxgenjs

**Option A: python-pptx (lightweight, pure Python)**

```python
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

prs = Presentation()
prs.slide_width = Inches(10)
prs.slide_height = Inches(7.5)

# Define colors
NAVY = RGBColor(30, 39, 97)
ICE_BLUE = RGBColor(202, 220, 252)
WHITE = RGBColor(255, 255, 255)
DARK_GRAY = RGBColor(64, 64, 64)

# Title slide
slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank layout
background = slide.background
fill = background.fill
fill.solid()
fill.fore_color.rgb = NAVY

# Add title
title_box = slide.shapes.add_textbox(Inches(0.5), Inches(2.5), Inches(9), Inches(2))
title_frame = title_box.text_frame
title_frame.text = "Presentation Title"
title_frame.paragraphs[0].font.size = Pt(54)
title_frame.paragraphs[0].font.bold = True
title_frame.paragraphs[0].font.color.rgb = WHITE

prs.save("output.pptx")
```

**Option B: pptxgenjs (Node.js, advanced styling)**

```bash
npm install -g pptxgenjs
```

```javascript
const PptxGenJS = require("pptxgenjs");
const prs = new PptxGenJS();

// Define master layout
const MASTER = {
  navy: "1E2761",
  ice: "CADCFC",
  white: "FFFFFF",
  font: "Calibri"
};

// Add slide
let slide = prs.addSlide();
slide.background = { color: MASTER.navy };
slide.addText("Presentation Title", {
  x: 0.5, y: 2.5, w: 9, h: 2,
  fontSize: 54, bold: true, color: MASTER.white, align: "center"
});

prs.writeFile({ fileName: "output.pptx" });
```

### Step 6: Content Organization Example

For a typical report:

1. **Slide 1**: Title (company/report name, date, author)
2. **Slide 2**: Table of Contents
3. **Slide 3**: Executive Summary (2-3 key bullets)
4. **Slides 4-6**: Main sections (one section per slide or multi-slide per section)
5. **Slide 7**: Key Findings (callout numbers)
6. **Slide 8**: Recommendations
7. **Slide 9**: Next Steps / Close

### Step 7: Add Visual Elements

**Images:**
```python
slide.shapes.add_picture("image.png", Inches(0.5), Inches(1), width=Inches(4))
```

**Tables:**
```python
rows, cols = len(data) + 1, len(data[0])
left = Inches(0.5)
top = Inches(1)
width = Inches(9)
height = Inches(4)
table_shape = slide.shapes.add_table(rows, cols, left, top, width, height).table

# Fill header
for col_idx, header in enumerate(headers):
    cell = table_shape.cell(0, col_idx)
    cell.text = header
    cell.fill.solid()
    cell.fill.fore_color.rgb = NAVY
    # Format text...

# Fill data
for row_idx, row_data in enumerate(data, start=1):
    for col_idx, value in enumerate(row_data):
        cell = table_shape.cell(row_idx, col_idx)
        cell.text = str(value)
```

**Shapes & icons:**
```python
# Add colored circle for emphasis
circle = slide.shapes.add_shape(
    MSO_SHAPE.OVAL,
    Inches(0.5), Inches(1),
    Inches(0.6), Inches(0.6)
)
circle.fill.solid()
circle.fill.fore_color.rgb = ICE_BLUE
```

### Step 8: Verification

After building, **always verify**:

1. **Text bounds**: Does all text fit in its box? (Most common error)
2. **Alignment**: Are elements evenly spaced?
3. **Overflow**: Check content doesn't bleed past margins
4. **Content completeness**: Extract all text and scan for missing data

```bash
extract-text output.pptx
# Review output for missing slides, truncated text, or errors
```

## Example: Quick Start for Data from Files

```python
#!/usr/bin/env python3
import os
import glob
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

def scan_directory(path):
    """Identify all files in directory"""
    files = {}
    for ext in ['*.pdf', '*.docx', '*.txt', '*.csv', '*.png', '*.jpg']:
        files[ext] = glob.glob(os.path.join(path, ext))
    return files

def create_presentation(directory_path, output_file="presentation.pptx"):
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)
    
    files = scan_directory(directory_path)
    
    # Title slide
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    # ... add content ...
    
    # For each file, extract and create slides
    for file_path in files['*.txt']:
        with open(file_path) as f:
            content = f.read()
        # Create slide from content
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        # ... format and add content ...
    
    prs.save(output_file)
    print(f"✓ Presentation saved: {output_file}")

if __name__ == "__main__":
    create_presentation("/path/to/files")
```

## Design Principles (Must Follow)

✓ **Pick ONE bold color palette** — all slides consistent
✓ **Visual element on every slide** — image, chart, icon, or shape
✓ **Vary layouts** — don't repeat the same structure
✓ **Left-align body text** — center only titles and numbers
✓ **Large size contrast** — 36pt+ titles, 14-16pt body
✓ **Breathing room** — margins 0.5"+, spacing 0.3-0.5"
✓ **Professional fonts** — Calibri + Georgia, not Arial

✗ **No plain text slides** — every slide needs visual content
✗ **No low contrast** — dark text on dark, light on light
✗ **No underlines under titles** — use whitespace instead
✗ **No colored bars/ribbons** — unless user explicitly requests

## Tools & Dependencies

Install before using:

```bash
pip install python-pptx python-docx pypdf pillow pandas openpyxl
npm install -g pptxgenjs  # Optional, for advanced styling
```

## References

- [python-pptx documentation](https://python-pptx.readthedocs.io/)
- [pptxgenjs documentation](https://gitbrent.github.io/PptxGenJS/)
- [PPTX Design Skill](../pptx/SKILL.md) — Color palettes and layout ideas
