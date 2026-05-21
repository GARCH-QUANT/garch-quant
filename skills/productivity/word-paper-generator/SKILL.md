---
name: word-paper-generator
description: 生成规范格式的中文学术论文Word文档（A股研究报告/金融建模等）
category: productivity
---
# Word论文生成器

生成规范格式的中文学术论文Word文档，自动适配黑体/宋体/Times New Roman混排。

## 使用场景
- 中文学术论文（A股研究报告、金融建模论文等）
- 需要复杂样式：多级标题、公式、表格、参考文献
- 定期生成结构化报告并推送Telegram

## 依赖
```
python-docx
```
安装：`/home/agentuser/.hermes/hermes-agent/venv/bin/python3 -m pip install python-docx -q`

## 核心样式定义

```python
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn

doc = Document()

# === 基础正文字体 ===
doc.styles['Normal'].font.name = '宋体'
doc.styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
doc.styles['Normal'].font.size = Pt(12)
doc.styles['Normal'].paragraph_format.line_spacing = 1.5
doc.styles['Normal'].paragraph_format.first_line_indent = Inches(0.5)

# === 标题样式（黑体）===
title_style = doc.styles.add_style('Title Style', WD_STYLE_TYPE.PARAGRAPH)
title_style.font.name = '黑体'
title_style._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
title_style.font.size = Pt(18)
title_style.font.bold = True
title_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
title_style.paragraph_format.space_after = Pt(12)
title_style.paragraph_format.first_line_indent = Inches(0)

# === 副标题样式（宋体）===
subtitle_style = doc.styles.add_style('Subtitle Style', WD_STYLE_TYPE.PARAGRAPH)
subtitle_style.font.name = '宋体'
subtitle_style._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
subtitle_style.font.size = Pt(14)
subtitle_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
subtitle_style.paragraph_format.space_after = Pt(18)
subtitle_style.paragraph_format.first_line_indent = Inches(0)

# === 一级标题（黑体16pt）===
h1_style = doc.styles.add_style('Heading 1 Style', WD_STYLE_TYPE.PARAGRAPH)
h1_style.font.name = '黑体'
h1_style._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
h1_style.font.size = Pt(16)
h1_style.font.bold = True
h1_style.paragraph_format.space_before = Pt(18)
h1_style.paragraph_format.space_after = Pt(12)
h1_style.paragraph_format.first_line_indent = Inches(0)

# === 二级标题（黑体14pt）===
h2_style = doc.styles.add_style('Heading 2 Style', WD_STYLE_TYPE.PARAGRAPH)
h2_style.font.name = '黑体'
h2_style._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
h2_style.font.size = Pt(14)
h2_style.font.bold = True
h2_style.paragraph_format.space_before = Pt(12)
h2_style.paragraph_format.space_after = Pt(6)
h2_style.paragraph_format.first_line_indent = Inches(0)

# === 摘要样式 ===
abstract_style = doc.styles.add_style('Abstract Style', WD_STYLE_TYPE.PARAGRAPH)
abstract_style.font.name = '宋体'
abstract_style._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
abstract_style.font.size = Pt(12)
abstract_style.paragraph_format.line_spacing = 1.5
abstract_style.paragraph_format.first_line_indent = Inches(0)

# === 参考文献样式（悬挂缩进）===
ref_style = doc.styles.add_style('Reference Style', WD_STYLE_TYPE.PARAGRAPH)
ref_style.font.name = '宋体'
ref_style._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
ref_style.font.size = Pt(10.5)
ref_style.paragraph_format.line_spacing = 1.25
ref_style.paragraph_format.first_line_indent = Inches(0)
ref_style.paragraph_format.hanging_indent = Inches(0.25)

# === 表格标题样式 ===
table_title_style = doc.styles.add_style('Table Title Style', WD_STYLE_TYPE.PARAGRAPH)
table_title_style.font.name = '宋体'
table_title_style._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
table_title_style.font.size = Pt(11)
table_title_style.font.bold = True
table_title_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
table_title_style.paragraph_format.space_before = Pt(6)
table_title_style.paragraph_format.space_after = Pt(6)
table_title_style.paragraph_format.first_line_indent = Inches(0)

# === 公式样式（居中Times New Roman）===
formula_style = doc.styles.add_style('Formula Style', WD_STYLE_TYPE.PARAGRAPH)
formula_style.font.name = 'Times New Roman'
formula_style.font.size = Pt(12)
formula_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
formula_style.paragraph_format.space_before = Pt(6)
formula_style.paragraph_format.space_after = Pt(6)
formula_style.paragraph_format.first_line_indent = Inches(0)

# === 列表样式（左缩进）===
list_style = doc.styles.add_style('List Style', WD_STYLE_TYPE.PARAGRAPH)
list_style.font.name = '宋体'
list_style._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
list_style.font.size = Pt(12)
list_style.paragraph_format.line_spacing = 1.5
list_style.paragraph_format.first_line_indent = Inches(0)
list_style.paragraph_format.left_indent = Inches(0.5)
```

## 标准论文结构模板

```python
# 标题
doc.add_paragraph('论文标题', style='Title Style')
doc.add_paragraph('副标题', style='Subtitle Style')

# 摘要
doc.add_paragraph('摘要', style='Heading 1 Style')
p = doc.add_paragraph(style='Abstract Style')
p.add_run('摘要正文...')
p.add_run('关键词：').bold = True
p.add_run('关键词内容')

# 1 一级章节
doc.add_paragraph('1 章节标题', style='Heading 1 Style')
# 1.1 二级章节
doc.add_paragraph('1.1 小节标题', style='Heading 2 Style')
doc.add_paragraph('正文内容...')

# 公式（居中）
doc.add_paragraph('r_t = ln(P_t/P_{t-1}) × 100  (1)', style='Formula Style')

# 列表
doc.add_paragraph('1. 列表项1', style='List Style')

# 表格
table = doc.add_table(rows=3, cols=3)
table.style = 'Table Grid'
# 设置表头...
doc.add_paragraph('表1 表格标题', style='Table Title Style')

# 参考文献
doc.add_paragraph('参考文献', style='Heading 1 Style')
for ref in refs:
    doc.add_paragraph(ref, style='Reference Style')

# 保存
doc.save('/tmp/output.docx')
```

## 推送到Telegram

```python
# 生成后用send_message推送
send_message(
    action='send',
    target='telegram:2064623932',  # 或用户指定的chat_id
    message='📄 论文标题\n\nMEDIA:/tmp/output.docx'
)
```

## 注意事项
- **中文字体**：只用黑体（标题）、宋体（正文）、Times New Roman（公式/英文）
- **公式下标**：Unicode下标字符（ᵢₜ²₋₁）或直接用 `_t-1` 风格
- **希腊字母**：Unicode字符（σᵢ,ₜ²）或直接写 `sigma_i,t^2`
- **pip安装**：必须用 venv 的 python：`/home/agentuser/.hermes/hermes-agent/venv/bin/python3`
- **脚本保存**：大型脚本写到 `/tmp/gen_paper.py` 再执行，避免terminal转义问题
