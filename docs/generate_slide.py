"""Generate a single summary slide for tiny-trtllm."""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

prs = Presentation()
prs.slide_width = Inches(16)
prs.slide_height = Inches(9)

slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank

# Background
bg = slide.background
fill = bg.fill
fill.solid()
fill.fore_color.rgb = RGBColor(0x0D, 0x11, 0x17)

# --- Title ---
txBox = slide.shapes.add_textbox(Inches(0.8), Inches(0.3), Inches(14), Inches(1.0))
tf = txBox.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
run1 = p.add_run()
run1.text = "tiny-"
run1.font.size = Pt(48)
run1.font.bold = True
run1.font.color.rgb = RGBColor(0xE6, 0xED, 0xF3)
run2 = p.add_run()
run2.text = "trtllm"
run2.font.size = Pt(48)
run2.font.bold = True
run2.font.color.rgb = RGBColor(0x58, 0xA6, 0xFF)

# Subtitle
txBox2 = slide.shapes.add_textbox(Inches(0.8), Inches(1.2), Inches(14), Inches(0.5))
tf2 = txBox2.text_frame
p2 = tf2.paragraphs[0]
run_sub = p2.add_run()
run_sub.text = "Minimal reimplementation of TensorRT-LLM's PyTorch backend  |  ~4K LOC  |  262 tests  |  86% coverage  |  0 C++  |  9x batched speedup"
run_sub.font.size = Pt(16)
run_sub.font.color.rgb = RGBColor(0x8B, 0x94, 0x9E)

# --- Divider line ---
shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.85), Inches(14.4), Pt(2))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(0x30, 0x36, 0x3D)
shape.line.fill.background()

# --- Left column: 16 Differentiators ---
left_x = Inches(0.8)

txBox3 = slide.shapes.add_textbox(left_x, Inches(2.0), Inches(7), Inches(0.4))
tf3 = txBox3.text_frame
p3 = tf3.paragraphs[0]
run_h = p3.add_run()
run_h.text = "16 Core Architectural Differentiators"
run_h.font.size = Pt(20)
run_h.font.bold = True
run_h.font.color.rgb = RGBColor(0x7E, 0xE7, 0x87)

features = [
    ("Engine", [
        "Two-tier scheduling (Capacity + MicroBatch)",
        "GUARANTEED_NO_EVICT / MAX_UTILIZATION policies",
        "EQUAL_PROGRESS / FCFS chunking policies",
        "Iteration-based executor loop",
        "Phase-segmented ScheduledRequests (4 lists)",
        "Async two-phase sampling + grouped strategies",
        "Overlap executor (CPU/GPU ping-pong)",
        "Await-responses (threading.Condition)",
        "Batched KV-cached inference (continuous batching)",
    ]),
    ("Layers", [
        "Plan/execute attention (plan \u2192 run)",
        "Hybrid attention backend dispatcher",
        "TRT-LLM thop.attention() kernel (via pip)",
        "MoE (gate \u2192 topk \u2192 fused dispatch)",
    ]),
    ("Infra", [
        "Pydantic config hierarchy",
        "ResourceManager lifecycle (prepare \u2192 update \u2192 free)",
        "Multi-GPU TP with NCCL",
    ]),
]

y = Inches(2.5)
for category, items in features:
    txBox_cat = slide.shapes.add_textbox(left_x, y, Inches(7), Inches(0.25))
    tf_cat = txBox_cat.text_frame
    p_cat = tf_cat.paragraphs[0]
    run_cat = p_cat.add_run()
    run_cat.text = category
    run_cat.font.size = Pt(13)
    run_cat.font.bold = True
    run_cat.font.color.rgb = RGBColor(0x58, 0xA6, 0xFF)
    y += Inches(0.25)

    for item in items:
        txBox_item = slide.shapes.add_textbox(left_x + Inches(0.15), y, Inches(6.8), Inches(0.22))
        tf_item = txBox_item.text_frame
        p_item = tf_item.paragraphs[0]
        run_item = p_item.add_run()
        run_item.text = f"  {item}"
        run_item.font.size = Pt(11)
        run_item.font.color.rgb = RGBColor(0x8B, 0x94, 0x9E)
        y += Inches(0.22)
    y += Inches(0.08)

# --- Right column: Architecture + Benchmark + Stats ---
right_x = Inches(8.5)

# Architecture header
txBox4 = slide.shapes.add_textbox(right_x, Inches(2.0), Inches(7), Inches(0.4))
tf4 = txBox4.text_frame
p4 = tf4.paragraphs[0]
run_h2 = p4.add_run()
run_h2.text = "Architecture"
run_h2.font.size = Pt(20)
run_h2.font.bold = True
run_h2.font.color.rgb = RGBColor(0xD2, 0xA8, 0xFF)

arch_text = """tinytrtllm/
  config.py          Pydantic config
  llm.py             Top-level API
  engine/
    scheduler.py     Two-tier scheduling
    executor.py      PyExecutor + Overlap
    hf_adapter.py    Batched KV-cache adapter
    sampler.py       Async two-phase
    block_manager.py Paged KV + prefix cache
    model_engine.py  Forward + CUDA graphs
  layers/
    attention*.py    5 backends + hybrid
    linear.py        TP-aware parallel
    moe.py           Gate + TopK + fused
  models/
    llama.py         LlamaForCausalLM
    qwen3.py         Qwen3ForCausalLM
    qwen3_moe.py     Qwen3MoEForCausalLM
  serve/server.py    FastAPI OpenAI-compat"""

txBox_arch = slide.shapes.add_textbox(right_x, Inches(2.5), Inches(7), Inches(3.0))
tf_arch = txBox_arch.text_frame
tf_arch.word_wrap = True
for line in arch_text.strip().split("\n"):
    p_line = tf_arch.add_paragraph()
    run_line = p_line.add_run()
    run_line.text = line
    run_line.font.size = Pt(10)
    run_line.font.name = "Courier New"
    run_line.font.color.rgb = RGBColor(0x8B, 0x94, 0x9E)

# --- Benchmark results ---
bench_y = Inches(5.6)

txBox_bh = slide.shapes.add_textbox(right_x, bench_y, Inches(7), Inches(0.4))
tf_bh = txBox_bh.text_frame
p_bh = tf_bh.paragraphs[0]
run_bh = p_bh.add_run()
run_bh.text = "Benchmark: L40S, Qwen3-0.6B, greedy, 32 tok"
run_bh.font.size = Pt(16)
run_bh.font.bold = True
run_bh.font.color.rgb = RGBColor(0xFF, 0xA6, 0x57)

bench_data = [
    ("1 req", "58", "59", "1.0x"),
    ("4 req", "61", "212", "3.5x"),
    ("8 req", "59", "363", "6.1x"),
    ("16 req", "59", "526", "9.0x"),
]

# Table header
header_y = bench_y + Inches(0.4)
cols = [
    (right_x, "Requests"),
    (right_x + Inches(1.5), "HF Serial"),
    (right_x + Inches(3.0), "Batched"),
    (right_x + Inches(4.5), "Speedup"),
]
for cx, label in cols:
    tb = slide.shapes.add_textbox(cx, header_y, Inches(1.4), Inches(0.22))
    tf_t = tb.text_frame
    p_t = tf_t.paragraphs[0]
    r_t = p_t.add_run()
    r_t.text = label
    r_t.font.size = Pt(10)
    r_t.font.bold = True
    r_t.font.color.rgb = RGBColor(0xE6, 0xED, 0xF3)

for row_i, (reqs, hf, batched, speedup) in enumerate(bench_data):
    row_y = header_y + Inches(0.22) + Inches(row_i * 0.22)
    vals = [reqs, f"{hf} tok/s", f"{batched} tok/s", speedup]
    for col_i, val in enumerate(vals):
        cx = cols[col_i][0]
        tb = slide.shapes.add_textbox(cx, row_y, Inches(1.4), Inches(0.22))
        tf_t = tb.text_frame
        p_t = tf_t.paragraphs[0]
        r_t = p_t.add_run()
        r_t.text = val
        r_t.font.size = Pt(10)
        if col_i == 2 and row_i > 0:  # highlight batched numbers
            r_t.font.bold = True
            r_t.font.color.rgb = RGBColor(0x7E, 0xE7, 0x87)
        elif col_i == 3 and row_i > 0:
            r_t.font.bold = True
            r_t.font.color.rgb = RGBColor(0xFF, 0xA6, 0x57)
        else:
            r_t.font.color.rgb = RGBColor(0x8B, 0x94, 0x9E)

# Correctness note
corr_y = header_y + Inches(1.15)
tb_c = slide.shapes.add_textbox(right_x, corr_y, Inches(7), Inches(0.25))
tf_c = tb_c.text_frame
p_c = tf_c.paragraphs[0]
r_c = p_c.add_run()
r_c.text = "Correctness: token-for-token IDENTICAL to HF model.generate()"
r_c.font.size = Pt(11)
r_c.font.bold = True
r_c.font.color.rgb = RGBColor(0x7E, 0xE7, 0x87)

# Stats boxes
stats = [
    ("4,015", "Production LOC", "7EE787"),
    ("3,506", "Test LOC", "58A6FF"),
    ("86%", "Coverage", "D2A8FF"),
    ("262", "Tests Passing", "FFA657"),
    ("9x", "Batch Speedup", "FF6B6B"),
]

stat_y = Inches(7.5)
for i, (val, label, color) in enumerate(stats):
    sx = right_x + Inches(i * 1.4)
    # Value
    txBox_v = slide.shapes.add_textbox(sx, stat_y, Inches(1.3), Inches(0.4))
    tf_v = txBox_v.text_frame
    p_v = tf_v.paragraphs[0]
    p_v.alignment = PP_ALIGN.CENTER
    run_v = p_v.add_run()
    run_v.text = val
    run_v.font.size = Pt(24)
    run_v.font.bold = True
    run_v.font.color.rgb = RGBColor(int(color[:2], 16), int(color[2:4], 16), int(color[4:], 16))
    # Label
    txBox_l = slide.shapes.add_textbox(sx, stat_y + Inches(0.38), Inches(1.3), Inches(0.25))
    tf_l = txBox_l.text_frame
    p_l = tf_l.paragraphs[0]
    p_l.alignment = PP_ALIGN.CENTER
    run_l = p_l.add_run()
    run_l.text = label
    run_l.font.size = Pt(9)
    run_l.font.color.rgb = RGBColor(0x8B, 0x94, 0x9E)

# Footer
txBox_f = slide.shapes.add_textbox(Inches(0.8), Inches(8.5), Inches(14), Inches(0.3))
tf_f = txBox_f.text_frame
p_f = tf_f.paragraphs[0]
run_f = p_f.add_run()
run_f.text = "github.com/venkywonka/tiny-trtllm  |  branch: dev/poc-1  |  Python + Triton only, zero C++"
run_f.font.size = Pt(12)
run_f.font.color.rgb = RGBColor(0x58, 0xA6, 0xFF)

prs.save("/home/ubuntu/tiny-trtllm/docs/tiny-trtllm-summary.pptx")
print("Saved: docs/tiny-trtllm-summary.pptx")
