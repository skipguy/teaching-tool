"""
app.py — Flask backend for the Teaching Preparation Tool.

Two-phase workflow:
  Phase 1 (fast, no API): cut questions → status "preview"
                           user reviews / splits / deletes
  Phase 2 (API):          OCR each image → DeepSeek → PPT → status "done"

Endpoints:
  POST /upload                   — upload files; returns file IDs
  POST /process                  — start phase 1 (cutting); returns job_id
  GET  /status/<job_id>          — poll status & progress
  GET  /preview/<job_id>         — thumbnails (available after phase 1)
  GET  /full_image/<job_id>/<i>  — full-res image for split modal
  POST /split/<job_id>/<i>       — two-line split: discard strip between lines
  POST /delete_question/<job_id>/<i> — remove a question image
  POST /analyze/<job_id>         — start phase 2 (OCR + AI + PPT)
  GET  /results/<job_id>         — AI analysis JSON
  GET  /download/<job_id>/<type> — ppt / markdown / json
  GET  /                         — frontend
"""

from __future__ import annotations

import os
# Must be set before numpy / onnxruntime are imported.
# On Windows, numpy and onnxruntime each ship their own OpenMP DLL
# (iomp5md.dll / libgomp.dll).  When both are loaded in the same process
# the second one fails with "DLL initialization routine failed".
# KMP_DUPLICATE_LIB_OK=TRUE tells Intel's OpenMP runtime to tolerate
# duplicate libraries instead of aborting.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import base64
import io
import json
import logging
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any

import fitz
from flask import Flask, jsonify, request, send_file, render_template, abort
from PIL import Image

from core.pdf_cutter import cut_questions, QuestionSlice, ocr_image_to_text
from core.ppt_gen import generate_ppt
from core.deepseek_client import analyze_questions
from core.pdf_cutter import _get_ocr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Pre-load OCR engine in the main thread.
# On Windows, onnxruntime DLLs must be initialised before any background
# threads are spawned; loading them lazily inside a daemon thread causes
# "DLL initialization routine failed" (0x8007045A).
try:
    _get_ocr()
    logger.info("OCR engine pre-loaded successfully.")
except Exception as _ocr_init_err:
    logger.warning("OCR engine pre-load failed (will retry on first use): %s", _ocr_init_err)

_files: dict[str, Path] = {}
_jobs:  dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/upload", methods=["POST"])
def upload():
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400
    uploaded = []
    for f in request.files.getlist("files"):
        if not f.filename:
            continue
        file_id = uuid.uuid4().hex
        safe_name = _safe_filename(f.filename)
        dest = UPLOAD_DIR / f"{file_id}_{safe_name}"
        f.save(str(dest))
        _files[file_id] = dest
        uploaded.append({"id": file_id, "name": f.filename, "size": dest.stat().st_size})
        logger.info("Uploaded %s → %s", f.filename, dest)
    return jsonify({"files": uploaded})


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Cut questions
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/process", methods=["POST"])
def process():
    data          = request.get_json(force=True)
    homework_id   = data.get("homework_id")
    reference_ids = data.get("reference_ids", [])
    supplement_ids= data.get("supplement_ids", [])
    api_key       = data.get("api_key", "").strip()
    sort_order    = data.get("sort_order", "number")
    mode          = data.get("mode", "homework")   # "homework" | "exam"

    if not homework_id or homework_id not in _files:
        return jsonify({"error": "Invalid homework file ID"}), 400
    if not api_key:
        return jsonify({"error": "API key is required"}), 400

    hw_path_obj = _files[homework_id]
    hw_name = _extract_pdf_title(str(hw_path_obj)) or hw_path_obj.name.split("_", 1)[-1].rsplit(".", 1)[0]
    ref_names = [
        _files[rid].name.split("_", 1)[-1].rsplit(".", 1)[0]
        for rid in reference_ids if rid in _files
    ]

    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "status":   "queued",
            "progress": 0,
            "message":  "等待处理…",
            # stored for phase 2
            "_api_key":        api_key,
            "_sort_order":     sort_order,
            "_mode":           mode,
            "_homework_id":    homework_id,
            "_reference_ids":  reference_ids,
            "_supplement_ids": supplement_ids,
            "_hw_name":        hw_name,
            "_ref_names":      ref_names,
            # results
            "results":         None,
            "ppt_bytes":       None,
            "markdown":        None,
            "question_images": [],
            "question_images_full": [],
            "slices_meta":     [],
            "error":           None,
        }

    threading.Thread(
        target=_run_cut, args=(job_id,), daemon=True
    ).start()

    return jsonify({"job_id": job_id})


def _run_cut(job_id: str) -> None:
    try:
        _set_status(job_id, "cutting")
        _set_progress(job_id, 5, "正在切割题目…")

        hw_path = str(_files[_jobs[job_id]["_homework_id"]])
        mode    = _jobs[job_id].get("_mode", "homework")
        slices: list[QuestionSlice] = cut_questions(hw_path, mode=mode)
        if not slices:
            raise RuntimeError("未检测到题目，请确认PDF含文字图层或可被OCR识别")

        thumbnails = []
        full_images = []
        for s in slices:
            thumb = _make_thumbnail(s.image_bytes, max_w=300, max_h=200)
            thumbnails.append(base64.b64encode(thumb).decode())
            full_images.append(base64.b64encode(s.image_bytes).decode())

        mode = _jobs[job_id].get("_mode", "homework")
        label = "题" if mode == "homework" else "道题"
        with _jobs_lock:
            _jobs[job_id].update({
                "status":   "preview",
                "progress": 100,
                "message":  f"切割完成，共 {len(slices)} {label}，请检查后点击「开始AI分析」",
                "question_images":      thumbnails,
                "question_images_full": full_images,
                "slices_meta": [
                    {"number": s.question_number, "page_range": list(s.page_range)}
                    for s in slices
                ],
            })
        logger.info("[%s] Phase 1 done: %d questions", job_id, len(slices))

    except Exception as exc:
        _set_error(job_id, str(exc), traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: OCR + AI + PPT  (triggered by user after reviewing)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/analyze/<job_id>", methods=["POST"])
def analyze(job_id: str):
    job = _get_job(job_id)
    if job["status"] not in ("preview", "done"):
        return jsonify({"error": "正在处理中，请等待当前任务完成"}), 400

    data = request.get_json(force=True, silent=True) or {}
    sort_order = job.get("_sort_order", "number")
    # "topic" sort always needs JSON analysis (topic groups drive the ordering)
    analyze_knowledge = bool(data.get("analyze_knowledge", True)) or (sort_order == "topic")
    generate_board    = bool(data.get("generate_board", True))

    with _jobs_lock:
        job["_analyze_knowledge"] = analyze_knowledge
        job["_generate_board"]    = generate_board

    threading.Thread(
        target=_run_analyze, args=(job_id,), daemon=True
    ).start()
    return jsonify({"ok": True})


def _run_analyze(job_id: str) -> None:
    try:
        _set_status(job_id, "analyzing")
        job = _jobs[job_id]

        api_key            = job["_api_key"]
        sort_order         = job["_sort_order"]
        mode               = job.get("_mode", "homework")
        reference_ids      = job["_reference_ids"]
        supplement_ids     = job["_supplement_ids"]
        hw_name            = job["_hw_name"]
        ref_names          = job["_ref_names"]
        analyze_knowledge  = job.get("_analyze_knowledge", True)
        generate_board     = job.get("_generate_board", True)

        full_images = job["question_images_full"]
        meta_list   = job["slices_meta"]
        q_numbers   = list(range(1, len(meta_list) + 1))

        need_ai = analyze_knowledge or generate_board

        # ── Step 1: Extract reference texts (only if calling AI) ──────────────
        ref_texts: list[str] = []
        if need_ai:
            _set_progress(job_id, 5, "正在提取课件内容…")
            for rid in reference_ids:
                if rid in _files:
                    ref_texts.append(_extract_any_text(_files[rid]))
            for sid in supplement_ids:
                if sid in _files:
                    ref_texts.append(_extract_any_text(_files[sid]))

        # ── Step 2: OCR each question image (only if calling AI) ──────────────
        q_texts: list[str] = []
        if need_ai:
            _set_progress(job_id, 15, f"正在识别 {len(full_images)} 道题目文字…")
            for i, b64 in enumerate(full_images):
                img_bytes = base64.b64decode(b64)
                text = ocr_image_to_text(img_bytes)
                q_texts.append(text)
                if (i + 1) % 5 == 0 or i + 1 == len(full_images):
                    _set_progress(job_id, 15 + int(25 * (i + 1) / len(full_images)),
                                  f"已识别 {i+1}/{len(full_images)} 题…")

        # ── Step 3: DeepSeek analysis ─────────────────────────────────────────
        analysis: dict = {"questions": [], "topic_groups": [], "board_writing": ""}
        if need_ai:
            ai_calls = []
            if generate_board:    ai_calls.append("板书")
            if analyze_knowledge: ai_calls.append("知识点分析")
            call_desc = "、".join(ai_calls)
            _set_progress(job_id, 40, f"正在进行AI分析（{call_desc}）…")
            analysis = analyze_questions(
                api_key=api_key,
                question_texts=q_texts,
                question_numbers=q_numbers,
                homework_name=hw_name,
                reference_names=ref_names if ref_names else None,
                reference_texts=ref_texts if ref_texts else None,
                generate_board=generate_board,
                analyze_json=analyze_knowledge,
                mode=mode,
            )
        else:
            _set_progress(job_id, 40, "跳过AI分析，直接生成PPT…")

        _set_progress(job_id, 75, "正在生成PPT…")

        # ── Step 4: Sort & PPT ────────────────────────────────────────────────
        topic_groups = analysis.get("topic_groups", [])
        num_to_idx   = {n: i for i, n in enumerate(q_numbers)}

        if sort_order == "topic" and topic_groups:
            # Reorder slides by topic group
            seen_idx: list[int] = []
            seen_set: set[int]  = set()
            for group in topic_groups:
                for qnum in group.get("numbers", []):
                    idx = num_to_idx.get(qnum)
                    if idx is not None and idx not in seen_set:
                        seen_idx.append(idx)
                        seen_set.add(idx)
            for i in range(len(q_numbers)):
                if i not in seen_set:
                    seen_idx.append(i)
            sorted_indices = seen_idx

            # Red label = topic group name
            num_to_label: dict[int, str] = {}
            for g_i, group in enumerate(topic_groups, 1):
                label = f"{g_i}. {group.get('topic', '')}"
                for qnum in group.get("numbers", []):
                    num_to_label.setdefault(qnum, label)
            topic_labels: list[str] | None = [
                num_to_label.get(q_numbers[i], "") for i in sorted_indices
            ]

        else:
            sorted_indices = list(range(len(q_numbers)))

            if analyze_knowledge and analysis.get("questions"):
                # Red label = primary_topic of each question (knowledge point)
                num_to_topic: dict[int, str] = {
                    q["number"]: q.get("primary_topic", "")
                    for q in analysis["questions"]
                    if q.get("primary_topic")
                }
                topic_labels = [
                    num_to_topic.get(q_numbers[i], "") for i in sorted_indices
                ]
            else:
                topic_labels = None

        ordered_images  = [base64.b64decode(full_images[i]) for i in sorted_indices]
        ordered_numbers = [q_numbers[i] for i in sorted_indices]
        ppt_bytes = generate_ppt(ordered_images, ordered_numbers, topic_labels=topic_labels)

        _set_progress(job_id, 95, "整理结果…")

        with _jobs_lock:
            _jobs[job_id].update({
                "status":   "done",
                "progress": 100,
                "message":  "处理完成！",
                "results":  analysis,
                "ppt_bytes": ppt_bytes,
                "markdown": analysis.get("board_writing", ""),
            })
        logger.info("[%s] Phase 2 done (analyze_knowledge=%s, generate_board=%s)",
                    job_id, analyze_knowledge, generate_board)

    except Exception as exc:
        _set_error(job_id, str(exc), traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# Status / Preview / Results / Download
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/status/<job_id>")
def status(job_id: str):
    job = _get_job(job_id)
    return jsonify({
        "status":   job["status"],
        "progress": job["progress"],
        "message":  job["message"],
        "error":    job.get("error"),
    })


@app.route("/preview/<job_id>")
def preview(job_id: str):
    job = _get_job(job_id)
    if job["status"] not in ("preview", "analyzing", "done"):
        return jsonify({"error": "Not ready"}), 400
    return jsonify({
        "images": job["question_images"],
        "meta":   job.get("slices_meta", []),
    })


@app.route("/full_image/<job_id>/<int:index>")
def full_image(job_id: str, index: int):
    job = _get_job(job_id)
    full_list = job.get("question_images_full", [])
    if index < 0 or index >= len(full_list):
        abort(404)
    return jsonify({"image": full_list[index]})


@app.route("/results/<job_id>")
def results(job_id: str):
    job = _get_job(job_id)
    if job["status"] != "done":
        return jsonify({"error": "Job not finished"}), 400
    return jsonify(job["results"])


@app.route("/download/<job_id>/<file_type>")
def download(job_id: str, file_type: str):
    job = _get_job(job_id)
    if job["status"] != "done":
        abort(404)

    hw_name = job.get("_hw_name", "作业").strip() or "作业"

    if file_type == "ppt":
        buf = io.BytesIO(job["ppt_bytes"])
        return send_file(buf, as_attachment=True,
                         download_name=f"{hw_name} 作业讲评.pptx",
                         mimetype="application/vnd.openxmlformats-officedocument.presentationml.presentation")
    elif file_type == "markdown":
        buf = io.BytesIO(job["markdown"].encode("utf-8"))
        return send_file(buf, as_attachment=True,
                         download_name=f"{hw_name} 笔记.md",
                         mimetype="text/markdown; charset=utf-8")
    elif file_type == "json":
        buf = io.BytesIO(json.dumps(job["results"], ensure_ascii=False, indent=2).encode("utf-8"))
        return send_file(buf, as_attachment=True,
                         download_name=f"{hw_name} 分析.json",
                         mimetype="application/json")
    else:
        abort(404)


# ─────────────────────────────────────────────────────────────────────────────
# Image editing: split (two-line) and delete
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/split/<job_id>/<int:index>", methods=["POST"])
def split_question(job_id: str, index: int):
    """
    Two-line split: top image = above line1, bottom image = below line2.
    The strip between line1 and line2 is discarded.
    Body: { "y_ratio_top": 0.4, "y_ratio_bot": 0.5, "new_number": 5 }
    """
    data        = request.get_json(force=True)
    y_ratio_top = float(data.get("y_ratio_top", 0.45))
    y_ratio_bot = float(data.get("y_ratio_bot", 0.55))
    new_num     = int(data.get("new_number", 0))

    job = _get_job(job_id)
    full_list = job.get("question_images_full", [])
    if index < 0 or index >= len(full_list):
        return jsonify({"error": "Index out of range"}), 400

    img_bytes = base64.b64decode(full_list[index])
    img = Image.open(io.BytesIO(img_bytes))
    w, h = img.size

    cut_top = max(0, min(int(h * y_ratio_top), h))
    cut_bot = max(0, min(int(h * y_ratio_bot), h))

    if cut_bot >= cut_top:
        # Normal mode: discard strip between cut_top and cut_bot
        cut_bot = max(cut_top + 1, cut_bot)
        top_img = img.crop((0, 0,       w, cut_top))
        bot_img = img.crop((0, cut_bot, w, h))
    else:
        # Overlap mode: blue is above red
        # Top question ends at cut_top (red line)
        # Bottom question starts at cut_bot (blue line, which is above red)
        # The strip between cut_bot and cut_top is assigned to the bottom question
        top_img = img.crop((0, 0,       w, cut_top))
        bot_img = img.crop((0, cut_bot, w, h))

    def b64png(pil_img):
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    top_b64 = b64png(top_img)
    bot_b64 = b64png(bot_img)
    top_thumb = base64.b64encode(_make_thumbnail(base64.b64decode(top_b64), 300, 200)).decode()
    bot_thumb = base64.b64encode(_make_thumbnail(base64.b64decode(bot_b64), 300, 200)).decode()

    meta_list = job.get("slices_meta", [])
    orig_num  = meta_list[index]["number"] if index < len(meta_list) else index + 1

    with _jobs_lock:
        job["question_images_full"].pop(index)
        job["question_images_full"].insert(index, bot_b64)
        job["question_images_full"].insert(index, top_b64)
        job["question_images"].pop(index)
        job["question_images"].insert(index, bot_thumb)
        job["question_images"].insert(index, top_thumb)
        if index < len(job["slices_meta"]):
            orig_meta = job["slices_meta"].pop(index)
            job["slices_meta"].insert(index, {"number": new_num,  "page_range": orig_meta["page_range"]})
            job["slices_meta"].insert(index, {"number": orig_num, "page_range": orig_meta["page_range"]})

    return jsonify({"ok": True})


@app.route("/merge_next/<job_id>/<int:index>", methods=["POST"])
def merge_next(job_id: str, index: int):
    """
    Merge question[index] and question[index+1] into one image (stacked vertically).
    The user can then re-split it correctly with the split tool.
    """
    job = _get_job(job_id)
    full_list = job.get("question_images_full", [])
    if index < 0 or index + 1 >= len(full_list):
        return jsonify({"error": "No next question to merge with"}), 400

    img_a = Image.open(io.BytesIO(base64.b64decode(full_list[index])))
    img_b = Image.open(io.BytesIO(base64.b64decode(full_list[index + 1])))

    max_w   = max(img_a.width, img_b.width)
    merged  = Image.new("RGB", (max_w, img_a.height + img_b.height), (255, 255, 255))
    merged.paste(img_a, (0, 0))
    merged.paste(img_b, (0, img_a.height))

    buf = io.BytesIO()
    merged.save(buf, format="PNG")
    merged_b64   = base64.b64encode(buf.getvalue()).decode()
    merged_thumb = base64.b64encode(_make_thumbnail(buf.getvalue(), 300, 200)).decode()

    meta_list = job.get("slices_meta", [])
    meta_a = meta_list[index]     if index     < len(meta_list) else {"number": index + 1,     "page_range": [1, 1]}
    meta_b = meta_list[index + 1] if index + 1 < len(meta_list) else {"number": index + 2, "page_range": [1, 1]}

    with _jobs_lock:
        job["question_images_full"].pop(index + 1)
        job["question_images_full"].pop(index)
        job["question_images_full"].insert(index, merged_b64)

        job["question_images"].pop(index + 1)
        job["question_images"].pop(index)
        job["question_images"].insert(index, merged_thumb)

        if index + 1 < len(job["slices_meta"]):
            job["slices_meta"].pop(index + 1)
        if index < len(job["slices_meta"]):
            job["slices_meta"].pop(index)
        job["slices_meta"].insert(index, {
            "number": meta_a["number"],
            "page_range": meta_a["page_range"],
        })

    return jsonify({"ok": True})


@app.route("/delete_question/<job_id>/<int:index>", methods=["POST"])
def delete_question(job_id: str, index: int):
    job = _get_job(job_id)
    full_list = job.get("question_images_full", [])
    if index < 0 or index >= len(full_list):
        return jsonify({"error": "Index out of range"}), 400
    with _jobs_lock:
        job["question_images_full"].pop(index)
        job["question_images"].pop(index)
        if index < len(job.get("slices_meta", [])):
            job["slices_meta"].pop(index)
    return jsonify({"ok": True})


# ─────────────────────────────────────────────────────────────────────────────
# Frontend
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_job(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        abort(404)
    return job


def _set_status(job_id: str, status: str) -> None:
    with _jobs_lock:
        _jobs[job_id]["status"] = status


def _set_progress(job_id: str, pct: int, msg: str) -> None:
    with _jobs_lock:
        _jobs[job_id]["progress"] = pct
        _jobs[job_id]["message"]  = msg
    logger.info("[%s] %d%% — %s", job_id, pct, msg)


def _set_error(job_id: str, msg: str, tb: str = "") -> None:
    logger.error("[%s] Error: %s\n%s", job_id, msg, tb)
    with _jobs_lock:
        _jobs[job_id].update({
            "status":   "error",
            "progress": 0,
            "message":  f"处理失败: {msg}",
            "error":    msg,
        })


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._- " else "_" for c in name)


def _extract_pdf_title(pdf_path: str) -> str:
    """
    Try to extract a homework title from the PDF.
    Strategy:
      1. PDF metadata 'title' field (if non-empty and not a generic tool name).
      2. First non-blank, non-header-looking text line on page 1 that looks like a title
         (short, centred or large font, or contains keywords like '作业'/'练习'/'试卷'/'卷').
      3. Return empty string if nothing useful found (caller will fall back to filename).
    """
    try:
        doc = fitz.open(pdf_path)
        # 1. Metadata title
        meta_title = (doc.metadata or {}).get("title", "").strip()
        if meta_title and len(meta_title) > 2 and not any(
            kw in meta_title.lower() for kw in ("microsoft", "word", "wps", "adobe", "pdf")
        ):
            doc.close()
            return meta_title

        # 2. First page text — find best candidate line
        if len(doc) == 0:
            doc.close()
            return ""
        page = doc[0]
        page_w = page.rect.width
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE).get("blocks", [])
        doc.close()
        candidates: list[tuple[float, str]] = []   # (font_size, text)
        for block in blocks[:15]:  # scan top blocks only
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                text = "".join(s["text"] for s in line.get("spans", [])).strip()
                if not text or len(text) > 60:
                    continue
                # font size = max span size in line
                sizes = [s.get("size", 0) for s in line.get("spans", [])]
                font_size = max(sizes) if sizes else 0
                # centre check: line bbox centre near page centre
                bbox = line.get("bbox", [0, 0, 0, 0])
                line_cx = (bbox[0] + bbox[2]) / 2
                centred = abs(line_cx - page_w / 2) < page_w * 0.20
                # keyword check
                kw_match = any(kw in text for kw in ("作业", "练习", "试卷", "卷", "测试", "考试", "题"))
                if font_size >= 12 or centred or kw_match:
                    candidates.append((font_size, text))

        if candidates:
            # Pick the line with largest font size
            candidates.sort(key=lambda x: -x[0])
            return candidates[0][1]
    except Exception as e:
        logger.warning("Could not extract PDF title from %s: %s", pdf_path, e)
    return ""


def _extract_pdf_text(pdf_path: str, max_chars: int = 6000) -> str:
    try:
        doc = fitz.open(pdf_path)
        texts = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(texts)[:max_chars]
    except Exception as e:
        logger.warning("Could not extract text from %s: %s", pdf_path, e)
        return ""


def _extract_any_text(file_path: Path, max_chars: int = 6000) -> str:
    """Extract text from PDF, DOCX, or image file."""
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf_text(str(file_path), max_chars)

    if suffix == ".docx":
        try:
            from docx import Document
            doc = Document(str(file_path))
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            return text[:max_chars]
        except ImportError:
            logger.warning("python-docx not installed, skipping %s", file_path)
            return ""
        except Exception as e:
            logger.warning("Could not extract docx %s: %s", file_path, e)
            return ""

    if suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"):
        try:
            from core.pdf_cutter import ocr_image_to_text
            img_bytes = file_path.read_bytes()
            return ocr_image_to_text(img_bytes)[:max_chars]
        except Exception as e:
            logger.warning("Could not OCR image %s: %s", file_path, e)
            return ""

    return ""


def _make_thumbnail(png_bytes: bytes, max_w: int, max_h: int) -> bytes:
    img = Image.open(io.BytesIO(png_bytes))
    img.thumbnail((max_w, max_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Teaching Preparation Tool")
    print("  访问地址: http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
