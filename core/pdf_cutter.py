"""
pdf_cutter.py — Question cutting via preprocess→stitch→cut pipeline.

Algorithm:
1. Render each PDF page to PIL Image at 200 DPI.
2. Strip header/footer from every page using pixel-level analysis.
3. Stitch the content strips into ONE long vertical image.
4. Find question-number anchors in original page coordinates (text layer first,
   then RapidOCR fallback), then map to stitched-image coordinates.
5. Cut questions from the stitched image with simple y-range slices.
6. Last question: scan backward from stitched bottom for last content row.
7. Return QuestionSlice list.
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import re
import io
import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional

import fitz
from PIL import Image

logger = logging.getLogger(__name__)

# ── Pre-load RapidOCR at module import time (main thread) ─────────────────────
# On Windows, onnxruntime's DLL must be loaded in the main thread.
# Importing it lazily inside a daemon thread causes:
#   "DLL load failed while importing onnxruntime_pybind11_state:
#    动态链接库(DLL)初始化例程失败。"
# Solution: import the RapidOCR *class* here (module level = main thread),
# so the DLL is registered before Flask spawns any background threads.
# The *engine instance* is still created lazily on first use.
try:
    from rapidocr_onnxruntime import RapidOCR as _RapidOCR
    logger.info("rapidocr_onnxruntime imported successfully (main thread).")
except Exception as _rapidocr_import_err:
    _RapidOCR = None  # type: ignore[assignment,misc]
    logger.warning("rapidocr_onnxruntime import failed: %s", _rapidocr_import_err)

# ── Exam section header detection ────────────────────────────────────────────
# Matches lines like "一、单选题", "第I卷（选择题...）", "第一部分" etc.
# Used in exam mode to find where actual question content starts on page 0,
# so that numbered exam instructions (1.答卷前… 2.回答选择题…) are ignored.
_EXAM_SECTION_PAT = re.compile(
    r'^(?:'
    r'第\s*[一二三四五六七八九十IVXLCDM\d]+\s*[卷题部分]'      # 第I卷、第二部分
    r'|[一二三四五六七八九十]+\s*[、．]\s*(?:单选|多选|选择|实验|计算|解答|填空|综合|不定项|判断)'
    r')'
)

# ── Regex patterns for question number detection ──────────────────────────────

# Homework mode: top-level questions are labeled (1)、(2)… or 1. 2.
_Q_PATTERNS_HOMEWORK = [
    re.compile(r'^\s*第\s*(\d+)\s*题'),                     # 第1题
    re.compile(r'^\s*【(\d+)】'),                            # 【1】
    re.compile(r'^\s*[（(]\s*(\d+)\s*[）)]\s*[、.．\s]'),   # (1)、 (1).
    re.compile(r'^\s*[（(]\s*(\d+)\s*[）)]\s*$'),           # (1) 独占一行
    re.compile(r'^\s*(\d{1,2})\s*[.．]\s*[^\d\s]'),         # 1.后接内容
    re.compile(r'^\s*(\d{1,2})\s*[、]\s*'),                 # 1、
    re.compile(r'^\s*(\d{1,2})\s*[.．]\s*$'),               # 1. 单独一行
]

# Exam mode: top-level questions are labeled 1. 2. 3. only.
# The (n) patterns are intentionally excluded because （1）（2）… are
# sub-questions inside calculation problems and must NOT become cut points.
_Q_PATTERNS_EXAM = [
    re.compile(r'^\s*第\s*(\d+)\s*题'),                          # 第1题
    re.compile(r'^\s*【(\d+)】'),                                 # 【1】
    re.compile(r'^\s*(\d{1,2})\s*[.．]\s*[^\d\s]'),              # 1.后接非数字
    # Some questions start with a year/number: "5．2023年…", "13．1879 年…"
    # Match digit+period+digits+(space?)+Chinese char — excludes "0.25 s", "3.2 m"
    re.compile(r'^\s*(\d{1,2})\s*[.．]\s*\d+(?:[\u4e00-\u9fff（【《]|\s+[\u4e00-\u9fff])'), # 5.2023年…
    re.compile(r'^\s*(\d{1,2})\s*[、]\s*'),                      # 1、
    re.compile(r'^\s*(\d{1,2})\s*[.．]\s*$'),                    # 1. 单独一行
]

DPI = 200
SCALE = DPI / 72.0
MARGIN = 7                # px above question anchor
WHITE_THRESHOLD = 245
PAGE_NUM_SCAN_PX = 120    # px from edge to scan for header/footer
MIN_ANCHOR_DIST_PX = 80         # homework: ~1 cm at 200 DPI
MIN_ANCHOR_DIST_PX_EXAM = 80    # exam: same distance; (n) sub-labels already excluded by pattern set

_ocr_engine = None


def _get_ocr():
    """Return the shared RapidOCR engine, creating it on first call."""
    global _ocr_engine
    if _ocr_engine is None:
        if _RapidOCR is None:
            raise RuntimeError(
                "RapidOCR 不可用：rapidocr_onnxruntime 模块导入失败，"
                "请检查 onnxruntime 安装。"
            )
        logger.info("Initializing RapidOCR engine...")
        _ocr_engine = _RapidOCR()
        logger.info("RapidOCR engine ready.")
    return _ocr_engine


@dataclass
class Anchor:
    page_idx: int
    y_pixel: int          # coordinate in ORIGINAL page image
    question_number: int
    raw_text: str


@dataclass
class QuestionSlice:
    question_number: int
    raw_number: str
    page_range: tuple[int, int]   # 1-based
    image_bytes: bytes             # PNG


# ── Public entry point ────────────────────────────────────────────────────────

def cut_questions(pdf_path: str, mode: str = "homework") -> list[QuestionSlice]:
    """
    Cut a PDF into per-question images.

    mode:
      "homework" — use all patterns including (n)-style; suitable for
                   homework sheets where top-level questions are (1)(2)…
      "exam"     — use stricter patterns (no (n) labels); suitable for
                   exam papers where (1)(2)… are sub-questions inside
                   calculation problems.
    """
    doc = fitz.open(pdf_path)
    try:
        # Step 1: Render pages
        page_images = _render_pages(doc)

        # Step 2: Strip header/footer from each page → get content strips + offsets
        strips, page_content_tops = _strip_headers_footers(page_images)

        # Step 3: Stitch content strips into one long image
        stitched, strip_y_offsets = _stitch_strips(strips)

        # Step 4: Find anchors in original page coordinates
        anchors = _find_anchors_text(doc, page_images, mode=mode)
        if not anchors:
            logger.info("No text-layer anchors found, using RapidOCR...")
            anchors = _find_anchors_ocr(page_images, mode=mode)

        if not anchors:
            logger.warning("No anchors detected, returning full stitched image as one question.")
            return [QuestionSlice(1, "1", (1, len(doc)), _to_png(stitched))]

        logger.info("Detected %d questions (mode=%s).", len(anchors), mode)

        # Step 5: Convert anchor y-coords to stitched-image coordinates
        stitched_anchors = _map_anchors_to_stitched(
            anchors, page_content_tops, strip_y_offsets, page_images
        )

        # Step 6: Cut questions from stitched image
        slices: list[QuestionSlice] = []
        stitched_h = stitched.height
        content_bottom = _find_content_bottom(stitched)

        for i, (anchor, sy) in enumerate(stitched_anchors):
            y0 = max(0, sy - MARGIN)

            if i + 1 < len(stitched_anchors):
                y1 = stitched_anchors[i + 1][1]
            else:
                # Last question: trim trailing blank space
                y1 = content_bottom

            if y1 <= y0 + 5:
                continue

            crop = stitched.crop((0, y0, stitched.width, y1))
            # page_range: approximate from anchor
            p_start = anchor.page_idx + 1
            if i + 1 < len(stitched_anchors):
                p_end = stitched_anchors[i + 1][0].page_idx + 1
            else:
                p_end = len(doc)

            slices.append(QuestionSlice(
                question_number=anchor.question_number,
                raw_number=anchor.raw_text,
                page_range=(p_start, p_end),
                image_bytes=_to_png(crop),
            ))

        return slices
    finally:
        doc.close()


# ── Render pages ──────────────────────────────────────────────────────────────

def _render_pages(doc) -> list[Image.Image]:
    mat = fitz.Matrix(SCALE, SCALE)
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images


# ── Header/footer stripping ───────────────────────────────────────────────────

def _find_page_num_margin(img: Image.Image, side: str) -> int:
    """
    Return the y-coordinate where real content begins (top) or ends (bottom),
    skipping page numbers, chapter headers, and blank margins.

    Skips rows that are:
    - Blank (no dark pixels)
    - Short (<15% dark) — page number digit
    - Wide (>40% dark) in the edge zone — chapter header
    """
    arr = np.array(img.convert('L'))
    h, w = arr.shape
    scan = min(PAGE_NUM_SCAN_PX, h // 5)

    def is_skippable(y: int, near_edge: bool) -> bool:
        dark = int(np.sum(arr[y] < WHITE_THRESHOLD))
        if dark == 0:
            return True
        if dark < w * 0.15:
            return True
        if near_edge and dark > w * 0.40:
            return True
        return False

    if side == 'top':
        boundary = 0
        for y in range(h):
            near_edge = y < scan
            if int(np.min(arr[y])) < WHITE_THRESHOLD:
                if near_edge and is_skippable(y, near_edge):
                    boundary = y + 1
                    continue
                boundary = y
                break
        return max(0, boundary - 2)
    else:  # bottom
        boundary = h
        for y in range(h - 1, -1, -1):
            near_edge = (h - y) < scan
            if int(np.min(arr[y])) < WHITE_THRESHOLD:
                if near_edge and is_skippable(y, near_edge):
                    boundary = y
                    continue
                boundary = y + 1
                break
        return min(h, boundary + 2)


def _strip_headers_footers(
    page_images: list[Image.Image],
) -> tuple[list[Image.Image], list[int]]:
    """
    For each page, crop to content-only region (strip header & footer).
    Returns:
      strips            — list of cropped content images
      page_content_tops — list of content_top y in original page coords
    """
    strips = []
    page_content_tops = []
    for img in page_images:
        top = _find_page_num_margin(img, 'top')
        bot = _find_page_num_margin(img, 'bottom')
        if bot <= top + 5:
            # Degenerate page: keep a thin slice so stitching still works
            bot = min(img.height, top + 10)
        strip = img.crop((0, top, img.width, bot))
        strips.append(strip)
        page_content_tops.append(top)
    return strips, page_content_tops


# ── Stitching ─────────────────────────────────────────────────────────────────

def _stitch_strips(
    strips: list[Image.Image],
) -> tuple[Image.Image, list[int]]:
    """
    Vertically stitch content strips.
    Returns:
      stitched        — the combined image
      strip_y_offsets — stitched y at which each strip begins
    """
    max_w = max(s.width for s in strips)
    total_h = sum(s.height for s in strips)
    result = Image.new("RGB", (max_w, total_h), (255, 255, 255))

    offsets: list[int] = []
    y = 0
    for strip in strips:
        offsets.append(y)
        if strip.width < max_w:
            canvas = Image.new("RGB", (max_w, strip.height), (255, 255, 255))
            canvas.paste(strip, (0, 0))
            strip = canvas
        result.paste(strip, (0, y))
        y += strip.height

    return result, offsets


# ── Anchor coordinate mapping ─────────────────────────────────────────────────

def _map_anchors_to_stitched(
    anchors: list[Anchor],
    page_content_tops: list[int],
    strip_y_offsets: list[int],
    page_images: list[Image.Image],
) -> list[tuple[Anchor, int]]:
    """
    Convert each anchor's (page_idx, y_pixel) to stitched-image y coordinate.
    Anchors whose y_pixel falls before the content top (inside stripped header)
    are clamped to the strip start.
    """
    result: list[tuple[Anchor, int]] = []
    for anchor in anchors:
        pi = anchor.page_idx
        if pi >= len(strip_y_offsets):
            continue
        content_top = page_content_tops[pi]
        # y relative to content strip
        y_in_strip = max(0, anchor.y_pixel - content_top)
        # cap to strip height
        strip_h = page_images[pi].height - content_top
        y_in_strip = min(y_in_strip, max(0, strip_h - 1))
        # convert to stitched coords
        stitched_y = strip_y_offsets[pi] + y_in_strip
        result.append((anchor, stitched_y))
    return result


# ── Content bottom detection ──────────────────────────────────────────────────

def _find_content_bottom(img: Image.Image) -> int:
    """Return y of last non-white row + small padding."""
    arr = np.array(img.convert('L'))
    for y in range(arr.shape[0] - 1, -1, -1):
        if int(np.min(arr[y])) < WHITE_THRESHOLD:
            return min(arr.shape[0], y + 8)
    return arr.shape[0]


# ── Text-layer anchor detection ───────────────────────────────────────────────

def _find_anchors_text(doc, page_images: list[Image.Image],
                       mode: str = "homework") -> list[Anchor]:
    patterns = _Q_PATTERNS_EXAM if mode == "exam" else _Q_PATTERNS_HOMEWORK
    min_dist = MIN_ANCHOR_DIST_PX_EXAM if mode == "exam" else MIN_ANCHOR_DIST_PX

    # In exam mode: find where actual questions start on page 0 by locating
    # the first section header (e.g. "一、单选题" / "第I卷").  Numbered items
    # before that line (exam instructions) are skipped.
    p0_min_y: int = 0
    if mode == "exam" and len(doc) > 0:
        p0_min_y = _find_exam_content_start(doc[0])

    anchors: list[Anchor] = []
    page_ys: dict[int, list[int]] = {}

    for pi, page in enumerate(doc):
        left_limit = page.rect.width * 0.35
        text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            if block["bbox"][0] > left_limit:
                continue
            y_px = max(0, min(int(block["bbox"][1] * SCALE),
                              page_images[pi].height - 1))
            # Skip pre-question blocks on page 0 in exam mode
            if pi == 0 and mode == "exam" and y_px < p0_min_y:
                continue
            for line in block.get("lines", [])[:2]:
                line_text = "".join(s["text"] for s in line.get("spans", [])).strip()
                qnum = _match_qnum(line_text, patterns)
                if qnum is None:
                    continue
                if any(abs(y_px - ey) < min_dist
                       for ey in page_ys.get(pi, [])):
                    continue
                page_ys.setdefault(pi, []).append(y_px)
                anchors.append(Anchor(pi, y_px, qnum, line_text[:20]))
                break

    anchors.sort(key=lambda a: (a.page_idx, a.y_pixel))
    return anchors


def _find_exam_content_start(page) -> int:
    """
    Return the pixel y-coordinate of the first section header on page 0
    (e.g. "一、单选题", "第I卷").  Anchors before this position on page 0
    are treated as exam instructions, not question numbers.
    Returns 0 if no section header is found (no filtering applied).
    """
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", [])[:1]:
            text = "".join(s["text"] for s in line.get("spans", [])).strip()
            if _EXAM_SECTION_PAT.match(text):
                return int(block["bbox"][1] * SCALE)
    return 0


# ── RapidOCR anchor detection ─────────────────────────────────────────────────

def _find_anchors_ocr(page_images: list[Image.Image],
                      mode: str = "homework") -> list[Anchor]:
    patterns = _Q_PATTERNS_EXAM if mode == "exam" else _Q_PATTERNS_HOMEWORK
    min_dist = MIN_ANCHOR_DIST_PX_EXAM if mode == "exam" else MIN_ANCHOR_DIST_PX

    ocr = _get_ocr()
    anchors: list[Anchor] = []
    page_ys: dict[int, list[int]] = {}

    for pi, img in enumerate(page_images):
        left_limit = img.width * 0.35
        arr = np.array(img)

        try:
            result, _ = ocr(arr)
        except Exception as e:
            logger.warning("RapidOCR failed page %d: %s", pi, e)
            continue

        if not result:
            continue

        for item in result:
            bbox, text, conf = item
            if conf < 0.5:
                continue

            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            if min(xs) > left_limit:
                continue

            qnum = _match_qnum(text.strip(), patterns)
            if qnum is None:
                continue

            y_top = max(0, min(int(min(ys)), img.height - 1))
            if any(abs(y_top - ey) < min_dist
                   for ey in page_ys.get(pi, [])):
                continue
            page_ys.setdefault(pi, []).append(y_top)
            anchors.append(Anchor(pi, y_top, qnum, text[:20]))

    anchors.sort(key=lambda a: (a.page_idx, a.y_pixel))
    return anchors


# ── Question number matching ──────────────────────────────────────────────────

def _match_qnum(text: str, patterns=None) -> Optional[int]:
    if patterns is None:
        patterns = _Q_PATTERNS_HOMEWORK
    for pat in patterns:
        m = pat.match(text)
        if m:
            try:
                return int(m.group(1))
            except (IndexError, ValueError):
                pass
    return None


# ── Utilities ─────────────────────────────────────────────────────────────────

def ocr_image_to_text(img_bytes: bytes) -> str:
    """
    OCR a question image and return its text content.
    Results are sorted top-to-bottom so the text reads naturally.
    """
    ocr = _get_ocr()
    img = Image.open(io.BytesIO(img_bytes))
    arr = np.array(img)
    try:
        result, _ = ocr(arr)
    except Exception as e:
        logger.warning("OCR failed on question image: %s", e)
        return ""
    if not result:
        return ""
    # Sort by top-y of each text box, then join
    items = sorted(result, key=lambda x: min(p[1] for p in x[0]))
    return "\n".join(item[1] for item in items if item[2] > 0.4)


def _to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()
