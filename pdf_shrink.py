#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF 경량화 스크립트
- method=mixed  : 벡터(텍스트/도면)는 유지, 내장 이미지(JPEG/PNG 등)만 다운스케일+재압축
- method=raster : 모든 페이지를 지정 DPI로 이미지화하여 새 PDF 생성
- method=auto   : mixed 시도 후 목표 용량 미달이면 raster를 순차 DPI로 재시도
- --raster-pages: 특정 페이지만 래스터 처리(예: 7 또는 7,10-12)
- --auto-split  : 최종 결과가 목표 용량을 넘으면 자동으로 파트 분할(각 파트가 target 이하)

필수 라이브러리: PyMuPDF (fitz), Pillow
pip install pymupdf pillow
"""

import io
import os
from pathlib import Path
from typing import List, Tuple, Optional

import fitz  # PyMuPDF
from PIL import Image
import argparse

# ----------------------------- 유틸 -----------------------------

def human_bytes(n: int) -> str:
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < step:
            return f"{n:.2f} {unit}"
        n /= step
    return f"{n:.2f} PB"

def file_size(path: Path) -> int:
    return path.stat().st_size

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def parse_pages(expr: str) -> List[int]:
    expr = (expr or "").strip()
    if not expr:
        return []
    pages = set()
    for token in expr.split(","):
        token = token.strip()
        if "-" in token:
            a, b = token.split("-", 1)
            a = int(a); b = int(b)
            for p in range(min(a, b), max(a, b) + 1):
                pages.add(p)
        else:
            pages.add(int(token))
    return sorted(pages)

# ----------------------- 혼합(이미지 재압축) --------------------

def _resize_keep_aspect(im: Image.Image, max_px: int) -> Image.Image:
    """긴 변이 max_px를 넘으면 비율 유지 축소, 아니면 원본 유지."""
    w, h = im.size
    long_side = max(w, h)
    if long_side <= max_px:
        return im
    scale = max_px / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return im.resize((new_w, new_h), Image.LANCZOS)

def _pil_bytes(im: Image.Image, prefer_format: str, quality: int) -> bytes:
    """
    PIL 이미지를 바이트로 직렬화.
    - prefer_format: 'JPEG' 또는 'PNG' (투명/모노/라인아트면 PNG가 유리)
    """
    buf = io.BytesIO()
    if prefer_format.upper() == "JPEG":
        # JPEG로 저장: 무손실 투명 없음, 포토/스캔류에 유리
        if im.mode in ("RGBA", "LA"):
            # 투명도 제거 (흰색 배경 합성)
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        elif im.mode not in ("RGB",):
            im = im.convert("RGB")
        im.save(buf, format="JPEG", quality=quality, optimize=True, subsampling=2)
    else:
        # PNG로 저장: 투명/라인아트/도면에 유리
        if im.mode in ("P", "1"):
            im = im.convert("RGBA")
        im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

def _should_use_png(src_mode: str, alpha: bool, is_lineart_hint: bool) -> bool:
    """
    라인아트/텍스트가 많거나 투명도가 있으면 PNG 선호.
    JPEG 압축 아티팩트가 눈에 띄기 쉬운 경우 PNG가 적합.
    """
    if alpha:
        return True
    if is_lineart_hint:
        return True
    if src_mode in ("1", "L"):  # 흑백/그레이 스캔
        return False  # 회색조 스캔은 JPEG가 보통 유리
    return False

def is_problematic_image(doc: fitz.Document, img_tuple) -> bool:
    """
    마스크(/SMask, /Mask) 동반, CMYK, 고비트, JPEG2000(JPX) 등은 재압축 시 검은 화면 등 문제가
    발생할 수 있어 보수적으로 '스킵'한다.
    """
    try:
        xref = img_tuple[0]
        smask_xref = img_tuple[1] if len(img_tuple) > 1 else 0
        bpc = img_tuple[4] if len(img_tuple) > 4 else None
        colorspace = img_tuple[5] if len(img_tuple) > 5 else None
        filter_name = img_tuple[7] if len(img_tuple) > 7 else ""
    except Exception:
        return True

    # 1) Soft Mask 또는 Color Key Mask 여부(객체 텍스트에 /SMask 또는 /Mask 직검)
    try:
        obj_str = doc.xref_object(xref, compressed=False) or ""
        if "/SMask" in obj_str or "/Mask" in obj_str:
            return True
    except Exception:
        pass

    # get_images가 smask xref를 바로 넘겨주는 경우
    if isinstance(smask_xref, int) and smask_xref > 0:
        return True

    # 2) CMYK 색공간
    if colorspace and ("CMYK" in str(colorspace).upper()):
        return True

    # 3) 높은 비트수: 8bpc 초과는 보수적으로 스킵
    if bpc and isinstance(bpc, int) and bpc > 8:
        return True

    # 4) JPEG2000(JPXDecode) 등 특수 필터는 스킵
    if filter_name and ("JPX" in str(filter_name).upper()):
        return True

    return False

def shrink_mixed(
    src: Path,
    dst: Path,
    max_px: int = 2000,
    quality: int = 75,
    skip_small_kb: int = 20,
    verbose: bool = True,
) -> None:
    """
    벡터 보존 + 이미지만 축소/재압축.
    - max_px: 이미지 긴 변 최대 픽셀
    - quality: JPEG 품질(대개 65~85 권장)
    - skip_small_kb: 너무 작은 이미지는 건너뜀(효과 미미/가독성 보호)
    """
    ensure_parent(dst)
    doc = fitz.open(src)
    seen = set()  # 중복 xref 처리 방지
    replaced = 0
    total_imgs = 0
    skipped_problematic = 0

    for page_num, page in enumerate(doc, start=1):
        images = page.get_images(full=True)
        total_imgs += len(images)
        for img in images:
            xref = img[0]
            if xref in seen:
                continue
            seen.add(xref)

            # 문제 소지가 있으면 스킵(마스크, CMYK, JPX, 8bpc 초과 등)
            if is_problematic_image(doc, img):
                skipped_problematic += 1
                continue

            info = doc.extract_image(xref)
            if not info:
                continue

            base = info["image"]
            orig_size = len(base)
            if orig_size < skip_small_kb * 1024:
                continue  # 너무 작은 이미지는 스킵

            # 원본 열기
            try:
                im = Image.open(io.BytesIO(base))
                im.load()
            except Exception:
                continue

            # 다운스케일
            im2 = _resize_keep_aspect(im, max_px=max_px)

            # 포맷 결정: PNG 또는 JPEG
            src_mode = im2.mode
            alpha = ("A" in src_mode)
            is_lineart_hint = (src_mode in ("1", "P"))
            use_png = _should_use_png(src_mode, alpha, is_lineart_hint)

            new_bytes = _pil_bytes(im2, "PNG" if use_png else "JPEG", quality=quality)

            # 새 바이트가 충분히 작을 때만 교체(보통 10% 이상 줄었을 때)
            if len(new_bytes) < orig_size * 0.9:
                doc.update_stream(xref, new_bytes)
                replaced += 1
                if verbose:
                    print(f"[page {page_num}] image xref {xref}: {human_bytes(orig_size)} → {human_bytes(len(new_bytes))}")

    # 메타/가비지 정리 후 저장
    doc.set_metadata({})  # 메타데이터 제거
    doc.save(dst.as_posix(), deflate=True, clean=True, garbage=4)
    doc.close()

    if verbose:
        print(f"[mixed] {replaced}/{total_imgs} images recompressed, skipped problematic: {skipped_problematic}")

# ------------------------ 래스터라이즈 방식 ----------------------

def _pixmap_to_rgb_image(pix: fitz.Pixmap) -> Image.Image:
    """
    PyMuPDF Pixmap을 안전하게 PIL 이미지(RGB)로 변환.
    - alpha 채널이 있으면 흰색 배경으로 합성
    - 일부 드라이버에서 발생하는 CMYK / 색상 이슈 방지
    """
    # PyMuPDF의 pix.samples는 bytes. 채널 수는 pix.n (alpha 포함 시 4 또는 2 등)
    # alpha 존재 여부는 pix.alpha로 확인 가능.
    if getattr(pix, "alpha", False) or (pix.n in (2, 4)):
        # RGBA로 만들고 흰 배경에 합성 → RGB
        im = Image.frombytes("RGBA", [pix.width, pix.height], pix.samples)
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        return bg
    else:
        # alpha 없음 → RGB 직변환
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def rasterize_pdf(
    src: Path,
    dst: Path,
    dpi: int = 120,
    image_format: str = "JPEG",
    jpeg_quality: int = 75,
    verbose: bool = True,
) -> None:
    """
    모든 페이지를 이미지로 렌더링 후 새 PDF에 삽입.
    - image_format: 'JPEG' 또는 'PNG' (JPEG가 용량 유리)
    """
    ensure_parent(dst)
    doc = fitz.open(src)
    out = fitz.open()

    for i, page in enumerate(doc, start=1):
        # alpha=True로 렌더하여 투명·마스크 포함 요소를 안전 처리
        pix = page.get_pixmap(dpi=dpi, alpha=True)
        im = _pixmap_to_rgb_image(pix)

        buf = io.BytesIO()
        if image_format.upper() == "PNG":
            im.save(buf, format="PNG", optimize=True)
        else:
            im.save(buf, format="JPEG", quality=jpeg_quality, optimize=True, subsampling=2)
        img_bytes = buf.getvalue()

        rect = fitz.Rect(0, 0, pix.width, pix.height)
        new_page = out.new_page(width=rect.width, height=rect.height)
        new_page.insert_image(rect, stream=img_bytes)
        if verbose:
            print(f"[raster] page {i} at {dpi}dpi")

    out.save(dst.as_posix(), deflate=True, clean=True, garbage=4)
    out.close()
    doc.close()

def rasterize_selected_pages(
    src: Path,
    dst: Path,
    pages: List[int],
    dpi: int = 120,
    image_format: str = "JPEG",
    jpeg_quality: int = 75,
    verbose: bool = True,
) -> None:
    """
    지정한 페이지 번호만 이미지로 렌더링하고, 나머지는 벡터 그대로 유지.
    페이지 번호는 1-based.
    """
    ensure_parent(dst)
    src_doc = fitz.open(src)
    out = fitz.open()

    page_set = set(pages)

    for i in range(src_doc.page_count):
        pno = i + 1
        if pno in page_set:
            page = src_doc.load_page(i)
            pix = page.get_pixmap(dpi=dpi, alpha=True)  # 안전 모드
            im = _pixmap_to_rgb_image(pix)
            buf = io.BytesIO()
            if image_format.upper() == "PNG":
                im.save(buf, format="PNG", optimize=True)
            else:
                im.save(buf, format="JPEG", quality=jpeg_quality, optimize=True, subsampling=2)
            img_bytes = buf.getvalue()
            rect = fitz.Rect(0, 0, pix.width, pix.height)
            new_page = out.new_page(width=rect.width, height=rect.height)
            new_page.insert_image(rect, stream=img_bytes)
            if verbose:
                print(f"[hybrid raster] page {pno} rasterized at {dpi}dpi")
        else:
            # 벡터 / 비문제 페이지는 그대로 복사
            out.insert_pdf(src_doc, from_page=i, to_page=i)

    out.save(dst.as_posix(), deflate=True, clean=True, garbage=4)
    out.close()
    src_doc.close()

# ------------------------ 자동 분할 기능 -------------------------

def split_by_size(src: Path, target_bytes: int) -> List[Path]:
    """
    src PDF를 여러 파트로 나눠 각 파트가 target_bytes 이하가 되도록 분할.
    단순 이진 탐색으로 페이지 수를 조절.
    """
    doc = fitz.open(src)
    total = doc.page_count
    out_paths: List[Path] = []
    start = 0
    part = 1

    while start < total:
        lo, hi = 1, total - start
        best_n = 1
        # 이진 탐색으로 최대 페이지 수 탐색
        while lo <= hi:
            mid = (lo + hi) // 2
            tmp = fitz.open()
            tmp.insert_pdf(doc, from_page=start, to_page=start + mid - 1)
            tmp_path = src.parent / f"{src.stem}_part{part}_tmp.pdf"
            tmp.save(tmp_path.as_posix(), deflate=True, clean=True, garbage=3)
            sz = file_size(tmp_path)
            tmp.close()

            if sz <= target_bytes:
                best_n = mid
                # 임시를 확정명으로 교체
                final_path = src.parent / f"{src.stem}_part{part}.pdf"
                if final_path.exists():
                    final_path.unlink()
                os.replace(tmp_path, final_path)
                lo = mid + 1
            else:
                tmp_path.unlink(missing_ok=True)
                hi = mid - 1

        out_paths.append(src.parent / f"{src.stem}_part{part}.pdf")
        start += best_n
        part += 1

    doc.close()
    return out_paths

# --------------------------- 메인 로직 ---------------------------

def run(
    input_path: Path,
    output_path: Path,
    method: str,
    target_mb: Optional[float],
    dpi_sequence: Tuple[int, ...],
    mixed_max_px: int,
    mixed_quality: int,
    raster_quality: int,
    auto_split: bool,
    hybrid_pages: Optional[List[int]] = None,
) -> None:
    ensure_parent(output_path)

    target_bytes = int(target_mb * 1024 * 1024) if target_mb else None

    if method == "mixed":
        shrink_mixed(input_path, output_path, max_px=mixed_max_px, quality=mixed_quality)
        print(f"[mixed] result: {human_bytes(file_size(output_path))}")

    elif method == "raster":
        dpi = dpi_sequence[0] if dpi_sequence else 120
        rasterize_pdf(
            input_path, output_path,
            dpi=dpi, image_format="JPEG", jpeg_quality=raster_quality
        )
        print(f"[raster {dpi}dpi] result: {human_bytes(file_size(output_path))}")

    elif method == "auto":
        tmp_mixed = output_path.with_suffix(".mixed.pdf")
        shrink_mixed(input_path, tmp_mixed, max_px=mixed_max_px, quality=mixed_quality)
        sz = file_size(tmp_mixed)
        print(f"[auto] after mixed: {human_bytes(sz)}")
        use_path = tmp_mixed

        if target_bytes is not None and sz > target_bytes:
            for dpi in dpi_sequence:
                tmp_r = output_path.with_suffix(f".r{dpi}.pdf")
                rasterize_pdf(
                    input_path, tmp_r,
                    dpi=dpi, image_format="JPEG", jpeg_quality=raster_quality
                )
                r_sz = file_size(tmp_r)
                print(f"[auto] raster {dpi}dpi: {human_bytes(r_sz)}")
                if r_sz <= target_bytes:
                    use_path = tmp_r
                    break
                else:
                    if r_sz < sz:
                        use_path = tmp_r
                        sz = r_sz

        if use_path != output_path:
            if output_path.exists():
                output_path.unlink()
            os.replace(use_path, output_path)

        print(f"[auto] result: {human_bytes(file_size(output_path))}")

    else:
        raise ValueError("--method 은 mixed | raster | auto 중 하나여야 합니다.")

    # ---- 부분 래스터(선택) : 특정 페이지만 rasterize ----
    if hybrid_pages:
        tmp_h = output_path.with_suffix(".hybrid.pdf")
        # 부분 래스터는 가독성 위해 auto 모드에서 가장 높은 DPI(목록 첫 항목) 사용
        dpi_for_hybrid = max(90, (dpi_sequence[0] if dpi_sequence else 120))
        rasterize_selected_pages(
            output_path, tmp_h,
            pages=hybrid_pages, dpi=dpi_for_hybrid,
            image_format="JPEG", jpeg_quality=raster_quality
        )
        if output_path.exists():
            output_path.unlink()
        os.replace(tmp_h, output_path)
        print(f"[hybrid] rasterized pages {hybrid_pages} → {human_bytes(file_size(output_path))}")

    # ---- 자동 분할 ----
    if auto_split and target_bytes is not None:
        final_sz = file_size(output_path)
        if final_sz > target_bytes:
            print(f"[split] {human_bytes(final_sz)} > target {human_bytes(target_bytes)} → splitting...")
            parts = split_by_size(output_path, target_bytes)
            for p in parts:
                print(f"  - {p.name}: {human_bytes(file_size(p))}")
            print("[split] done. 원본은 그대로 두었습니다.")

# ----------------------------- CLI -------------------------------

def main():
    ap = argparse.ArgumentParser(description="PDF 용량 줄이기 (혼합/래스터/자동)")
    ap.add_argument("input", type=str, help="입력 PDF 경로")
    ap.add_argument("-o", "--output", type=str, required=True, help="출력 PDF 경로")

    ap.add_argument("--method", type=str, default="auto",
                    choices=["mixed", "raster", "auto"],
                    help="혼합(mixed), 래스터(raster), 자동(auto)")

    ap.add_argument("--target-mb", type=float, default=4.0,
                    help="목표 용량(MB). auto/mixed에서 참고")

    # 혼합 방식 파라미터
    ap.add_argument("--max-px", type=int, default=2000,
                    help="혼합 방식: 이미지 긴 변 최대 픽셀 (권장 1200~2500)")
    ap.add_argument("--quality", type=int, default=75,
                    help="혼합 방식: JPEG 품질 (권장 65~85)")

    # 래스터 방식 파라미터
    ap.add_argument("--dpi", type=int, nargs="*", default=[150, 120, 100, 90, 72],
                    help="래스터 방식(및 자동모드): 시도할 DPI 목록(내림차순 권장)")
    ap.add_argument("--raster-quality", type=int, default=75,
                    help="래스터 방식: JPEG 품질 (권장 65~85)")

    ap.add_argument("--auto-split", action="store_true",
                    help="최종 결과가 target-mb 초과 시 자동 분할")

    ap.add_argument("--raster-pages", type=str, default="",
                    help="특정 페이지만 래스터 처리 (예: '7' 또는 '7,10-12')")

    args = ap.parse_args()

    run(
        input_path=Path(args.input),
        output_path=Path(args.output),
        method=args.method,
        target_mb=args.target_mb,
        dpi_sequence=tuple(args.dpi),
        mixed_max_px=args.max_px,
        mixed_quality=args.quality,
        raster_quality=args.raster_quality,
        auto_split=args.auto_split,
        hybrid_pages=parse_pages(args.raster_pages),
    )

if __name__ == "__main__":
    main()
