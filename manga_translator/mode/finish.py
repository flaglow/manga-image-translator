import argparse
import asyncio
import gc
import os
import pickle
import re
import shutil
import sys
import time

import numpy as np
from PIL import Image

from manga_translator import MangaTranslator, Context, Config
from ..manga_translator import set_main_logger
from ..save import save_result
from ..utils import get_logger, natural_sort, init_logging, set_log_level
from ..utils.generic import load_image, dump_image

import logging
import json

logger = get_logger('finish')


def parse_args():
    p = argparse.ArgumentParser(
        prog='manga_translator.mode.finish',
        description='Pass B: translate and render from GPU-preprocessed cache (Pass A output) without a GPU')
    p.add_argument('-i', '--input', required=True, type=str, help='Path to the source image folder (Pass A source)')
    p.add_argument('-o', '--dest', default='', type=str, help='Path to the destination folder for finished images')
    p.add_argument('--cache', default='', type=str,
                   help='Preprocess cache dir (default: input path with leading data/ replaced by tmp/gpu_pre/)')
    p.add_argument('--config-file', default=None, type=str, help='Path to the config file')
    p.add_argument('--font-path', default='', type=str, help='Path to font file')
    p.add_argument('--batch-size', default=12, type=int, help='Number of pages to translate concurrently per chunk')
    p.add_argument('--context-size', default=0, type=int, help='Pages of context are needed for translating the current page')
    p.add_argument('--post-dict', default=None, type=str, help='Path to the post-translation dictionary file')
    p.add_argument('--overwrite', action='store_true', help='Overwrite already finished images')
    p.add_argument('--from-cache', action='store_true',
                   help='Repair mode: render from gpu_pre cache only (original images may be gone). '
                        'Output resolution follows the cache png.')
    p.add_argument('-v', '--verbose', action='store_true', help='Print debug info')
    return p.parse_args()


def default_cache_dir(input_path: str) -> str:
    norm = input_path.rstrip('/\\')
    parts = re.split(r'[/\\]', norm)
    if parts and parts[0] == 'data':
        parts[0] = os.path.join('tmp', 'gpu_pre')
        return os.path.join(*parts)
    return os.path.join('tmp', 'gpu_pre', os.path.basename(norm))


def load_config(config_file_path):
    if not config_file_path:
        return Config()
    try:
        with open(config_file_path, 'r', encoding='utf-8') as file:
            config_content = file.read()
    except Exception as e:
        print("Couldnt read file")
        raise e
    config_extension = os.path.splitext(config_file_path)[1].lower()

    try:
        if config_extension == ".toml":
            import tomllib
            config_dict = tomllib.loads(config_content)
        elif config_extension == ".json":
            config_dict = json.loads(config_content)
        else:
            raise ValueError("Unsupported configuration file format")
    except Exception as e:
        print("Failed to load configuration file")
        raise e
    return Config(**config_dict)


def count_cache_pkls(cache_dir: str) -> int:
    n = 0
    if not os.path.isdir(cache_dir):
        return n
    for _, _, files in os.walk(cache_dir):
        n += sum(1 for f in files if f.lower().endswith('.pkl'))
    return n


def classify_from_cache(cache_dir, dest, overwrite):
    """수리(refit) 모드 — 원본 없이 캐시(png+pkl)만으로 렌더 대상을 구성한다.
    캐시가 없으면 스킵(원본 카피 불가), 출력은 캐시 해상도 PNG."""
    render_pages = []
    image_count = skipped = 0
    for root, _subdirs, files in os.walk(cache_dir):
        for f in natural_sort(files):
            if not f.lower().endswith('.pkl'):
                continue
            image_count += 1
            stem = os.path.splitext(f)[0]
            png_path = os.path.join(root, stem + '.png')
            dest_path = os.path.join(dest, os.path.relpath(os.path.join(root, stem), cache_dir) + '.png')
            if not overwrite and os.path.exists(dest_path):
                logger.debug(f'Skipping already finished: "{dest_path}"')
                skipped += 1
                continue
            if not os.path.exists(png_path):
                logger.warning(f'Cache png missing, cannot repair: "{stem}"')
                continue
            try:
                with open(os.path.join(root, f), 'rb') as fp:
                    regions = pickle.load(fp).get('regions') or []
            except Exception:
                logger.warning(f'Cache pkl corrupt: "{stem}"')
                continue
            if regions:
                render_pages.append((None, dest_path, png_path, regions))
    return render_pages, [], image_count, skipped


def classify_pages(input_path, cache_dir, dest, overwrite):
    """Split source pages into render (valid cache pair) and copy-original pages."""
    render_pages = []   # (file_path, dest_path, cache_png, regions)
    copy_pages = []     # (file_path, dest_path)
    image_count = 0
    skipped = 0
    for root, subdirs, files in os.walk(input_path):
        files = natural_sort(files)
        for f in files:
            if f.lower() == '.thumb':
                continue
            file_path = os.path.join(root, f)
            rel = os.path.relpath(file_path, input_path)
            stem = os.path.splitext(rel)[0]
            ext = os.path.splitext(f)[1].lower()
            dest_path = os.path.join(dest, stem + ext)

            if not overwrite and os.path.exists(dest_path):
                logger.debug(f'Skipping already finished: "{dest_path}"')
                skipped += 1
                image_count += 1
                continue

            # Same rule as local mode: only PIL-openable files count as pages
            try:
                with Image.open(file_path) as im:
                    im.verify()
            except Exception as e:
                logger.warning(f'Failed to open image: {file_path}, error: {e}')
                continue
            image_count += 1

            pkl_path = os.path.join(cache_dir, stem + '.pkl')
            png_path = os.path.join(cache_dir, stem + '.png')
            reason = None
            regions = None

            if not os.path.exists(pkl_path):
                reason = 'cache pkl missing'
            elif not os.path.exists(png_path):
                reason = 'cache png missing'
            else:
                try:
                    with Image.open(png_path) as im:
                        im.verify()
                except Exception:
                    reason = 'cache png corrupt'
            if not reason:
                try:
                    with open(pkl_path, 'rb') as fp:
                        data = pickle.load(fp)
                    if isinstance(data, dict):
                        regions = data.get('regions') or []
                    else:
                        regions = None
                except Exception:
                    regions = None
                    reason = 'cache pkl corrupt'
            if not reason and not regions:
                reason = 'no text regions'
            if reason:
                logger.warning(f'Copying original, cache unusable: "{rel}" ({reason})')
                copy_pages.append((file_path, dest_path))
            else:
                render_pages.append((file_path, dest_path, png_path, regions))
    return render_pages, copy_pages, image_count, skipped


async def main():
    args = parse_args()
    init_logging()
    set_log_level(logging.DEBUG if args.verbose else logging.INFO)
    set_main_logger(logger)

    input_path = os.path.abspath(os.path.expanduser(args.input))
    dest = os.path.abspath(os.path.expanduser(args.dest)) if args.dest else input_path + '-finished'
    cache_dir = os.path.abspath(os.path.expanduser(args.cache or default_cache_dir(args.input)))

    if args.from_cache:
        render_pages, copy_pages, image_count, skipped = classify_from_cache(cache_dir, dest, args.overwrite)
    else:
        if not os.path.isdir(input_path):
            logger.error(f'Input folder not found: "{input_path}"')
            sys.exit(1)
        render_pages, copy_pages, image_count, skipped = classify_pages(input_path, cache_dir, dest, args.overwrite)

    if args.font_path and not os.path.exists(args.font_path):
        logger.error(f'Font file not found: "{args.font_path}"')
        sys.exit(1)

    try:
        config = load_config(args.config_file)
    except Exception as e:
        logger.error(f'Failed to load configuration file: {e}')
        sys.exit(1)

    mt = MangaTranslator({
        'use_gpu': False,
        'kernel_size': 3,
        'batch_size': args.batch_size,
        'batch_concurrent': True,
        'context_size': args.context_size,
        'post_dict': args.post_dict,
        'models_ttl': 0,
        'input': [args.input],
        'save_text': False,
        'load_text': False,
        'font_path': args.font_path or None,
        'verbose': args.verbose,
    })

    pkl_count = count_cache_pkls(cache_dir)
    if pkl_count > image_count:
        logger.warning(f'Cache pkl count ({pkl_count}) exceeds source image count ({image_count})')

    if not render_pages and not copy_pages:
        logger.info('No images found to translate')
        return

    logger.info(f'Found {len(render_pages) + len(copy_pages)} pages: {len(render_pages)} to render, {len(copy_pages)} to copy original')
    if skipped:
        logger.info(f'Skipped {skipped} already finished pages (use --overwrite to redo)')

    start_time = time.time()
    saved = 0

    # 세그먼트 로그 — 책 단위 jsonl: 페이지별 (원문, 번역) 쌍. 향후 문제 세그먼트만
    # 골라 수리(/api/repair)하는 근거가 된다. rel 규약: data/·tmp/gpu_pre/ 접두 제거.
    rel_in = args.input.rstrip('/\\').replace('\\', '/')
    for _prefix in ('data/', 'tmp/gpu_pre/'):
        if rel_in.startswith(_prefix):
            rel_in = rel_in[len(_prefix):]
            break
    os.makedirs('tmp/hymt2-log', exist_ok=True)
    seg_log = open(os.path.join('tmp/hymt2-log', rel_in + '.jsonl'), 'a', encoding='utf-8')

    # Chunked translation + rendering, order preserved
    for chunk_start in range(0, len(render_pages), args.batch_size):
        chunk = render_pages[chunk_start:chunk_start + args.batch_size]

        pairs = []
        pair_metas = []
        for (file_path, dest_path, cache_png, regions) in chunk:
            try:
                cache_img = Image.open(cache_png)
                ctx = Context()
                if file_path is None:
                    # 수리 모드 — 원본 없음: 캐시 png를 원본·인페인트 겸용 (출력=캐시 해상도)
                    img_rgb = np.array(cache_img.convert('RGB'))
                    ctx.input = cache_img
                    ctx.img_rgb = img_rgb
                    ctx.img_alpha = None
                    ctx.img_inpainted = img_rgb
                else:
                    img = Image.open(file_path)
                    img_rgb, img_alpha = load_image(img)
                    ctx.input = img
                    ctx.img_rgb = img_rgb
                    ctx.img_alpha = img_alpha
                    ctx.img_inpainted = np.array(cache_img.convert('RGB'))
                ctx.text_regions = regions
                ctx.from_lang = 'auto'
                ctx.verbose = mt.verbose
                pairs.append((ctx, config))
                pair_metas.append((file_path, dest_path))
            except Exception as e:
                logger.warning(f'Failed to load page for render: {file_path}, error: {e}')
                try:
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.copyfile(file_path, dest_path)
                    logger.info(f'Saving(original) "{dest_path}"')
                    saved += 1
                except Exception as copy_error:
                    logger.error(f'Failed to copy original: {file_path}, error: {copy_error}')

        if not pairs:
            continue

        try:
            translated = await mt._concurrent_translate_contexts(pairs)
        except Exception as e:
            logger.error(f'Chunk translation failed: {e}')
            raise

        # Render and save each page; filtered-to-empty regions fall back to original copy
        for (file_path, dest_path), (ctx, _config) in zip(pair_metas, translated):
            try:
                if ctx.text_regions:
                    ctx.img_rendered = await mt._run_text_rendering(config, ctx)
                    result = dump_image(ctx.input, ctx.img_rendered, ctx.img_alpha)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    save_ctx = Context()
                    save_ctx.save_quality = 100
                    save_result(result, dest_path, save_ctx)
                    logger.info(f'Saving "{dest_path}"')
                elif file_path is not None:
                    shutil.copyfile(file_path, dest_path)
                    logger.info(f'Saving(original) "{dest_path}"')
                saved += 1
                # 세그먼트 로그 — 페이지별 (원문, 번역) 기록. 향후 문제 세그먼트만 골라
                # 수리(/api/repair)하는 근거 데이터.
                seg_log.write(json.dumps({
                    'page': os.path.splitext(os.path.basename(dest_path))[0],
                    'segs': [[r.text, r.translation] for r in (ctx.text_regions or [])],
                }, ensure_ascii=False) + '\n')
                seg_log.flush()
            except Exception as e:
                logger.error(f'Failed to process page: {file_path}, error: {e}')

        # Same page-context accumulation as translate_batch
        for ctx, _config in translated:
            if ctx.text_regions:
                mt.all_page_translations.append({r.text: r.translation for r in ctx.text_regions})

        gc.collect()

    # Copy-original pages (no usable cache pair)
    for file_path, dest_path in copy_pages:
        try:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copyfile(file_path, dest_path)
            logger.info(f'Saving(original) "{dest_path}"')
            saved += 1
        except Exception as e:
            logger.error(f'Failed to copy original: {file_path}, error: {e}')

    total_time = time.time() - start_time
    seg_log.close()
    logger.info(f'Done! Finished {saved} images in {total_time:.1f}s')
    return saved


if __name__ == '__main__':
    asyncio.run(main())
