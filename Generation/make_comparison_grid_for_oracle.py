"""Create Oracle qualitative-comparison contact sheets.

Each class is one column. The two rows are:
    Ground truth
    Oracle
"""

import argparse
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def natural_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def image_files(directory):
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
        ],
        key=natural_key,
    )


def class_name_from_gt(folder_name):
    """Remove the numeric prefix before the first underscore."""
    return (
        folder_name.split("_", 1)[1]
        if "_" in folder_name
        else folder_name
    )


def load_cell(image_path, size):
    with Image.open(image_path) as image:
        image = image.convert("RGB")

        return ImageOps.pad(
            image,
            (size, size),
            method=Image.Resampling.LANCZOS,
            color="white",
        )


def choose_generated_image(class_dir, sample_index):
    expected = class_dir / f"{sample_index}.png"

    if expected.exists():
        return expected

    files = image_files(class_dir)

    if sample_index >= len(files):
        raise IndexError(
            f"{class_dir}: sample_index={sample_index}, "
            f"but only {len(files)} generated images were found"
        )

    return files[sample_index]


def load_font(size):
    font_path = Path(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )

    if font_path.exists():
        return ImageFont.truetype(
            str(font_path),
            size=size,
        )

    return ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gt_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--oracle_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--classes_per_page",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
    )

    args = parser.parse_args()

    # 入力ディレクトリの確認
    for directory in (
        args.gt_dir,
        args.oracle_dir,
    ):
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Directory not found: {directory}"
            )

    # 正解画像のクラスフォルダ
    gt_class_dirs = sorted(
        [
            path
            for path in args.gt_dir.iterdir()
            if path.is_dir()
        ],
        key=lambda path: path.name,
    )

    if not gt_class_dirs:
        raise RuntimeError(
            f"No class directories found in {args.gt_dir}"
        )

    # 正解画像とOracle画像の対応付け
    rows = []
    missing = []

    for gt_class_dir in gt_class_dirs:
        class_name = class_name_from_gt(
            gt_class_dir.name
        )

        oracle_class_dir = (
            args.oracle_dir / class_name
        )

        gt_images = image_files(gt_class_dir)

        if (
            not gt_images
            or not oracle_class_dir.is_dir()
        ):
            missing.append(class_name)
            continue

        oracle_image = choose_generated_image(
            oracle_class_dir,
            args.sample_index,
        )

        rows.append(
            (
                class_name,
                gt_images[0],
                oracle_image,
            )
        )

    if missing:
        preview = ", ".join(missing[:10])

        raise RuntimeError(
            f"{len(missing)} classes could not be matched. "
            f"First entries: {preview}"
        )

    args.save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_size = args.image_size
    margin = 16
    label_width = 150
    header_height = 58
    row_gap = 8
    row_height = image_size + row_gap

    page_count = math.ceil(
        len(rows) / args.classes_per_page
    )

    class_font = load_font(12)
    row_font = load_font(17)

    row_labels = (
        "Test image",
        "Oracle",
    )

    # ページ単位で比較画像を作成
    for page_index in range(page_count):
        page_rows = rows[
            page_index * args.classes_per_page:
            (page_index + 1) * args.classes_per_page
        ]

        page_width = (
            label_width
            + len(page_rows) * image_size
            + 2 * margin
        )

        page_height = (
            header_height
            + 2 * row_height
            + margin
        )

        canvas = Image.new(
            "RGB",
            (page_width, page_height),
            "white",
        )

        draw = ImageDraw.Draw(canvas)

        page_start = (
            page_index * args.classes_per_page
        )

        for column_index, (
            class_name,
            gt_path,
            oracle_path,
        ) in enumerate(page_rows):

            x = (
                label_width
                + margin
                + column_index * image_size
            )

            class_number = (
                page_start
                + column_index
                + 1
            )

            short_name = (
                class_name
                if len(class_name) <= 16
                else class_name[:15] + "…"
            )

            heading = (
                f"{class_number:03d}\n"
                f"{short_name}"
            )

            heading_box = draw.multiline_textbbox(
                (0, 0),
                heading,
                font=class_font,
                align="center",
            )

            heading_width = (
                heading_box[2]
                - heading_box[0]
            )

            draw.multiline_text(
                (
                    x
                    + (image_size - heading_width) / 2,
                    8,
                ),
                heading,
                fill="black",
                font=class_font,
                align="center",
                spacing=2,
            )

            # 上段：正解画像、下段：Oracle画像
            for row_index, image_path in enumerate(
                (
                    gt_path,
                    oracle_path,
                )
            ):
                y = (
                    header_height
                    + row_index * row_height
                )

                canvas.paste(
                    load_cell(
                        image_path,
                        image_size,
                    ),
                    (x, y),
                )

            draw.line(
                (
                    x + image_size - 1,
                    header_height,
                    x + image_size - 1,
                    page_height - margin,
                ),
                fill=(215, 215, 215),
                width=1,
            )

        # 左側の行ラベル
        for row_index, row_label in enumerate(
            row_labels
        ):
            y = (
                header_height
                + row_index * row_height
            )

            text_box = draw.textbbox(
                (0, 0),
                row_label,
                font=row_font,
            )

            text_height = (
                text_box[3]
                - text_box[1]
            )

            draw.text(
                (
                    margin,
                    y
                    + (image_size - text_height) / 2,
                ),
                row_label,
                fill="black",
                font=row_font,
            )

            draw.line(
                (
                    margin,
                    y
                    + image_size
                    + row_gap // 2,
                    page_width - margin,
                    y
                    + image_size
                    + row_gap // 2,
                ),
                fill=(215, 215, 215),
                width=1,
            )

        output_path = (
            args.save_dir
            / f"oracle_comparison_{page_index + 1:02d}.png"
        )

        canvas.save(output_path)

        print(f"Saved: {output_path}")

    print(
        f"Done: {len(rows)} classes, "
        f"{page_count} pages, "
        f"Oracle sample index={args.sample_index}"
    )


if __name__ == "__main__":
    main()