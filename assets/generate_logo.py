"""
Generates assets/logo.png: a small geometric mark (a magnifying glass over a
checkmark) used as the app's browser-tab icon and header logo, replacing the
emoji placeholders the UI used at first.

Drawn programmatically with Pillow rather than hand-authored as a binary
asset, so the design is versioned as readable code and easy to tweak (colors,
proportions) without an external image editor. Supersampled 4x then
downscaled for anti-aliased edges, since Pillow's basic drawing primitives
are aliased at native resolution.

Run: python assets/generate_logo.py
"""

from pathlib import Path

from PIL import Image, ImageDraw

OUTPUT_PATH = Path(__file__).parent / "logo.png"

SUPERSAMPLE = 4
SIZE = 128 * SUPERSAMPLE

LENS_BLUE = (37, 99, 235, 255)      # magnifying glass ring + handle
CHECK_GREEN = (22, 163, 74, 255)    # checkmark inside the lens


def draw_logo() -> Image.Image:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Lens: a ring (circle outline), centered slightly up-left to leave room
    # for the handle in the bottom-right corner.
    lens_center = (SIZE * 0.42, SIZE * 0.42)
    lens_radius = SIZE * 0.28
    ring_width = int(SIZE * 0.07)
    bbox = [
        lens_center[0] - lens_radius,
        lens_center[1] - lens_radius,
        lens_center[0] + lens_radius,
        lens_center[1] + lens_radius,
    ]
    draw.ellipse(bbox, outline=LENS_BLUE, width=ring_width)

    # Handle: a thick rounded line from the lens edge out to the corner.
    handle_start_r = lens_radius + ring_width * 0.3
    import math

    angle = math.radians(45)
    handle_start = (
        lens_center[0] + handle_start_r * math.cos(angle),
        lens_center[1] + handle_start_r * math.sin(angle),
    )
    handle_end = (SIZE * 0.88, SIZE * 0.88)
    handle_width = int(SIZE * 0.08)
    draw.line([handle_start, handle_end], fill=LENS_BLUE, width=handle_width)
    # Round the handle's two ends (PIL's line join is not rounded by default).
    r = handle_width / 2
    for pt in (handle_start, handle_end):
        draw.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill=LENS_BLUE)

    # Checkmark inside the lens: a short-then-long stroke, "verified result".
    check_width = int(SIZE * 0.065)
    p1 = (lens_center[0] - lens_radius * 0.45, lens_center[1] + lens_radius * 0.05)
    p2 = (lens_center[0] - lens_radius * 0.1, lens_center[1] + lens_radius * 0.4)
    p3 = (lens_center[0] + lens_radius * 0.5, lens_center[1] - lens_radius * 0.35)
    draw.line([p1, p2], fill=CHECK_GREEN, width=check_width)
    draw.line([p2, p3], fill=CHECK_GREEN, width=check_width)
    for pt in (p1, p2, p3):
        r = check_width / 2
        draw.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill=CHECK_GREEN)

    return img.resize((128, 128), Image.LANCZOS)


if __name__ == "__main__":
    logo = draw_logo()
    logo.save(OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}")
