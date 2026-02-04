#!/usr/bin/env python3
"""
Sprite Generator - Simulates Gambo's asset generation pipeline
Uses Google Gemini (Nano Banana) for image generation with template-based approach
"""

import argparse
import base64
import json
import os
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple

# ML-based background removal
try:
    from rembg import remove as rembg_remove
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False
    print("Warning: rembg not installed, falling back to chromakey")

@dataclass
class SpriteConfig:
    """Configuration for sprite generation"""
    character: str          # Character description
    animation: str          # Animation type (idle, walk, attack, etc.)
    frame_count: int        # Number of frames
    frame_size: int         # Size of each frame (square)
    background: str         # "white" or "magenta" for chromakey

    @property
    def template_width(self) -> int:
        return self.frame_count * self.frame_size

    @property
    def template_height(self) -> int:
        return self.frame_size


class SpriteGenerator:
    """Main sprite generation pipeline"""

    ANIMATION_PROMPTS = {
        "idle": "idle animation, slight breathing movement, frame {i}: {desc}",
        "walk": "walk cycle animation, frame {i}: {desc}",
        "run": "running animation, frame {i}: {desc}",
        "attack": "attack animation, frame {i}: {desc}",
        "jump": "jump animation, frame {i}: {desc}",
        "die": "death animation, frame {i}: {desc}",
    }

    FRAME_DESCRIPTIONS = {
        "idle": ["neutral pose", "slight inhale", "neutral pose", "slight exhale"],
        "walk": ["left foot forward", "feet together", "right foot forward", "feet together"],
        "run": ["left foot extended", "airborne", "right foot extended", "airborne"],
        "attack": ["wind up", "mid swing", "follow through", "recovery"],
        "jump": ["crouch", "launch", "apex", "falling"],
        "die": ["hit reaction", "falling", "on ground", "faded"],
    }

    def __init__(self, api_key: str, output_dir: str = "./output"):
        self.api_key = api_key
        self.output_dir = output_dir
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent"
        self.vision_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        os.makedirs(output_dir, exist_ok=True)

    def analyze_segmentation(self, raw_image: Image.Image, config: SpriteConfig) -> dict:
        """Use LLM vision to analyze raw output and guide segmentation"""
        print("        Using vision to analyze frame layout...")

        buf = BytesIO()
        raw_image.save(buf, format='PNG')
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        prompt = f"""Analyze this sprite sheet image for segmentation.

I need to extract {config.frame_count} individual character frames from this image.

Please respond in JSON format:
{{
  "total_sprites_visible": <number of distinct character sprites you can see>,
  "rows": <number of rows of sprites (1 or 2)>,
  "has_magenta_separators": true/false,
  "sprites_well_separated": true/false (clear gaps between sprites?),
  "any_clipping": true/false (any sprites cut off at edges?),
  "layout_description": "brief description of the layout",
  "recommended_approach": "separators" or "contours" or "even_split"
}}

Be precise about counting the sprites."""

        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        payload = {
            "contents": [{
                "parts": [
                    {"inlineData": {"mimeType": "image/png", "data": img_b64}},
                    {"text": prompt}
                ]
            }]
        }

        try:
            response = requests.post(self.vision_url, headers=headers, json=payload)
            data = response.json()

            if 'candidates' in data:
                text = data['candidates'][0]['content']['parts'][0].get('text', '')
                import re
                json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                    print(f"        Sprites found: {analysis.get('total_sprites_visible', '?')}, "
                          f"Rows: {analysis.get('rows', '?')}, "
                          f"Separators: {analysis.get('has_magenta_separators', '?')}")
                    return analysis

            return {"recommended_approach": "even_split"}

        except Exception as e:
            print(f"        Vision analysis failed: {e}")
            return {"recommended_approach": "even_split"}

    def assess_quality(self, sprite_sheet: Image.Image, config: SpriteConfig) -> dict:
        """Use LLM vision to assess sprite sheet quality"""
        print("        Running LLM quality assessment...")

        # Encode image
        buf = BytesIO()
        sprite_sheet.save(buf, format='PNG')
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        prompt = f"""Analyze this sprite sheet for a {config.animation} animation.

Check for these issues and respond in JSON format:
{{
  "quality_score": 1-10,
  "issues": ["list of specific problems found"],
  "frame_count_correct": true/false (expected {config.frame_count} frames),
  "character_consistent": true/false (same character across frames),
  "animation_smooth": true/false (poses flow naturally),
  "has_clipping": true/false (any sprites cut off at edges),
  "recommendation": "pass" or "regenerate"
}}

Be strict - only score 8+ if the sprite sheet is production-ready for a game."""

        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        payload = {
            "contents": [{
                "parts": [
                    {"inlineData": {"mimeType": "image/png", "data": img_b64}},
                    {"text": prompt}
                ]
            }]
        }

        try:
            response = requests.post(self.vision_url, headers=headers, json=payload)
            data = response.json()

            if 'candidates' in data:
                text = data['candidates'][0]['content']['parts'][0].get('text', '')
                # Extract JSON from response
                import re
                json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
                if json_match:
                    assessment = json.loads(json_match.group())
                    return assessment

            return {"quality_score": 5, "recommendation": "unknown", "issues": ["Could not parse assessment"]}

        except Exception as e:
            print(f"        Quality assessment failed: {e}")
            return {"quality_score": 5, "recommendation": "unknown", "issues": [str(e)]}

    def create_template(self, config: SpriteConfig) -> Image.Image:
        """Create a sprite sheet template with STRONG magenta separators"""
        width = config.template_width
        height = config.template_height

        # White background
        img = Image.new('RGB', (width, height), (255, 255, 255))

        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)

        # Draw THICK magenta separator lines between frames
        # These will be preserved by the AI and make segmentation trivial
        separator_width = 4
        for i in range(1, config.frame_count):
            x = i * config.frame_size
            draw.rectangle(
                [x - separator_width//2, 0, x + separator_width//2, height],
                fill=(255, 0, 255)  # Magenta
            )

        return img

    def build_prompt(self, config: SpriteConfig) -> str:
        """Build the generation prompt"""
        frame_descs = self.FRAME_DESCRIPTIONS.get(config.animation, ["frame"] * config.frame_count)

        # Ensure we have enough descriptions
        while len(frame_descs) < config.frame_count:
            frame_descs = frame_descs + frame_descs
        frame_descs = frame_descs[:config.frame_count]

        frame_details = ", ".join([f"Frame {i+1}: {desc}" for i, desc in enumerate(frame_descs)])

        prompt = f"""Game Asset. Pixel art sprite sheet with {config.frame_count} SEPARATE frames.

CRITICAL: The template image has MAGENTA (#FF00FF) vertical lines separating the frames.
You MUST keep these magenta separator lines in your output.
Each character MUST stay INSIDE their frame - do NOT cross the magenta lines.

Character: {config.character}

Animation: {config.animation} animation
{frame_details}

Technical requirements:
- {config.frame_count} frames in a SINGLE HORIZONTAL ROW
- KEEP the magenta (#FF00FF) separator lines between frames
- WHITE background within each frame
- Each sprite stays INSIDE its frame boundary
- Side view, facing RIGHT
- Consistent character design across ALL frames
- Clear pixel art style"""

        return prompt

    def call_api(self, template: Image.Image, prompt: str, reference: Image.Image = None) -> Image.Image:
        """Call Gemini API with template, prompt, and optional reference image"""
        # Encode template
        buf = BytesIO()
        template.save(buf, format='PNG')
        template_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        # Build parts list
        parts = [
            {"inlineData": {"mimeType": "image/png", "data": template_b64}},
        ]

        # Add reference image if provided (for character consistency)
        if reference:
            ref_buf = BytesIO()
            reference.save(ref_buf, format='PNG')
            ref_b64 = base64.b64encode(ref_buf.getvalue()).decode('utf-8')
            parts.append({"inlineData": {"mimeType": "image/png", "data": ref_b64}})
            prompt = f"IMPORTANT: Match the character design from the reference image exactly.\n\n{prompt}"

        parts.append({"text": prompt})

        payload = {
            "contents": [{
                "parts": parts
            }]
        }

        print("  Calling Gemini API...")
        response = requests.post(self.api_url, headers=headers, json=payload)
        data = response.json()

        if 'candidates' not in data:
            raise Exception(f"API error: {json.dumps(data, indent=2)[:500]}")

        # Extract image from response
        for part in data['candidates'][0]['content']['parts']:
            if 'inlineData' in part:
                img_data = base64.b64decode(part['inlineData']['data'])
                return Image.open(BytesIO(img_data))

        raise Exception("No image in API response")

    def resize_to_target(self, img: Image.Image, config: SpriteConfig) -> Image.Image:
        """Resize generated image to target dimensions, handling aspect ratio mismatches"""
        target_w = config.template_width
        target_h = config.template_height
        target_ratio = target_w / target_h  # e.g., 4.0 for 256x64

        img_ratio = img.width / img.height

        # If aspect ratios are very different, the API ignored our template
        if abs(img_ratio - target_ratio) > 0.5:
            print(f"        Warning: API returned {img.width}x{img.height} (ratio {img_ratio:.1f}), expected ratio {target_ratio:.1f}")
            print(f"        Attempting intelligent crop...")

            # Try to detect if the image has horizontal sprite layout
            # by looking for content regions
            arr = np.array(img.convert('RGB'))
            gray = np.mean(arr, axis=2)

            # Find rows/cols with significant content (not white)
            col_activity = np.mean(gray < 240, axis=0)  # % of non-white pixels per column
            row_activity = np.mean(gray < 240, axis=1)  # % of non-white pixels per row

            # Find content bounds
            active_cols = np.where(col_activity > 0.01)[0]
            active_rows = np.where(row_activity > 0.01)[0]

            if len(active_cols) > 0 and len(active_rows) > 0:
                left = max(0, active_cols[0] - 5)
                right = min(img.width, active_cols[-1] + 5)
                top = max(0, active_rows[0] - 5)
                bottom = min(img.height, active_rows[-1] + 5)

                # Crop to content
                img = img.crop((left, top, right, bottom))
                print(f"        Cropped to content: {img.width}x{img.height}")

        # Now resize to target
        resized = img.resize((target_w, target_h), Image.Resampling.NEAREST)
        return resized

    def remove_background(self, img: Image.Image, config: SpriteConfig, tolerance: int = 30, use_ml: bool = True) -> Image.Image:
        """Remove background using ML (rembg) or chromakey fallback"""

        if use_ml and HAS_REMBG:
            print("        Using ML (rembg) for background removal...")
            # rembg works best with RGB input
            if img.mode != 'RGB':
                rgb_img = img.convert('RGB')
            else:
                rgb_img = img

            # Apply ML-based background removal
            result = rembg_remove(rgb_img)
            return result

        # Fallback: chromakey
        print("        Using chromakey for background removal...")
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        arr = np.array(img)

        # Target color to remove
        if config.background == "white":
            target = np.array([255, 255, 255])
        else:  # magenta
            target = np.array([255, 0, 255])

        # Calculate distance from target color
        rgb = arr[:, :, :3].astype(np.float32)
        dist = np.sqrt(np.sum((rgb - target) ** 2, axis=2))

        # Create alpha mask
        alpha = np.where(dist < tolerance, 0, 255).astype(np.uint8)

        # Apply alpha
        arr[:, :, 3] = alpha

        return Image.fromarray(arr)

    def detect_magenta_separators(self, img: Image.Image) -> List[Tuple[int, int]]:
        """Detect magenta separator lines in the image"""
        arr = np.array(img.convert('RGB'))
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

        # Magenta: high R, low G, high B
        magenta_mask = (r > 200) & (g < 100) & (b > 200)
        magenta_cols = np.mean(magenta_mask, axis=0) > 0.3

        if not np.any(magenta_cols):
            return []

        # Group consecutive magenta columns
        positions = np.where(magenta_cols)[0]
        separators = []
        start = positions[0]
        prev = positions[0]

        for pos in positions[1:]:
            if pos - prev > 5:
                separators.append((int(start), int(prev)))
                start = pos
            prev = pos
        separators.append((int(start), int(prev)))

        return separators

    def detect_row_split(self, img: Image.Image) -> int:
        """Detect if image has multiple rows, return y position of split"""
        arr = np.array(img.convert('RGB'))
        gray = np.mean(arr, axis=2)
        row_content = np.mean(gray < 250, axis=1)

        # Look for gap in middle third
        mid_start = img.height // 3
        mid_end = 2 * img.height // 3
        mid_region = row_content[mid_start:mid_end]

        if len(mid_region) > 0 and np.min(mid_region) < 0.1:
            return np.argmin(mid_region) + mid_start

        return img.height  # No split found

    def detect_frame_boundaries_cv(self, img: Image.Image, expected_frames: int) -> List[Tuple[int, int, int, int]]:
        """Detect sprite bounding boxes using OpenCV contour detection"""
        arr = np.array(img)

        # Create binary mask from alpha channel or color
        if img.mode == 'RGBA':
            mask = (arr[:, :, 3] > 10).astype(np.uint8) * 255
        else:
            # Convert to grayscale and threshold
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            # Assume white background
            mask = (gray < 250).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get bounding boxes and filter small noise
        min_area = (img.width * img.height) / (expected_frames * 20)  # Min 5% of expected frame area
        bboxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area > min_area:
                bboxes.append((x, y, w, h))

        # Sort by x position (left to right)
        bboxes.sort(key=lambda b: b[0])

        # Merge overlapping or adjacent boxes
        merged = []
        for bbox in bboxes:
            x, y, w, h = bbox
            if merged and x < merged[-1][0] + merged[-1][2] + 10:  # Adjacent or overlapping
                # Merge with previous
                px, py, pw, ph = merged[-1]
                new_x = min(px, x)
                new_y = min(py, y)
                new_w = max(px + pw, x + w) - new_x
                new_h = max(py + ph, y + h) - new_y
                merged[-1] = (new_x, new_y, new_w, new_h)
            else:
                merged.append(bbox)

        print(f"        Detected {len(merged)} sprite regions (expected {expected_frames})")

        return merged

    def detect_frame_boundaries(self, img: Image.Image, expected_frames: int) -> List[Tuple[int, int]]:
        """Detect actual frame boundaries - returns (x_start, x_end) tuples"""
        # Try CV-based detection first
        bboxes = self.detect_frame_boundaries_cv(img, expected_frames)

        if len(bboxes) == expected_frames:
            # Convert bboxes to x boundaries
            return [(b[0], b[0] + b[2]) for b in bboxes]

        # Fallback: column-based detection
        arr = np.array(img)

        if img.mode == 'RGBA':
            content_mask = arr[:, :, 3] > 0
        else:
            rgb = arr[:, :, :3]
            diff = np.max(np.abs(rgb.astype(np.int16) - 255), axis=2)
            content_mask = diff > 30

        col_has_content = np.any(content_mask, axis=0)

        boundaries = []
        in_content = False
        content_start = 0

        for x, has_content in enumerate(col_has_content):
            if has_content and not in_content:
                content_start = x
                in_content = True
            elif not has_content and in_content:
                boundaries.append((content_start, x))
                in_content = False

        if in_content:
            boundaries.append((content_start, len(col_has_content)))

        if len(boundaries) == expected_frames:
            return boundaries

        # Final fallback: even division
        print(f"        Fallback: using even division")
        frame_width = img.width // expected_frames
        return [(i * frame_width, (i + 1) * frame_width) for i in range(expected_frames)]

    def slice_frames(self, img: Image.Image, config: SpriteConfig) -> List[Image.Image]:
        """Slice sprite sheet into individual frames with content-aware detection"""
        # Try to detect actual frame boundaries
        boundaries = self.detect_frame_boundaries(img, config.frame_count)

        frames = []
        for i, (x_start, x_end) in enumerate(boundaries):
            # Extract the content region
            content = img.crop((x_start, 0, x_end, img.height))

            # Create a frame of the target size, centered
            frame = Image.new('RGBA', (config.frame_size, config.frame_size), (0, 0, 0, 0))

            # Scale content to fit if needed, maintaining aspect ratio
            content_w, content_h = content.size
            scale = min(config.frame_size / content_w, config.frame_size / content_h)

            if scale < 1:
                new_w = int(content_w * scale)
                new_h = int(content_h * scale)
                content = content.resize((new_w, new_h), Image.Resampling.NEAREST)

            # Center in frame
            paste_x = (config.frame_size - content.width) // 2
            paste_y = (config.frame_size - content.height) // 2
            frame.paste(content, (paste_x, paste_y), content if content.mode == 'RGBA' else None)

            frames.append(frame)

        return frames

    def save_outputs(self,
                     sprite_sheet: Image.Image,
                     frames: List[Image.Image],
                     config: SpriteConfig,
                     name: str) -> dict:
        """Save all outputs and generate metadata"""

        # Create output subdirectory
        output_path = os.path.join(self.output_dir, name)
        os.makedirs(output_path, exist_ok=True)

        # Save sprite sheet
        sheet_path = os.path.join(output_path, f"{name}_sheet.png")
        sprite_sheet.save(sheet_path)

        # Save individual frames
        frame_paths = []
        for i, frame in enumerate(frames):
            frame_path = os.path.join(output_path, f"{name}_frame_{i+1}.png")
            frame.save(frame_path)
            frame_paths.append(frame_path)

        # Generate metadata
        metadata = {
            "name": name,
            "character": config.character,
            "animation": config.animation,
            "frame_count": config.frame_count,
            "frame_size": config.frame_size,
            "sprite_sheet": sheet_path,
            "frames": [
                {
                    "index": i,
                    "path": path,
                    "x": i * config.frame_size,
                    "y": 0,
                    "width": config.frame_size,
                    "height": config.frame_size
                }
                for i, path in enumerate(frame_paths)
            ]
        }

        # Save metadata
        meta_path = os.path.join(output_path, f"{name}_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def generate_reference(self, character: str, size: int = 128) -> Image.Image:
        """Generate a character reference sheet for consistency"""
        print(f"\n=== Generating Reference Sheet ===")
        print(f"  Character: {character}")

        prompt = f"""Character reference sheet for: {character}

Create a single character shown from the SIDE VIEW (profile).
The character should be in a neutral standing pose.
Pixel art style, {size}x{size} pixels.
WHITE background.
This will be used as a reference for animation consistency."""

        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }

        print("  Calling Gemini API for reference...")
        response = requests.post(self.api_url, headers=headers, json=payload)
        data = response.json()

        if 'candidates' not in data:
            raise Exception(f"API error: {json.dumps(data, indent=2)[:500]}")

        for part in data['candidates'][0]['content']['parts']:
            if 'inlineData' in part:
                img_data = base64.b64decode(part['inlineData']['data'])
                img = Image.open(BytesIO(img_data))
                # Resize to target size
                img = img.resize((size, size), Image.Resampling.NEAREST)
                print(f"  Reference generated: {img.width}x{img.height}")
                return img

        raise Exception("No image in API response")

    def generate(self, config: SpriteConfig, name: str, reference: Image.Image = None,
                 max_attempts: int = 3, min_quality: int = 6) -> dict:
        """Main generation pipeline with quality assessment and auto-retry"""

        for attempt in range(max_attempts):
            print(f"\n=== Generating: {name} (attempt {attempt + 1}/{max_attempts}) ===")
            print(f"  Character: {config.character}")
            print(f"  Animation: {config.animation} ({config.frame_count} frames)")
            print(f"  Frame size: {config.frame_size}x{config.frame_size}")
            if reference:
                print(f"  Using reference image for consistency")

            # Step 1: Create template
            print("  [1/6] Creating template...")
            template = self.create_template(config)

            # Step 2: Build prompt
            print("  [2/6] Building prompt...")
            prompt = self.build_prompt(config)

            # Step 3: Call API
            print("  [3/6] Generating with Nano Banana...")
            raw_output = self.call_api(template, prompt, reference)
            print(f"        Raw output: {raw_output.width}x{raw_output.height}")

            # Step 4: Smart segmentation using magenta separators
            print("  [4/6] Segmenting frames...")

            # Use vision to analyze the layout first
            seg_analysis = self.analyze_segmentation(raw_output, config)

            # Detect magenta separators
            separators = self.detect_magenta_separators(raw_output)

            # Use vision recommendation if no separators found
            if not separators and seg_analysis.get('recommended_approach') == 'contours':
                print("        Vision recommends contour-based detection")

            if separators:
                print(f"        Found {len(separators)} magenta separators")
                # Calculate frame boundaries from separators
                frame_bounds = []
                prev_end = 0
                for sep_start, sep_end in separators:
                    if sep_start > prev_end + 10:
                        frame_bounds.append((prev_end, sep_start))
                    prev_end = sep_end + 1
                if prev_end < raw_output.width - 10:
                    frame_bounds.append((prev_end, raw_output.width))

                # Detect row split
                row_split = self.detect_row_split(raw_output)
                if row_split < raw_output.height:
                    print(f"        Detected row split at y={row_split}, using top row")

                # Extract frames with aspect-ratio preservation
                frames = []
                for x_start, x_end in frame_bounds[:config.frame_count]:
                    frame_region = raw_output.crop((x_start, 0, x_end, row_split))
                    frame_clean = self.remove_background(frame_region, config)

                    # Find actual sprite content bounds
                    frame_arr = np.array(frame_clean)
                    if frame_clean.mode == 'RGBA':
                        alpha = frame_arr[:, :, 3]
                        rows_with_content = np.any(alpha > 10, axis=1)
                        cols_with_content = np.any(alpha > 10, axis=0)

                        if rows_with_content.any() and cols_with_content.any():
                            top = int(np.argmax(rows_with_content))
                            bottom = int(len(rows_with_content) - np.argmax(rows_with_content[::-1]))
                            left = int(np.argmax(cols_with_content))
                            right = int(len(cols_with_content) - np.argmax(cols_with_content[::-1]))

                            # Crop to content
                            sprite = frame_clean.crop((left, top, right, bottom))

                            # Scale to fit target while preserving aspect ratio
                            scale = min(config.frame_size / sprite.width,
                                       config.frame_size / sprite.height) * 0.9
                            new_w = max(1, int(sprite.width * scale))
                            new_h = max(1, int(sprite.height * scale))
                            sprite_scaled = sprite.resize((new_w, new_h), Image.Resampling.NEAREST)

                            # Center in target frame
                            final_frame = Image.new('RGBA', (config.frame_size, config.frame_size), (0, 0, 0, 0))
                            paste_x = (config.frame_size - new_w) // 2
                            paste_y = (config.frame_size - new_h) // 2
                            final_frame.paste(sprite_scaled, (paste_x, paste_y), sprite_scaled)
                            frames.append(final_frame)
                            continue

                    # Fallback: simple resize (shouldn't normally reach here)
                    frame_resized = frame_clean.resize(
                        (config.frame_size, config.frame_size),
                        Image.Resampling.NEAREST
                    )
                    frames.append(frame_resized)
            else:
                # Fallback to original approach
                print("        No separators found, using fallback segmentation")
                resized = self.resize_to_target(raw_output, config)
                processed = self.remove_background(resized, config)
                frames = self.slice_frames(processed, config)

            # Create sprite sheet from frames
            sheet = Image.new('RGBA', (config.template_width, config.template_height), (0, 0, 0, 0))
            for i, frame in enumerate(frames):
                sheet.paste(frame, (i * config.frame_size, 0))

            # Step 5: Quality assessment
            print("  [5/6] Quality assessment...")
            assessment = self.assess_quality(sheet, config)
            quality = assessment.get('quality_score', 5)
            recommendation = assessment.get('recommendation', 'unknown')
            issues = assessment.get('issues', [])

            print(f"        Quality score: {quality}/10")
            print(f"        Recommendation: {recommendation}")
            if issues:
                print(f"        Issues: {', '.join(issues[:3])}")

            # Check if quality is acceptable
            if quality >= min_quality or attempt == max_attempts - 1:
                # Step 6: Save outputs
                print("  [6/6] Saving outputs...")
                metadata = self.save_outputs(sheet, frames, config, name)
                metadata['quality_assessment'] = assessment
                metadata['generation_attempts'] = attempt + 1

                print(f"  Done! Output: {self.output_dir}/{name}/")
                return metadata
            else:
                print(f"        Quality too low ({quality} < {min_quality}), regenerating...")

        return metadata  # Return last attempt even if quality is low

    def generate_character(self, character: str, name: str,
                           animations: List[str] = None,
                           frame_count: int = 4,
                           frame_size: int = 64,
                           background: str = "white") -> dict:
        """Generate a complete character with multiple animations using a reference for consistency"""
        if animations is None:
            animations = ["idle", "walk", "attack", "die"]

        print(f"\n{'='*50}")
        print(f"GENERATING COMPLETE CHARACTER: {name}")
        print(f"{'='*50}")

        # Step 1: Generate reference sheet
        reference = self.generate_reference(character, frame_size * 2)

        # Save reference
        ref_path = os.path.join(self.output_dir, name, f"{name}_reference.png")
        os.makedirs(os.path.dirname(ref_path), exist_ok=True)
        reference.save(ref_path)
        print(f"  Saved reference: {ref_path}")

        # Step 2: Generate each animation using the reference
        all_metadata = {
            "character_name": name,
            "character_description": character,
            "reference": ref_path,
            "animations": {}
        }

        for anim in animations:
            config = SpriteConfig(
                character=character,
                animation=anim,
                frame_count=frame_count,
                frame_size=frame_size,
                background=background
            )
            anim_name = f"{name}_{anim}"
            metadata = self.generate(config, anim_name, reference)
            all_metadata["animations"][anim] = metadata

        # Save combined metadata
        meta_path = os.path.join(self.output_dir, name, f"{name}_character.json")
        with open(meta_path, 'w') as f:
            json.dump(all_metadata, f, indent=2)

        print(f"\n{'='*50}")
        print(f"CHARACTER COMPLETE: {name}")
        print(f"Output: {self.output_dir}/{name}/")
        print(f"{'='*50}")

        return all_metadata


def main():
    parser = argparse.ArgumentParser(description="Generate game sprite sheets using AI")
    parser.add_argument("--api-key", required=True, help="Google Gemini API key")
    parser.add_argument("--character", required=True, help="Character description")
    parser.add_argument("--name", required=True, help="Output name for the sprite")
    parser.add_argument("--output", default="./output", help="Output directory")

    # Mode selection
    parser.add_argument("--full-character", action="store_true",
                        help="Generate complete character with all animations")
    parser.add_argument("--animations", nargs="+",
                        default=["idle", "walk", "attack", "die"],
                        help="Animations to generate (for --full-character)")

    # Single animation mode
    parser.add_argument("--animation", default="idle",
                        choices=["idle", "walk", "run", "attack", "jump", "die"],
                        help="Animation type (for single animation mode)")

    # Common options
    parser.add_argument("--frames", type=int, default=4, help="Number of frames")
    parser.add_argument("--size", type=int, default=64, help="Frame size in pixels")
    parser.add_argument("--background", default="white", choices=["white", "magenta"],
                        help="Background color for chromakey")

    args = parser.parse_args()

    generator = SpriteGenerator(args.api_key, args.output)

    if args.full_character:
        # Generate complete character with reference-based consistency
        metadata = generator.generate_character(
            character=args.character,
            name=args.name,
            animations=args.animations,
            frame_count=args.frames,
            frame_size=args.size,
            background=args.background
        )
    else:
        # Single animation mode
        config = SpriteConfig(
            character=args.character,
            animation=args.animation,
            frame_count=args.frames,
            frame_size=args.size,
            background=args.background
        )
        metadata = generator.generate(config, args.name)

    print("\n=== Final Metadata ===")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
