# Bananabread

AI-powered sprite sheet generator using Google Gemini (Nano Banana) for 2D game asset creation.

## Features

- **Template-based generation** — Magenta separators ensure reliable frame segmentation
- **ML background removal** — Clean transparency using rembg (U²-Net)
- **Vision-guided segmentation** — LLM analyzes layout before extraction
- **Quality assessment** — Automatic scoring and retry on low quality
- **Aspect ratio preservation** — Content-aware extraction prevents distortion
- **Multi-animation support** — Generate complete characters with consistent design

## Installation

```bash
# Clone the repo
git clone <repo-url>
cd bananabread

# Install with pip
pip install .

# Or with uv (faster)
uv sync
```

## Usage

### Single Animation

```bash
python sprite_generator.py \
  --api-key "YOUR_GEMINI_API_KEY" \
  --character "pixel art knight with sword and shield" \
  --animation walk \
  --frames 4 \
  --size 64 \
  --name "knight_walk"
```

### Complete Character (Multiple Animations)

```bash
python sprite_generator.py \
  --api-key "YOUR_GEMINI_API_KEY" \
  --character "pixel art wizard with blue robe and staff" \
  --name "wizard" \
  --full-character \
  --animations idle walk attack die
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--api-key` | Google Gemini API key | Required |
| `--character` | Character description | Required |
| `--name` | Output name | Required |
| `--animation` | Animation type (idle/walk/run/attack/jump/die) | idle |
| `--frames` | Number of frames | 4 |
| `--size` | Frame size in pixels | 64 |
| `--full-character` | Generate all animations with reference | False |
| `--animations` | List of animations for full character | idle walk attack die |
| `--output` | Output directory | ./output |

## Output

```
output/
└── knight_walk/
    ├── knight_walk_sheet.png    # Combined sprite sheet
    ├── knight_walk_frame_1.png  # Individual frames
    ├── knight_walk_frame_2.png
    ├── knight_walk_frame_3.png
    ├── knight_walk_frame_4.png
    └── knight_walk_meta.json    # Metadata with positions
```

### Metadata Format

```json
{
  "name": "knight_walk",
  "animation": "walk",
  "frame_count": 4,
  "frame_size": 64,
  "frames": [
    {"index": 0, "x": 0, "y": 0, "width": 64, "height": 64},
    {"index": 1, "x": 64, "y": 0, "width": 64, "height": 64}
  ],
  "quality_assessment": {
    "quality_score": 8,
    "character_consistent": true,
    "has_clipping": false
  }
}
```

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│  1. TEMPLATE        Create image with magenta separator lines   │
│  2. PROMPT          Build animation-specific generation prompt  │
│  3. NANO BANANA     Generate sprites via Gemini Image API       │
│  4. VISION ANALYSIS LLM examines layout, counts sprites         │
│  5. SEGMENTATION    Detect separators → extract frames          │
│  6. POST-PROCESS    ML background removal, preserve aspect ratio│
│  7. QUALITY CHECK   LLM scores output, retry if needed          │
│  8. OUTPUT          Save PNGs + JSON metadata                   │
└─────────────────────────────────────────────────────────────────┘
```

## API Key

Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

Set via environment variable to avoid passing on command line:

```bash
export GEMINI_API_KEY="your-key-here"
```

## Requirements

- Python 3.9+
- Google Gemini API access
- ~500MB disk space (for rembg model download on first run)

## License

MIT
