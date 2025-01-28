# PafView

A high-performance GPU-accelerated viewer for PAF (Pairwise mApping Format) alignment files.

## Features

- Fast rendering of large-scale genomic alignments using wgpu
- Interactive visualization of alignment details
- Support for CIGAR strings and alignment metadata
- Customizable color schemes with dark mode support
- Real-time navigation and zooming capabilities

## Installation

```bash
cargo install pafview
```

Or build from source:

```bash
git clone https://github.com/your-username/pafview
cd pafview
cargo build --release
```

## Usage

Basic usage:
```bash
pafview <input.paf>
```

## Navigation Controls

- **Left Click + Drag**: Pan/scroll the view
- **Mouse Wheel**: Zoom in/out
- **PgUp/PgDn**: Quick zoom in/out
- **Arrow Keys**: Move view up/down/left/right
- **Right Click + Drag**: Box zoom selection
- **Right Click + Ctrl**: Measure distances with ruler

## Visualization Features

- Color-coded alignment operations (matches, mismatches, insertions, deletions)
- Automatic sequence name display
- Detailed alignment information on hover
- Support for forward and reverse strand alignments
- Optional fields display from PAF metadata

## File Format Support

- Standard PAF format
- PAF with CIGAR strings (cg:Z: tag)
- Additional optional fields (tags)

## Performance

PafView is designed for efficient handling of large-scale genomic alignments:

- **Memory Efficiency**:
  - Memory-mapped PAF file access for minimal memory footprint
  - BGZF index support for compressed files
  - Lazy loading of CIGAR strings and optional fields
  - Efficient bimap-based sequence name indexing

- **Fast Rendering**:
  - GPU-accelerated graphics using wgpu
  - Quadtree-based viewport culling (QBVH)
  - Adaptive tile-based rendering system
  - Efficient pixel buffer management
  - Dynamic level-of-detail rendering

- **Alignment Processing**:
  - Zero-copy CIGAR string parsing
  - Optimized coordinate space transformations
  - Efficient interval tree queries for alignment lookups
  - Fast strand-aware coordinate mapping

## Requirements

- Graphics card with Vulkan, Metal, DX12, or WebGPU support
- Rust 1.70 or later

## License

MIT License

Copyright (c) 2024 PafView Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use PafView in your research, please cite:

[Citation information to be added]

