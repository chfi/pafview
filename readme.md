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

PafView is optimized for handling large PAF files efficiently:
- GPU-accelerated rendering
- Memory-mapped file handling
- Efficient alignment indexing
- Dynamic tile-based rendering

## Requirements

- Graphics card with Vulkan, Metal, DX12, or WebGPU support
- Rust 1.70 or later

## License

[Insert License Information]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

