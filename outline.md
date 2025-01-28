# Essential Files for PafView Paper

## Core Implementation Files

These files demonstrate the key technical innovations:

1. `src/cigar.rs` - Core CIGAR string handling and zero-copy parsing
2. `src/cigar/implicit.rs` - Implicit CIGAR representation for memory efficiency
3. `src/render/exact.rs` - Exact rendering algorithm implementation
4. `src/render/color.rs` - Color scheme management and visualization
5. `src/pixels.rs` - GPU-accelerated pixel buffer management
6. `src/annotations.rs` - Annotation system for interactive features

## Performance Critical Components

1. `src/cigar/memmap.rs` - Memory-mapped file handling
2. `src/paf.rs` - PAF format parsing and indexing
3. `src/sequences.rs` - Efficient sequence data management

## Key Features to Highlight

1. Memory efficiency through:
   - Memory-mapped file access
   - Zero-copy CIGAR parsing
   - Efficient sequence name indexing

2. Rendering performance via:
   - GPU acceleration
   - Quadtree-based viewport culling
   - Adaptive tile-based rendering
   - Dynamic level-of-detail

3. Interactive features:
   - Real-time navigation
   - Zoom capabilities
   - Distance measurements
   - Alignment information display

## Benchmark Data Needed

1. Memory usage comparisons:
   - Loading various genome sizes
   - With/without CIGAR strings
   - Peak memory during visualization

2. Rendering performance:
   - Frame rates at different zoom levels
   - Time to initial display
   - Interaction responsiveness

3. Comparison metrics:
   - Against existing tools (IGV, etc.)
   - Memory footprint
   - Load times
   - Interactive performance

## Example Datasets

1. Human-to-human whole genome alignment
2. Multi-strain bacterial genome comparison
3. Large structural variant examples
4. Complex rearrangement scenarios

## Figures Needed

1. System architecture diagram
2. Memory usage plots
3. Performance benchmarks
4. Interface screenshots:
   - Whole genome view
   - Zoomed alignment detail
   - Feature highlighting
   - Interactive tools

## Supplementary Materials

1. Detailed installation instructions
2. Example workflow documentation
3. Benchmark scripts and data
4. Test datasets
