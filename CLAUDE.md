# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Solver for the **n-cop Shannon Switching game** on complete graphs. Cop and robber alternate adding edges to their respective graphs. The robber wins by connecting vertices 0 and 1; the cop wins by preventing this. In the n-cop variant, the cop can add up to n edges per turn.

**`ncop2-rs/` is the fastest solver** (Rust, nightly). `ncop-rs/` is the original Rust port. `ncop3-rs/` is an alternative graph-canonical approach. `ncop.cpp` is the oldest C++ prototype.

## Known Results

| Cops (n) | Cop wins on… | Robber wins on… | Time | How we know |
|-----------|-------------|-----------------|------|-------------|
| 2 | K1–K6 | **K7** | fast | C++ prototype |
| 3 | K1–K8 | **K9** | 4.9s | ncop2-rs (single-edge cop sub-steps) |
| 4 | K1–**K11** | ? (K12 untested) | ~615s | ncop2-rs, 128M TT, 621M nodes |

**Key findings:**
- Single-edge cop sub-steps: 15x speedup (K9 3-cop: 72s → 4.9s)
- Parallel cop sub-steps *hurt* performance — cop is an OR node, sequential-first is optimal
- K11 4-cop: Cop wins in ~10 min with 128M TT (local Apple Silicon faster than 32-core remote Haswell)
- TT sizing matters: 16M→128M gave 4.8x fewer nodes on K11; 128M→1G gave negligible improvement
- Local machine (Apple Silicon) beats remote (32-core Haswell) because cop sub-steps are sequential — extra cores mostly idle

## Build & Run

### ncop2-rs (Rust nightly — fastest solver)

Requires **nightly** Rust (`#![feature(generic_const_exprs)]`).

```bash
cd ncop2-rs
rustup override set nightly          # if not already
RUSTFLAGS="-C target-cpu=native" cargo build --release
./target/release/ncop2-rs
```

CLI: `ncop2-rs SIZE COPS [PAR_DEPTH]` (dispatches to monomorphized versions for sizes 3–16, cops 1–6). Falls back to compile-time defaults if no args.

Configure defaults by editing `src/main.rs`:
```rust
const SIZE: u8 = 9;        // K_9
const NCOP: usize = 4;     // 4 cops
const TT_SIZE_LOG2: usize = 27; // 128M entries (16 bytes each → 2GB cop + 512MB robber)
```
`PAR_MAX_DEPTH` is auto-set by `build.rs` based on available CPUs.

Run tests:
```bash
cd ncop2-rs && RUSTFLAGS="-C target-cpu=native" cargo +nightly test --release
```

### ncop-rs (Rust — original port)

```bash
cd ncop-rs
RUSTFLAGS="-C target-cpu=native" cargo build --release
./target/release/ncop-rs
```

Run tests:
```bash
cd ncop-rs && cargo test
```

Configure by editing the constants at the top of `ncop-rs/src/main.rs`:
```rust
const SIZE: u8 = 9;        // graph size (K_n)
const NCOP: usize = 3;     // number of cops
const CACHE_DEPTH: usize = 10;   // transposition table depth
const PARALLEL_DEPTH: usize = 0; // rayon parallelism kicks in at this depth
```

### ncop3-rs (Rust — graph-canonical DAG approach)

```bash
cd ncop3-rs
cargo build --release                              # single-threaded
cargo build --release --features parallel           # with Rayon
```

Uses `nauty-pet` for graph canonicalization and `petgraph` for graph representation. Fundamentally different algorithm: backward induction over a DAG of canonical game states.

### ncop.cpp (C++ prototype)

```bash
g++ -O3 -march=native -ffast-math -funroll-loops -flto ncop.cpp -o ncop
./ncop <num_cops>                        # eval mode: sizes 1–8
./ncop <num_cops> <graph_size> cop       # play mode: human as cop
```

### Web UI

Interactive 2-player game at **https://ncop.meghanto.me/** (or open `webui/index.html` locally).
Features: AI cop/robber, move browser, force-directed layouts, GIF/video export.

## Architecture

### ncop2-rs/src/main.rs (fastest — ~1470 lines)

**Graph representation**: AVX2 `__m256i` (256-bit), a 16×16 adjacency matrix as sixteen `u16` rows packed into one SIMD register. Same as ncop-rs but with additional optimizations:
- **Connected component contraction**: Before caching, robber's connected components are compressed into single nodes. The cache key is the adjacency matrix of unclaimed edges between/within contracted components (`contract_and_hash`).
- **Lockless transposition table**: Fixed-size array with splitmix64 hashing, 128-bit XOR trick for torn-read detection. Separate cop TT (full size) and robber TT (1/4 size).
- **Single-edge cop sub-steps**: `cop_step(picks_left)` makes sequential single-edge moves instead of picking N edges at once. TT caches intermediate states via `hash_with_picks(hash, picks_left)`. This dramatically improves TT hit rates — different N-edge combinations share intermediate states.
- **Component-pair dedup**: For single edges, two edges connecting the same component pair produce identical contracted states. `[[bool; MAX_COMP]; MAX_COMP]` stack array provides O(1) zero-allocation dedup.
- **Rayon parallelism**: At depths < `PAR_MAX_DEPTH`, robber nodes use sequential-first + `par_iter().find_map_any()` verification. Cop nodes stay sequential (OR node — parallel speculation wastes work).

### ncop-rs/src/main.rs (original — 601 lines)

Uses AVX2 `__m256i` for adjacency matrices, pre-computed `STAR_GRAPHS[16]` for fast edge operations, and SIMD BFS for connectivity checking.

### ncop3-rs/src/main.rs (canonical DAG — 486 lines)

Uses `nauty-pet` for graph isomorphism canonicalization and `petgraph` for representation. Solves by backward induction over a DAG of canonical game states rather than forward minimax. Optional parallelism via `--features parallel` (swaps `Rc` for `Arc`, `Weak` for `sync::Weak`). Uses `weak-table` for memory-efficient canonical state deduplication.

### ncop.cpp (prototype — 457 lines)

- No transposition table, no parallelism
- NCOP hardcoded to 1/2/3 via nested loops
- Eval mode iterates sizes 1–8, stops at first robber win
- Interactive play mode (human as cop vs AI robber)

## Optimization History

Starting from the original `ncop2-rs` (single-threaded, no contraction, basic TT):

| Change | K8 3-cop time | Speedup |
|--------|--------------|---------|
| Baseline (ncop-rs) | 82s | — |
| + Component contraction + lockless TT + Rayon | 0.5s | **160x** |
| K9 3-cop (previously intractable) | 72s | ∞ |
| Single-edge cop sub-steps | 4.9s | **15x** vs 72s |

Work-stealing (parallelism at multiple depths) was tried but regressed performance due to `par_bridge()` overhead. Root-only parallelism remains the default.

Parallel cop sub-steps were tested (both all-at-once and batched `nproc/COPS`): they generated ~60% more nodes (OR node wastes parallel work on speculative branches). Sequential cop remains optimal.

TT sizing: 128M entries (TT_SIZE_LOG2=27) is the sweet spot for K11. Going to 1G (LOG2=30, 20GB RAM) gave negligible improvement. Going below 128M causes severe eviction (16M→128M = 4.8x fewer nodes).

## Key Constraints

- All solvers require Rust nightly (edition 2024)
- `ncop-rs` and `ncop2-rs` require AVX2 (`target-cpu=native`)
- Graph sizes capped at 16 (256-bit bitboard = 16 × u16)
- Search is exhaustive minimax — runtime grows exponentially

### Running on Remote Server
We have access to a 32-core Windows remote machine (Haswell), but **local Apple Silicon is faster** for this workload because cop sub-steps are sequential — the extra cores mostly sit idle. Remote is only useful if effective parallelism is added to the cop search.
- **SSH Target**: `meghanto@172.23.44.77`
- **Copying over code**: `scp ncop2-rs/src/main.rs meghanto@172.23.44.77:'C:\Users\meghanto\ncop2-rs\src\main.rs'`
- **Building**: `ssh meghanto@172.23.44.77 'cd C:\Users\meghanto\ncop2-rs && set RUSTFLAGS=-C target-cpu=native && cargo +nightly build --release'`
- **Running**: `ssh meghanto@172.23.44.77 'cd C:\Users\meghanto\ncop2-rs && target\release\ncop2-rs.exe 11 4'`
