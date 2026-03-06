# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Solver for the **n-cop Shannon Switching game** on complete graphs. Cop and robber alternate adding edges to their respective graphs. The robber wins by connecting vertices 0 and 1; the cop wins by preventing this. In the n-cop variant, the cop can add up to n edges per turn.

**`ncop2-rs/` is the fastest solver** (Rust, nightly). `ncop-rs/` is the original Rust port. `ncop3-rs/` is an alternative graph-canonical approach. `ncop.cpp` is the oldest C++ prototype.

## Known Results

| Cops (n) | Cop wins on… | Robber wins on… | How we know |
|----------|-------------|-----------------|-------------|
| 2 | K1–K6 | **K7** | C++ prototype, fast |
| 3 | K1–K8 | **K9** | ncop2-rs, 72 seconds |
| 4 | K1–K10 (likely) | **K11?** | ncop2-rs run killed before finishing (~8 min in, 1.39B nodes) |

**The key open question: What is the smallest K where 4 cops lose?** K11 4-cop was running on a 32-core remote machine with dedup (341,055 raw root moves → 708 unique), processing ~2.8M nodes/sec. It was killed before completing.

## Build & Run

### ncop2-rs (Rust nightly — fastest solver)

Requires **nightly** Rust (`#![feature(generic_const_exprs)]`).

```bash
cd ncop2-rs
rustup override set nightly          # if not already
RUSTFLAGS="-C target-cpu=native" cargo build --release
./target/release/ncop2-rs
```

Configure by editing `src/main.rs`:
```rust
type Game = GameState<Size11, 4>;       // K_11 with 4 cops
const TT_SIZE_LOG2: usize = 24;        // 16M TT entries
```
`PAR_MAX_DEPTH` is auto-set by `build.rs` based on available CPUs.

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

### ncop2-rs/src/main.rs (fastest — 1443 lines)

**Graph representation**: `BitGraph<Size>` using the `bitvec` crate — no AVX2 dependency, trusts compiler auto-vectorization. Size is encoded as a type-level natural (`Size11 = ((), Size10)` etc.) with const generic expressions.

**Key optimizations** (achieved 160x speedup over ncop-rs on K8 3-cop):
1. **Connected component contraction**: Before caching, robber's connected components are compressed into single nodes. The cache key is the adjacency matrix of unclaimed edges between/within contracted components. This collapses huge numbers of strategically equivalent positions.
2. **Lockless transposition table**: Fixed-size array with splitmix64 hashing — no locks, no allocation overhead. Collisions cause re-evaluation (safe, just slower).
3. **Rayon parallelism**: `par_bridge()` at depths < `PAR_MAX_DEPTH` (auto-tuned by `build.rs`).
4. **Dedup pre-filter**: At the root level, canonically equivalent cop moves are deduplicated before evaluation (e.g., 341K → 708 for K11 4-cop).

**Search**: Same minimax structure as ncop-rs (`cops_evaluate` / `robbers_evaluate`), but with component contraction making the TT dramatically more effective.

### ncop-rs/src/main.rs (original — 601 lines)

**Graph representation**: `AdjMatrix = __m256i` (256-bit AVX2), a 16×16 adjacency matrix as sixteen `u16` rows. Pre-computed `STAR_GRAPHS[16]` (one per vertex) make add/remove/has_edge a few SIMD ops. `is_0_1_connected()` does BFS entirely in SIMD via horizontal-OR propagation.

**GameState**: `GameState<const K: u8, const COPS: usize>` — graph size and cop count are compile-time constants. Holds `cop` and `robber` adjacency matrices.

**Search** (`SearchForWinner` trait):
- Cop enumerates all `C(remaining_edges, COPS)` combinations using `itertools::array_combinations::<COPS>()`
- Transposition table: `RwLock<HashMap<[u64;8], Victor>>`, consulted at depths < `CACHE_DEPTH`
- `NextFewerK` trait + macro: when fewer than COPS edges remain, falls back to `GameState<K, COPS-1>`

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

Work-stealing (parallelism at multiple depths) was tried but regressed performance due to `par_bridge()` overhead. Root-only parallelism remains the default.

## Key Constraints

- `ncop-rs` requires AVX2 (`target-cpu=native`)
- `ncop2-rs` requires Rust nightly (`generic_const_exprs`)
- Graph sizes capped at 16 in ncop-rs (256-bit bitboard = 16 × u16); ncop2-rs uses `bitvec` so the limit is higher but runtime is the bottleneck
- Search is exhaustive minimax — runtime grows exponentially
- The three Rust crates (`ncop-rs/`, `ncop2-rs/`, `ncop3-rs/`) are separate git repos, not yet integrated as submodules into the main repo

### Running on Remote Server
We have access to a 32-core Windows remote machine specifically for running large jobs like K=11 4-cop.
- **SSH Target**: `meghanto@172.23.44.77`
- **Copying over code**: Use `scp` or `rsync` from the local machine.
- **Running**: The remote has Rust installed. You can compile and run directly on it via SSH commands (e.g. `ssh meghanto@172.23.44.77 'cd C:\Users\meghanto\ncop2-rs && cargo run --release'`).
