use itertools::Itertools;
use parking_lot::RwLock;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelBridge, ParallelIterator};
use std::arch::x86_64::{
    __m256i, _mm256_alignr_epi8, _mm256_and_si256, _mm256_andnot_si256, _mm256_cmpeq_epi16,
    _mm256_extract_epi16, _mm256_or_si256, _mm256_set1_epi16, _mm256_set1_epi32,
    _mm256_setzero_si256, _mm256_testz_si256, _mm256_xor_si256,
};
#[cfg(target_feature = "avx512vl")]
use std::arch::x86_64::_mm256_ternarylogic_epi64;
use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::HashSet;
use std::io::{Write, stdout};
use std::iter;
use std::mem::{transmute, transmute_copy};
use std::num::NonZero;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

macro_rules! env_usize {
    ($name:expr) => {{
        const S: &str = env!($name);
        const N: usize = const {
            let bytes = S.as_bytes();
            let mut i = 0;
            let mut n: usize = 0;
            while i < bytes.len() {
                n = n * 10 + (bytes[i] - b'0') as usize;
                i += 1;
            }
            n
        };
        N
    }};
}

// ============================================================
// Configuration
// ============================================================

const SIZE: u8 = 9;
const NCOP: usize = 4;
const TT_SIZE_LOG2: usize = 27; // 128M entries

/// Default from build.rs based on available CPUs: floor(log2(cpus)).
/// Overridable at runtime via 4th command-line arg.
const PAR_MAX_DEPTH_DEFAULT: usize = env_usize!("PAR_MAX_DEPTH");

/// Runtime-configurable parallel depth. Set in main(), read in search.
static PAR_DEPTH: AtomicUsize = AtomicUsize::new(0);

static NODES_EVALUATED: AtomicU64 = AtomicU64::new(0);
static TT_HITS: AtomicU64 = AtomicU64::new(0);
static TT_STORES: AtomicU64 = AtomicU64::new(0);
static RTT_HITS: AtomicU64 = AtomicU64::new(0);
static RTT_STORES: AtomicU64 = AtomicU64::new(0);

// ============================================================
// AVX2 SIMD graph representation (from ncop-rs)
// ============================================================

type AdjMatrix = __m256i;

const fn star_graph(n: usize) -> AdjMatrix {
    let mut data = [1u16 << n; 16]; // fill the column
    data[n] = 0xffff; // fill the row
    unsafe { transmute(data) }
}

const STAR_GRAPHS: [AdjMatrix; 16] = [
    star_graph(0),
    star_graph(1),
    star_graph(2),
    star_graph(3),
    star_graph(4),
    star_graph(5),
    star_graph(6),
    star_graph(7),
    star_graph(8),
    star_graph(9),
    star_graph(10),
    star_graph(11),
    star_graph(12),
    star_graph(13),
    star_graph(14),
    star_graph(15),
];

trait Graph: Sized {
    fn add_edge(&mut self, u: usize, v: usize);
    fn remove_edge(&mut self, u: usize, v: usize);
    fn has_edge(&self, u: usize, v: usize) -> bool;
    fn is_0_1_connected(&self) -> bool;
}

impl Graph for AdjMatrix {
    fn add_edge(&mut self, u: usize, v: usize) {
        *self = unsafe {
            _mm256_or_si256(
                _mm256_and_si256(STAR_GRAPHS[u], STAR_GRAPHS[v]),
                *self,
            )
        }
    }

    fn remove_edge(&mut self, u: usize, v: usize) {
        *self = unsafe {
            _mm256_andnot_si256(
                _mm256_and_si256(STAR_GRAPHS[u], STAR_GRAPHS[v]),
                *self,
            )
        }
    }

    fn has_edge(&self, u: usize, v: usize) -> bool {
        unsafe {
            _mm256_testz_si256(
                _mm256_and_si256(STAR_GRAPHS[u], STAR_GRAPHS[v]),
                *self,
            ) == 0
        }
    }

    fn is_0_1_connected(&self) -> bool {
        unsafe {
            let mut matches = _mm256_set1_epi16(_mm256_extract_epi16(*self, 0) as i16);
            let mut last = _mm256_setzero_si256();
            loop {
                // no change? not connected
                let cmp = _mm256_xor_si256(last, matches);
                if _mm256_testz_si256(cmp, cmp) != 0 {
                    break false;
                }
                last = matches;

                // find rows intersecting with frontier
                matches = _mm256_and_si256(matches, *self);
                matches = _mm256_cmpeq_epi16(matches, _mm256_setzero_si256());
                matches = _mm256_xor_si256(matches, _mm256_set1_epi32(-1));

                // get their contents
                matches = _mm256_and_si256(matches, *self);
                if _mm256_testz_si256(matches, STAR_GRAPHS[1]) == 0 {
                    break true; // did we see the one row?
                }

                // merge into new columns
                matches = _mm256_or_si256(_mm256_alignr_epi8(matches, matches, 2), matches);
                matches = _mm256_or_si256(_mm256_alignr_epi8(matches, matches, 4), matches);
                matches = _mm256_or_si256(_mm256_alignr_epi8(matches, matches, 8), matches);
                matches = _mm256_or_si256(_mm256_alignr_epi8(matches, matches, 16), matches);
            }
        }
    }
}

// ============================================================
// GameState
// ============================================================

#[derive(Copy, Clone)]
struct GameState<const K: u8, const COPS: usize> {
    robber: AdjMatrix,
    cop: AdjMatrix,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Victor {
    Robber,
    Cop,
}

fn robber_initial(k: usize) -> AdjMatrix {
    let mut data = [0u16; 16];
    for i in 0..k {
        data[i] = 1 << i;
    }
    unsafe { transmute(data) }
}

fn cop_initial(k: usize) -> AdjMatrix {
    let mut data = [0u16; 16];
    for i in 0..k {
        data[i] = (1 << k) - 1;
    }
    unsafe {
        _mm256_or_si256(
            _mm256_xor_si256(transmute(data), _mm256_set1_epi32(-1)),
            robber_initial(k),
        )
    }
}

impl<const K: u8, const COPS: usize> GameState<K, COPS> {
    fn new() -> Self {
        Self {
            robber: robber_initial(K as usize),
            cop: cop_initial(K as usize),
        }
    }

    fn size(&self) -> usize {
        K as usize
    }

    fn cops(&self) -> usize {
        COPS
    }

    fn did_robber_win(&self) -> bool {
        self.robber.is_0_1_connected()
    }

    fn did_cop_win(&self) -> bool {
        unsafe { !_mm256_xor_si256(self.cop, _mm256_set1_epi32(-1)).is_0_1_connected() }
    }

    #[inline(always)]
    #[cfg(target_feature = "avx512vl")]
    fn remaining_edges(&self) -> impl Iterator<Item = (usize, usize)> + Clone {
        let edges: [u16; 16] = unsafe {
            transmute(_mm256_ternarylogic_epi64(
                self.cop,
                self.robber,
                _mm256_setzero_si256(),
                0x01,
            ))
        };
        edges.into_iter().enumerate().flat_map(|(u, bits)| {
            iter::successors(NonZero::new(bits), |bits| {
                NonZero::new(bits.get() & (bits.get() - 1))
            })
            .map(move |bits| (u, bits.trailing_zeros() as usize))
            .filter(move |(u, v)| *u < *v && *v < K as usize)
        })
    }

    #[inline(always)]
    #[cfg(not(target_feature = "avx512vl"))]
    fn remaining_edges(&self) -> impl Iterator<Item = (usize, usize)> + Clone {
        unsafe {
            let edges = _mm256_xor_si256(
                _mm256_or_si256(self.cop, self.robber),
                _mm256_set1_epi32(-1),
            );
            let edges = transmute::<_, [u16; 16]>(edges);
            edges.into_iter().enumerate().flat_map(|(u, bits)| {
                iter::successors(NonZero::new(bits), |bits| {
                    NonZero::new(bits.get() & (bits.get() - 1))
                })
                .map(move |bits| (u, bits.trailing_zeros() as usize))
                .filter(move |(u, v)| *u < *v && *v < K as usize)
            })
        }
    }
}

// ============================================================
// Contracted representation
// ============================================================

const MAX_COMP: usize = 16;
const MAX_EDGES: usize = MAX_COMP * (MAX_COMP - 1) / 2; // 120

fn pack_row(row: &[u8], n: usize) -> u64 {
    let mut packed: u64 = 0;
    for i in 0..n {
        packed |= (row[i] as u64) << (i * 6);
    }
    packed
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

struct Prng {
    state: u64,
}

impl Prng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_usize(&mut self, bound: usize) -> usize {
        self.state = splitmix64(self.state);
        (self.state % bound as u64) as usize
    }
}

fn shuffle<T>(slice: &mut [T], rng: &mut Prng) {
    if slice.is_empty() { return; }
    for i in (1..slice.len()).rev() {
        let j = rng.next_usize(i + 1);
        slice.swap(i, j);
    }
}

struct ContractedInfo {
    comp: [u8; MAX_COMP],
    n_comps: u8,
    adj: [[u8; MAX_COMP]; MAX_COMP],
}

fn compute_components<const K: u8, const COPS: usize>(
    state: &GameState<K, COPS>,
) -> Option<ContractedInfo> {
    let n = K as usize;

    let mut comp = [0u8; MAX_COMP];
    let mut n_comps: u8 = 0;
    let mut visited = [false; MAX_COMP];

    for start in 0..n {
        if visited[start] {
            continue;
        }
        let id = n_comps;
        n_comps += 1;
        visited[start] = true;
        comp[start] = id;
        let mut queue_buf = [0u8; MAX_COMP];
        queue_buf[0] = start as u8;
        let mut head = 0;
        let mut tail = 1;
        while head < tail {
            let u = queue_buf[head] as usize;
            head += 1;
            for v in 0..n {
                if !visited[v] && v != u && state.robber.has_edge(u, v) {
                    visited[v] = true;
                    comp[v] = id;
                    queue_buf[tail] = v as u8;
                    tail += 1;
                }
            }
        }
    }

    if comp[0] == comp[1] {
        return None;
    }

    let mut adj = [[0u8; MAX_COMP]; MAX_COMP];
    for (u, v) in state.remaining_edges() {
        let ci = comp[u] as usize;
        let cv = comp[v] as usize;
        adj[ci][cv] += 1;
        if ci != cv {
            adj[cv][ci] += 1;
        }
    }

    Some(ContractedInfo { comp, n_comps, adj })
}

fn edge_score(u: usize, v: usize, info: &ContractedInfo) -> isize {
    let terminal: isize = if u <= 1 || v <= 1 { 100 } else { 0 };
    let weight = info.adj[info.comp[u] as usize][info.comp[v] as usize] as isize;
    terminal - weight
}

/// Score offset so all scores map to non-negative bucket indices.
/// Max weight = C(MAX_COMP-1, 2) = 105 (unclaimed edges within one component).
/// Score range: [-105, 100], offset by 105 → [0, 205].
const SCORE_OFFSET: isize = (MAX_COMP as isize - 1) * (MAX_COMP as isize - 2) / 2;
const SCORE_BUCKETS: usize = (SCORE_OFFSET + 101) as usize;

/// Counting-sort edges by score, descending. Returns the number of edges.
fn sort_edges_by_score(
    edges: &[(usize, usize); MAX_EDGES],
    n: usize,
    info: &ContractedInfo,
    out: &mut [(usize, usize); MAX_EDGES],
) -> usize {
    let mut counts = [0u16; SCORE_BUCKETS];
    let mut scores = [0u16; MAX_EDGES]; // bucket index per edge
    for i in 0..n {
        let (u, v) = edges[i];
        let bucket = (edge_score(u, v, info) + SCORE_OFFSET) as u16;
        scores[i] = bucket;
        counts[bucket as usize] += 1;
    }
    // Prefix sum descending: high buckets get low output indices
    let mut prefix = [0u16; SCORE_BUCKETS];
    let mut sum = 0u16;
    for bucket in (0..SCORE_BUCKETS).rev() {
        prefix[bucket] = sum;
        sum += counts[bucket];
    }
    // Place edges into output
    for i in 0..n {
        let bucket = scores[i] as usize;
        out[prefix[bucket] as usize] = edges[i];
        prefix[bucket] += 1;
    }
    n
}

fn contract_and_hash(info: &ContractedInfo) -> [u64; 2] {
    let nc = info.n_comps as usize;
    let mut adj = info.adj;

    let c0 = info.comp[0] as usize;
    let c1 = info.comp[1] as usize;
    let mut remap = [0u8; MAX_COMP];
    remap[c0] = 0;
    remap[c1] = 1;
    let mut next_id = 2u8;
    for i in 0..nc {
        if i != c0 && i != c1 {
            remap[i] = next_id;
            next_id += 1;
        }
    }

    let old_adj = adj;
    for i in 0..nc {
        for j in 0..nc {
            adj[remap[i] as usize][remap[j] as usize] = old_adj[i][j];
        }
    }

    // Zero out diagonal — within-component edges are never played by either side,
    // so positions differing only in within-component edge counts are equivalent.
    for i in 0..nc {
        adj[i][i] = 0;
    }

    if adj[0][..nc] > adj[1][..nc] {
        adj.swap(0, 1);
        for row in adj[..nc].iter_mut() {
            row.swap(0, 1);
        }
    }

    for i in 2..nc.saturating_sub(1) {
        let mut min_idx = i;
        for j in (i + 1)..nc {
            if adj[j][..nc] < adj[min_idx][..nc] {
                min_idx = j;
            }
        }
        if min_idx != i {
            adj.swap(i, min_idx);
            for row in adj[..nc].iter_mut() {
                row.swap(i, min_idx);
            }
        }
    }

    // Hash 0: standard splitmix64 chain
    let mut h0 = splitmix64(nc as u64);
    for i in 0..nc {
        let packed = pack_row(&adj[i], nc);
        h0 = splitmix64(h0 ^ packed);
    }

    // Hash 1: independent chain with different seed
    let mut h1 = splitmix64(nc as u64 ^ 0x517cc1b727220a95);
    for i in 0..nc {
        let packed = pack_row(&adj[i], nc);
        h1 = splitmix64(h1 ^ packed);
    }

    [h0, h1]
}

/// Mix picks_left into a base hash to differentiate TT entries for
/// the same contracted state at different cop sub-step counts.
fn hash_with_picks(hash: [u64; 2], picks_left: usize) -> [u64; 2] {
    [
        splitmix64(hash[0] ^ (picks_left as u64)),
        splitmix64(hash[1] ^ (picks_left as u64).wrapping_mul(0x517cc1b727220a95)),
    ]
}

// ============================================================
// Lockless transposition table (128-bit hash, XOR trick)
// ============================================================
// Entry layout: [word0, word1]
//   word0 = hash[0] with bit 0 stolen for Victor (1=Cop, 0=Robber)
//   word1 = hash[1] ^ word0   (XOR trick: torn reads produce hash mismatch)
//
// Probe: read word0, word1. Verify (word0 & !1) == (hash[0] & !1)
//        AND (word1 ^ word0) == hash[1]. Extract victor from word0 bit 0.
// Collision requires BOTH 63-bit hash[0] AND 64-bit hash[1] to match: ~1/2^127.
// Torn read requires the mismatched word to accidentally verify: ~1/2^64.

struct TranspositionTable {
    entries: Box<[[AtomicU64; 2]]>,
    mask: usize,
}

impl TranspositionTable {
    fn new(size_log2: usize) -> Self {
        let size = 1usize << size_log2;
        let mut entries = Vec::with_capacity(size);
        for _ in 0..size {
            entries.push([AtomicU64::new(0), AtomicU64::new(0)]);
        }
        TranspositionTable {
            entries: entries.into_boxed_slice(),
            mask: size - 1,
        }
    }

    fn probe(&self, hash: [u64; 2]) -> Option<Victor> {
        let idx = (hash[0] as usize) & self.mask;
        let word0 = self.entries[idx][0].load(Ordering::Relaxed);
        if word0 == 0 {
            return None;
        }
        let word1 = self.entries[idx][1].load(Ordering::Relaxed);
        // Verify both halves: hash[0] match (63 bits) + XOR trick (64 bits)
        if (word0 & !1) == (hash[0] & !1) && (word1 ^ word0) == hash[1] {
            Some(if word0 & 1 == 1 {
                Victor::Cop
            } else {
                Victor::Robber
            })
        } else {
            None
        }
    }

    fn store(&self, hash: [u64; 2], result: Victor) {
        let idx = (hash[0] as usize) & self.mask;
        let word0 = (hash[0] & !1)
            | match result {
                Victor::Cop => 1,
                Victor::Robber => 0,
            };
        if word0 == 0 {
            return; // skip the ~1/2^63 edge case (sentinel collision)
        }
        let word1 = hash[1] ^ word0;
        self.entries[idx][0].store(word0, Ordering::Relaxed);
        self.entries[idx][1].store(word1, Ordering::Relaxed);
    }
}

// ============================================================
// Search with work-stealing parallelism
// ============================================================

trait SearchForWinner: Sized + Copy + Send + Sync {
    fn cop_step(&self, picks_left: usize, depth: usize, tt: &TranspositionTable, rtt: &TranspositionTable, rng: &mut Prng) -> Victor;
    fn cops_evaluate(&self, depth: usize, tt: &TranspositionTable, rtt: &TranspositionTable, rng: &mut Prng) -> Victor;
    fn robbers_evaluate(&self, depth: usize, tt: &TranspositionTable, rtt: &TranspositionTable, rng: &mut Prng) -> Victor;
}

impl<const K: u8, const COPS: usize> SearchForWinner for GameState<K, COPS> {
    fn cop_step(&self, picks_left: usize, depth: usize, tt: &TranspositionTable, rtt: &TranspositionTable, rng: &mut Prng) -> Victor {
        // No more picks → check win, then hand to robber
        if picks_left == 0 {
            if self.did_cop_win() {
                return Victor::Cop;
            }
            return self.robbers_evaluate(depth, tt, rtt, rng);
        }

        NODES_EVALUATED.fetch_add(1, Ordering::Relaxed);

        // Early cutoff: cop already won, remaining picks don't matter
        if self.did_cop_win() {
            return Victor::Cop;
        }

        let info = compute_components(self);
        if info.is_none() {
            return Victor::Robber;
        }
        let info = info.unwrap();

        let threats = info.adj[info.comp[0] as usize][info.comp[1] as usize] as usize;
        if threats > picks_left {
            return Victor::Robber;
        }

        // TT probe with picks_left mixed in
        let base_hash = contract_and_hash(&info);
        let h = hash_with_picks(base_hash, picks_left);
        if let Some(v) = tt.probe(h) {
            TT_HITS.fetch_add(1, Ordering::Relaxed);
            return v;
        }

        // Collect cross-component edges only
        let mut raw_edges = [(0usize, 0usize); MAX_EDGES];
        let mut n_edges = 0;
        for (u, v) in self.remaining_edges() {
            if info.comp[u] == info.comp[v] {
                continue;
            }
            raw_edges[n_edges] = (u, v);
            n_edges += 1;
        }

        // Shuffle before stable sort to randomize equal-score edges
        shuffle(&mut raw_edges[..n_edges], rng);
        
        // Sort edges by score for better cutoffs
        let mut edges = [(0usize, 0usize); MAX_EDGES];
        sort_edges_by_score(&raw_edges, n_edges, &info, &mut edges);

        // Component-pair dedup: edges connecting the same component pair
        // produce identical contracted states, so only try one per pair.
        let mut seen_pair = [[false; MAX_COMP]; MAX_COMP];

        let victor = if n_edges == 0 {
            // No cross-component remaining edges → robber can never connect 0-1
            Victor::Cop
        } else {
            let mut victor = Victor::Robber;
            for i in 0..n_edges {
                let (u, v) = edges[i];
                let cu = info.comp[u] as usize;
                let cv = info.comp[v] as usize;
                let (lo, hi) = if cu <= cv { (cu, cv) } else { (cv, cu) };
                if seen_pair[lo][hi] {
                    continue;
                }
                seen_pair[lo][hi] = true;

                let mut state = *self;
                state.cop.add_edge(u, v);
                if state.cop_step(picks_left - 1, depth, tt, rtt, rng) == Victor::Cop {
                    victor = Victor::Cop;
                    break;
                }
            }
            victor
        };

        tt.store(h, victor);
        TT_STORES.fetch_add(1, Ordering::Relaxed);

        victor
    }

    fn cops_evaluate(&self, depth: usize, tt: &TranspositionTable, rtt: &TranspositionTable, rng: &mut Prng) -> Victor {
        self.cop_step(COPS, depth, tt, rtt, rng)
    }

    fn robbers_evaluate(&self, depth: usize, tt: &TranspositionTable, rtt: &TranspositionTable, rng: &mut Prng) -> Victor {
        // compute_components returns None when comp[0]==comp[1] → robber has won
        let info = compute_components(self);
        if info.is_none() {
            return Victor::Robber;
        }

        if let Some(ref ci) = info {
            if ci.adj[ci.comp[0] as usize][ci.comp[1] as usize] > 0 {
                return Victor::Robber;
            }
        }

        // Robber TT probe
        if let Some(ref ci) = info {
            let h = contract_and_hash(ci);
            if let Some(v) = rtt.probe(h) {
                RTT_HITS.fetch_add(1, Ordering::Relaxed);
                return v;
            }
        }

        let mut raw_edges = [(0usize, 0usize); MAX_EDGES];
        let mut n_edges = 0;
        for (u, v) in self.remaining_edges() {
            if let Some(ref ci) = info {
                if ci.comp[u] == ci.comp[v] {
                    continue;
                }
            }
            raw_edges[n_edges] = (u, v);
            n_edges += 1;
        }
        let mut edges = raw_edges;

        let mut victor = Victor::Cop;
        if n_edges > 0 {
            let edges_slice = &mut edges[..n_edges];
            shuffle(edges_slice, rng);
            
            for &(u, v) in edges_slice.iter() {
                let mut state = *self;
                state.robber.add_edge(u, v);
                if state.did_robber_win() || state.cops_evaluate(depth + 1, tt, rtt, rng) == Victor::Robber {
                    if depth == 0 {
                        println!(", but Robber can win by adding edge ({u}, {v})");
                    }
                    victor = Victor::Robber;
                    break;
                }
            }
        };

        // Robber TT store
        if let Some(ref ci) = info {
            rtt.store(contract_and_hash(ci), victor);
            RTT_STORES.fetch_add(1, Ordering::Relaxed);
        }

        victor
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod test {
    use crate::{
        GameState, Graph, SearchForWinner, TranspositionTable, Victor,
        compute_components, contract_and_hash, solve_bfs,
    };
    use itertools::Itertools;

    fn tt() -> TranspositionTable {
        TranspositionTable::new(16)
    }

    fn rtt() -> TranspositionTable {
        TranspositionTable::new(16)
    }

    #[test]
    fn boards_work() {
        let mut state = GameState::<4, 1>::new();

        state.cop.add_edge(0, 2);
        state.robber.add_edge(1, 3);

        assert!(state.cop.has_edge(0, 2));
        assert!(!state.cop.has_edge(1, 3));
        assert!(state.robber.has_edge(1, 3));
        assert!(!state.robber.has_edge(0, 2));
        assert!(!state.cop.has_edge(0, 1));
        assert!(!state.robber.has_edge(0, 1));

        state.cop.remove_edge(0, 2);
        assert!(!state.cop.has_edge(0, 2));

        assert!(!state.robber.is_0_1_connected());
        state.robber.add_edge(2, 3);
        assert!(!state.robber.is_0_1_connected());
        state.robber.add_edge(2, 0);
        assert!(state.robber.is_0_1_connected());
    }

    #[test]
    fn simple_victory() {
        let mut state = GameState::<6, 2>::new();
        state.cop.add_edge(0, 1);
        state.cop.add_edge(0, 2);
        state.robber.add_edge(4, 5);

        state.cop.add_edge(0, 3);
        state.cop.add_edge(0, 4);
        state.robber.add_edge(0, 5);

        state.cop.add_edge(1, 4);
        state.cop.add_edge(1, 5);
        state.robber.add_edge(1, 2);

        assert_eq!(state.cops_evaluate(0, &tt(), &rtt(), &mut crate::Prng::new(12345)), Victor::Cop);
    }

    #[test]
    fn cop_1_wins_4_clique() {
        let mut state = GameState::<4, 1>::new();
        assert_eq!(state.cops_evaluate(0, &tt(), &rtt(), &mut crate::Prng::new(12345)), Victor::Robber);

        state.cop.add_edge(0, 1);
        state.robber.add_edge(0, 2);
        assert_eq!(state.cops_evaluate(0, &tt(), &rtt(), &mut crate::Prng::new(12345)), Victor::Robber);
    }

    #[test]
    fn connectivity_1() {
        let mut state = GameState::<6, 2>::new();
        state.robber.add_edge(0, 5);
        assert!(!state.robber.is_0_1_connected());
        state.robber.add_edge(1, 4);
        assert!(!state.robber.is_0_1_connected());
        state.robber.add_edge(2, 3);
        assert!(!state.robber.is_0_1_connected());
        state.robber.add_edge(5, 2);
        assert!(!state.robber.is_0_1_connected());
        state.robber.add_edge(5, 3);
        assert!(!state.robber.is_0_1_connected());
        state.robber.add_edge(4, 2);
        assert!(state.robber.is_0_1_connected());
    }

    #[test]
    fn edge_biterator_works() {
        let mut state = GameState::<4, 1>::new();
        let edges = state.remaining_edges().collect::<Vec<_>>();
        let mut expected = (0..state.size())
            .tuple_combinations()
            .collect::<Vec<(_, _)>>();
        assert_eq!(expected, edges);

        for _ in 0..expected.len() {
            let (u, v) = expected.pop().unwrap();
            state.cop.add_edge(u, v);
            assert_eq!(expected, state.remaining_edges().collect::<Vec<_>>());
        }

        let mut state = GameState::<4, 1>::new();
        let mut expected = (0..state.size())
            .tuple_combinations()
            .collect::<Vec<(_, _)>>();

        for _ in 0..expected.len() {
            let (u, v) = expected.pop().unwrap();
            state.robber.add_edge(u, v);
            assert_eq!(expected, state.remaining_edges().collect::<Vec<_>>());
        }

        let mut state = GameState::<4, 1>::new();
        let mut expected = (0..state.size())
            .tuple_combinations()
            .collect::<Vec<(_, _)>>();

        for i in 0..expected.len() {
            let (u, v) = expected.pop().unwrap();
            if i % 2 == 0 {
                state.robber.add_edge(u, v);
            } else {
                state.cop.add_edge(u, v);
            }
            assert_eq!(expected, state.remaining_edges().collect::<Vec<_>>());
        }
    }

    #[test]
    fn contraction_symmetry() {
        let mut state_a = GameState::<4, 1>::new();
        state_a.cop.add_edge(0, 2);
        state_a.cop.add_edge(1, 3);

        let mut state_b = GameState::<4, 1>::new();
        state_b.cop.add_edge(0, 3);
        state_b.cop.add_edge(1, 2);

        let hash_a = contract_and_hash(&compute_components(&state_a).unwrap());
        let hash_b = contract_and_hash(&compute_components(&state_b).unwrap());
        assert_eq!(hash_a, hash_b);
    }

    #[test]
    fn bfs_k4_1cop_robber_wins() {
        assert_eq!(solve_bfs(GameState::<4, 1>::new()), Victor::Robber);
    }

    #[test]
    fn bfs_k6_2cop_cop_wins() {
        assert_eq!(solve_bfs(GameState::<6, 2>::new()), Victor::Cop);
    }

    #[test]
    fn bfs_k3_1cop_cop_wins() {
        assert_eq!(solve_bfs(GameState::<3, 1>::new()), Victor::Cop);
    }

    #[test]
    fn bfs_vs_dfs_k4_1cop() {
        let state = GameState::<4, 1>::new();
        let dfs_result = state.cops_evaluate(0, &tt(), &rtt(), &mut crate::Prng::new(12345));
        let bfs_result = solve_bfs(state);
        assert_eq!(dfs_result, bfs_result);
    }

    #[test]
    fn bfs_vs_dfs_k6_2cop() {
        let state = GameState::<6, 2>::new();
        let dfs_result = state.cops_evaluate(0, &tt(), &rtt(), &mut crate::Prng::new(12345));
        let bfs_result = solve_bfs(state);
        assert_eq!(dfs_result, bfs_result);
    }

    #[test]
    fn bfs_vs_dfs_k7_2cop() {
        let state = GameState::<7, 2>::new();
        let dfs_result = state.cops_evaluate(0, &tt(), &rtt(), &mut crate::Prng::new(12345));
        let bfs_result = solve_bfs(state);
        assert_eq!(dfs_result, bfs_result);
    }

    #[test]
    fn contraction_robber_components() {
        let mut state_a = GameState::<6, 1>::new();
        state_a.robber.add_edge(2, 3);
        state_a.cop.add_edge(0, 4);

        let mut state_b = GameState::<6, 1>::new();
        state_b.robber.add_edge(2, 3);
        state_b.cop.add_edge(0, 5);

        let hash_a = contract_and_hash(&compute_components(&state_a).unwrap());
        let hash_b = contract_and_hash(&compute_components(&state_b).unwrap());
        assert_eq!(hash_a, hash_b);
    }
}

// ============================================================
// DAG/BFS solver
// ============================================================

struct DagNode {
    parents: Vec<u32>,
    n_unresolved: u32,
    resolved: Option<Victor>,
    is_cop_turn: bool,
}

/// BFS expansion + backward propagation.
/// Each depth = one edge claim (cop or robber). Round = COPS cop sub-steps + 1 robber step.
/// Deduplication via contracted multigraph hash + depth_mod.
fn solve_bfs<const K: u8, const COPS: usize>(initial: GameState<K, COPS>) -> Victor {
    let start = Instant::now();

    if initial.did_cop_win() {
        return Victor::Cop;
    }
    if initial.did_robber_win() {
        return Victor::Robber;
    }

    let root_info = match compute_components(&initial) {
        Some(ci) => ci,
        None => return Victor::Robber,
    };
    let root_hash = contract_and_hash(&root_info);
    let round_len = COPS + 1;

    let mut nodes: Vec<DagNode> = Vec::new();
    let mut hash_to_idx: HashMap<([u64; 2], u8), u32> = HashMap::new();
    let mut states: Vec<GameState<K, COPS>> = Vec::new();

    let root_idx = 0u32;
    nodes.push(DagNode {
        parents: Vec::new(),
        n_unresolved: 0,
        resolved: None,
        is_cop_turn: true,
    });
    states.push(initial);
    hash_to_idx.insert((root_hash, 0), root_idx);

    let mut frontier: Vec<u32> = vec![root_idx];
    let mut depth: usize = 0;

    loop {
        if frontier.is_empty() {
            eprintln!("Frontier exhausted at depth {depth} → Cop wins.");
            return Victor::Cop;
        }

        let depth_mod = (depth % round_len) as u8;
        let is_cop_turn = (depth_mod as usize) < COPS;
        let next_depth_mod = ((depth + 1) % round_len) as u8;
        let next_is_cop_turn = (next_depth_mod as usize) < COPS;

        eprintln!(
            "[{:>6.1}s] depth {depth}: {} frontier ({}) | DAG: {} nodes, {} MB",
            start.elapsed().as_secs_f64(),
            frontier.len(),
            if is_cop_turn { format!("cop {}/{}", depth_mod as usize + 1, COPS) } else { "robber".into() },
            nodes.len(),
            (nodes.len() * std::mem::size_of::<DagNode>()
                + states.len() * std::mem::size_of::<GameState<K, COPS>>())
                / (1024 * 1024),
        );

        let mut next_frontier: Vec<u32> = Vec::new();
        let mut newly_resolved: Vec<u32> = Vec::new();

        for &parent_idx in &frontier {
            if nodes[parent_idx as usize].resolved.is_some() {
                continue;
            }

            let state = states[parent_idx as usize];
            let info = compute_components(&state);
            let edges: Vec<(usize, usize)> = state.remaining_edges()
                .filter(|&(u, v)| {
                    if let Some(ref ci) = info {
                        ci.comp[u] != ci.comp[v]
                    } else {
                        true
                    }
                })
                .collect();

            let mut n_unresolved = 0u32;
            let mut has_any_child = false;
            let mut all_resolved_same = true;
            let mut parent_terminal = false;

            if is_cop_turn {
                for &(u, v) in &edges {
                    let mut cs = state;
                    cs.cop.add_edge(u, v);

                    if cs.did_cop_win() {
                        nodes[parent_idx as usize].resolved = Some(Victor::Cop);
                        newly_resolved.push(parent_idx);
                        parent_terminal = true;
                        break;
                    }

                    let child_info = compute_components(&cs);
                    let child_hash = match &child_info {
                        Some(ci) => contract_and_hash(ci),
                        None => {
                            nodes[parent_idx as usize].resolved = Some(Victor::Cop);
                            newly_resolved.push(parent_idx);
                            parent_terminal = true;
                            break;
                        }
                    };

                    match hash_to_idx.get(&(child_hash, next_depth_mod)) {
                        Some(&idx) => {
                            has_any_child = true;
                            if let Some(v) = nodes[idx as usize].resolved {
                                if v == Victor::Cop {
                                    nodes[parent_idx as usize].resolved = Some(Victor::Cop);
                                    newly_resolved.push(parent_idx);
                                    parent_terminal = true;
                                    break;
                                }
                                continue;
                            }
                            all_resolved_same = false;
                            nodes[idx as usize].parents.push(parent_idx);
                            n_unresolved += 1;
                        }
                        None => {
                            has_any_child = true;
                            all_resolved_same = false;
                            let idx = nodes.len() as u32;
                            nodes.push(DagNode {
                                parents: vec![parent_idx],
                                n_unresolved: 0,
                                resolved: None,
                                is_cop_turn: next_is_cop_turn,
                            });
                            states.push(cs);
                            hash_to_idx.insert((child_hash, next_depth_mod), idx);
                            next_frontier.push(idx);
                            n_unresolved += 1;
                        }
                    };
                }

                if !parent_terminal {
                    if !has_any_child {
                        nodes[parent_idx as usize].resolved = Some(Victor::Cop);
                        newly_resolved.push(parent_idx);
                    } else if n_unresolved == 0 && all_resolved_same {
                        nodes[parent_idx as usize].resolved = Some(Victor::Robber);
                        newly_resolved.push(parent_idx);
                    }
                }
            } else {
                for &(u, v) in &edges {
                    let mut cs = state;
                    cs.robber.add_edge(u, v);

                    if cs.did_robber_win() {
                        nodes[parent_idx as usize].resolved = Some(Victor::Robber);
                        newly_resolved.push(parent_idx);
                        parent_terminal = true;
                        break;
                    }

                    let child_info = compute_components(&cs);
                    let child_hash = match &child_info {
                        Some(ci) => contract_and_hash(ci),
                        None => unreachable!(),
                    };

                    match hash_to_idx.get(&(child_hash, next_depth_mod)) {
                        Some(&idx) => {
                            has_any_child = true;
                            if let Some(v) = nodes[idx as usize].resolved {
                                if v == Victor::Robber {
                                    nodes[parent_idx as usize].resolved = Some(Victor::Robber);
                                    newly_resolved.push(parent_idx);
                                    parent_terminal = true;
                                    break;
                                }
                                continue;
                            }
                            all_resolved_same = false;
                            nodes[idx as usize].parents.push(parent_idx);
                            n_unresolved += 1;
                        }
                        None => {
                            has_any_child = true;
                            all_resolved_same = false;
                            let idx = nodes.len() as u32;
                            nodes.push(DagNode {
                                parents: vec![parent_idx],
                                n_unresolved: 0,
                                resolved: None,
                                is_cop_turn: next_is_cop_turn,
                            });
                            states.push(cs);
                            hash_to_idx.insert((child_hash, next_depth_mod), idx);
                            next_frontier.push(idx);
                            n_unresolved += 1;
                        }
                    };
                }

                if !parent_terminal {
                    if !has_any_child {
                        nodes[parent_idx as usize].resolved = Some(Victor::Cop);
                        newly_resolved.push(parent_idx);
                    } else if n_unresolved == 0 && all_resolved_same {
                        nodes[parent_idx as usize].resolved = Some(Victor::Cop);
                        newly_resolved.push(parent_idx);
                    }
                }
            }

            if !parent_terminal {
                nodes[parent_idx as usize].n_unresolved = n_unresolved;
            }
        }

        // Backward propagation
        while !newly_resolved.is_empty() {
            let batch = std::mem::take(&mut newly_resolved);
            for resolved_idx in batch {
                let result = match nodes[resolved_idx as usize].resolved {
                    Some(v) => v,
                    None => continue,
                };

                if resolved_idx == root_idx {
                    eprintln!(
                        "[{:>6.1}s] Root resolved: {result:?} wins. DAG: {} nodes",
                        start.elapsed().as_secs_f64(),
                        nodes.len(),
                    );
                    return result;
                }

                let parents = nodes[resolved_idx as usize].parents.clone();
                for parent_idx in parents {
                    if nodes[parent_idx as usize].resolved.is_some() {
                        continue;
                    }

                    let parent_is_cop = nodes[parent_idx as usize].is_cop_turn;
                    let is_or = (parent_is_cop && result == Victor::Cop)
                        || (!parent_is_cop && result == Victor::Robber);

                    if is_or {
                        nodes[parent_idx as usize].resolved = Some(result);
                        newly_resolved.push(parent_idx);
                    } else {
                        nodes[parent_idx as usize].n_unresolved -= 1;
                        if nodes[parent_idx as usize].n_unresolved == 0 {
                            nodes[parent_idx as usize].resolved = Some(result);
                            newly_resolved.push(parent_idx);
                        }
                    }
                }
            }
        }

        if let Some(v) = nodes[root_idx as usize].resolved {
            eprintln!(
                "[{:>6.1}s] Root resolved: {v:?} wins. DAG: {} nodes",
                start.elapsed().as_secs_f64(),
                nodes.len(),
            );
            return v;
        }

        frontier = next_frontier;
        depth += 1;
    }
}

// ============================================================
// HashMap-based perfect cache (no eviction)
// ============================================================

struct HashMapCache {
    map: RwLock<HashMap<[u64; 2], Victor>>,
}

impl HashMapCache {
    fn new() -> Self {
        Self {
            map: RwLock::new(HashMap::new()),
        }
    }

    fn probe(&self, hash: [u64; 2]) -> Option<Victor> {
        self.map.read().get(&hash).copied()
    }

    fn store(&self, hash: [u64; 2], result: Victor) {
        self.map.write().insert(hash, result);
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.map.read().len()
    }
}

trait SearchWithCache: Sized + Copy + Send + Sync {
    fn cops_eval(&self, depth: usize, cache: &HashMapCache) -> Victor;
    fn robbers_eval(&self, depth: usize, cache: &HashMapCache) -> Victor;
}

impl<const K: u8, const COPS: usize> SearchWithCache for GameState<K, COPS> {
    fn cops_eval(&self, depth: usize, cache: &HashMapCache) -> Victor {
        NODES_EVALUATED.fetch_add(1, Ordering::Relaxed);
        if self.did_cop_win() {
            return Victor::Cop;
        }
        if self.did_robber_win() {
            return Victor::Robber;
        }

        let info = compute_components(self);
        if let Some(ref ci) = info {
            let h = contract_and_hash(ci);
            if let Some(v) = cache.probe(h) {
                TT_HITS.fetch_add(1, Ordering::Relaxed);
                return v;
            }
        }

        let mut raw_edges = [(0usize, 0usize); MAX_EDGES];
        let mut n_edges = 0;
        for (u, v) in self.remaining_edges() {
            if let Some(ref ci) = info {
                if ci.comp[u] == ci.comp[v] {
                    continue;
                }
            }
            raw_edges[n_edges] = (u, v);
            n_edges += 1;
        }

        let mut cop_choices = raw_edges[..n_edges]
            .iter()
            .copied()
            .array_combinations::<COPS>()
            .peekable();

        let victor = if cop_choices.peek().is_none() {
            Victor::Cop
        } else if depth < PAR_DEPTH.load(Ordering::Relaxed) {
            let found_cop_win = cop_choices
                .par_bridge()
                .any(|choices| {
                    let mut state = *self;
                    for (u, v) in choices {
                        state.cop.add_edge(u, v);
                    }
                    let eval = if state.did_cop_win() {
                        Victor::Cop
                    } else {
                        state.robbers_eval(depth, cache)
                    };
                    eval == Victor::Cop
                });
            if found_cop_win { Victor::Cop } else { Victor::Robber }
        } else {
            let mut victor = Victor::Robber;
            for choices in cop_choices {
                let mut state = *self;
                for (u, v) in choices {
                    state.cop.add_edge(u, v);
                }
                let eval = if state.did_cop_win() {
                    Victor::Cop
                } else {
                    state.robbers_eval(depth, cache)
                };
                if eval == Victor::Cop {
                    victor = Victor::Cop;
                    break;
                }
            }
            victor
        };

        if let Some(ref ci) = info {
            cache.store(contract_and_hash(ci), victor);
            TT_STORES.fetch_add(1, Ordering::Relaxed);
        }

        victor
    }

    fn robbers_eval(&self, depth: usize, cache: &HashMapCache) -> Victor {
        if self.did_cop_win() {
            return Victor::Cop;
        }
        if self.did_robber_win() {
            return Victor::Robber;
        }

        let info = compute_components(self);
        let mut raw_edges = [(0usize, 0usize); MAX_EDGES];
        let mut n_edges = 0;
        for (u, v) in self.remaining_edges() {
            if let Some(ref ci) = info {
                if ci.comp[u] == ci.comp[v] {
                    continue;
                }
            }
            raw_edges[n_edges] = (u, v);
            n_edges += 1;
        }

        for &(u, v) in &raw_edges[..n_edges] {
            let mut state = *self;
            state.robber.add_edge(u, v);
            if state.did_robber_win() || state.cops_eval(depth + 1, cache) == Victor::Robber {
                return Victor::Robber;
            }
        }
        Victor::Cop
    }
}

// ============================================================
// Main
// ============================================================

fn run<const K: u8, const COPS: usize>() {
    let state = GameState::<K, COPS>::new();
    println!(
        "Playing with {} cops on a clique of size {} (par_depth={})",
        state.cops(),
        state.size(),
        PAR_DEPTH.load(Ordering::Relaxed),
    );

    let tt = TranspositionTable::new(TT_SIZE_LOG2);
    let rtt = TranspositionTable::new(TT_SIZE_LOG2.saturating_sub(2).max(16)); // robber TT: 1/4 size
    let start = Instant::now();
    static DONE: AtomicBool = AtomicBool::new(false);
    std::thread::spawn(move || {
        let mut last_nodes = 0u64;
        loop {
            std::thread::sleep(std::time::Duration::from_secs(10));
            if DONE.load(Ordering::Relaxed) {
                break;
            }
            let nodes = NODES_EVALUATED.load(Ordering::Relaxed);
            let hits = TT_HITS.load(Ordering::Relaxed);
            let stores = TT_STORES.load(Ordering::Relaxed);
            let elapsed = start.elapsed().as_secs();
            let rate = (nodes - last_nodes) / 10;
            let rhits = RTT_HITS.load(Ordering::Relaxed);
            let rstores = RTT_STORES.load(Ordering::Relaxed);
            let hit_rate = if stores > 0 {
                hits as f64 / (hits + stores) as f64 * 100.0
            } else {
                0.0
            };
            let rhit_rate = if rstores > 0 {
                rhits as f64 / (rhits + rstores) as f64 * 100.0
            } else {
                0.0
            };
            eprintln!(
                "[{elapsed:>4}s] nodes: {nodes:>12}  cop_tt: {hits:>10}/{stores:<10} ({hit_rate:.1}%)  rob_tt: {rhits:>10}/{rstores:<10} ({rhit_rate:.1}%)  nodes/s: {rate:>10}",
            );
            last_nodes = nodes;
        }
    });

    let num_threads = PAR_DEPTH.load(Ordering::Relaxed);
    let num_threads = if num_threads == 0 { 1 } else { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) };

    let (tx, rx) = std::sync::mpsc::channel();
    rayon::scope(|s| {
        for i in 0..num_threads {
            let tx = tx.clone();
            let tt = &tt;
            let rtt = &rtt;
            s.spawn(move |_| {
                let mut rng = Prng::new(1337 + i as u64 * 0x9e3779b97f4a7c15);
                let victor = state.cops_evaluate(0, tt, rtt, &mut rng);
                let _ = tx.send(victor);
            });
        }
    });    
    let victor = rx.recv().unwrap();
    DONE.store(true, Ordering::Relaxed);

    let elapsed = start.elapsed().as_secs_f64();
    let nodes = NODES_EVALUATED.load(Ordering::Relaxed);
    let hits = TT_HITS.load(Ordering::Relaxed);
    let stores = TT_STORES.load(Ordering::Relaxed);
    let rhits = RTT_HITS.load(Ordering::Relaxed);
    let rstores = RTT_STORES.load(Ordering::Relaxed);
    println!("{victor:?} wins in {elapsed:.1}s  (nodes: {nodes}, cop_tt: {hits}/{stores}, rob_tt: {rhits}/{rstores})");
}

macro_rules! dispatch_cops {
    ($size:literal, $cops:expr, [$($c:literal),+]) => {
        match $cops {
            $( $c => run::<$size, $c>(), )+
            _ => eprintln!("Unsupported cop count: {}. Supported: {}", $cops, stringify!($($c),+)),
        }
    };
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (size, cops) = match args.len() {
        1 => (SIZE as usize, NCOP),
        3 | 4 => {
            let s = args[1].parse::<usize>().expect("SIZE must be a number");
            let c = args[2].parse::<usize>().expect("NCOP must be a number");
            (s, c)
        }
        _ => {
            eprintln!("Usage: {} [SIZE NCOP [PAR_DEPTH]]", args[0]);
            eprintln!("  No args: uses compile-time defaults (SIZE={}, NCOP={})", SIZE, NCOP);
            eprintln!("  Example: {} 9 3", args[0]);
            eprintln!("  Example: {} 9 3 0   # serial (no parallelism)", args[0]);
            std::process::exit(1);
        }
    };

    // Set parallel depth: from 4th arg, or compile-time default
    let par_depth = args.get(3)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(PAR_MAX_DEPTH_DEFAULT);
    PAR_DEPTH.store(par_depth, Ordering::Relaxed);

    // Dispatch to monomorphized versions (sizes 3-16, cops 1-6)
    match size {
        3  => dispatch_cops!( 3, cops, [1,2,3,4,5,6]),
        4  => dispatch_cops!( 4, cops, [1,2,3,4,5,6]),
        5  => dispatch_cops!( 5, cops, [1,2,3,4,5,6]),
        6  => dispatch_cops!( 6, cops, [1,2,3,4,5,6]),
        7  => dispatch_cops!( 7, cops, [1,2,3,4,5,6]),
        8  => dispatch_cops!( 8, cops, [1,2,3,4,5,6]),
        9  => dispatch_cops!( 9, cops, [1,2,3,4,5,6]),
        10 => dispatch_cops!(10, cops, [1,2,3,4,5,6]),
        11 => dispatch_cops!(11, cops, [1,2,3,4,5,6]),
        12 => dispatch_cops!(12, cops, [1,2,3,4,5,6]),
        13 => dispatch_cops!(13, cops, [1,2,3,4,5,6]),
        14 => dispatch_cops!(14, cops, [1,2,3,4,5,6]),
        15 => dispatch_cops!(15, cops, [1,2,3,4,5,6]),
        16 => dispatch_cops!(16, cops, [1,2,3,4,5,6]),
        _  => eprintln!("Unsupported graph size: {}. Supported: 3-16", size),
    }
}
