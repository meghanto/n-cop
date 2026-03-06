#![feature(generic_const_exprs)]

use bitvec::array::BitArray;
use bitvec::macros::internal::funty::Fundamental;
use itertools::Itertools;
use rayon::iter::{ParallelBridge, ParallelIterator};
use std::fmt::{Display, Formatter};
use std::io::{Write, stdout};
use std::iter;
use std::ops::{Deref, DerefMut, Not};
use std::sync::atomic::{AtomicU64, Ordering};

type Game = GameState<Size10, 3>;
const TT_SIZE_LOG2: usize = 24; // 16M entries
const PARALLEL_DEPTH: usize = 0;

type BitBacking = usize;
type Size1 = ((), ());
type Size2 = ((), Size1);
type Size3 = ((), Size2);
type Size4 = ((), Size3);
type Size5 = ((), Size4);
type Size6 = ((), Size5);
type Size7 = ((), Size6);
type Size8 = ((), Size7);
type Size9 = ((), Size8);
type Size10 = ((), Size9);
type Size11 = ((), Size10);
type Size12 = ((), Size11);
type Size13 = ((), Size12);
type Size14 = ((), Size13);
type Size15 = ((), Size14);

#[derive(Hash, Eq, PartialEq, Copy, Clone)]
struct BitGraph<Repr>
where
    Repr: Count,
    [(); <Repr as Count>::N]:,
    [(); <Repr as Count>::N.div_ceil(size_of::<BitBacking>() * 8)]:,
{
    storage: [BitArray<[BitBacking; <Repr as Count>::N.div_ceil(size_of::<BitBacking>() * 8)]>;
        <Repr as Count>::N],
}

impl<Repr> Display for BitGraph<Repr>
where
    Repr: Count,
    [(); <Repr as Count>::N]:,
    [(); <Repr as Count>::N.div_ceil(size_of::<BitBacking>() * 8)]:,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for row in self.storage.iter().rev() {
            let mut bits = row[..Repr::N].iter();
            write!(f, "{}", bits.next().unwrap().as_usize())?;
            while let Some(next) = bits.next() {
                write!(f, " {}", next.as_usize())?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl<Repr> Default for BitGraph<Repr>
where
    Repr: Count,
    [(); <Repr as Count>::N]:,
    [(); <Repr as Count>::N.div_ceil(size_of::<BitBacking>() * 8)]:,
{
    fn default() -> Self {
        Self {
            storage: [BitArray::default(); <Repr as Count>::N],
        }
    }
}

impl<Repr> Deref for BitGraph<Repr>
where
    Repr: Count,
    [(); <Repr as Count>::N]:,
    [(); <Repr as Count>::N.div_ceil(size_of::<BitBacking>() * 8)]:,
{
    type Target = [BitArray<[BitBacking; <Repr as Count>::N.div_ceil(size_of::<BitBacking>() * 8)]>;
        <Repr as Count>::N];

    fn deref(&self) -> &Self::Target {
        &self.storage
    }
}

impl<Repr> DerefMut for BitGraph<Repr>
where
    Repr: Count,
    [(); <Repr as Count>::N]:,
    [(); <Repr as Count>::N.div_ceil(size_of::<BitBacking>() * 8)]:,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.storage
    }
}

impl<Repr> BitGraph<Repr>
where
    Repr: Count,
    [(); <Repr as Count>::N]:,
    [(); <Repr as Count>::N.div_ceil(size_of::<BitBacking>() * 8)]:,
{
    fn add_edge(&mut self, u: usize, v: usize) {
        self[u].set(v, true);
        self[v].set(u, true);
    }

    fn remove_edge(&mut self, u: usize, v: usize) {
        self[u].set(v, false);
        self[v].set(u, false);
    }

    fn has_edge(&self, u: usize, v: usize) -> bool {
        self[u][v]
    }

    fn is_0_1_connected(&self) -> bool {
        let mut matches = [self[0]; Repr::N];
        let mut last = [BitArray::default(); Repr::N];
        while last != matches {
            last = matches;

            let mut new_matches = BitArray::default();
            for (row, result) in self.iter().zip(matches.iter_mut()) {
                *result &= row;
                if result.any() {
                    new_matches |= *row;
                }
            }
            if new_matches[1] {
                return true;
            }

            matches = [new_matches; Repr::N];
        }
        false
    }
}

trait Count {
    const N: usize;
}

impl Count for () {
    const N: usize = 0;
}

impl<Tail> Count for ((), Tail)
where
    Tail: Count,
{
    const N: usize = Tail::N + 1;
}

#[derive(Hash, Eq, PartialEq, Copy, Clone)]
struct GameState<Repr, const COPS: usize>
where
    Repr: Count,
    [(); <Repr as Count>::N]:,
    [(); <Repr as Count>::N.div_ceil(size_of::<BitBacking>() * 8)]:,
{
    robber: BitGraph<Repr>,
    cop: BitGraph<Repr>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Victor {
    Robber,
    Cop,
}

// ============================================================
// Contracted representation: compress robber connected components
// into single nodes. The adj matrix stores unclaimed edge counts
// between/within components. Component sizes don't matter for
// game value — only the unclaimed counts determine the game tree.
// ============================================================

const MAX_COMP: usize = 16;

/// Pack up to 10 values (each 0..45, i.e. 6 bits) into a single u64.
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

/// Compute contracted canonical hash for a game state.
/// Returns None if robber already won (0 and 1 in same component).
fn contract_and_hash<Repr, const COPS: usize>(
    state: &GameState<Repr, COPS>,
) -> Option<u64>
where
    Repr: Count,
    [(); <Repr as Count>::N]:,
    [(); <Repr as Count>::N.div_ceil(size_of::<BitBacking>() * 8)]:,
{
    let n = Repr::N;

    // Step 1: Find connected components in robber graph (BFS, ignoring self-loops)
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

    let nc = n_comps as usize;

    // Check if 0 and 1 are in the same component (robber already won)
    if comp[0] == comp[1] {
        return None;
    }

    // Step 2: Build adjacency matrix of unclaimed edges between components
    let mut adj = [[0u8; MAX_COMP]; MAX_COMP];
    for (u, v) in state.remaining_edges() {
        let ci = comp[u] as usize;
        let cv = comp[v] as usize;
        adj[ci][cv] += 1;
        if ci != cv {
            adj[cv][ci] += 1;
        }
    }

    // Step 3: Remap so component of vertex 0 → index 0, vertex 1 → index 1
    let c0 = comp[0] as usize;
    let c1 = comp[1] as usize;
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

    // Apply remap
    let old_adj = adj;
    for i in 0..nc {
        for j in 0..nc {
            adj[remap[i] as usize][remap[j] as usize] = old_adj[i][j];
        }
    }

    // Step 4: Canonicalize
    // Sort 0 and 1 relative to each other
    if adj[0][..nc] > adj[1][..nc] {
        adj.swap(0, 1);
        for row in adj[..nc].iter_mut() {
            row.swap(0, 1);
        }
    }

    // Sort 2..nc by adjacency row (selection sort)
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

    // Step 5: Hash using splitmix64 over packed rows
    let mut hash = splitmix64(nc as u64);
    for i in 0..nc {
        let packed = pack_row(&adj[i], nc);
        hash = splitmix64(hash ^ packed);
    }

    Some(hash)
}

// ============================================================
// Lockless transposition table
// Entry layout: single u64
//   - 0 = empty
//   - nonzero: bits[63..1] = hash verification, bit[0] = result (1=Cop, 0=Robber)
// ============================================================

struct TranspositionTable {
    entries: Box<[AtomicU64]>,
    mask: usize,
}

impl TranspositionTable {
    fn new(size_log2: usize) -> Self {
        let size = 1usize << size_log2;
        let mut entries = Vec::with_capacity(size);
        for _ in 0..size {
            entries.push(AtomicU64::new(0));
        }
        TranspositionTable {
            entries: entries.into_boxed_slice(),
            mask: size - 1,
        }
    }

    fn probe(&self, hash: u64) -> Option<Victor> {
        let idx = (hash as usize) & self.mask;
        let entry = self.entries[idx].load(Ordering::Relaxed);
        if entry == 0 {
            return None;
        }
        // Verify: upper 63 bits must match
        if (entry & !1) == (hash & !1) {
            Some(if entry & 1 == 1 {
                Victor::Cop
            } else {
                Victor::Robber
            })
        } else {
            None
        }
    }

    fn store(&self, hash: u64, result: Victor) {
        let idx = (hash as usize) & self.mask;
        let value = (hash & !1)
            | match result {
                Victor::Cop => 1,
                Victor::Robber => 0,
            };
        // Don't store if hash happens to encode as 0 (empty sentinel)
        if value != 0 {
            self.entries[idx].store(value, Ordering::Relaxed);
        }
    }
}

// ============================================================
// Search
// ============================================================

trait SearchForWinner: Sized + Copy + Send + Sync {
    fn cops_evaluate(&self, depth: usize, tt: &TranspositionTable) -> Victor;
    fn robbers_evaluate(&self, depth: usize, tt: &TranspositionTable) -> Victor;
}

// Base case: 0-sized graph (unreachable)
impl<const COPS: usize> SearchForWinner for GameState<(), COPS> {
    fn cops_evaluate(&self, _depth: usize, _tt: &TranspositionTable) -> Victor {
        unreachable!("Zero-sized graph in search");
    }
    fn robbers_evaluate(&self, _depth: usize, _tt: &TranspositionTable) -> Victor {
        unreachable!("Zero-sized graph in search");
    }
}

impl<Tail, const COPS: usize> SearchForWinner for GameState<((), Tail), COPS>
where
    Tail: Count,
    [(); <((), Tail) as Count>::N]:,
    [(); <((), Tail) as Count>::N.div_ceil(size_of::<BitBacking>() * 8)]:,
    Self: Copy + Send + Sync,
{
    fn cops_evaluate(&self, depth: usize, tt: &TranspositionTable) -> Victor {
        if self.did_cop_win() {
            return Victor::Cop;
        }
        if self.did_robber_win() {
            return Victor::Robber;
        }

        // TT lookup via contracted canonical hash
        let hash = contract_and_hash(self);
        if let Some(h) = hash {
            if let Some(v) = tt.probe(h) {
                return v;
            }
        }

        let mut cop_choices = self
            .remaining_edges()
            .array_combinations::<COPS>()
            .peekable();

        let victor = if cop_choices.peek().is_none() {
            // Fewer than COPS edges remain — cop takes all, game over
            Victor::Cop
        } else if depth == PARALLEL_DEPTH {
            let found_cop_win = cop_choices
                .par_bridge()
                .any(|choices| {
                    let mut state = *self;
                    for (u, v) in choices {
                        state.cop.add_edge(u, v);
                    }
                    if depth == 0 {
                        let coords = choices
                            .iter()
                            .map(|(u, v)| format!("({u}, {v})"))
                            .join(", ");
                        print!("Cop removes edges {coords}");
                        stdout().flush().unwrap();
                    }
                    let eval = if state.did_cop_win() {
                        Victor::Cop
                    } else {
                        state.robbers_evaluate(depth, tt)
                    };
                    if depth == 0 {
                        if eval == Victor::Cop {
                            println!(", and wins!");
                        } else {
                            println!();
                        }
                    }
                    eval == Victor::Cop
                });
            if found_cop_win {
                Victor::Cop
            } else {
                Victor::Robber
            }
        } else {
            let mut victor = Victor::Robber;
            for choices in cop_choices {
                let mut state = *self;
                for (u, v) in choices {
                    state.cop.add_edge(u, v);
                }
                if depth == 0 {
                    let coords = choices
                        .iter()
                        .map(|(u, v)| format!("({u}, {v})"))
                        .join(", ");
                    print!("Cop removes edges {coords}");
                    stdout().flush().unwrap();
                }
                let eval = if state.did_cop_win() {
                    Victor::Cop
                } else {
                    state.robbers_evaluate(depth, tt)
                };
                if depth == 0 {
                    if eval == Victor::Cop {
                        println!(", and wins!");
                    } else {
                        println!();
                    }
                }
                if eval == Victor::Cop {
                    victor = Victor::Cop;
                    break;
                }
            }
            victor
        };

        // Store in TT
        if let Some(h) = hash {
            tt.store(h, victor);
        }

        victor
    }

    fn robbers_evaluate(&self, depth: usize, tt: &TranspositionTable) -> Victor {
        if self.did_cop_win() {
            return Victor::Cop;
        }
        if self.did_robber_win() {
            return Victor::Robber;
        }

        for (u, v) in self.remaining_edges() {
            let mut state = *self;
            state.robber.add_edge(u, v);
            if state.did_robber_win() || state.cops_evaluate(depth + 1, tt) == Victor::Robber {
                if depth == 0 {
                    println!(", but Robber can win by adding edge ({u}, {v})");
                }
                return Victor::Robber;
            }
        }
        Victor::Cop
    }
}

// ============================================================
// GameState basics
// ============================================================

impl<Repr, const COPS: usize> GameState<Repr, COPS>
where
    Repr: Count,
    [(); <Repr as Count>::N]:,
    [(); <Repr as Count>::N.div_ceil(size_of::<BitBacking>() * 8)]:,
{
    fn size(&self) -> usize {
        Repr::N
    }

    fn cops(&self) -> usize {
        COPS
    }

    fn initial_robber() -> BitGraph<Repr> {
        let mut robber = BitGraph {
            storage: [BitArray::default(); Repr::N],
        };
        for i in 0..Repr::N {
            robber[i].set(i, true);
        }
        robber
    }

    fn initial_cop() -> BitGraph<Repr> {
        let mut cop = BitGraph {
            storage: [!BitArray::ZERO; Repr::N],
        };
        for (i, row) in cop.iter_mut().take(Repr::N).enumerate() {
            row[..Repr::N].fill(false);
            row.set(i, true);
        }
        cop
    }

    fn new() -> Self {
        Self {
            robber: Self::initial_robber(),
            cop: Self::initial_cop(),
        }
    }

    fn did_robber_win(&self) -> bool {
        self.robber.is_0_1_connected()
    }

    fn did_cop_win(&self) -> bool {
        let mut inverted = BitGraph {
            storage: self.cop.clone(),
        };
        for row in inverted[..Repr::N].iter_mut() {
            let _ = row[..Repr::N].not();
        }
        !inverted.is_0_1_connected()
    }

    fn remaining_edges(&self) -> impl Iterator<Item = (usize, usize)> + Clone {
        let mut remaining = self.cop.storage;
        for (d, r) in remaining.iter_mut().zip(self.robber.iter()) {
            *d |= r;
            *d = d.not();
        }
        remaining.into_iter().enumerate().flat_map(|(u, mut bits)| {
            bits[..u].fill(false);
            iter::from_fn(move || {
                let idx = bits.leading_zeros();
                if idx < Repr::N {
                    bits.set(idx, false);
                    Some((u, idx))
                } else {
                    None
                }
            })
        })
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod test {
    use crate::{
        GameState, SearchForWinner, Size4, Size6, TranspositionTable, Victor, TT_SIZE_LOG2,
        contract_and_hash,
    };
    use itertools::Itertools;

    fn tt() -> TranspositionTable {
        TranspositionTable::new(16) // small TT for tests
    }

    #[test]
    fn boards_work() {
        let mut cop = GameState::<Size4, 1>::initial_cop();
        let mut robber = GameState::<Size4, 1>::initial_robber();

        cop.add_edge(0, 2);
        robber.add_edge(1, 3);

        assert!(cop.has_edge(0, 2));
        assert!(!cop.has_edge(1, 3));
        assert!(robber.has_edge(1, 3));
        assert!(!robber.has_edge(0, 2));
        assert!(!cop.has_edge(0, 1));
        assert!(!robber.has_edge(0, 1));

        cop.remove_edge(0, 2);
        assert!(!cop.has_edge(0, 2));

        assert!(!robber.is_0_1_connected());
        robber.add_edge(2, 3);
        assert!(!robber.is_0_1_connected());
        robber.add_edge(2, 0);
        assert!(robber.is_0_1_connected());
    }

    #[test]
    fn simple_victory() {
        let mut state = GameState::<Size6, 2>::new();
        state.cop.add_edge(0, 1);
        state.cop.add_edge(0, 2);
        state.robber.add_edge(4, 5);

        state.cop.add_edge(0, 3);
        state.cop.add_edge(0, 4);
        state.robber.add_edge(0, 5);

        state.cop.add_edge(1, 4);
        state.cop.add_edge(1, 5);
        state.robber.add_edge(1, 2);

        assert_eq!(state.cops_evaluate(0, &tt()), Victor::Cop);
    }

    #[test]
    fn cop_1_wins_4_clique() {
        let mut state = GameState::<Size4, 1>::new();
        assert_eq!(state.cops_evaluate(0, &tt()), Victor::Robber);

        state.cop.add_edge(0, 1);
        state.robber.add_edge(0, 2);
        assert_eq!(state.cops_evaluate(0, &tt()), Victor::Robber);
    }

    #[test]
    fn connectivity_1() {
        let mut robber = GameState::<((), ((), ((), ((), ((), ((), ())))))), 2>::initial_robber();
        robber.add_edge(0, 5);
        assert!(!robber.is_0_1_connected());
        robber.add_edge(1, 4);
        assert!(!robber.is_0_1_connected());
        robber.add_edge(2, 3);
        assert!(!robber.is_0_1_connected());
        robber.add_edge(5, 2);
        assert!(!robber.is_0_1_connected());
        robber.add_edge(5, 3);
        assert!(!robber.is_0_1_connected());
        robber.add_edge(4, 2);
        assert!(robber.is_0_1_connected());
    }

    #[test]
    fn edge_biterator_works() {
        let mut state = GameState::<Size4, 1>::new();
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

        let mut state = GameState::<Size4, 1>::new();
        let mut expected = (0..state.size())
            .tuple_combinations()
            .collect::<Vec<(_, _)>>();

        for _ in 0..expected.len() {
            let (u, v) = expected.pop().unwrap();
            state.robber.add_edge(u, v);
            assert_eq!(expected, state.remaining_edges().collect::<Vec<_>>());
        }

        let mut state = GameState::<Size4, 1>::new();
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
        // Two states that differ only by swapping vertices 2 and 3
        // should produce the same contracted hash
        let mut state_a = GameState::<Size4, 1>::new();
        state_a.cop.add_edge(0, 2);
        state_a.cop.add_edge(1, 3);

        let mut state_b = GameState::<Size4, 1>::new();
        state_b.cop.add_edge(0, 3);
        state_b.cop.add_edge(1, 2);

        let hash_a = contract_and_hash(&state_a);
        let hash_b = contract_and_hash(&state_b);
        assert_eq!(hash_a, hash_b);
    }

    #[test]
    fn contraction_robber_components() {
        // After robber connects 2-3, those vertices form a component.
        // Swapping the cop's edge targets within that component
        // shouldn't change the hash.
        let mut state_a = GameState::<Size6, 1>::new();
        state_a.robber.add_edge(2, 3);
        state_a.cop.add_edge(0, 4);

        let mut state_b = GameState::<Size6, 1>::new();
        state_b.robber.add_edge(2, 3);
        state_b.cop.add_edge(0, 5);

        // These aren't exactly the same contracted state because
        // 4 and 5 are in different singleton components.
        // But swapping 4 and 5 should give the same canonical hash
        // since they're both singletons with identical connectivity.
        let hash_a = contract_and_hash(&state_a);
        let hash_b = contract_and_hash(&state_b);
        assert_eq!(hash_a, hash_b);
    }
}

fn main() {
    let state = Game::new();
    println!(
        "Playing with {} cops on a clique of size {}",
        state.cops(),
        state.size()
    );

    let tt = TranspositionTable::new(TT_SIZE_LOG2);
    let victor = state.cops_evaluate(0, &tt);

    println!("{victor:?} wins");
}
