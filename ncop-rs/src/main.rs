use itertools::Itertools;
use rayon::iter::ParallelBridge;
use rayon::iter::ParallelIterator;
use std::arch::x86_64::{
    __m256i, _mm256_alignr_epi8, _mm256_and_si256, _mm256_andnot_si256, _mm256_cmpeq_epi16,
    _mm256_extract_epi16, _mm256_or_si256, _mm256_set1_epi16, _mm256_set1_epi32,
    _mm256_setzero_si256, _mm256_testz_si256, _mm256_xor_si256,
};
use std::collections::HashMap;
use std::error::Error;
use std::io::{Write, stdout};
use std::iter;
use std::mem::{transmute, transmute_copy};
use std::num::NonZero;
use std::sync::RwLock;

const SIZE: u8 = 10;
const NCOP: usize = 3;
const CACHE_DEPTH: usize = 10;
const PARALLEL_DEPTH: usize = 0;

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
    fn add_edge(&mut self, u: u8, v: u8);
    fn remove_edge(&mut self, u: u8, v: u8);

    fn has_edge(&self, u: u8, v: u8) -> bool;

    fn is_0_1_connected(&self) -> bool;
}

impl Graph for AdjMatrix {
    fn add_edge(&mut self, u: u8, v: u8) {
        *self = unsafe {
            _mm256_or_si256(
                _mm256_and_si256(STAR_GRAPHS[u as usize], STAR_GRAPHS[v as usize]),
                *self,
            )
        }
    }

    fn remove_edge(&mut self, u: u8, v: u8) {
        *self = unsafe {
            _mm256_andnot_si256(
                _mm256_and_si256(STAR_GRAPHS[u as usize], STAR_GRAPHS[v as usize]),
                *self,
            )
        }
    }

    fn has_edge(&self, u: u8, v: u8) -> bool {
        unsafe {
            _mm256_testz_si256(
                _mm256_and_si256(STAR_GRAPHS[u as usize], STAR_GRAPHS[v as usize]),
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

fn print_graph(graph: AdjMatrix) {
    let data = unsafe { transmute::<_, [u16; 16]>(graph) };
    for i in (0..16).rev() {
        for j in 0..16 {
            if data[i] & (1 << j) != 0 {
                print!("1 ");
            } else {
                print!("0 ");
            }
        }
        println!();
    }
    println!();
}

fn robber_initial_size_k(k: usize) -> AdjMatrix {
    let mut data = [0u16; 16];
    for i in 0..k {
        data[i] = 1 << i;
    }
    unsafe { transmute(data) }
}

fn cop_initial_size_k(k: usize) -> AdjMatrix {
    let mut data = [0u16; 16];
    for i in 0..k {
        data[i] = (1 << k) - 1;
    }
    unsafe {
        _mm256_or_si256(
            _mm256_xor_si256(transmute(data), _mm256_set1_epi32(-1)),
            robber_initial_size_k(k),
        )
    }
}

#[derive(Copy, Clone)]
struct GameState<const K: u8, const COPS: usize> {
    cop: AdjMatrix,
    robber: AdjMatrix,
}

#[derive(Ord, PartialOrd, Eq, PartialEq, Copy, Clone, Debug)]
enum Victor {
    Cop,
    Robber,
}

// no specialization :(
trait NextFewerK {
    type Reduced;

    fn reduced(self) -> Self::Reduced;
}

macro_rules! define_nexts {
    ($u: literal, $v: literal) => {
        impl<const K: u8> NextFewerK for GameState<K, $u> {
            type Reduced = GameState<K, $v>;

            fn reduced(self) -> Self::Reduced {
                GameState::<K, $v> {
                    cop: self.cop,
                    robber: self.robber,
                }
            }
        }
    };
}
define_nexts!(5, 4);
define_nexts!(4, 3);
define_nexts!(3, 2);
define_nexts!(2, 1);
define_nexts!(1, 0);

// #[derive(Clone)]
// enum CacheEntry {
//     Unset,
//     Known(Victor),
//     Unknown(Box<[CacheEntry]>),
// }
//
// impl CacheEntry {
//     fn entry(&mut self, query: &mut [u64; 4], depth: usize) -> Option<&RwLock<CacheEntry>> {
//         if matches!(self, CacheEntry::Known(_)) {
//             return Some(self);
//         }
//
//         let mut idx = 0;
//         for e in query.iter_mut() {
//             let l = e.trailing_ones();
//             if l < 64 {
//                 idx += l;
//                 *e |= 1 << l;
//                 break;
//             }
//             idx += 64;
//         }
//         let idx = idx as usize;
//         if idx < 256 - depth {
//             let list = match self {
//                 CacheEntry::Unknown(l) => l,
//                 e @ CacheEntry::Unset => {
//                     *e = CacheEntry::Unknown(
//                         vec![CacheEntry::Unset; 256 - depth].into_boxed_slice(),
//                     );
//                     let CacheEntry::Unknown(list) = e else {
//                         unreachable!("By earlier assignment");
//                     };
//                     list
//                 }
//                 CacheEntry::Known(_) => unreachable!("By earlier check"),
//             };
//             list[idx as usize].entry(query, depth + 1)
//         } else {
//             Some(self)
//         }
//     }
//
//     fn get(&mut self, entry: &AdjMatrix) -> Option<&Victor> {
//         let mut query: [u64; 4] = unsafe { transmute_copy(entry) };
//         query.reverse();
//         match self.entry(&mut query, 0) {
//             None => None,
//             Some(CacheEntry::Known(v)) => Some(v),
//             Some(CacheEntry::Unknown(_) | CacheEntry::Unset) => None,
//         }
//     }
//
//     fn insert(&mut self, entry: AdjMatrix, victor: Victor) {
//         let mut query: [u64; 4] = unsafe { transmute(entry) };
//         query.reverse();
//         if let Some(entry) = self.entry(&mut query, 0) {
//             *entry = CacheEntry::Known(victor);
//         }
//     }
// }

type CacheEntry = HashMap<[u64; 8], Victor>;

trait SearchForWinner {
    fn robbers_turn_evaluate(&self, depth: usize, cache: &RwLock<CacheEntry>) -> Victor;

    fn cops_turn_evaluate(&self, depth: usize, cache: &RwLock<CacheEntry>) -> Victor;
    fn cops_turn_evaluate_inner(&self, depth: usize, cache: &RwLock<CacheEntry>) -> Victor;
}

impl<const K: u8, const COPS: usize> SearchForWinner for GameState<K, COPS>
where
    Self: NextFewerK,
    <Self as NextFewerK>::Reduced: SearchForWinner,
{
    fn cops_turn_evaluate(&self, depth: usize, cache: &RwLock<CacheEntry>) -> Victor {
        if self.did_cop_win() {
            Victor::Cop
        } else if self.did_robber_win() {
            Victor::Robber
        } else if depth < CACHE_DEPTH {
            let repr = self.flattened_repr();
            if let Some(result) = cache.read().unwrap().get(&repr).copied() {
                result
            } else {
                let result = self.cops_turn_evaluate_inner(depth, cache);
                cache.write().unwrap().insert(repr, result);
                result
            }
        } else {
            self.cops_turn_evaluate_inner(depth, cache)
        }
    }

    fn cops_turn_evaluate_inner(&self, depth: usize, cache: &RwLock<CacheEntry>) -> Victor {
        let mut cop_choices = self
            .remaining_edges()
            .array_combinations::<COPS>()
            .peekable();
        if cop_choices.peek().is_none() {
            self.reduced().cops_turn_evaluate_inner(depth, cache)
        } else if depth == PARALLEL_DEPTH {
            let victor = cop_choices
                .par_bridge()
                .fold_with(Victor::Robber, |existing, cop_choices| {
                    let victor = self.cops_turn_evaluate_innermost(depth, cache, cop_choices);
                    match (existing, victor) {
                        (_, Victor::Cop) | (Victor::Cop, _) => Victor::Cop,
                        _ => Victor::Robber,
                    }
                })
                .reduce(
                    || Victor::Robber,
                    |v1, v2| match (v1, v2) {
                        (_, Victor::Cop) | (Victor::Cop, _) => Victor::Cop,
                        _ => Victor::Robber,
                    },
                );
            victor
        } else {
            for cop_choices in cop_choices {
                if self.cops_turn_evaluate_innermost(depth, cache, cop_choices) == Victor::Cop {
                    return Victor::Cop;
                }
            }
            Victor::Robber
        }
    }

    fn robbers_turn_evaluate(&self, depth: usize, cache: &RwLock<CacheEntry>) -> Victor {
        if self.did_cop_win() {
            Victor::Cop
        } else if self.did_robber_win() {
            Victor::Robber
        } else {
            for (u, v) in self.remaining_edges() {
                debug_assert!(self.is_move_legal(u, v));
                let mut state = *self;
                state.robber.add_edge(u, v);
                if state.cops_turn_evaluate(depth + 1, cache) == Victor::Robber {
                    if depth == 0 {
                        println!(", but Robber can win by adding edge ({u}, {v})");
                    }
                    return Victor::Robber;
                }
            }
            Victor::Cop
        }
    }
}

// base case
impl<const K: u8> SearchForWinner for GameState<K, 0> {
    fn robbers_turn_evaluate(&self, _depth: usize, _cache: &RwLock<CacheEntry>) -> Victor {
        unreachable!("There are no edges left, but nobody has won?");
    }

    fn cops_turn_evaluate_inner(&self, _depth: usize, _cache: &RwLock<CacheEntry>) -> Victor {
        unreachable!("There are no edges left, but nobody has won?");
    }

    fn cops_turn_evaluate(&self, _depth: usize, _cache: &RwLock<CacheEntry>) -> Victor {
        if self.did_cop_win() {
            Victor::Cop
        } else if self.did_robber_win() {
            Victor::Robber
        } else {
            unreachable!("There are no edges left, but nobody has won?");
        }
    }
}

impl<const K: u8, const COPS: usize> GameState<K, COPS> {
    fn new() -> Self {
        Self {
            cop: cop_initial_size_k(K as usize),
            robber: robber_initial_size_k(K as usize),
        }
    }

    fn did_robber_win(&self) -> bool {
        self.robber.is_0_1_connected()
    }

    fn did_cop_win(&self) -> bool {
        let result =
            unsafe { !_mm256_xor_si256(self.cop, _mm256_set1_epi32(-1)).is_0_1_connected() };
        result
    }

    fn is_move_legal(&self, u: u8, v: u8) -> bool {
        u != v && unsafe { !_mm256_or_si256(self.cop, self.robber).has_edge(u, v) }
    }

    fn flattened_repr(&self) -> [u64; 8] {
        unsafe { transmute::<_, [u64; 8]>(*self) }
    }

    fn remaining_edges_graph(&self) -> AdjMatrix {
        unsafe {
            _mm256_xor_si256(
                _mm256_or_si256(self.cop, self.robber),
                _mm256_set1_epi32(-1),
            )
        }
    }

    fn remaining_edges(&self) -> impl Iterator<Item = (u8, u8)> + Clone {
        unsafe {
            let edges = self.remaining_edges_graph();
            let edges = transmute::<_, [u16; 16]>(edges);
            edges.into_iter().enumerate().flat_map(|(u, bits)| {
                iter::successors(NonZero::new(bits), |bits| {
                    NonZero::new(bits.get() & (bits.get() - 1))
                })
                .map(move |bits| (u as u8, bits.trailing_zeros() as u8))
                .filter(|(u, v)| *u < *v)
            })
        }
    }

    fn cops_turn_evaluate_innermost(
        &self,
        depth: usize,
        cache: &RwLock<CacheEntry>,
        cop_choices: [(u8, u8); COPS],
    ) -> Victor
    where
        Self: SearchForWinner,
    {
        let mut state = *self;
        for (u, v) in cop_choices {
            debug_assert!(state.is_move_legal(u, v));
            state.cop.add_edge(u, v);
        }
        if depth == 0 {
            let coords = cop_choices
                .iter()
                .map(|(u, v)| format!("({u}, {v})"))
                .join(", ");
            print!("Cop removes edges {coords}");
            stdout().flush().unwrap();
        }
        let eval = state.robbers_turn_evaluate(depth, cache);
        if eval == Victor::Cop {
            if depth == 0 {
                println!(", and wins!");
            }
            Victor::Cop
        } else {
            Victor::Robber
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        CacheEntry, GameState, Graph, SearchForWinner, Victor, cop_initial_size_k, print_graph,
        robber_initial_size_k,
    };
    use itertools::Itertools;
    use std::collections::HashSet;
    use std::sync::RwLock;

    #[test]
    fn boards_work() {
        let mut cop = cop_initial_size_k(4);
        let mut robber = robber_initial_size_k(4);

        print_graph(cop);
        print_graph(robber);

        cop.add_edge(0, 2);
        robber.add_edge(1, 3);

        // Test edge existence
        assert!(cop.has_edge(0, 2));
        assert!(!cop.has_edge(1, 3));
        assert!(robber.has_edge(1, 3));
        assert!(!robber.has_edge(0, 2));
        assert!(!cop.has_edge(0, 1));
        assert!(!robber.has_edge(0, 1));

        // Test removing edges
        cop.remove_edge(0, 2);
        assert!(!cop.has_edge(0, 2));

        // Test connectivity
        assert!(!robber.is_0_1_connected());
        robber.add_edge(2, 3);
        assert!(!robber.is_0_1_connected());
        robber.add_edge(2, 0);
        assert!(robber.is_0_1_connected());
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

        assert_eq!(
            state.cops_turn_evaluate(0, &RwLock::<CacheEntry>::default()),
            Victor::Cop
        );
    }

    #[test]
    fn cop_1_wins_4_clique() {
        let mut state = GameState::<4, 1>::new();
        assert_eq!(
            state.cops_turn_evaluate(0, &RwLock::<CacheEntry>::default()),
            Victor::Robber
        );

        state.cop.add_edge(0, 1);
        state.robber.add_edge(0, 2);
        assert_eq!(
            state.cops_turn_evaluate(0, &RwLock::<CacheEntry>::default()),
            Victor::Robber
        );
    }

    #[test]
    fn connectivity_1() {
        let mut robber = robber_initial_size_k(6);
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
        let mut state = GameState::<15, 1>::new();
        let edges = state.remaining_edges().collect::<Vec<_>>();
        let mut expected = (0..15u8).tuple_combinations().collect::<Vec<(_, _)>>();
        assert_eq!(expected, edges);

        for _ in 0..expected.len() {
            let (u, v) = expected.pop().unwrap();
            state.cop.add_edge(u, v);
            assert_eq!(expected, state.remaining_edges().collect::<Vec<_>>());
        }

        let mut state = GameState::<4, 1>::new();
        let mut expected = (0..4u8).tuple_combinations().collect::<Vec<(_, _)>>();

        for _ in 0..expected.len() {
            let (u, v) = expected.pop().unwrap();
            state.robber.add_edge(u, v);
            assert_eq!(expected, state.remaining_edges().collect::<Vec<_>>());
        }
    }

    #[test]
    fn iterator_equivalent() {
        let expected = (0..15u8)
            .tuple_combinations()
            .array_combinations()
            .collect::<HashSet<[(_, _); 2]>>();
        let mut collected = HashSet::new();
        for u in 0..15 {
            for v in (u + 1)..15 {
                for m in u..15 {
                    let n_start = if m == u { v } else { m } + 1;
                    for n in n_start..15 {
                        if m == u && n == v {
                            continue;
                        }
                        let mut result = [(u, v), (m, n)];
                        result.sort();
                        collected.insert(result);
                    }
                }
            }
        }
        assert_eq!(expected, collected);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let state = GameState::<SIZE, NCOP>::new();
    let result = state.cops_turn_evaluate(0, &RwLock::<CacheEntry>::default());
    println!("{result:?} wins");
    Ok(())
}
