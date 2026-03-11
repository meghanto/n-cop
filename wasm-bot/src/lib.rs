use wasm_bindgen::prelude::*;
use std::collections::HashMap;

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct AdjMatrix {
    rows: [u16; 16],
}

impl AdjMatrix {
    fn new() -> Self {
        Self { rows: [0; 16] }
    }
    
    fn full(k: u8) -> Self {
        let mut m = Self::new();
        let mask = (1 << k) - 1;
        for i in 0..k as usize {
            m.rows[i] = mask & !(1 << i);
        }
        m
    }

    fn add_edge(&mut self, u: usize, v: usize) {
        self.rows[u] |= 1 << v;
        self.rows[v] |= 1 << u;
    }
    
    fn has_edge(&self, u: usize, v: usize) -> bool {
        (self.rows[u] & (1 << v)) != 0
    }
    
    fn is_0_1_connected(&self) -> bool {
        let mut visited: u16 = 1;
        let mut frontier: u16 = 1;
        
        while frontier != 0 {
            if (visited & 2) != 0 {
                return true;
            }
            let mut next_frontier: u16 = 0;
            let mut f = frontier;
            while f != 0 {
                let bit = f.trailing_zeros() as usize;
                next_frontier |= self.rows[bit];
                f &= f - 1;
            }
            next_frontier &= !visited;
            visited |= next_frontier;
            frontier = next_frontier;
        }
        (visited & 2) != 0
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct GameState {
    k: u8,
    n: usize,
    blue: AdjMatrix, // cop deleted edges
    red: AdjMatrix,  // robber claimed edges
}

const INF: i32 = 1_000_000_000;
const COP_WIN: i32 = 100_000_000;
const ROBBER_WIN: i32 = -100_000_000;

impl GameState {
    fn did_cop_win(&self) -> bool {
        let mut available = AdjMatrix::full(self.k);
        for u in 0..self.k as usize {
            available.rows[u] &= !self.blue.rows[u];
        }
        !available.is_0_1_connected()
    }

    fn did_robber_win(&self) -> bool {
        self.red.is_0_1_connected()
    }

    fn remaining_edges(&self) -> Vec<(usize, usize)> {
        let mut edges = Vec::with_capacity(120);
        for u in 0..self.k as usize {
            let mut avail = ((1 << self.k) - 1) & !self.blue.rows[u] & !self.red.rows[u];
            avail &= !((1 << (u + 1)) - 1); // u < v
            while avail != 0 {
                let v = avail.trailing_zeros() as usize;
                edges.push((u, v));
                avail &= avail - 1;
            }
        }
        edges
    }
    
    // Very fast unweighted min cut from 0 to 1
    fn eval_cop(&self) -> i32 {
        if self.did_cop_win() { return COP_WIN; }
        if self.did_robber_win() { return ROBBER_WIN; }
        
        let k = self.k as usize;
        let mut cap = [[0i32; 16]; 16];
        for u in 0..k {
            for v in u+1..k {
                let c = if self.blue.has_edge(u, v) {
                    0
                } else if self.red.has_edge(u, v) {
                    10000 // effectively INF
                } else {
                    1
                };
                cap[u][v] = c;
                cap[v][u] = c;
            }
        }
        
        let mut flow = 0;
        loop {
            let mut parent = [-1i32; 16];
            let mut q = Vec::with_capacity(16);
            parent[0] = -2;
            q.push(0);
            
            let mut head = 0;
            let mut reached = false;
            while head < q.len() {
                let u = q[head];
                head += 1;
                
                if u == 1 {
                    reached = true;
                    break;
                }
                
                for v in 0..k {
                    if parent[v] == -1 && cap[u][v] > 0 {
                        parent[v] = u as i32;
                        q.push(v);
                    }
                }
            }
            
            if !reached {
                break;
            }
            
            let mut push = 10000;
            let mut curr = 1;
            while curr != 0 {
                let p = parent[curr] as usize;
                push = push.min(cap[p][curr]);
                curr = p;
            }
            
            flow += push;
            curr = 1;
            while curr != 0 {
                let p = parent[curr] as usize;
                cap[p][curr] -= push;
                cap[curr][p] += push;
                curr = p;
            }
        }
        
        -flow * 1000
    }
}

fn js_now() -> f64 {
    js_sys::Date::now()
}

struct Searcher {
    tt: HashMap<u64, TTEntry>,
    start_time: f64,
    time_limit_ms: f64,
    nodes: usize,
    timeout: bool,
}

#[derive(Copy, Clone)]
struct TTEntry {
    depth: i32,
    flag: u8,
    val: i32,
}

impl Searcher {
    fn new(time_limit_ms: f64) -> Self {
        Self {
            tt: HashMap::new(),
            start_time: js_now(),
            time_limit_ms,
            nodes: 0,
            timeout: false,
        }
    }

    fn check_time(&mut self) -> bool {
        if self.nodes & 4095 == 0 {
            if js_now() - self.start_time > self.time_limit_ms {
                self.timeout = true;
                return true;
            }
        }
        false
    }

    fn hash_state(state: &GameState, cop_turn: bool) -> u64 {
        let mut h = 0u64;
        for i in 0..state.k as usize {
            h ^= (state.blue.rows[i] as u64) << (i * 2);
            h ^= (state.red.rows[i] as u64) << (i * 2 + 1);
        }
        if cop_turn { h ^= 0x123456789ABCDEF0; }
        h
    }

    fn minimax(&mut self, state: &GameState, depth: i32, mut alpha: i32, mut beta: i32, is_cop: bool, picks_left: usize) -> i32 {
        self.nodes += 1;
        if self.check_time() { return 0; }

        if state.did_cop_win() { return COP_WIN + depth; }
        if state.did_robber_win() { return ROBBER_WIN - depth; }

        if depth == 0 {
            return state.eval_cop();
        }

        let whole_turn = !is_cop || picks_left == state.n;
        let hash = Self::hash_state(state, is_cop);
        let orig_alpha = alpha;
        
        if whole_turn {
            if let Some(entry) = self.tt.get(&hash) {
                if entry.depth >= depth {
                    if entry.flag == 0 { return entry.val; }
                    if entry.flag == 1 { alpha = alpha.max(entry.val); }
                    if entry.flag == 2 { beta = beta.min(entry.val); }
                    if alpha >= beta { return entry.val; }
                }
            }
        }

        let mut edges = state.remaining_edges();
        if edges.is_empty() { return state.eval_cop(); }

        // basic move ordering for robber turns
        if !is_cop {
            edges.sort_by_key(|&(u, v)| {
                let mut next = *state;
                next.red.add_edge(u, v);
                if next.did_robber_win() { -100000 } else { 0 }
            });
        }

        let val;
        if is_cop {
            let mut best = -INF;
            for &(u, v) in &edges {
                let mut next = *state;
                next.blue.add_edge(u, v);
                
                let score = if picks_left > 1 {
                    self.minimax(&next, depth, alpha, beta, true, picks_left - 1)
                } else {
                    self.minimax(&next, depth - 1, alpha, beta, false, 0)
                };
                
                if self.timeout { return 0; }
                best = best.max(score);
                alpha = alpha.max(best);
                if beta <= alpha { break; }
            }
            val = best;
        } else {
            let mut best = INF;
            for &(u, v) in &edges {
                let mut next = *state;
                next.red.add_edge(u, v);
                
                let score = self.minimax(&next, depth - 1, alpha, beta, true, state.n);
                
                if self.timeout { return 0; }
                best = best.min(score);
                beta = beta.min(best);
                if beta <= alpha { break; }
            }
            val = best;
        }

        if whole_turn && !self.timeout {
            let flag = if val <= orig_alpha { 2 } else if val >= beta { 1 } else { 0 };
            self.tt.insert(hash, TTEntry { depth, flag, val });
        }

        val
    }
}

fn generate_subsets(edges: &[(usize, usize)], size: usize) -> Vec<Vec<(usize, usize)>> {
    let mut result = Vec::new();
    fn backtrack(
        edges: &[(usize, usize)],
        size: usize,
        start: usize,
        current: &mut Vec<(usize, usize)>,
        result: &mut Vec<Vec<(usize, usize)>>
    ) {
        if current.len() == size || (start == edges.len() && !current.is_empty()) {
            if current.len() > 0 && current.len() == size {
                result.push(current.clone());
            } else if start == edges.len() && !current.is_empty() && current.len() < size {
                // If we don't have enough edges to fill `size`, just push what we have
                result.push(current.clone());
            }
            return;
        }
        for i in start..edges.len() {
            current.push(edges[i]);
            backtrack(edges, size, i + 1, current, result);
            current.pop();
        }
    }
    backtrack(edges, size, 0, &mut Vec::new(), &mut result);
    result
}

#[wasm_bindgen]
pub fn cop_best_move_wasm(k: u8, n: usize, blue_edges_flat: &[u8], red_edges_flat: &[u8], time_limit_ms: f64) -> js_sys::Int32Array {
    let mut state = GameState {
        k,
        n,
        blue: AdjMatrix::new(),
        red: AdjMatrix::new(),
    };

    for i in (0..blue_edges_flat.len()).step_by(2) {
        state.blue.add_edge(blue_edges_flat[i] as usize, blue_edges_flat[i+1] as usize);
    }
    for i in (0..red_edges_flat.len()).step_by(2) {
        state.red.add_edge(red_edges_flat[i] as usize, red_edges_flat[i+1] as usize);
    }

    let mut edges = state.remaining_edges();
    if edges.is_empty() {
        return js_sys::Int32Array::new_with_length(0);
    }

    // Heuristic sort: evaluate cop flow, but hardcode priority for 0-1 edges to prevent truncation blindspots!
    edges.sort_by_key(|&(u, v)| {
        let mut score = 0;
        
        // HIGHEST priority: the direct edge 0-1
        if (u == 0 && v == 1) || (u == 1 && v == 0) {
            score += 1000000;
        } 
        // HIGH priority: incident to 0 or 1
        else if u <= 1 || v <= 1 {
            score += 50000;
        }
        
        let mut mut_state = state;
        mut_state.blue.add_edge(u, v);
        score += mut_state.eval_cop(); // typically around -11000
        
        score
    });
    edges.reverse();

    // Dynamically size the candidate pool to avoid exploding subsets
    let pool_size = if n == 1 {
        50
    } else if n == 2 {
        32
    } else if n == 3 {
        20
    } else if n <= 6 {
        16
    } else {
        n + 4
    };

    let mut top_candidates = Vec::new();
    for e in edges.iter().take(pool_size) {
        top_candidates.push(*e);
    }
    
    // Always guarantee the direct edge is in the pool if it's available!
    let direct_edge = (0, 1);
    if state.remaining_edges().contains(&direct_edge) && !top_candidates.contains(&direct_edge) {
        top_candidates.push(direct_edge);
    }

    let subsets = generate_subsets(&top_candidates, n);
    
    let mut searcher = Searcher::new(time_limit_ms);
    let mut best_move = Vec::new();

    // Iterative deepening
    for depth in 1..=100 {
        if searcher.timeout { break; }
        
        let mut local_best_move = Vec::new();
        let mut best_score = -INF;
        
        for subset in &subsets {
            if searcher.timeout { break; }
            let mut next = state;
            for &(u, v) in subset {
                next.blue.add_edge(u, v);
            }
            
            let score = if next.did_cop_win() {
                COP_WIN + 1000
            } else {
                searcher.minimax(&next, depth - 1, -INF, INF, false, 0)
            };
            
            if score > best_score {
                best_score = score;
                local_best_move = subset.clone();
            }
            
            if best_score >= COP_WIN {
                break; // Found forced win
            }
        }
        
        if !searcher.timeout && !local_best_move.is_empty() {
            best_move = local_best_move;
        }
        
        if best_score >= COP_WIN || best_score <= ROBBER_WIN {
            break; // Stop deepening if win/loss forced
        }
    }

    // Fallback if we timed out on depth 1 and didn't find anything
    if best_move.is_empty() && !subsets.is_empty() {
        best_move = subsets[0].clone();
    }

    let arr = js_sys::Int32Array::new_with_length((best_move.len() * 2) as u32);
    for (i, &(u, v)) in best_move.iter().enumerate() {
        arr.set_index((i * 2) as u32, u as i32);
        arr.set_index((i * 2 + 1) as u32, v as i32);
    }
    arr
}
