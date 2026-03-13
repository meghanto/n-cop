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
    
    fn get_component(&self, start: usize) -> u16 {
        let mut visited: u16 = 1 << start;
        let mut frontier: u16 = 1 << start;
        
        while frontier != 0 {
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
        visited
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
    fn get_cop_subsets(&self) -> Vec<Vec<(usize, usize)>> {
        let mut comp_map = [0; 16];
        let mut visited = 0;
        for i in 0..self.k as usize {
            if (visited & (1 << i)) == 0 {
                let comp = self.red.get_component(i);
                visited |= comp;
                let mut f = comp;
                while f != 0 {
                    let b = f.trailing_zeros() as usize;
                    comp_map[b] = i;
                    f &= f - 1;
                }
            }
        }
        let root0 = comp_map[0];
        let root1 = comp_map[1];

        let mut edges = self.remaining_edges();
        if edges.is_empty() { return Vec::new(); }

        edges.sort_by_key(|&(u, v)| {
            let mut score = 0;
            let cu = comp_map[u];
            let cv = comp_map[v];
            if (cu == root0 && cv == root1) || (cu == root1 && cv == root0) {
                score += 500_000_000;
            } else if cu == root0 || cu == root1 || cv == root0 || cv == root1 {
                score += 50_000_000;
            }
            let mut mut_state = *self;
            mut_state.blue.add_edge(u, v);
            score += mut_state.eval_cop();
            score
        });
        edges.reverse();

        let pool_size = if self.n == 1 { 50 } else if self.n == 2 { 32 } else if self.n == 3 { 20 } else if self.n <= 6 { 16 } else { self.n + 4 };
        let mut top_candidates = Vec::new();
        for e in edges.iter().take(pool_size) {
            top_candidates.push(*e);
        }
        for &(u, v) in &edges {
            let cu = comp_map[u];
            let cv = comp_map[v];
            if (cu == root0 && cv == root1) || (cu == root1 && cv == root0) {
                if !top_candidates.contains(&(u, v)) {
                    top_candidates.push((u, v));
                }
            }
        }
        generate_subsets(&top_candidates, self.n)
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

    fn hash_state(state: &GameState, cop_turn: bool, picks_left: usize) -> u64 {
        let mut h = 0u64;
        for i in 0..state.k as usize {
            h ^= (state.blue.rows[i] as u64) << (i * 2);
            h ^= (state.red.rows[i] as u64) << (i * 2 + 1);
        }
        if cop_turn { h ^= 0x123456789ABCDEF0; }
        h ^= (picks_left as u64) << 50;
        h
    }

    fn minimax(&mut self, state: &GameState, depth: i32, mut alpha: i32, mut beta: i32, is_cop: bool, picks_left: usize) -> i32 {
        self.nodes += 1;
        if self.check_time() { return 0; }

        if state.did_cop_win() { return COP_WIN + depth; }
        if state.did_robber_win() { return ROBBER_WIN - depth; }

        let mut comp_map = [0; 16];
        let mut visited = 0;
        for i in 0..state.k as usize {
            if (visited & (1 << i)) == 0 {
                let comp = state.red.get_component(i);
                visited |= comp;
                let mut f = comp;
                while f != 0 {
                    let b = f.trailing_zeros() as usize;
                    comp_map[b] = i;
                    f &= f - 1;
                }
            }
        }
        
        let root0 = comp_map[0];
        let root1 = comp_map[1];

        let mut threats = 0;
        let mut e0 = [0i32; 16]; 
        let mut e1 = [0i32; 16]; 

        let mask = (1 << state.k) - 1;
        for u in 0..state.k as usize {
            let cu = comp_map[u];
            let mut avail = mask & !state.blue.rows[u] & !state.red.rows[u];
            avail &= !((1 << (u + 1)) - 1); // Only count each edge once
            while avail != 0 {
                let v = avail.trailing_zeros() as usize;
                let cv = comp_map[v];
                
                if (cu == root0 && cv == root1) || (cu == root1 && cv == root0) {
                    threats += 1;
                } else if cu == root0 {
                    e0[cv] += 1;
                } else if cv == root0 {
                    e0[cu] += 1;
                } else if cu == root1 {
                    e1[cv] += 1;
                } else if cv == root1 {
                    e1[cu] += 1;
                }
                avail &= avail - 1;
            }
        }
        
        let effective_picks = if is_cop { picks_left as i32 } else { 0 };
        
        let mut required_picks = threats;
        for i in 0..state.k as usize {
            if comp_map[i] == i && i != root0 && i != root1 {
                let cost0 = e0[i];
                let cost1 = e1[i];
                let cost_both = std::cmp::max(0, e0[i] - state.n as i32) + std::cmp::max(0, e1[i] - state.n as i32);
                required_picks += cost0.min(cost1).min(cost_both);
            }
        }

        if required_picks > effective_picks {
            return ROBBER_WIN - depth;
        }

        if depth == 0 {
            return state.eval_cop();
        }

        let whole_turn = !is_cop || picks_left == state.n;
        
        // --- Structural Fast-Path (Density / Fractional Arboricity check) ---
        // If we are at the root of a full turn evaluating the whole graph,
        // and the number of edges the robber could potentially use is overwhelmingly
        // large compared to the vertices, the graph is too dense for the Cop to win.
        // Nash-Williams: A graph has k edge-disjoint spanning trees only if |E| >= k(|V| - 1).
        // If we can verify 2n spanning trees, the robber mathematically wins.
        if whole_turn && depth == 0 {
            let mut avail_edges = 0;
            let mut avail_nodes_mask = 0u16;
            for u in 0..state.k as usize {
                let mut avail = ((1 << state.k) - 1) & !state.blue.rows[u];
                avail &= !((1 << (u + 1)) - 1); // only count u < v once
                let count = avail.count_ones();
                if count > 0 {
                    avail_edges += count;
                    avail_nodes_mask |= 1 << u;
                    avail_nodes_mask |= avail;
                }
            }
            
            let v_count = avail_nodes_mask.count_ones();
            if v_count > 1 {
                let required_edges_for_2n_trees = 2 * (state.n as u32) * (v_count - 1);
                
                // If there literally aren't enough edges left in the entire graph
                // to form the required spanning trees, we can't mathematically prove a win this way.
                // However, if the graph is *massively* dense (e.g., > 2n*(V-1)), we could 
                // return early. Since exactly packing them is slow, we use this as a hard cutoff
                // if we were to implement a full Matroid intersection later. 
                // For now, if E is incredibly small, we could conversely prove the Cop wins,
                // but the Minimax will find that instantly anyway.
            }
        }
        
        let hash = Self::hash_state(state, is_cop, picks_left);
        let orig_alpha = alpha;
        
        if let Some(entry) = self.tt.get(&hash) {
            if entry.depth >= depth {
                if entry.flag == 0 { return entry.val; }
                if entry.flag == 1 { alpha = alpha.max(entry.val); }
                if entry.flag == 2 { beta = beta.min(entry.val); }
                if alpha >= beta { return entry.val; }
            }
        }

        let val;
        if is_cop {
            let mut best = -INF;
            let subsets = state.get_cop_subsets();
            if subsets.is_empty() { return state.eval_cop(); }
            
            for subset in &subsets {
                let mut next = *state;
                for &(u, v) in subset {
                    next.blue.add_edge(u, v);
                }
                
                let score = self.minimax(&next, depth - 1, alpha, beta, false, 0);
                
                if self.timeout { return 0; }
                best = best.max(score);
                alpha = alpha.max(best);
                if beta <= alpha { break; }
            }
            val = best;
        } else {
            let mut edges = state.remaining_edges();
            if edges.is_empty() { return state.eval_cop(); }
            
            edges.sort_by_key(|&(u, v)| {
                let mut next = *state;
                next.red.add_edge(u, v);
                if next.did_robber_win() { -100000 } else { 0 }
            });
            
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

        if !self.timeout {
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

    let subsets = state.get_cop_subsets();
    if subsets.is_empty() {
        return js_sys::Int32Array::new_with_length(0);
    }
    
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

#[wasm_bindgen]
pub fn robber_best_move_wasm(k: u8, n: usize, blue_edges_flat: &[u8], red_edges_flat: &[u8]) -> js_sys::Int32Array {
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
    
    let mut comp_map = [0; 16];
    let mut visited = 0;
    for i in 0..state.k as usize {
        if (visited & (1 << i)) == 0 {
            let comp = state.red.get_component(i);
            visited |= comp;
            let mut f = comp;
            while f != 0 {
                let b = f.trailing_zeros() as usize;
                comp_map[b] = i;
                f &= f - 1;
            }
        }
    }
    
    let root0 = comp_map[0];
    let root1 = comp_map[1];
    
    let mut mu = [[0i32; 16]; 16];
    let mask = (1 << state.k) - 1;
    for u in 0..state.k as usize {
        let cu = comp_map[u];
        let mut avail = mask & !state.blue.rows[u] & !state.red.rows[u];
        avail &= !((1 << (u + 1)) - 1);
        while avail != 0 {
            let v = avail.trailing_zeros() as usize;
            let cv = comp_map[v];
            if cu != cv {
                mu[cu][cv] += 1;
                mu[cv][cu] += 1;
            }
            avail &= avail - 1;
        }
    }
    
    let get_any_edge = |a: usize, b: usize| -> Option<(usize, usize)> {
        for u in 0..state.k as usize {
            if comp_map[u] == a || comp_map[u] == b {
                let mut avail = mask & !state.blue.rows[u] & !state.red.rows[u];
                while avail != 0 {
                    let v = avail.trailing_zeros() as usize;
                    if (comp_map[u] == a && comp_map[v] == b) || (comp_map[u] == b && comp_map[v] == a) {
                        return Some((u, v));
                    }
                    avail &= avail - 1;
                }
            }
        }
        None
    };

    let m_r = |c: usize, my_mu: &[[i32; 16]; 16]| -> i32 { my_mu[root0][c] };
    let m_t = |c: usize, my_mu: &[[i32; 16]; 16]| -> i32 { my_mu[root1][c] };
    
    let gamma_t = |c: usize, my_mu: &[[i32; 16]; 16]| -> i32 {
        let mut sum = 0;
        for d in 0..state.k as usize {
            if comp_map[d] == d && d != root0 && d != root1 && d != c {
                sum += my_mu[c][d] * m_r(d, my_mu);
            }
        }
        sum
    };
    
    let gamma_r = |c: usize, my_mu: &[[i32; 16]; 16]| -> i32 {
        let mut sum = 0;
        for d in 0..state.k as usize {
            if comp_map[d] == d && d != root0 && d != root1 && d != c {
                sum += my_mu[c][d] * m_t(d, my_mu);
            }
        }
        sum
    };
    
    let sigma = |c: usize, my_mu: &[[i32; 16]; 16]| -> i32 {
        let mr = m_r(c, my_mu);
        let mt = m_t(c, my_mu);
        let c3 = std::cmp::max(0, mr - n as i32) + std::cmp::max(0, mt - n as i32);
        mr.min(mt).min(c3)
    };
    
    let lambda_after_move = |new_mu: &[[i32; 16]; 16]| -> i32 {
        let mut max_val = 0;
        for x in 0..state.k as usize {
            if comp_map[x] == x && x != root0 && x != root1 {
                max_val = max_val.max(m_r(x, new_mu).max(m_t(x, new_mu)));
            }
        }
        max_val
    };

    let mut o_comps = Vec::new();
    for i in 0..state.k as usize {
        if comp_map[i] == i && i != root0 && i != root1 {
            o_comps.push(i);
        }
    }
    
    if mu[root0][root1] > 0 {
        if let Some(e) = get_any_edge(root0, root1) {
            let arr = js_sys::Int32Array::new_with_length(2);
            arr.set_index(0, e.0 as i32);
            arr.set_index(1, e.1 as i32);
            return arr;
        }
    }
    
    let mut best_move = None;
    let mut best_score = -1_000_000_000;
    let mut best_tiebreak = -1_000_000_000;
    
    for &c in &o_comps {
        if m_t(c, &mu) > 0 {
            let mut s = -1_000_000_000;
            if m_r(c, &mu) > n as i32 {
                s = 1_000_000_000;
            } else {
                let mut new_mu = mu.clone();
                for d in 0..state.k as usize {
                    if d != c && d != root1 {
                        new_mu[root1][d] += new_mu[c][d];
                        new_mu[d][root1] += new_mu[c][d];
                        new_mu[c][d] = 0;
                        new_mu[d][c] = 0;
                    }
                }
                new_mu[c][root1] = 0; new_mu[root1][c] = 0;
                
                let lambda = lambda_after_move(&new_mu);
                s = gamma_t(c, &mu) - m_r(c, &mu) * m_t(c, &mu) - (n as i32 - m_r(c, &mu)) * lambda;
            }
            if s > best_score {
                best_score = s;
                best_move = Some(("absorb_T", c, c));
                best_tiebreak = -1_000_000_000;
            }
        }
        
        if m_r(c, &mu) > 0 {
            let mut s = -1_000_000_000;
            if m_t(c, &mu) > n as i32 {
                s = 1_000_000_000;
            } else {
                let mut new_mu = mu.clone();
                for d in 0..state.k as usize {
                    if d != c && d != root0 {
                        new_mu[root0][d] += new_mu[c][d];
                        new_mu[d][root0] += new_mu[c][d];
                        new_mu[c][d] = 0;
                        new_mu[d][c] = 0;
                    }
                }
                new_mu[c][root0] = 0; new_mu[root0][c] = 0;
                
                let lambda = lambda_after_move(&new_mu);
                s = gamma_r(c, &mu) - m_r(c, &mu) * m_t(c, &mu) - (n as i32 - m_t(c, &mu)) * lambda;
            }
            if s > best_score {
                best_score = s;
                best_move = Some(("absorb_R", c, c));
                best_tiebreak = -1_000_000_000;
            }
        }
    }
    
    for i in 0..o_comps.len() {
        for j in i+1..o_comps.len() {
            let a = o_comps[i];
            let b = o_comps[j];
            if mu[a][b] > 0 {
                let mr_e = m_r(a, &mu) + m_r(b, &mu);
                let mt_e = m_t(a, &mu) + m_t(b, &mu);
                
                let mut new_mu = mu.clone();
                for d in 0..state.k as usize {
                    if d != a && d != b {
                        new_mu[a][d] += new_mu[b][d];
                        new_mu[d][a] += new_mu[b][d];
                        new_mu[b][d] = 0;
                        new_mu[d][b] = 0;
                    }
                }
                new_mu[a][b] = 0; new_mu[b][a] = 0;
                
                let tiebreak = sigma(a, &new_mu);
                let mut s = -1_000_000_000;
                
                if mr_e > n as i32 && mt_e > n as i32 {
                    s = 1_000_000_000;
                } else {
                    let c_val = m_r(a, &mu) * m_t(b, &mu) + m_t(a, &mu) * m_r(b, &mu);
                    let lambda = lambda_after_move(&new_mu);
                    s = c_val - (n as i32) * lambda;
                }
                
                if s > best_score {
                    best_score = s;
                    best_tiebreak = tiebreak;
                    best_move = Some(("merge", a, b));
                } else if s == best_score && tiebreak > best_tiebreak {
                    best_tiebreak = tiebreak;
                    best_move = Some(("merge", a, b));
                }
            }
        }
    }
    
    let arr = js_sys::Int32Array::new_with_length(2);
    if let Some((kind, a, b)) = best_move {
        let edge = match kind {
            "absorb_T" => get_any_edge(a, root1),
            "absorb_R" => get_any_edge(a, root0),
            "merge" => get_any_edge(a, b),
            _ => None
        };
        if let Some(e) = edge {
            arr.set_index(0, e.0 as i32);
            arr.set_index(1, e.1 as i32);
            return arr;
        }
    }
    
    let edges = state.remaining_edges();
    if !edges.is_empty() {
        arr.set_index(0, edges[0].0 as i32);
        arr.set_index(1, edges[0].1 as i32);
    } else {
        return js_sys::Int32Array::new_with_length(0);
    }
    arr
}
