mod solver {
    #![allow(dead_code, unused_imports)]
    include!("../main.rs");

    use std::cmp::Ordering as CmpOrdering;
    #[derive(Default)]
    struct WitnessStore {
        cop: std::collections::HashMap<[u64; 2], (usize, usize)>,
        robber: std::collections::HashMap<[u64; 2], (usize, usize)>,
    }

    #[derive(Clone)]
    enum PvMove {
        Cop(Vec<(usize, usize)>),
        Robber((usize, usize)),
    }

    #[derive(Clone)]
    struct PvNode {
        side: &'static str,
        mv: PvMove,
        winner: Option<&'static str>,
        children: Vec<PvNode>,
    }

    fn reset_counters_local() {
        NODES_EVALUATED.store(0, Ordering::Relaxed);
        TT_HITS.store(0, Ordering::Relaxed);
        TT_STORES.store(0, Ordering::Relaxed);
        RTT_HITS.store(0, Ordering::Relaxed);
        RTT_STORES.store(0, Ordering::Relaxed);
    }

    fn component_distances(info: &ContractedInfo, start: usize) -> [u8; MAX_COMP] {
        let mut dist = [u8::MAX; MAX_COMP];
        let mut queue = [0usize; MAX_COMP];
        let mut head = 0usize;
        let mut tail = 0usize;
        dist[start] = 0;
        queue[tail] = start;
        tail += 1;

        while head < tail {
            let cur = queue[head];
            head += 1;
            let next_dist = dist[cur].saturating_add(1);
            for nxt in 0..info.n_comps as usize {
                if cur == nxt || info.adj[cur][nxt] == 0 || dist[nxt] != u8::MAX {
                    continue;
                }
                dist[nxt] = next_dist;
                queue[tail] = nxt;
                tail += 1;
            }
        }

        dist
    }

    fn orient_components(
        cu: usize,
        cv: usize,
        dist0: &[u8; MAX_COMP],
        dist1: &[u8; MAX_COMP],
    ) -> (usize, usize) {
        match dist0[cu].cmp(&dist0[cv]) {
            CmpOrdering::Less => (cu, cv),
            CmpOrdering::Greater => (cv, cu),
            CmpOrdering::Equal => match dist1[cv].cmp(&dist1[cu]) {
                CmpOrdering::Less => (cu, cv),
                CmpOrdering::Greater => (cv, cu),
                CmpOrdering::Equal => {
                    if cu <= cv {
                        (cu, cv)
                    } else {
                        (cv, cu)
                    }
                }
            },
        }
    }

    fn robber_threat_mode<const COPS: usize>(info: &ContractedInfo) -> bool {
        let c1 = info.comp[1] as usize;
        (0..info.n_comps as usize).any(|comp| comp != c1 && info.adj[comp][c1] > 0 && info.adj[comp][c1] as usize <= COPS)
    }

    fn robber_score<const COPS: usize>(
        u: usize,
        v: usize,
        info: &ContractedInfo,
        dist0: &[u8; MAX_COMP],
        dist1: &[u8; MAX_COMP],
    ) -> isize {
        let cu = info.comp[u] as usize;
        let cv = info.comp[v] as usize;
        let (a, b) = orient_components(cu, cv, dist0, dist1);
        let c0 = info.comp[0] as usize;
        let c1 = info.comp[1] as usize;
        if robber_threat_mode::<COPS>(info) {
            info.adj[b][c0] as isize - info.adj[c1][a] as isize
        } else {
            info.adj[b][c1] as isize - info.adj[c0][a] as isize
        }
    }

    fn collect_cop_edges<const K: u8, const COPS: usize>(
        state: &GameState<K, COPS>,
        info: &ContractedInfo,
        out: &mut [(usize, usize); MAX_EDGES],
    ) -> usize {
        let mut n_edges = 0usize;
        for (u, v) in state.remaining_edges() {
            if info.comp[u] == info.comp[v] {
                continue;
            }
            out[n_edges] = (u, v);
            n_edges += 1;
        }
        n_edges
    }

    fn find_first_cop_edge<const K: u8, const COPS: usize>(
        state: &GameState<K, COPS>,
        info: &ContractedInfo,
    ) -> Option<(usize, usize)> {
        let mut raw_edges = [(0usize, 0usize); MAX_EDGES];
        let n_edges = collect_cop_edges(state, info, &mut raw_edges);
        let mut edges = [(0usize, 0usize); MAX_EDGES];
        sort_edges_by_score(&raw_edges, n_edges, info, &mut edges);
        edges[..n_edges].first().copied()
    }

    fn robber_ordered_edges<const K: u8, const COPS: usize>(
        state: &GameState<K, COPS>,
        info: &ContractedInfo,
    ) -> Vec<(usize, usize)> {
        let c0 = info.comp[0] as usize;
        let c1 = info.comp[1] as usize;
        let mut direct = Vec::new();
        let mut other = Vec::new();
        let dist0 = component_distances(info, c0);
        let dist1 = component_distances(info, c1);

        for (u, v) in state.remaining_edges() {
            if info.comp[u] == info.comp[v] {
                continue;
            }
            let cu = info.comp[u] as usize;
            let cv = info.comp[v] as usize;
            if (cu == c0 && cv == c1) || (cu == c1 && cv == c0) {
                direct.push((u, v));
            } else {
                other.push((u, v));
            }
        }

        direct.sort_unstable();
        other.sort_unstable_by(|&(u1, v1), &(u2, v2)| {
            robber_score::<COPS>(u2, v2, info, &dist0, &dist1)
                .cmp(&robber_score::<COPS>(u1, v1, info, &dist0, &dist1))
                .then_with(|| (u1, v1).cmp(&(u2, v2)))
        });
        direct.extend(other);
        direct
    }

    fn cop_eval_pv<const K: u8, const COPS: usize>(
        state: &GameState<K, COPS>,
        picks_left: usize,
        tt: &TranspositionTable,
        rtt: &TranspositionTable,
        witnesses: &mut WitnessStore,
    ) -> Victor {
        if picks_left == 0 {
            if state.did_cop_win() {
                return Victor::Cop;
            }
            return robber_eval_pv(state, tt, rtt, witnesses);
        }

        NODES_EVALUATED.fetch_add(1, Ordering::Relaxed);

        if state.did_cop_win() {
            return Victor::Cop;
        }

        let info = match compute_components(state) {
            Some(info) => info,
            None => return Victor::Robber,
        };

        let h = hash_with_picks(contract_and_hash(&info), picks_left);
        if let Some(v) = tt.probe(h) {
            TT_HITS.fetch_add(1, Ordering::Relaxed);
            return v;
        }

        let threats = info.adj[info.comp[0] as usize][info.comp[1] as usize] as usize;
        if threats > picks_left {
            if let Some(edge) = find_first_cop_edge(state, &info) {
                witnesses.cop.entry(h).or_insert(edge);
            }
            tt.store(h, Victor::Robber);
            TT_STORES.fetch_add(1, Ordering::Relaxed);
            return Victor::Robber;
        }

        let mut raw_edges = [(0usize, 0usize); MAX_EDGES];
        let n_edges = collect_cop_edges(state, &info, &mut raw_edges);
        let mut edges = [(0usize, 0usize); MAX_EDGES];
        sort_edges_by_score(&raw_edges, n_edges, &info, &mut edges);

        if n_edges == 0 {
            tt.store(h, Victor::Cop);
            TT_STORES.fetch_add(1, Ordering::Relaxed);
            return Victor::Cop;
        }

        let mut victor = Victor::Robber;
        let mut chosen = None;
        let mut seen_pair = [[false; MAX_COMP]; MAX_COMP];
        for &(u, v) in &edges[..n_edges] {
            let cu = info.comp[u] as usize;
            let cv = info.comp[v] as usize;
            let (lo, hi) = if cu <= cv { (cu, cv) } else { (cv, cu) };
            if seen_pair[lo][hi] {
                continue;
            }
            seen_pair[lo][hi] = true;

            let mut next = *state;
            next.cop.add_edge(u, v);
            match cop_eval_pv(&next, picks_left - 1, tt, rtt, witnesses) {
                Victor::Cop => {
                    victor = Victor::Cop;
                    chosen = None;
                    break;
                }
                Victor::Robber => {
                    if chosen.is_none() {
                        chosen = Some((u, v));
                    }
                }
            }
        }

        if victor == Victor::Robber {
            if let Some(edge) = chosen {
                witnesses.cop.insert(h, edge);
            }
        }
        tt.store(h, victor);
        TT_STORES.fetch_add(1, Ordering::Relaxed);
        victor
    }

    fn robber_eval_pv<const K: u8, const COPS: usize>(
        state: &GameState<K, COPS>,
        tt: &TranspositionTable,
        rtt: &TranspositionTable,
        witnesses: &mut WitnessStore,
    ) -> Victor {
        let info = match compute_components(state) {
            Some(info) => info,
            None => return Victor::Robber,
        };

        let h = contract_and_hash(&info);
        if let Some(v) = rtt.probe(h) {
            RTT_HITS.fetch_add(1, Ordering::Relaxed);
            return v;
        }

        if info.adj[info.comp[0] as usize][info.comp[1] as usize] > 0 {
            let edge = robber_ordered_edges::<K, COPS>(state, &info)[0];
            witnesses.robber.insert(h, edge);
            rtt.store(h, Victor::Robber);
            RTT_STORES.fetch_add(1, Ordering::Relaxed);
            return Victor::Robber;
        }

        let edges = robber_ordered_edges::<K, COPS>(state, &info);
        let mut victor = Victor::Cop;
        for (u, v) in edges {
            let mut next = *state;
            next.robber.add_edge(u, v);
            if next.did_robber_win() || cop_eval_pv(&next, COPS, tt, rtt, witnesses) == Victor::Robber {
                witnesses.robber.insert(h, (u, v));
                victor = Victor::Robber;
                break;
            }
        }

        rtt.store(h, victor);
        RTT_STORES.fetch_add(1, Ordering::Relaxed);
        victor
    }

    fn ensure_cop_witness<const K: u8, const COPS: usize>(
        state: &GameState<K, COPS>,
        picks_left: usize,
        tt: &TranspositionTable,
        rtt: &TranspositionTable,
        witnesses: &mut WitnessStore,
    ) -> Option<(usize, usize)> {
        if picks_left == 0 {
            return None;
        }
        let info = compute_components(state)?;
        let h = hash_with_picks(contract_and_hash(&info), picks_left);
        if let Some(&edge) = witnesses.cop.get(&h) {
            return Some(edge);
        }
        if cop_eval_pv(state, picks_left, tt, rtt, witnesses) != Victor::Robber {
            return None;
        }
        witnesses.cop.get(&h).copied()
    }

    fn ensure_robber_witness<const K: u8, const COPS: usize>(
        state: &GameState<K, COPS>,
        tt: &TranspositionTable,
        rtt: &TranspositionTable,
        witnesses: &mut WitnessStore,
    ) -> Option<(usize, usize)> {
        let info = compute_components(state)?;
        let h = contract_and_hash(&info);
        if let Some(&edge) = witnesses.robber.get(&h) {
            return Some(edge);
        }
        if robber_eval_pv(state, tt, rtt, witnesses) != Victor::Robber {
            return None;
        }
        witnesses.robber.get(&h).copied()
    }

    fn build_pv_tree<const K: u8, const COPS: usize>(
        state: GameState<K, COPS>,
        tt: &TranspositionTable,
        rtt: &TranspositionTable,
        witnesses: &mut WitnessStore,
    ) -> Option<PvNode> {
        if cop_eval_pv(&state, COPS, tt, rtt, witnesses) != Victor::Robber {
            return None;
        }

        let mut after_cop = state;
        let mut picks_left = COPS;
        let mut cop_moves = Vec::with_capacity(COPS);
        while picks_left > 0 {
            let edge = ensure_cop_witness(&after_cop, picks_left, tt, rtt, witnesses)?;
            cop_moves.push(edge);
            after_cop.cop.add_edge(edge.0, edge.1);
            picks_left -= 1;
            if after_cop.did_cop_win() {
                return None;
            }
        }

        let robber_edge = ensure_robber_witness(&after_cop, tt, rtt, witnesses)?;
        let mut after_robber = after_cop;
        after_robber.robber.add_edge(robber_edge.0, robber_edge.1);
        let terminal = after_robber.did_robber_win();

        let mut cop_node = PvNode {
            side: "Cop",
            mv: PvMove::Cop(cop_moves),
            winner: None,
            children: vec![PvNode {
                side: "Robber",
                mv: PvMove::Robber(robber_edge),
                winner: if terminal { Some("Robber") } else { None },
                children: Vec::new(),
            }],
        };

        if !terminal {
            let child = build_pv_tree(after_robber, tt, rtt, witnesses)?;
            cop_node.children[0].children.push(child);
        }

        Some(cop_node)
    }

    fn write_indent(out: &mut String, depth: usize) {
        for _ in 0..depth {
            out.push_str("  ");
        }
    }

    fn write_move(out: &mut String, mv: &PvMove) {
        match mv {
            PvMove::Cop(edges) => {
                out.push('[');
                for (i, &(u, v)) in edges.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push('[');
                    out.push_str(&u.to_string());
                    out.push_str(", ");
                    out.push_str(&v.to_string());
                    out.push(']');
                }
                out.push(']');
            }
            PvMove::Robber((u, v)) => {
                out.push('[');
                out.push_str(&u.to_string());
                out.push_str(", ");
                out.push_str(&v.to_string());
                out.push(']');
            }
        }
    }

    fn write_node(out: &mut String, node: &PvNode, depth: usize) {
        write_indent(out, depth);
        out.push_str("{\n");
        write_indent(out, depth + 1);
        out.push_str("\"side\": \"");
        out.push_str(node.side);
        out.push_str("\",\n");
        write_indent(out, depth + 1);
        out.push_str("\"move\": ");
        write_move(out, &node.mv);
        out.push_str(",\n");
        if let Some(winner) = node.winner {
            write_indent(out, depth + 1);
            out.push_str("\"winner\": \"");
            out.push_str(winner);
            out.push_str("\",\n");
        }
        write_indent(out, depth + 1);
        out.push_str("\"children\": [");
        if node.children.is_empty() {
            out.push_str("]\n");
        } else {
            out.push('\n');
            for (i, child) in node.children.iter().enumerate() {
                if i > 0 {
                    out.push_str(",\n");
                }
                write_node(out, child, depth + 2);
            }
            out.push('\n');
            write_indent(out, depth + 1);
            out.push_str("]\n");
        }
        write_indent(out, depth);
        out.push('}');
    }

    pub fn run_pv<const K: u8, const COPS: usize>() {
        let tt = TranspositionTable::new(TT_SIZE_LOG2);
        let rtt = TranspositionTable::new(TT_SIZE_LOG2.saturating_sub(2).max(16));
        let state = GameState::<K, COPS>::new();
        let mut witnesses = WitnessStore::default();

        reset_counters_local();
        let start = Instant::now();
        if cop_eval_pv(&state, COPS, &tt, &rtt, &mut witnesses) != Victor::Robber {
            eprintln!("Initial position is not a robber win for K{} with {} cops.", K, COPS);
            std::process::exit(1);
        }

        let Some(root) = build_pv_tree(state, &tt, &rtt, &mut witnesses) else {
            eprintln!("Failed to construct a robber principal variation.");
            std::process::exit(1);
        };

        let elapsed = start.elapsed().as_secs_f64();
        let nodes = NODES_EVALUATED.load(Ordering::Relaxed);
        let hits = TT_HITS.load(Ordering::Relaxed);
        let stores = TT_STORES.load(Ordering::Relaxed);
        let rhits = RTT_HITS.load(Ordering::Relaxed);
        let rstores = RTT_STORES.load(Ordering::Relaxed);

        let mut out = String::new();
        out.push_str("{\n");
        out.push_str("  \"size\": ");
        out.push_str(&K.to_string());
        out.push_str(",\n");
        out.push_str("  \"cops\": ");
        out.push_str(&COPS.to_string());
        out.push_str(",\n");
        out.push_str("  \"winner\": \"Robber\",\n");
        out.push_str("  \"elapsed_secs\": ");
        out.push_str(&format!("{elapsed:.3}"));
        out.push_str(",\n");
        out.push_str("  \"nodes\": ");
        out.push_str(&nodes.to_string());
        out.push_str(",\n");
        out.push_str("  \"cop_tt\": [");
        out.push_str(&hits.to_string());
        out.push_str(", ");
        out.push_str(&stores.to_string());
        out.push_str("],\n");
        out.push_str("  \"rob_tt\": [");
        out.push_str(&rhits.to_string());
        out.push_str(", ");
        out.push_str(&rstores.to_string());
        out.push_str("],\n");
        out.push_str("  \"root\": \n");
        write_node(&mut out, &root, 1);
        out.push_str("\n}\n");
        print!("{out}");
    }
}

macro_rules! dispatch_cops {
    ($size:literal, $cops:expr, [$($c:literal),+]) => {
        match $cops {
            $( $c => solver::run_pv::<$size, $c>(), )+
            _ => {
                eprintln!("Unsupported cop count: {}. Supported: {}", $cops, stringify!($($c),+));
                std::process::exit(1);
            }
        }
    };
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} SIZE NCOP", args[0]);
        std::process::exit(1);
    }

    let size = args[1].parse::<usize>().expect("SIZE must be a number");
    let cops = args[2].parse::<usize>().expect("NCOP must be a number");

    match size {
        3 => dispatch_cops!(3, cops, [1, 2, 3, 4, 5, 6]),
        4 => dispatch_cops!(4, cops, [1, 2, 3, 4, 5, 6]),
        5 => dispatch_cops!(5, cops, [1, 2, 3, 4, 5, 6]),
        6 => dispatch_cops!(6, cops, [1, 2, 3, 4, 5, 6]),
        7 => dispatch_cops!(7, cops, [1, 2, 3, 4, 5, 6]),
        8 => dispatch_cops!(8, cops, [1, 2, 3, 4, 5, 6]),
        9 => dispatch_cops!(9, cops, [1, 2, 3, 4, 5, 6]),
        10 => dispatch_cops!(10, cops, [1, 2, 3, 4, 5, 6]),
        11 => dispatch_cops!(11, cops, [1, 2, 3, 4, 5, 6]),
        12 => dispatch_cops!(12, cops, [1, 2, 3, 4, 5, 6]),
        13 => dispatch_cops!(13, cops, [1, 2, 3, 4, 5, 6]),
        14 => dispatch_cops!(14, cops, [1, 2, 3, 4, 5, 6]),
        15 => dispatch_cops!(15, cops, [1, 2, 3, 4, 5, 6]),
        16 => dispatch_cops!(16, cops, [1, 2, 3, 4, 5, 6]),
        _ => {
            eprintln!("Unsupported graph size: {}. Supported: 3-16", size);
            std::process::exit(1);
        }
    }
}
