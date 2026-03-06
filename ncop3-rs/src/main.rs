use itertools::Itertools;
use nauty_pet::graph::CanonUnGraph;
use petgraph::algo::has_path_connecting;
use petgraph::graph::{DefaultIx, NodeIndex, UnGraph};
use petgraph::visit::{EdgeRef, IntoNodeReferences};
use std::collections::HashSet;
use std::ops::{ControlFlow, Deref, Not};
use std::str::FromStr;
use std::{iter, mem};
use weak_table::ptr_weak_key_hash_map::Entry;
use weak_table::{PtrWeakHashSet, PtrWeakKeyHashMap, WeakHashSet};

#[cfg(feature = "parallel")]
use rayon::iter::IntoParallelIterator;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;
#[cfg(feature = "parallel")]
use std::sync::RwLock;

#[cfg(feature = "parallel")]
type Rc<T> = std::sync::Arc<T>;
#[cfg(feature = "parallel")]
type Weak<T> = std::sync::Weak<T>;
#[cfg(not(feature = "parallel"))]
type Rc<T> = std::rc::Rc<T>;
#[cfg(not(feature = "parallel"))]
type Weak<T> = std::rc::Weak<T>;

type ImmutableGame = CanonUnGraph<bool, bool, DefaultIx>;
type MutableGame = UnGraph<bool, bool, DefaultIx>;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Victor {
    Robber,
    Cop,
}

impl Not for Victor {
    type Output = Self;

    fn not(self) -> Self::Output {
        match self {
            Victor::Robber => Victor::Cop,
            Victor::Cop => Victor::Robber,
        }
    }
}

trait HasObjectiveNodes {
    type NodeIdx;

    fn objective_nodes(&self) -> (Self::NodeIdx, Self::NodeIdx);
}

impl HasObjectiveNodes for MutableGame {
    type NodeIdx = NodeIndex<DefaultIx>;

    fn objective_nodes(&self) -> (Self::NodeIdx, Self::NodeIdx) {
        self.node_references()
            .filter_map(|(n, w)| w.then_some(n))
            .collect_tuple()
            .unwrap()
    }
}

trait VictoryConditions {
    fn did_robber_win(&self) -> bool;
    fn did_cop_win(&self) -> bool;
}

impl VictoryConditions for MutableGame {
    fn did_robber_win(&self) -> bool {
        let filtered = self.filter_map(|_n, w| Some(*w), |_e, w| w.then_some(*w));
        let (zero, one) = filtered.objective_nodes();
        has_path_connecting(&filtered, zero, one, None)
    }

    fn did_cop_win(&self) -> bool {
        let (zero, one) = self.objective_nodes();
        !has_path_connecting(self, zero, one, None)
    }
}

trait Game: Sized {
    fn init(k: u32) -> Self;

    fn evaluate_robber(&self) -> ControlFlow<(), HashSet<Self>>;
    fn evaluate_cop(&self, cops: u32) -> ControlFlow<(), HashSet<Self>>;
}

impl Game for ImmutableGame {
    fn init(k: u32) -> Self {
        assert!(k > 1);
        let mut graph = UnGraph::default();
        // add zero and one as objective nodes
        graph.add_node(true);
        graph.add_node(true);
        // add the remaining nodes
        for _ in 2..k {
            graph.add_node(false);
        }
        let indices = graph.node_indices().collect::<Vec<_>>();
        for (n1, n2) in indices.into_iter().tuple_combinations() {
            graph.add_edge(n1, n2, false);
        }

        ImmutableGame::from(graph)
    }

    fn evaluate_robber(&self) -> ControlFlow<(), HashSet<Self>> {
        let mut next = HashSet::new();
        for available in self
            .edge_references()
            .filter_map(|e| (!*e.weight()).then_some(e.id()))
        {
            let mut state = self.deref().clone();
            state[available] = true;
            if state.did_robber_win() {
                return ControlFlow::Break(());
            }
            next.insert(ImmutableGame::from(state));
        }
        // for state in next {
        //     if state.evaluate_cop(cops, cache) == Victor::Robber {
        //         return Victor::Robber; // cop will update the cache
        //     }
        // }
        ControlFlow::Continue(next)
    }

    fn evaluate_cop(&self, cops: u32) -> ControlFlow<(), HashSet<Self>> {
        let mut work = HashSet::new();
        work.insert(self.clone());
        for _ in 0..cops {
            // canonicalization is _really_ expensive; it is cheaper to use intermediaries here
            let mut next = HashSet::new();
            for state in work {
                for available in state
                    .edge_references()
                    .filter_map(|e| (!*e.weight()).then_some(e.id()))
                {
                    let mut state = state.deref().clone();
                    state.remove_edge(available);
                    if state.did_cop_win() {
                        return ControlFlow::Break(());
                    }
                    next.insert(ImmutableGame::from(state));
                }
            }
            work = next;
        }
        if work.is_empty() {
            return ControlFlow::Break(()); // we exhausted the queue
        }
        // for state in work {
        //     if state.evaluate_robber(cops, cache) == Victor::Cop {
        //         return Victor::Cop; // robber will update the cache
        //     }
        // }
        ControlFlow::Continue(work)
    }
}

struct Progress {
    cops: u32,
    root: Rc<ImmutableGame>,
    cache: WeakHashSet<Weak<ImmutableGame>>,
    nodes: PtrWeakKeyHashMap<
        Weak<ImmutableGame>,
        (
            PtrWeakHashSet<Weak<ImmutableGame>>,
            HashSet<Rc<ImmutableGame>>,
        ),
    >,
    work: Vec<Weak<ImmutableGame>>,
    turn: Victor,
    depth: usize,
}

impl Progress {
    fn new(state: ImmutableGame, cops: u32) -> Result<Self, Victor> {
        if state.did_cop_win() {
            Err(Victor::Cop)
        } else if state.did_robber_win() || cops == 0 {
            Err(Victor::Robber)
        } else {
            match state.evaluate_cop(cops) {
                ControlFlow::Continue(collected) => {
                    let mut cache = WeakHashSet::new();

                    let root = Rc::new(state);
                    cache.insert(root.clone());

                    let mut nodes = PtrWeakKeyHashMap::new();
                    let mut outgoing = HashSet::with_capacity(collected.len());
                    for out in collected {
                        let out = Rc::new(out);
                        cache.insert(out.clone());
                        outgoing.insert(out.clone());
                        nodes.insert(out, (iter::once(root.clone()).collect(), HashSet::new()));
                    }
                    let work = outgoing.iter().map(Rc::downgrade).collect();
                    nodes.insert(root.clone(), (PtrWeakHashSet::new(), outgoing));

                    Ok(Self {
                        cops,
                        root,
                        cache,
                        nodes,
                        work,
                        turn: Victor::Robber,
                        depth: 1,
                    })
                }
                ControlFlow::Break(_) => {
                    Err(Victor::Cop) // the cop trivially wins on the first move
                }
            }
        }
    }

    #[cfg(feature = "parallel")]
    fn step(mut self) -> ControlFlow<Victor, Self> {
        assert!(!self.work.is_empty());
        println!("Stepping depth {}", self.depth);
        let nodes = RwLock::new(self.nodes);
        let cache = RwLock::new(self.cache);
        let (work, mut losers) = mem::take(&mut self.work)
            .into_par_iter()
            .fold(
                || {
                    ControlFlow::Continue((
                        PtrWeakHashSet::<Weak<ImmutableGame>>::new(),
                        PtrWeakHashSet::<Weak<ImmutableGame>>::new(),
                    ))
                },
                |mut maybe_work, current| {
                    if let ControlFlow::Continue((work, losers)) = &mut maybe_work {
                        if let Some(current) = current.upgrade()
                            && nodes
                                .read()
                                .unwrap()
                                .get(&current)
                                .map_or(false, |(_, outgoing)| outgoing.is_empty())
                        {
                            let result = match self.turn {
                                Victor::Robber => current.deref().clone().evaluate_robber(),
                                Victor::Cop => current.deref().clone().evaluate_cop(self.cops),
                            };
                            match result {
                                ControlFlow::Continue(collected) => {
                                    if nodes
                                        .read()
                                        .unwrap()
                                        .get(&current)
                                        .map_or(true, |(_, outgoing)| !outgoing.is_empty())
                                    {
                                        return maybe_work; // this branch is already handled in another thread
                                    }
                                    let collected = collected
                                        .into_iter()
                                        .map(|g| {
                                            if let Some(existing) = cache.read().unwrap().get(&g) {
                                                existing
                                            } else {
                                                let rcd = Rc::new(g);
                                                let mut guard = cache.write().unwrap();
                                                if let Some(existing) = guard.get(&rcd) {
                                                    existing
                                                } else {
                                                    guard.insert(rcd.clone());
                                                    rcd
                                                }
                                            }
                                        })
                                        .collect::<HashSet<_>>();
                                    work.extend(collected.iter().cloned());
                                    {
                                        let mut guard = nodes.write().unwrap();
                                        if let Some((_, outgoing)) = guard.get_mut(&current) {
                                            if outgoing.is_empty() {
                                                *outgoing = collected.clone();
                                            } else {
                                                return maybe_work; // another thread sniped us
                                            }
                                        }
                                        for out in collected {
                                            match guard.entry(out) {
                                                Entry::Occupied(mut o) => {
                                                    o.get_mut().0.insert(current.clone());
                                                }
                                                Entry::Vacant(v) => {
                                                    v.insert((
                                                        iter::once(current.clone()).collect(),
                                                        HashSet::new(),
                                                    ));
                                                }
                                            }
                                        }
                                    }
                                }
                                ControlFlow::Break(_) => {
                                    losers.insert(current);
                                }
                            }
                        }
                    }
                    maybe_work
                },
            )
            .reduce(
                || {
                    ControlFlow::Continue((
                        PtrWeakHashSet::<Weak<ImmutableGame>>::new(),
                        PtrWeakHashSet::<Weak<ImmutableGame>>::new(),
                    ))
                },
                |mut r1, r2| {
                    if let ControlFlow::Continue((w1, l1)) = &mut r1 {
                        if let ControlFlow::Continue((w2, l2)) = r2 {
                            w1.extend(w2);
                            l1.extend(l2);
                        } else {
                            return r2;
                        }
                    }
                    r1
                },
            )?;
        self.nodes = nodes.into_inner().unwrap();
        let mut prop_step = 1usize;
        while !losers.is_empty() {
            let drained = mem::take(&mut losers);
            losers.reserve(drained.capacity());
            for loser in drained {
                if Rc::ptr_eq(&loser, &self.root) {
                    return ControlFlow::Break(self.turn);
                }
                if let Some((incoming, _outgoing)) = self.nodes.remove(&loser) {
                    for parent in incoming {
                        if self.nodes.get(&parent).is_none() {
                            continue; // already resolved
                        }
                        if prop_step % 2 == 0 {
                            // OR: one winning child suffices
                            losers.insert(parent);
                        } else {
                            // AND: all children must lose
                            if let Some((_, outgoing)) = self.nodes.get_mut(&parent) {
                                outgoing.remove(&loser);
                                if outgoing.is_empty() {
                                    losers.insert(parent);
                                }
                            }
                        }
                    }
                }
            }
            prop_step += 1;
        }
        self.cache = cache.into_inner().unwrap();
        self.work = work.into_iter().map(|rc| Rc::downgrade(&rc)).collect();
        self.turn = !self.turn;
        self.depth += 1;
        ControlFlow::Continue(self)
    }

    #[cfg(not(feature = "parallel"))]
    fn step(mut self) -> ControlFlow<Victor, Self> {
        assert!(!self.work.is_empty());
        println!("Stepping depth {}", self.depth);
        let mut losers = PtrWeakHashSet::<Weak<ImmutableGame>>::new();
        for current in mem::take(&mut self.work) {
            if let Some(current) = current.upgrade()
                && let Some((_, outgoing)) = self.nodes.get(&current)
                && outgoing.is_empty()
            {
                let result = match self.turn {
                    Victor::Robber => current.deref().clone().evaluate_robber(),
                    Victor::Cop => current.deref().clone().evaluate_cop(self.cops),
                };
                match result {
                    ControlFlow::Continue(collected) => {
                        if let Some((_, outgoing)) = self.nodes.get_mut(&current) {
                            assert!(outgoing.is_empty());
                            *outgoing = collected
                                .into_iter()
                                .map(|g| {
                                    if let Some(existing) = self.cache.get(&g) {
                                        existing
                                    } else {
                                        let rcd = Rc::new(g);
                                        self.cache.insert(rcd.clone());
                                        rcd
                                    }
                                })
                                .collect::<HashSet<_>>();
                            for out in outgoing.clone() {
                                self.work.push(Rc::downgrade(&out));
                                match self.nodes.entry(out) {
                                    Entry::Occupied(mut o) => {
                                        o.get_mut().0.insert(current.clone());
                                    }
                                    Entry::Vacant(v) => {
                                        v.insert((
                                            iter::once(current.clone()).collect(),
                                            HashSet::new(),
                                        ));
                                    }
                                }
                            }
                        }
                    }
                    ControlFlow::Break(_) => {
                        losers.insert(current);
                        let mut prop_step = 1usize;
                        while !losers.is_empty() {
                            let drained = mem::take(&mut losers);
                            losers.reserve(drained.capacity());
                            for loser in drained {
                                if Rc::ptr_eq(&loser, &self.root) {
                                    return ControlFlow::Break(self.turn);
                                }
                                if let Some((incoming, _outgoing)) = self.nodes.remove(&loser) {
                                    for parent in incoming {
                                        if self.nodes.get(&parent).is_none() {
                                            continue; // already resolved
                                        }
                                        if prop_step % 2 == 0 {
                                            // OR: one winning child suffices
                                            losers.insert(parent);
                                        } else {
                                            // AND: all children must lose
                                            if let Some((_, outgoing)) = self.nodes.get_mut(&parent)
                                            {
                                                outgoing.remove(&loser);
                                                if outgoing.is_empty() {
                                                    losers.insert(parent);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            prop_step += 1;
                        }
                    }
                }
            }
        }
        self.turn = !self.turn;
        self.depth += 1;
        ControlFlow::Continue(self)
    }
}

fn main() {
    let cops = std::env::args()
        .skip(1)
        .map(|n| u32::from_str(&n).expect("Invalid number of cops"))
        .next()
        .expect("Need a number of cops");
    // let cops = 2;

    for k in 3.. {
        let game = CanonUnGraph::init(k);
        let mut progress = match Progress::new(game, cops) {
            Ok(progress) => progress,
            Err(victor) => {
                println!("{victor:?} trivially wins for n={cops}, k={k}");
                continue;
            }
        };
        println!("Solving for n={cops}, k={k}...");
        let victor = loop {
            progress = match progress.step() {
                ControlFlow::Continue(progress) => progress,
                ControlFlow::Break(victor) => break victor,
            };
        };
        println!("{victor:?} wins for n={cops}, k={k}");
        if victor == Victor::Robber {
            break;
        }
    }
}
