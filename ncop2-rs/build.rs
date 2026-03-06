fn main() {
    let par_depth = std::env::var("PAR_MAX_DEPTH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| {
            let cpus = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            (usize::BITS - 1 - cpus.leading_zeros()).max(1) as usize
        });
    println!("cargo::rustc-env=PAR_MAX_DEPTH={par_depth}");
    println!("cargo::rerun-if-env-changed=PAR_MAX_DEPTH");
    println!("cargo::rerun-if-changed=build.rs");
}
