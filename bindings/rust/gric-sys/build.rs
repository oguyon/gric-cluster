use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let build_dir = manifest_dir.join("../../../build");

    if build_dir.exists() {
        println!("cargo:rustc-link-search=native={}", build_dir.display());
    }

    println!("cargo:rustc-link-lib=gric");
    println!("cargo:rerun-if-changed=../../../include/gric/gric.h");
}
