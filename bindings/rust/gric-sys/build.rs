use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    if let Ok(dir) = env::var("GRIC_BUILD_DIR") {
        println!("cargo:rustc-link-search=native={}", dir);
    } else if let Ok(dir) = env::var("GRIC_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", dir);
    } else {
        let candidates = [
            manifest_dir.join("../../../build"),
            manifest_dir.join("../../../build-asan"),
            manifest_dir.join("../../../build-milk"),
        ];
        for dir in &candidates {
            if dir.exists() {
                println!("cargo:rustc-link-search=native={}", dir.display());
            }
        }
    }

    println!("cargo:rustc-link-lib=gric");
    println!("cargo:rerun-if-changed=../../../include/gric/gric.h");
}
