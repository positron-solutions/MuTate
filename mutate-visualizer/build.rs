use std::fs;
use std::path::Path;
use std::process::Command;

fn main() {
    let shader_dir = "shaders";

    // Tell Cargo to rerun if anything in shaders/ changes
    println!("cargo:rerun-if-changed={}", shader_dir);

    let out_dir = std::env::var("OUT_DIR").unwrap();

    for entry in fs::read_dir(shader_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().map(|s| s == "slang").unwrap_or(false) {
            let output_path = Path::new(&out_dir)
                .join(path.file_name().unwrap())
                .with_extension("spv");

            let needs_build = if output_path.exists() {
                let input_meta = fs::metadata(&path).unwrap();
                let output_meta = fs::metadata(&output_path).unwrap();
                input_meta.modified().unwrap() > output_meta.modified().unwrap()
            } else {
                true
            };

            if needs_build {
                Command::new("slangc")
                    .arg(&path)
                    .arg("-o")
                    .arg(&output_path)
                    .status()
                    .unwrap();
            }
        }
    }
}
