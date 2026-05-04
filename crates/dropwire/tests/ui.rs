// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

fn main() {
    let mut config = ui_test::Config::rustc("tests/ui");

    let deps = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/debug/deps");

    config.program.args.push("-L".into());
    config.program.args.push(deps.as_os_str().into());

    let rlib = std::fs::read_dir(&deps)
        .expect("deps dir missing")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("libmutate_dropwire") && n.ends_with(".rlib"))
                .unwrap_or(false)
        })
        .expect("libmutate_dropwire*.rlib not found");

    config.program.args.push("--extern".into());
    config
        .program
        .args
        .push(format!("mutate_dropwire={}", rlib.display()).into());

    config.bless_command = Some("cargo test --test ui -- --bless".into());

    ui_test::run_tests(config).unwrap();
}
