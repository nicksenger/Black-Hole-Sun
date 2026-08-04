#![cfg(feature = "postgres")]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use testcontainers::runners::AsyncRunner;
use testcontainers_modules::postgres::Postgres;

#[tokio::test]
#[ignore = "manual helper: regenerates SQLx offline cache into black-hole-void/.sqlx"]
async fn regenerate_sqlx_offline_schema() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    assert!(crate_dir.join("Cargo.toml").is_file());

    let workspace_root = crate_dir
        .parent()
        .expect("workspace root should be parent of black-hole-void");
    assert!(workspace_root.join("Cargo.toml").is_file());

    let postgres = Postgres::default()
        .start()
        .await
        .expect("postgres testcontainer should start");
    let pg_port = postgres
        .get_host_port_ipv4(5432)
        .await
        .expect("postgres mapped port should be available");
    let connection_string = format!("postgres://postgres:postgres@127.0.0.1:{pg_port}/postgres");

    // 1) Fresh container is up, run migrations against it.
    let pool = sqlx::PgPool::connect(&connection_string)
        .await
        .expect("postgres pool should initialize for sqlx prepare");
    black_hole_void::migrate::migrate_postgres_v0(&pool)
        .await
        .expect("sqlx prepare schema migrations should initialize");
    let schema_version =
        sqlx::query_scalar::<_, i32>("SELECT version FROM void_schema_metadata WHERE id = 1")
            .fetch_optional(&pool)
            .await
            .expect("schema metadata query should succeed after migration");
    assert_eq!(
        schema_version,
        Some(0),
        "migration should initialize schema version row"
    );
    pool.close().await;

    // 2) Compile black-hole-void against the migrated container with SQLX_OFFLINE_DIR set.
    let source_sqlx_dir = workspace_root.join("target").join("sqlx");
    if source_sqlx_dir.exists() {
        fs::remove_dir_all(&source_sqlx_dir)
            .expect("existing target/sqlx directory should be removable");
    }
    fs::create_dir_all(&source_sqlx_dir).expect("target/sqlx directory should be creatable");

    let status = Command::new("cargo")
        .current_dir(workspace_root)
        .env("SQLX_OFFLINE", "false")
        .env("SQLX_OFFLINE_DIR", &source_sqlx_dir)
        .env("DATABASE_URL", &connection_string)
        .args(["check", "-p", "black-hole-void", "--features", "postgres"])
        .status()
        .expect("cargo check should execute");
    assert!(status.success(), "cargo check failed with status: {status}");

    let generated_files =
        list_files_recursive(&source_sqlx_dir).expect("target/sqlx should be readable");
    assert!(
        !generated_files.is_empty(),
        "expected sqlx cache files under target/sqlx, but found none"
    );

    // 3) Copy generated cache into black-hole-void/.sqlx.
    let target_sqlx_dir = crate_dir.join(".sqlx");
    if target_sqlx_dir.exists() {
        fs::remove_dir_all(&target_sqlx_dir).expect("existing .sqlx directory should be removable");
    }
    fs::create_dir_all(&target_sqlx_dir).expect(".sqlx directory should be creatable");
    copy_dir_all(&source_sqlx_dir, &target_sqlx_dir)
        .expect("target/sqlx should copy to black-hole-void/.sqlx");

    // 4) Verify the offline schema was generated.
    let copied_files =
        list_files_recursive(&target_sqlx_dir).expect(".sqlx directory should be readable");
    assert!(
        !copied_files.is_empty(),
        "expected generated SQLx offline schema under black-hole-void/.sqlx, but found none"
    );
    assert_eq!(
        copied_files.len(),
        generated_files.len(),
        "copied SQLx cache file count should match generated file count"
    );

    println!(
        "Generated {} SQLx offline schema file(s) into black-hole-void/.sqlx",
        copied_files.len()
    );
}

fn copy_dir_all(src: &Path, dst: &Path) -> std::io::Result<()> {
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());

        if file_type.is_dir() {
            fs::create_dir_all(&dst_path)?;
            copy_dir_all(&src_path, &dst_path)?;
        } else {
            fs::copy(&src_path, &dst_path)?;
        }
    }

    Ok(())
}

fn list_files_recursive(path: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let entry_path = entry.path();
        if file_type.is_dir() {
            files.extend(list_files_recursive(&entry_path)?);
        } else {
            files.push(entry_path);
        }
    }
    Ok(files)
}
