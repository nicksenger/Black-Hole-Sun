mod common;

use common::*;

#[tokio::test]
async fn sun() {
    init_tracing();

    let model_path = match require_model_path("sun") {
        Some(path) => path,
        None => return,
    };

    todo!("sun test: {}", model_path);
}
