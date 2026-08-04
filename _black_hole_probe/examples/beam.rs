#[cfg(not(test))]
#[path = "../tests/sun.rs"]
mod sun;

#[cfg(not(test))]
fn main() {
    sun::run_beam();
}
