#[cfg(not(test))]
#[path = "../tests/primordia.rs"]
mod primordia;

#[cfg(not(test))]
fn main() {
    primordia::run_beam();
}
