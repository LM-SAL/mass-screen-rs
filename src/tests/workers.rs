use crate::screen::mass_screen_core;
use crate::workers::WorkerPool;
use rayon::ThreadPoolBuilder;
use std::cell::Cell;

thread_local! {
    pub(crate) static OBSERVED_WIDTH: Cell<Option<usize>> = const { Cell::new(None) };
}

#[test]
fn the_kernel_inherits_the_selected_pool() {
    let single = WorkerPool(ThreadPoolBuilder::new().num_threads(1).build().unwrap());
    let ambient = ThreadPoolBuilder::new().num_threads(2).build().unwrap();
    ambient.install(|| {
        let mut mask = [0.0];
        mass_screen_core(&[1.0], &[1.0], &mut mask, 1, 1, 1, 1.0, 0.0, Some(&single));
        assert_eq!(single.0.install(|| OBSERVED_WIDTH.take()), Some(1));
        mass_screen_core(&[1.0], &[1.0], &mut mask, 1, 1, 1, 1.0, 0.0, None);
        assert_eq!(OBSERVED_WIDTH.take(), Some(2));
    });
}
