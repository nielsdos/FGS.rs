/// Clamp method, because it's not yet available in Rust stable at the time of writing.
pub fn clamp<T: Ord>(x: T, min: T, max: T) -> T {
    x.min(max).max(min)
}

/// Clamp, but for f64. f64 has no "Ord" trait.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub fn clampf64(mut x: f64, min: f64, max: f64) -> f64 {
    // Due to the way NaNs work, using a simple if-else or even min & max, does result in sub-optimal
    // assembly code. Using a mutable variable in this way with this if-else construction generates
    // good assembly.
    if !(x > min) {
        x = min;
    }
    if !(x < max) {
        x = max;
    }
    x
}

/// Convert a character byte to uppercase.
#[inline]
pub fn upper(x: u8) -> u8 {
    x & !32
}
