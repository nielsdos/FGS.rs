use crate::viterbi::STATE_COUNT;

/// A reference to a matrix row.
type RowRef<'a, T> = &'a mut [T; STATE_COUNT];

/// References to matrix entries we want to access.
/// Avoids bounds checks on the row, and reduces data accesses.
pub struct MatrixRefs<'a, T> {
    pub t_1: RowRef<'a, T>,
    pub t0: RowRef<'a, T>,
    pub t1: Option<RowRef<'a, T>>,
    pub t2: Option<RowRef<'a, T>>,
}

/// Split matrix in different pieces.
pub fn split_matrix<T>(matrix: &mut [[T; STATE_COUNT]], idx: usize) -> MatrixRefs<T> {
    let slice = &mut matrix[idx..];

    let (t_1, t0, t1, t2) = match slice {
        [tm1, t, t1, t2, ..] => (tm1, t, Some(t1), Some(t2)),
        [tm1, t, t1] => (tm1, t, Some(t1), None),
        [tm1, t] => (tm1, t, None, None),
        _ => panic!("invalid slice input"),
    };

    MatrixRefs {
        // First two always exist
        t_1,
        t0,
        // These don't always exist, it depends on how far we are in the array
        t1,
        t2,
    }
}
