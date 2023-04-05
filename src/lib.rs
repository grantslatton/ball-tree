use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A `Point` is something that exists in some sort of metric space, and
/// can thus calculate its distance to another `Point`, and can be moved
/// a certain distance towards another `Point`.
pub trait Point: Sized + PartialEq {
    /// Distances should be positive, finite `f64`s. It is undefined behavior to
    /// return a negative, infinite, or `NaN` result.
    ///
    /// Distance should satisfy the triangle inequality. That is, `a.distance(c)`
    /// must be less or equal to than `a.distance(b) + b.distance(c)`.
    fn distance(&self, other: &Self) -> f64;

    /// If `d` is `0`, a point equal to the `self` should be returned. If `d` is equal
    /// to `self.distance(other)`, a point equal to `other` should be returned.
    /// Intermediate distances should be linearly interpolated between the two points,
    /// so if `d` is equal to `self.distance(other) / 2.0`, the midpoint should be
    /// returned.
    /// It is undefined behavior to use a distance that is negative, `NaN`, or greater
    /// than `self.distance(other)`.
    fn move_towards(&self, other: &Self, d: f64) -> Self;
}

fn midpoint<P: Point>(a: &P, b: &P) -> P {
    let d = a.distance(b);
    a.move_towards(b, d / 2.0)
}

/// Implement `Point` in the normal `D` dimensional Euclidean way for all arrays of floats. For example, a 2D point
/// would be a `[f64; 2]`.
impl<const D: usize> Point for [f64; D] {
    fn distance(&self, other: &Self) -> f64 {
        self.iter()
            .zip(other)
            .map(|(a, b)| (*a - *b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn move_towards(&self, other: &Self, d: f64) -> Self {
        let mut result = self.clone();

        let distance = self.distance(other);

        // Don't want to get a NaN in the division below
        if distance == 0.0 {
            return result;
        }

        let scale = d / self.distance(other);

        for i in 0..D {
            result[i] += scale * (other[i] - self[i]);
        }

        result
    }
}

// A little helper to allow us to use comparative functions on `f64`s by asserting that
// `NaN` isn't present.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
struct OrdF64(f64);
impl OrdF64 {
    fn new(x: f64) -> Self {
        assert!(!x.is_nan());
        OrdF64(x)
    }
}
impl Eq for OrdF64 {}
impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct Sphere<C> {
    center: C,
    radius: f64,
}

// Implementation of the "bouncing bubble" algorithm which essentially works like this:
// * Pick a point `a` that is farthest from `points[0]`
// * Pick a point `b` that is farthest from `a`
// * Use these two points to create an initial sphere centered at their midpoint and with
//   enough radius to encompass them
// * While there is still a point outside of this sphere, move the sphere towards that
//   point just enough to encompass that point, and grow the sphere radius by 1%
//
// This process will produce a non-optimal, but relatively snug fitting bounding sphere.

fn bounding_sphere<P: Point>(points: &[P]) -> Sphere<P> {
    assert!(points.len() >= 2);

    let a = &points
        .iter()
        .max_by_key(|a| OrdF64::new(points[0].distance(a)))
        .unwrap();
    let b = &points
        .iter()
        .max_by_key(|b| OrdF64::new(a.distance(b)))
        .unwrap();

    let mut center: P = midpoint(a, b);
    let mut radius = center.distance(b).max(std::f64::EPSILON);

    loop {
        match points.iter().filter(|p| center.distance(p) > radius).next() {
            None => break Sphere { center, radius },
            Some(p) => {
                let c_to_p = center.distance(&p);
                let d = c_to_p - radius;
                center = center.move_towards(p, d);
                radius = radius * 1.01;
            }
        }
    }
}

// Produce a partition of the given points with the following process:
// * Pick a point `a` that is farthest from `points[0]`
// * Pick a point `b` that is farthest from `a`
// * Partition the points into two groups: those closest to `a` and those closest to `b`
//
// This doesn't necessarily form the best partition, since `a` and `b` are not guaranteed
// to be the most distant pair of points, but it's usually sufficient.
fn partition<P: Point, V>(
    mut points: Vec<P>,
    mut values: Vec<V>,
) -> ((Vec<P>, Vec<V>), (Vec<P>, Vec<V>)) {
    assert!(points.len() >= 2);
    assert_eq!(points.len(), values.len());

    let a_i = points
        .iter()
        .enumerate()
        .max_by_key(|(_, a)| OrdF64::new(points[0].distance(a)))
        .unwrap()
        .0;

    let b_i = points
        .iter()
        .enumerate()
        .max_by_key(|(_, b)| OrdF64::new(points[a_i].distance(b)))
        .unwrap()
        .0;

    let (a_i, b_i) = (a_i.max(b_i), a_i.min(b_i));

    let (mut aps, mut avs) = (vec![points.swap_remove(a_i)], vec![values.swap_remove(a_i)]);
    let (mut bps, mut bvs) = (vec![points.swap_remove(b_i)], vec![values.swap_remove(b_i)]);

    for (p, v) in points.into_iter().zip(values) {
        if aps[0].distance(&p) < bps[0].distance(&p) {
            aps.push(p);
            avs.push(v);
        } else {
            bps.push(p);
            bvs.push(v);
        }
    }

    ((aps, avs), (bps, bvs))
}

#[derive(Debug, Clone)]
enum BallTreeInner<P, V> {
    Empty,
    Leaf(P, Vec<V>),
    // The sphere is a bounding sphere that encompasses this node (both children)
    Branch(
        Sphere<P>,
        Box<BallTreeInner<P, V>>,
        Box<BallTreeInner<P, V>>,
    ),
}

impl<P: Point, V> Default for BallTreeInner<P, V> {
    fn default() -> Self {
        BallTreeInner::Empty
    }
}

impl<P: Point, V> BallTreeInner<P, V> {
    fn new(mut points: Vec<P>, values: Vec<V>) -> Self {
        assert_eq!(
            points.len(),
            values.len(),
            "Given two vectors of differing lengths. points: {}, values: {}",
            points.len(),
            values.len()
        );

        if points.is_empty() {
            BallTreeInner::Empty
        } else if points.iter().all(|p| p == &points[0]) {
            BallTreeInner::Leaf(points.pop().unwrap(), values)
        } else {
            let sphere = bounding_sphere(&points);
            let ((aps, avs), (bps, bvs)) = partition(points, values);
            let (a_tree, b_tree) = (BallTreeInner::new(aps, avs), BallTreeInner::new(bps, bvs));
            BallTreeInner::Branch(sphere, Box::new(a_tree), Box::new(b_tree))
        }
    }

    fn distance(&self, p: &P) -> f64 {
        match self {
            BallTreeInner::Empty => std::f64::INFINITY,
            // The distance to a leaf is the distance to the single point inside of it
            BallTreeInner::Leaf(p0, _) => p.distance(p0),
            // The distance to a branch is the distance to the edge of the bounding sphere
            BallTreeInner::Branch(sphere, _, _) => p.distance(&sphere.center) - sphere.radius,
        }
    }
}

// We need a little wrapper to hold our priority queue elements for two reasons:
// * Rust's BinaryHeap is a max-heap, and we need a min-heap, so we invert the
//   ordering
// * We only want to order based on the first element, so we need a custom
//   implementation rather than deriving the order (which would require the value
//   to be orderable which is not necessary).
#[derive(Debug, Copy, Clone)]
struct Item<T>(f64, T);
impl<T> PartialEq for Item<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<T> Eq for Item<T> {}
impl<T> PartialOrd for Item<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0
            .partial_cmp(&other.0)
            .map(|ordering| ordering.reverse())
    }
}
impl<T> Ord for Item<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

// Maintain a priority queue of the nodes that are closest to the provided `point`. If we
// pop a leaf from the queue, that leaf is necessarily the next closest point. If we
// pop a branch from the queue, add its children. The priority of a node is its
// `distance` as defined above.
#[derive(Debug)]
struct Iter<'tree, 'query, P, V> {
    point: &'query P,
    balls: &'query mut BinaryHeap<Item<&'tree BallTreeInner<P, V>>>,
    i: usize,
    max_radius: f64,
}

impl<'tree, 'query, P: Point, V> Iterator for Iter<'tree, 'query, P, V> {
    type Item = (&'tree P, f64, &'tree V);

    fn next(&mut self) -> Option<Self::Item> {
        while self.balls.len() > 0 {
            // Peek in the leaf case, because we might need to visit this leaf multiple
            // times (if it has multiple values).
            if let Item(d, BallTreeInner::Leaf(p, vs)) = self.balls.peek().unwrap() {
                if self.i < vs.len() && *d <= self.max_radius {
                    self.i += 1;
                    return Some((p, *d, &vs[self.i - 1]));
                }
            }
            // Reset index for the next leaf we encounter
            self.i = 0;
            // Expand branch nodes
            if let Item(_, BallTreeInner::Branch(_, a, b)) = self.balls.pop().unwrap() {
                let d_a = a.distance(self.point);
                let d_b = b.distance(self.point);
                if d_a <= self.max_radius {
                    self.balls.push(Item(d_a, a));
                }
                if d_b <= self.max_radius {
                    self.balls.push(Item(d_b, b));
                }
            }
        }
        None
    }
}

impl<'tree, 'query, P, V> Drop for Iter<'tree, 'query, P, V> {
    fn drop(&mut self) {
        self.balls.clear();
    }
}

/// A `BallTree` is a space-partitioning data-structure that allows for finding
/// nearest neighbors in logarithmic time.
///
/// It does this by partitioning data into a series of nested bounding spheres
/// ("balls" in the literature). Spheres are used because it is trivial to
/// compute the distance between a point and a sphere (distance to the sphere's
/// center minus thte radius). The key observation is that a potential neighbor
/// is necessarily closer than all neighbors that are located inside of a
/// bounding sphere that is farther than the aforementioned neighbor.
///
/// Graphically:
/// ```text
///
///    A -
///    |  ----         distance(A, B) = 4
///    |      - B      distance(A, S) = 6
///     |
///      |
///      |    S
///        --------
///      /        G \
///     /   C        \
///    |           D |
///    |       F     |
///     \ E         /
///      \_________/
///```
///
/// In the diagram, `A` is closer to `B` than to `S`, and because `S` bounds
/// `C`, `D`, `E`, `F`, and `G`, it can be determined that `A` it is necessarily
/// closer to `B` than the other points without even computing exact distances
/// to them.
///
/// Ball trees are most commonly used as a form of predictive model where the
/// points are features and each point is associated with a value or label. Thus,
/// This implementation allows the user to associate a value with each point. If
/// this functionality is unneeded, `()` can be used as a value.
///
/// This implementation returns the nearest neighbors, their distances, and their
/// associated values. Returning the distances allows the user to perform some
/// sort of weighted interpolation of the neighbors for predictive purposes.
#[derive(Debug, Clone)]
pub struct BallTree<P, V>(BallTreeInner<P, V>);

impl<P: Point, V> Default for BallTree<P, V> {
    fn default() -> Self {
        BallTree(BallTreeInner::default())
    }
}

impl<P: Point, V> BallTree<P, V> {
    /// Construct this `BallTree`. Construction is somewhat expensive, so `BallTree`s
    /// are best constructed once and then used repeatedly.
    ///
    /// `panic` if `points.len() != values.len()`
    pub fn new(points: Vec<P>, values: Vec<V>) -> Self {
        BallTree(BallTreeInner::new(points, values))
    }

    /// Query this `BallTree`. The `Query` object provides a nearest-neighbor API and internally re-uses memory to avoid
    /// allocations on repeated queries.
    pub fn query(&self) -> Query<P, V> {
        Query {
            ball_tree: self,
            balls: Default::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Query<'tree, P, V> {
    ball_tree: &'tree BallTree<P, V>,
    balls: BinaryHeap<Item<&'tree BallTreeInner<P, V>>>,
}

impl<'tree, P: Point, V> Query<'tree, P, V> {
    /// Given a `point`, return an `Iterator` that yields neighbors from closest to
    /// farthest. To get the K nearest neighbors, simply `take` K from the iterator.
    ///
    /// The neighbor, its distance, and associated value are returned.
    pub fn nn<'query>(
        &'query mut self,
        point: &'query P,
    ) -> impl Iterator<Item = (&'tree P, f64, &'tree V)> + 'query {
        self.nn_within(point, f64::INFINITY)
    }

    /// The same as `nn` but only consider neighbors whose distance is `<= max_radius`.
    pub fn nn_within<'query>(
        &'query mut self,
        point: &'query P,
        max_radius: f64,
    ) -> impl Iterator<Item = (&'tree P, f64, &'tree V)> + 'query {
        let balls = &mut self.balls;
        balls.push(Item(self.ball_tree.0.distance(point), &self.ball_tree.0));
        Iter {
            point,
            balls,
            i: 0,
            max_radius,
        }
    }

    /// Return the size in bytes of the memory this `Query` is keeping internally to avoid allocation.
    pub fn allocated_size(&self) -> usize {
        self.balls.capacity() * std::mem::size_of::<Item<&'tree BallTreeInner<P, V>>>()
    }

    /// The `Query` object re-uses memory internally to avoid allocation. This method deallocates that memory.
    pub fn deallocate_memory(&mut self) {
        assert!(self.balls.is_empty());
        self.balls.shrink_to_fit();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaChaRng;
    use std::collections::HashSet;

    #[test]
    fn test_3d_points() {
        let mut rng: ChaChaRng = SeedableRng::seed_from_u64(0xcb42c94d23346e96);

        macro_rules! random_small_f64 {
            () => {
                rng.gen_range(-100.0, 100.0)
            };
        }

        macro_rules! random_3d_point {
            () => {
                [
                    random_small_f64!(),
                    random_small_f64!(),
                    random_small_f64!(),
                ]
            };
        }

        for _ in 0..1000 {
            let point_count = rng.gen_range(0, 100usize);

            let mut points = vec![];
            let mut values = vec![];

            for _ in 0..point_count {
                let point = random_3d_point!();
                let value = rng.gen::<u64>();
                points.push(point);
                values.push(value);
            }

            let tree = BallTree::new(points.clone(), values.clone());

            let mut query = tree.query();

            for _ in 0..100 {
                let point = random_3d_point!();
                let max_radius = rng.gen_range(0.0, 110.0);

                let expected_values = points
                    .iter()
                    .zip(&values)
                    .filter(|(p, _)| p.distance(&point) <= max_radius)
                    .map(|(_, v)| v)
                    .cloned()
                    .collect::<HashSet<_>>();

                let mut found_values = HashSet::new();

                let mut previous_d = 0.0;
                for (p, d, v) in query.nn_within(&point, max_radius) {
                    assert_eq!(point.distance(p), d);
                    assert!(d >= previous_d);
                    assert!(d <= max_radius);
                    previous_d = d;
                    found_values.insert(*v);
                }

                assert_eq!(expected_values, found_values);
            }

            assert!(query.allocated_size() > 0);
            // 2 (branching factor) * 8 (pointer size) * point count rounded up (max of 4 due to minimum vec sizing)
            assert!(query.allocated_size() <= 2 * 8 * point_count.next_power_of_two().max(4));

            query.deallocate_memory();
            assert_eq!(query.allocated_size(), 0);
        }
    }

    #[test]
    fn test_point_array_impls() {
        assert_eq!([5.0].distance(&[7.0]), 2.0);
        assert_eq!([5.0].move_towards(&[3.0], 1.0), [4.0]);

        assert_eq!([5.0, 3.0].distance(&[7.0, 5.0]), 2.0 * 2f64.sqrt());
        assert_eq!(
            [5.0, 3.0].move_towards(&[3.0, 1.0], 2f64.sqrt()),
            [4.0, 2.0]
        );

        assert_eq!([0.0, 0.0, 0.0, 0.0].distance(&[2.0, 2.0, 2.0, 2.0]), 4.0);
        assert_eq!(
            [0.0, 0.0, 0.0, 0.0].move_towards(&[2.0, 2.0, 2.0, 2.0], 8.0),
            [4.0, 4.0, 4.0, 4.0]
        );
    }
}
