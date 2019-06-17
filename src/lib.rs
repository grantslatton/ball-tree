use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// A `Point` is something that exists in some sort of metric space, and
/// can thus calculate its distance to another `Point`, and can be moved
/// a certain distance towards another `Point`.
pub trait Point : Sized {
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

// A little helper to allow us to use comparative functions on `f64`s by asserting that
// `NaN` isn't present.
#[derive(PartialEq, PartialOrd)]
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
    let mut radius = center.distance(b);

    loop {
        match points.iter().filter(|p| center.distance(p) > radius).next() {
            None => break Sphere { center, radius },
            Some(p) => {
                let c_to_p = center.distance(&p);
                let d = c_to_p - radius;
                center = center.move_towards(p, d);
                radius = radius * 1.01;
            },
        }
    }
}

// Produce a partition of the given points with the following process:
// * Pick a point `a` that is farthest from `points[0]`
// * Pick a point `b` that is farthest from `a`
// * Partition the points into two groups: those closest to `a` and those closest to `b`
//
// This doesn't necessarily form the best partition, since `a` and `b` are not guaranteed
// to be the most distance pair of points, but it's usually sufficient.
fn partition<P: Point, V>(mut points: Vec<P>, mut values: Vec<V>) -> ((Vec<P>, Vec<V>), (Vec<P>, Vec<V>)) {
    assert!(points.len() >= 2);
    assert_eq!(points.len(), values.len());
    let a_i = points
        .iter()
        .enumerate()
        .max_by_key(|(_,a)| OrdF64::new(points[0].distance(a)))
        .unwrap().0;

    let b_i = points
        .iter()
        .enumerate()
        .max_by_key(|(_,b)| OrdF64::new(points[a_i].distance(b)))
        .unwrap().0;
    
    let (a_i, b_i) = (a_i.max(b_i.max(1)), a_i.min(b_i).min(points.len()-2));

    let (mut aps, mut avs) = (vec![points.swap_remove(a_i)], vec![values.swap_remove(a_i)]);
    let (mut bps, mut bvs) = (vec![points.swap_remove(b_i)], vec![values.swap_remove(b_i)]);

    for (p,v) in points.into_iter().zip(values) {
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

// We could add a `Null` variant to support empty trees, but would that actually be used/useful?
enum BallTreeInner<P, V> {
    Leaf(P, V),
    // The sphere is a bounding sphere that encompasses this node (both children)
    Branch(Sphere<P>, Box<BallTreeInner<P, V>>, Box<BallTreeInner<P, V>>),
}

impl <P: Point, V> BallTreeInner<P, V> {
    fn new(mut points: Vec<P>, mut values: Vec<V>) -> Self {
        assert!(points.len() > 0, "Cannot construct a ball-tree with zero points");
        assert_eq!(
            points.len(), values.len(), 
            "Given two vectors of differing lengths. points: {}, values: {}", 
            points.len(), 
            values.len()
        );

        if points.len() == 1 {
            let (p, v) = (points.swap_remove(0), values.swap_remove(0));
            BallTreeInner::Leaf(p, v)
        } else {
            let sphere = bounding_sphere(&points);
            let ((aps, avs), (bps, bvs)) = partition(points, values);
            let (a_tree, b_tree) = (BallTreeInner::new(aps, avs), BallTreeInner::new(bps, bvs));
            BallTreeInner::Branch(sphere, Box::new(a_tree), Box::new(b_tree))
        }
    }

    fn distance(&self, p: &P) -> f64 {
        match self {
            // The distance to a leaf is the distance to the single point inside of it
            BallTreeInner::Leaf(p0, _) => p.distance(p0),

            // The distance to a branch is the distance to the edge of the bounding sphere
            BallTreeInner::Branch(sphere, _, _) => p.distance(&sphere.center) - sphere.radius
        }
    }

    // Maintain a priority queue of the nodes that are closest to the provided `point`. If we
    // pop a leaf from the queue, that leaf is necessarily the next closest point. If we
    // pop a branch from the queue, add its children. The priority of a node is its 
    // `distance` as defined above.
    fn knn(&self, point: &P, mut k: usize, mut result: impl FnMut(&P, f64, &V)) {

        // We need a little wrapper to hold our priority queue elements for two reasons:
        // * Rust's BinaryHeap is a max-heap, and we need a min-heap, so we invert the
        //   ordering
        // * We only want to order based on the first element, so we need a custom 
        //   implementation rather than deriving the order (which would require the value
        //   to be orderable which is not necessary).

        struct Item<T>(f64, T);
        impl <T> PartialEq for Item<T> {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }
        impl <T> Eq for Item<T> {}
        impl <T> PartialOrd for Item<T> {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                self.0.partial_cmp(&other.0).map(|ordering| ordering.reverse())
            }
        }
        impl <T> Ord for Item<T> {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap()
            }
        }

        // Initialize the queue with our children as our only elements
        let mut balls: BinaryHeap<_> = vec![Item(0.0, self)].into();

        while let Some(Item(d, ball)) = balls.pop() {
            if k == 0 {
                break;
            }
            match ball {
                BallTreeInner::Leaf(p, v) => {
                    result(p, d, v);
                    k -= 1;
                },
                BallTreeInner::Branch(_, a, b) => {
                    let d_a = a.distance(point);
                    let d_b = b.distance(point);
                    balls.push(Item(d_a, a));
                    balls.push(Item(d_b, b));
                },
            }
        }
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
pub struct BallTree<P, V>(BallTreeInner<P, V>);

impl <P: Point, V> BallTree<P, V> {

    /// Construct this `BallTree`. Construction is somewhat expensive, so `BallTree`s
    /// are best constructed once and then used repeatedly.
    ///
    /// `panic` if `points.len() != values.len()`
    pub fn new(points: Vec<P>, values: Vec<V>) -> Self {
        BallTree(BallTreeInner::new(points, values))
    }

    /// Given a `point` and a number of neigbors to look for, find the `k` nearest
    /// neighbors (or fewer, if the ball tree contains fewer than `k` points).
    ///
    /// The neighbor, its distance, and associated value is returned.
    ///
    /// We allow the caller to pass in a result-consuming closure rather than
    /// return a `Vec` to allow the caller to control allocation.
    pub fn knn(&self, point: &P, k: usize, result: impl FnMut(&P, f64, &V)) {
        self.0.knn(point, k, result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct TestPoint(f64); 

    impl Point for TestPoint {
        fn distance(&self, other: &Self) -> f64 {
            (self.0 - other.0).abs()
        }

        fn move_towards(&self, other: &Self, d: f64) -> Self {
            if self.0 > other.0 {
                TestPoint(self.0 - d)
            } else {
                TestPoint(self.0 + d)
            }
        }
    }

    #[test]
    fn test() {
        let points = (0_u32..1000).map(|x| TestPoint(x as f64)).collect::<Vec<_>>();
        let values = (0_u32..1000).map(|x| x.pow(2)).collect::<Vec<_>>();

        let tree = BallTree::new(points.clone(), values.clone());

        let n = 10;

        for p in vec![123.4, 567.8, 99999.9] {
            let mut results = vec![];
            tree.knn(&TestPoint(p), n, |k, d, v| results.push((*k, d, *v)));
            
            let mut reference = points
                .clone()
                .into_iter()
                .zip(values.clone())
                .map(|(k,v)| (k, (k.0 - p).abs(), v))
                .collect::<Vec<_>>();

            reference.sort_by_key(|(_, d, _)| OrdF64::new(*d));

            let expected_results = reference
                .into_iter()
                .take(n)
                .collect::<Vec<_>>();

            assert_eq!(expected_results, results);
        }

    }

}
