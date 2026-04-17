use ark_ff::{BigInteger, PrimeField};
use ark_goldilocks::Goldilocks as F;
use ark_goldilocks::Goldilocks;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use hash::SelectedHasher;
use hash::selected::HASH_BYTES;
use hash::sha3::Digest;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// =======================
/// Serialization helpers
/// =======================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SerFr(pub F);

impl From<F> for SerFr {
    fn from(x: F) -> Self {
        SerFr(x)
    }
}

impl From<SerFr> for F {
    fn from(w: SerFr) -> F {
        w.0
    }
}

impl Serialize for SerFr {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut out = [0u8; 8];
        let bytes = self.0.into_bigint().to_bytes_le();
        let n = bytes.len().min(8);
        out[..n].copy_from_slice(&bytes[..n]);
        serializer.serialize_bytes(&out)
    }
}

impl<'de> Deserialize<'de> for SerFr {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes: Vec<u8> = Deserialize::deserialize(deserializer)?;
        let mut v = 0u64;
        for (i, &b) in bytes.iter().take(8).enumerate() {
            v |= (b as u64) << (i * 8);
        }
        Ok(SerFr(Goldilocks::from(v)))
    }
}

pub fn field_to_bytes(field: &Goldilocks) -> [u8; 8] {
    let mut out = [0u8; 8];
    let bytes = field.into_bigint().to_bytes_le();
    let n = bytes.len().min(8);
    out[..n].copy_from_slice(&bytes[..n]);
    out
}

pub fn bytes_to_field(bytes: &[u8; 8]) -> Goldilocks {
    let mut v = 0u64;
    for (i, &b) in bytes.iter().enumerate() {
        v |= (b as u64) << (i * 8);
    }
    Goldilocks::from(v)
}

/// =======================
/// Hash finalization helper
/// =======================

#[inline]
fn finalize_hash(h: SelectedHasher) -> [u8; HASH_BYTES] {
    let result = h.finalize();
    let slice = result.as_slice();
    assert!(
        slice.len() >= HASH_BYTES,
        "Hasher output ({} bytes) shorter than HASH_BYTES ({})",
        slice.len(),
        HASH_BYTES,
    );
    let mut out = [0u8; HASH_BYTES];
    out.copy_from_slice(&slice[..HASH_BYTES]);
    out
}

/// =======================
/// Domain separation
/// =======================

#[derive(Clone, Copy, Debug)]
pub struct DsLabel {
    pub arity: usize,
    pub level: u32,
    pub position: u64,
    pub tree_label: u64,
}

impl DsLabel {
    pub fn to_bytes(self) -> [u8; 32] {
        let mut out = [0u8; 32];
        out[0..8].copy_from_slice(&(self.arity as u64).to_le_bytes());
        out[8..16].copy_from_slice(&(self.level as u64).to_le_bytes());
        out[16..24].copy_from_slice(&self.position.to_le_bytes());
        out[24..32].copy_from_slice(&self.tree_label.to_le_bytes());
        out
    }
}

pub const LEAF_LEVEL_DS: u32 = u32::MAX;

/// =======================
/// Merkle config
/// =======================

#[derive(Clone)]
pub struct MerkleChannelCfg {
    pub layer_arities: Vec<usize>,
    pub tree_label: u64,
}

impl MerkleChannelCfg {
    pub fn new(layer_arities: Vec<usize>, tree_label: u64) -> Self {
        Self { layer_arities, tree_label }
    }
}

/// =======================
/// Merkle opening
/// =======================

#[derive(Clone, Debug)]
pub struct MerkleOpening {
    pub leaf: [u8; HASH_BYTES],
    pub path: Vec<Vec<[u8; HASH_BYTES]>>,
    pub index: usize,
}

/// =======================
/// Free helpers
/// =======================

fn compress_leaf_standalone(
    arity: usize,
    tree_label: u64,
    position: u64,
    values: &[F],
) -> [u8; HASH_BYTES] {
    let ds = DsLabel {
        arity,
        level: LEAF_LEVEL_DS,
        position,
        tree_label,
    };
    let mut h = SelectedHasher::new();
    Digest::update(&mut h, ds.to_bytes());
    for v in values {
        Digest::update(&mut h, field_to_bytes(v));
    }
    finalize_hash(h)
}

fn compress_node_standalone(
    arity: usize,
    tree_label: u64,
    level: u32,
    position: u64,
    children: &[[u8; HASH_BYTES]],
) -> [u8; HASH_BYTES] {
    let ds = DsLabel {
        arity,
        level,
        position,
        tree_label,
    };
    let mut h = SelectedHasher::new();
    Digest::update(&mut h, ds.to_bytes());
    for c in children {
        Digest::update(&mut h, c);
    }
    finalize_hash(h)
}

/// Fold a chunk of leaves through `depth` binary levels, returning a single
/// node hash.  The chunk's working set stays in L2 cache (~48 KB for 1024
/// leaves × 48 bytes).
fn fold_chunk(
    base: usize,
    chunk_size: usize,
    depth: usize,
    leaf_arity: usize,
    tree_label: u64,
    all_values: &[Vec<F>],
    n: usize,
) -> [u8; HASH_BYTES] {
    debug_assert!(chunk_size.is_power_of_two());
    debug_assert!(depth == chunk_size.trailing_zeros() as usize);

    // Use a double-buffer to avoid per-level allocation
    let mut buf_a = Vec::with_capacity(chunk_size);
    let mut buf_b = Vec::with_capacity(chunk_size / 2);

    // Hash leaves into buf_a
    let end = (base + chunk_size).min(n);
    for i in base..end {
        buf_a.push(compress_leaf_standalone(
            leaf_arity, tree_label, i as u64, &all_values[i],
        ));
    }
    // Pad to chunk_size by repeating the last real leaf hash
    if buf_a.len() < chunk_size {
        let last = *buf_a.last().unwrap();
        buf_a.resize(chunk_size, last);
    }

    // Fold through `depth` binary levels
    for d in 0..depth {
        buf_b.clear();
        let parent_level = (d + 1) as u32;
        let parent_base = base >> (d + 1);
        let half = buf_a.len() / 2;
        for j in 0..half {
            buf_b.push(compress_node_standalone(
                2,
                tree_label,
                parent_level,
                (parent_base + j) as u64,
                &[buf_a[2 * j], buf_a[2 * j + 1]],
            ));
        }
        std::mem::swap(&mut buf_a, &mut buf_b);
    }

    debug_assert_eq!(buf_a.len(), 1);
    buf_a[0]
}

/// Rebuild the bottom `depth` binary levels for the chunk containing `index`,
/// returning all intermediate level nodes for that chunk.
fn rebuild_chunk_levels(
    index: usize,
    chunk_size: usize,
    depth: usize,
    leaf_arity: usize,
    tree_label: u64,
    all_values: &[Vec<F>],
    n: usize,
) -> Vec<Vec<[u8; HASH_BYTES]>> {
    let chunk_idx = index / chunk_size;
    let base = chunk_idx * chunk_size;
    let end = (base + chunk_size).min(n);

    let mut levels: Vec<Vec<[u8; HASH_BYTES]>> = Vec::with_capacity(depth + 1);

    // Level 0: leaf hashes
    let mut leaves: Vec<[u8; HASH_BYTES]> = (base..end)
        .map(|i| compress_leaf_standalone(leaf_arity, tree_label, i as u64, &all_values[i]))
        .collect();
    if leaves.len() < chunk_size {
        let last = *leaves.last().unwrap();
        leaves.resize(chunk_size, last);
    }
    levels.push(leaves);

    // Fold through binary levels, keeping every intermediate level
    for d in 0..depth {
        let prev = &levels[d];
        let parent_level = (d + 1) as u32;
        let parent_base = base >> (d + 1);
        let half = prev.len() / 2;
        let next: Vec<[u8; HASH_BYTES]> = (0..half)
            .map(|j| {
                compress_node_standalone(
                    2,
                    tree_label,
                    parent_level,
                    (parent_base + j) as u64,
                    &[prev[2 * j], prev[2 * j + 1]],
                )
            })
            .collect();
        levels.push(next);
    }

    levels
}

/// Minimum node count to bother spawning rayon tasks
const PARALLEL_THRESHOLD: usize = 256;

/// Number of bottom arity-2 levels to elide from storage.
/// 10 → chunk_size = 1024 → ~48 KB working set per chunk (fits in L2).
/// Opening cost: ~2 048 hashes per query (< 0.01 s).
const COMPACT_SKIP: usize = 10;

/// =======================
/// Merkle tree
/// =======================

pub struct MerkleTreeChannel {
    cfg: MerkleChannelCfg,
    /// How many bottom arity-2 levels are NOT stored.
    skip: usize,
    /// `levels[i]` corresponds to tree level `skip + i`.
    levels: Vec<Vec<[u8; HASH_BYTES]>>,
}

impl MerkleTreeChannel {
    pub fn new(cfg: MerkleChannelCfg, _trace_hash: [u8; HASH_BYTES]) -> Self {
        Self {
            cfg,
            skip: 0,
            levels: Vec::new(),
        }
    }

    // ------------------------------------------------------------------
    //  Legacy API (full tree in memory) — kept for backward compatibility
    // ------------------------------------------------------------------

    pub fn push_leaf(&mut self, values: &[F]) {
        if self.levels.is_empty() {
            self.levels.push(Vec::new());
        }
        let idx = self.levels[0].len();
        let leaf = compress_leaf_standalone(
            self.cfg.layer_arities[0],
            self.cfg.tree_label,
            idx as u64,
            values,
        );
        self.levels[0].push(leaf);
    }

    pub fn push_leaves_parallel(&mut self, all_values: &[Vec<F>]) {
        if self.levels.is_empty() {
            self.levels.push(Vec::new());
        }
        let arity = self.cfg.layer_arities[0];
        let tree_label = self.cfg.tree_label;

        #[cfg(feature = "parallel")]
        let leaves: Vec<[u8; HASH_BYTES]> = if all_values.len() >= PARALLEL_THRESHOLD {
            all_values
                .par_iter()
                .enumerate()
                .map(|(idx, values)| compress_leaf_standalone(arity, tree_label, idx as u64, values))
                .collect()
        } else {
            all_values
                .iter()
                .enumerate()
                .map(|(idx, values)| compress_leaf_standalone(arity, tree_label, idx as u64, values))
                .collect()
        };

        #[cfg(not(feature = "parallel"))]
        let leaves: Vec<[u8; HASH_BYTES]> = all_values
            .iter()
            .enumerate()
            .map(|(idx, values)| compress_leaf_standalone(arity, tree_label, idx as u64, values))
            .collect();

        self.levels[0] = leaves;
    }

    pub fn finalize(&mut self) -> [u8; HASH_BYTES] {
        let mut level = 0;
        while self.levels[level].len() > 1 {
            let arity = self.cfg.layer_arities[level];

            let cur_len = self.levels[level].len();
            if cur_len % arity != 0 {
                let last = *self.levels[level].last().unwrap();
                let pad = arity - cur_len % arity;
                self.levels[level].extend(std::iter::repeat(last).take(pad));
            }

            let tree_label = self.cfg.tree_label;
            let parent_level = level as u32 + 1;
            let num_parents = self.levels[level].len() / arity;

            let parents: Vec<[u8; HASH_BYTES]> = {
                let cur = &self.levels[level];

                #[cfg(feature = "parallel")]
                {
                    if num_parents >= PARALLEL_THRESHOLD {
                        cur.par_chunks(arity)
                            .enumerate()
                            .map(|(i, c)| {
                                compress_node_standalone(arity, tree_label, parent_level, i as u64, c)
                            })
                            .collect()
                    } else {
                        cur.chunks(arity)
                            .enumerate()
                            .map(|(i, c)| {
                                compress_node_standalone(arity, tree_label, parent_level, i as u64, c)
                            })
                            .collect()
                    }
                }

                #[cfg(not(feature = "parallel"))]
                {
                    cur.chunks(arity)
                        .enumerate()
                        .map(|(i, c)| {
                            compress_node_standalone(arity, tree_label, parent_level, i as u64, c)
                        })
                        .collect()
                }
            };

            self.levels.push(parents);
            level += 1;
        }
        self.levels.last().unwrap()[0]
    }

    pub fn open(&self, index: usize) -> MerkleOpening {
        let mut idx = index;
        let mut path = Vec::new();

        for level in 0..self.levels.len() - 1 {
            let nodes = &self.levels[level];
            let arity = self.cfg.layer_arities[self.skip + level];
            let group_start = (idx / arity) * arity;

            let siblings: Vec<[u8; HASH_BYTES]> = (0..arity)
                .filter_map(|i| {
                    let pos = group_start + i;
                    if pos != idx {
                        let node = if pos < nodes.len() {
                            nodes[pos]
                        } else {
                            *nodes.last().unwrap()
                        };
                        Some(node)
                    } else {
                        None
                    }
                })
                .collect();

            path.push(siblings);
            idx /= arity;
        }

        MerkleOpening {
            leaf: self.levels[0][index],
            path,
            index,
        }
    }

    // ------------------------------------------------------------------
    //  Compact API — elides bottom `skip` binary levels from storage
    // ------------------------------------------------------------------

    /// Combined leaf hashing + tree construction that keeps only the top
    /// levels in memory, reducing the Merkle footprint by ~1000×.
    ///
    /// Requirements:
    ///   - `all_values.len()` must be a power of two.
    ///   - The caller must pass the same `all_values` to `open_compact`.
    ///
    /// Each chunk of 2^COMPACT_SKIP leaves (~48 KB working set) is hashed
    /// and folded entirely within L2 cache, then only the resulting single
    /// node is kept.  This replaces the `push_leaves_parallel` + `finalize`
    /// sequence.
    pub fn commit_compact(&mut self, all_values: &[Vec<F>]) -> [u8; HASH_BYTES] {
        let n = all_values.len();
        assert!(n > 0, "commit_compact: need at least one leaf");
        assert!(
            n.is_power_of_two(),
            "commit_compact: n must be a power of two (got {n})"
        );

        let tree_label = self.cfg.tree_label;
        let arities = self.cfg.layer_arities.clone();

        // How many consecutive arity-2 levels from the bottom?
        let arity2_depth = arities.iter().take_while(|&&a| a == 2).count();
        let log2_n = n.trailing_zeros() as usize;
        let skip = arity2_depth.min(COMPACT_SKIP).min(log2_n);
        self.skip = skip;

        let chunk_size = 1usize << skip; // leaves per chunk
        let num_chunks = n / chunk_size;
        let leaf_arity = arities[0];

        // ---- Phase 1: parallel chunk folding ----
        // Each chunk hashes `chunk_size` leaves and folds through `skip`
        // binary levels, producing one node.  Working set ≈ chunk_size × 48
        // bytes, comfortably in L2 cache.

        #[cfg(feature = "parallel")]
        let chunk_nodes: Vec<[u8; HASH_BYTES]> = if num_chunks >= PARALLEL_THRESHOLD {
            (0..num_chunks)
                .into_par_iter()
                .map(|c| {
                    fold_chunk(
                        c * chunk_size,
                        chunk_size,
                        skip,
                        leaf_arity,
                        tree_label,
                        all_values,
                        n,
                    )
                })
                .collect()
        } else {
            (0..num_chunks)
                .map(|c| {
                    fold_chunk(
                        c * chunk_size,
                        chunk_size,
                        skip,
                        leaf_arity,
                        tree_label,
                        all_values,
                        n,
                    )
                })
                .collect()
        };

        #[cfg(not(feature = "parallel"))]
        let chunk_nodes: Vec<[u8; HASH_BYTES]> = (0..num_chunks)
            .map(|c| {
                fold_chunk(
                    c * chunk_size,
                    chunk_size,
                    skip,
                    leaf_arity,
                    tree_label,
                    all_values,
                    n,
                )
            })
            .collect();

        // ---- Phase 2: build remaining levels from the chunk nodes ----
        // chunk_nodes are at tree level `skip`.  Typically ~32 K entries
        // (~1.5 MB) — tiny compared to the full leaf level.

        self.levels.clear();
        self.levels.push(chunk_nodes);

        let mut level = 0; // index into self.levels
        while self.levels[level].len() > 1 {
            let tree_level = self.skip + level;
            let arity = arities[tree_level];

            let cur_len = self.levels[level].len();
            if cur_len % arity != 0 {
                let last = *self.levels[level].last().unwrap();
                let pad = arity - cur_len % arity;
                self.levels[level].extend(std::iter::repeat(last).take(pad));
            }

            let parent_tree_level = (tree_level + 1) as u32;
            let num_parents = self.levels[level].len() / arity;

            let parents: Vec<[u8; HASH_BYTES]> = {
                let cur = &self.levels[level];

                #[cfg(feature = "parallel")]
                {
                    if num_parents >= PARALLEL_THRESHOLD {
                        cur.par_chunks(arity)
                            .enumerate()
                            .map(|(i, c)| {
                                compress_node_standalone(
                                    arity,
                                    tree_label,
                                    parent_tree_level,
                                    i as u64,
                                    c,
                                )
                            })
                            .collect()
                    } else {
                        cur.chunks(arity)
                            .enumerate()
                            .map(|(i, c)| {
                                compress_node_standalone(
                                    arity,
                                    tree_label,
                                    parent_tree_level,
                                    i as u64,
                                    c,
                                )
                            })
                            .collect()
                    }
                }

                #[cfg(not(feature = "parallel"))]
                {
                    cur.chunks(arity)
                        .enumerate()
                        .map(|(i, c)| {
                            compress_node_standalone(
                                arity,
                                tree_label,
                                parent_tree_level,
                                i as u64,
                                c,
                            )
                        })
                        .collect()
                }
            };

            self.levels.push(parents);
            level += 1;
        }

        self.levels.last().unwrap()[0]
    }

    /// Generate an opening, recomputing the bottom `skip` levels on the fly
    /// from the original leaf values.
    ///
    /// Cost: ~2 × 2^skip hashes (≈ 2 048 for COMPACT_SKIP = 10).
    /// For 32–64 queries this adds < 0.1 s total.
    pub fn open_compact(&self, index: usize, all_values: &[Vec<F>]) -> MerkleOpening {
        let tree_label = self.cfg.tree_label;
        let arities = &self.cfg.layer_arities;
        let skip = self.skip;
        let chunk_size = 1usize << skip;
        let n = all_values.len();
        let leaf_arity = arities[0];

        // Rebuild the bottom `skip` levels for the chunk containing `index`
        let chunk_levels = rebuild_chunk_levels(
            index, chunk_size, skip, leaf_arity, tree_label, all_values, n,
        );

        let chunk_base = (index / chunk_size) * chunk_size;
        let mut idx = index;
        let mut path = Vec::new();

        // Bottom `skip` levels — from the recomputed chunk data
        for depth in 0..skip {
            let g = idx; // global index at this tree level
            let local = g - (chunk_base >> depth);
            let arity = 2;
            let local_group = (local / arity) * arity;

            let siblings: Vec<[u8; HASH_BYTES]> = (0..arity)
                .filter(|&i| local_group + i != local)
                .map(|i| chunk_levels[depth][local_group + i])
                .collect();

            path.push(siblings);
            idx /= arity;
        }

        // Top levels — from stored data
        for stored_level in 0..self.levels.len() - 1 {
            let tree_level = skip + stored_level;
            let arity = arities[tree_level];
            let group_start = (idx / arity) * arity;
            let nodes = &self.levels[stored_level];

            let siblings: Vec<[u8; HASH_BYTES]> = (0..arity)
                .filter(|&i| group_start + i != idx)
                .map(|i| {
                    let pos = group_start + i;
                    if pos < nodes.len() {
                        nodes[pos]
                    } else {
                        *nodes.last().unwrap()
                    }
                })
                .collect();

            path.push(siblings);
            idx /= arity;
        }

        MerkleOpening {
            leaf: chunk_levels[0][index - chunk_base],
            path,
            index,
        }
    }

    // ------------------------------------------------------------------
    //  Verification (works with both legacy and compact openings)
    // ------------------------------------------------------------------

    pub fn verify_opening(
        cfg: &MerkleChannelCfg,
        root: [u8; HASH_BYTES],
        opening: &MerkleOpening,
        _trace_hash: &[u8; HASH_BYTES],
    ) -> bool {
        let mut cur = opening.leaf;
        let mut idx = opening.index;

        for (level, siblings) in opening.path.iter().enumerate() {
            let arity = cfg.layer_arities[level];
            let pos = idx % arity;

            let mut children: Vec<[u8; HASH_BYTES]> = Vec::with_capacity(arity);
            let mut sibs = siblings.iter();

            for i in 0..arity {
                if i == pos {
                    children.push(cur);
                } else {
                    match sibs.next() {
                        Some(&x) => children.push(x),
                        None => return false,
                    }
                }
            }

            let ds = DsLabel {
                arity,
                level: level as u32 + 1,
                position: (idx / arity) as u64,
                tree_label: cfg.tree_label,
            };

            let mut h = SelectedHasher::new();
            Digest::update(&mut h, ds.to_bytes());
            for c in &children {
                Digest::update(&mut h, c);
            }
            cur = finalize_hash(h);

            idx /= arity;
        }

        cur == root
    }
}

/// =======================
/// Free function: compute a single leaf hash
/// =======================

pub fn compute_leaf_hash(cfg: &MerkleChannelCfg, index: usize, values: &[F]) -> [u8; HASH_BYTES] {
    compress_leaf_standalone(
        cfg.layer_arities[0],
        cfg.tree_label,
        index as u64,
        values,
    )
}