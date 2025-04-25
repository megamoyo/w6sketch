use fnv::{FnvHashMap, FnvHashSet, FnvHasher};
use lazy_static::lazy_static;
use probminhash::superminhasher::SuperMinHash;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::hash::BuildHasherDefault;

#[pyfunction]
pub fn is_release_build() -> bool {
    !cfg!(debug_assertions)
}

#[pyclass]
pub struct LSH {
    candidates: FnvHashMap<Vec<u8>, FnvHashSet<usize>>,
    hashes: Vec<Vec<Vec<u8>>>,
    ids: Vec<String>,
    id_map: FnvHashMap<String, usize>,
}

fn similarity_threshold(a: &[Vec<u8>], b: &[Vec<u8>]) -> f64 {
    let mut count = 0;
    for i in 0..a.len() {
        if a[i] == b[i] {
            count += 1;
        }
    }
    count as f64 / a.len() as f64
}

#[pymethods]
impl LSH {
    #[new]
    fn new() -> Self {
        LSH {
            candidates: FnvHashMap::default(),
            hashes: Vec::new(),
            ids: Vec::new(),
            id_map: FnvHashMap::default(),
        }
    }

    fn keys(&self) -> Vec<String> {
        self.ids.clone()
    }

    fn values(&self) -> Vec<Vec<f32>> {
        self.hashes
            .iter()
            .map(|x| {
                x.iter()
                    .map(|y| {
                        let mut bytes = [0u8; 4];
                        bytes.copy_from_slice(&y[..4]);
                        f32::from_le_bytes(bytes)
                    })
                    .collect()
            })
            .collect()
    }

    fn length(&self) -> usize {
        self.ids.len()
    }
    #[pyo3(signature = (data, threshold = 0.5))]
    #[inline]
    fn check(&self, data: Vec<f32>, threshold: f64) -> FnvHashMap<String, f64> {
        let data_bytes: Vec<Vec<u8>> = data
            .iter()
            .map(|x| {
                let mut bytes = [0u8; 4];
                bytes.copy_from_slice(&x.to_le_bytes());
                bytes.to_vec()
            })
            .collect();
        let candidates: FnvHashSet<usize> = data_bytes
            .iter()
            .flat_map(|x| {
                if let Some(c) = self.candidates.get(x) {
                    c.iter().cloned().collect()
                } else {
                    vec![]
                }
            })
            .collect();
        let mut result = FnvHashMap::default();
        for i in candidates {
            if let Some(hash) = self.hashes.get(i) {
                let similarity = similarity_threshold(&data_bytes, hash);
                if similarity >= threshold {
                    if let Some(id) = self.ids.get(i) {
                        result.insert(id.clone(), similarity);
                    }
                }
            }
        }
        result
    }

    #[pyo3(
        signature = (new_id, data, threshold = 0.5, add_if_dup = false),
    )]
    #[inline]
    fn check_and_add(
        &mut self,
        new_id: &str,
        data: Vec<f32>,
        threshold: f64,
        add_if_dup: bool,
    ) -> FnvHashMap<String, f64> {
        let data_bytes: Vec<Vec<u8>> = data
            .iter()
            .map(|x| {
                let mut bytes = [0u8; 4];
                bytes.copy_from_slice(&x.to_le_bytes());
                bytes.to_vec()
            })
            .collect();
        let result = self.check(data, threshold);
        if result.is_empty() || add_if_dup {
            let len_id = self.ids.len();
            self.id_map.insert(new_id.to_string(), len_id);
            self.ids.push(new_id.to_string());
            for i in 0..data_bytes.len() {
                self.candidates
                    .entry(data_bytes[i].clone())
                    .or_insert(FnvHashSet::default())
                    .insert(len_id);
            }
            self.hashes.push(data_bytes);
        }
        result
    }
}

#[pyclass]
pub struct SuperMinHasher {
    minhash: SuperMinHash<f32, Vec<char>, FnvHasher>,
    n_gram: usize,
    lowercase: bool,
    unicode_normalize: bool,
    zh_conv: bool,
    punct_norm: bool,
}

lazy_static! {
    static ref ICU_NORMALIZER: icu::normalizer::ComposingNormalizer =
        icu::normalizer::ComposingNormalizer::new_nfkc();
    static ref SP_PUNCT_RE: regex::Regex = regex::Regex::new(r"[\s\p{Punctuation}]+").unwrap();
}

#[pymethods]
impl SuperMinHasher {
    #[new]
    #[pyo3(
        signature = (size, n_gram = 5, lowercase = true, unicode_normalize = true, zh_conv = true, punct_norm = true),
    )]
    fn new(
        size: usize,
        n_gram: usize,
        lowercase: bool,
        unicode_normalize: bool,
        zh_conv: bool,
        punct_norm: bool,
    ) -> PyResult<Self> {
        if size == 0 {
            return Err(PyValueError::new_err("size must be greater than 0"));
        }
        if n_gram == 0 {
            return Err(PyValueError::new_err("n_gram must be greater than 0"));
        }

        let bh = BuildHasherDefault::<FnvHasher>::default();
        let minhash = SuperMinHash::new(size, bh);
        Ok(SuperMinHasher {
            minhash,
            n_gram,
            lowercase,
            unicode_normalize,
            zh_conv,
            punct_norm,
        })
    }

    #[inline]
    fn sketch(&mut self, mut s: String) {
        if self.unicode_normalize {
            s = ICU_NORMALIZER.normalize(&s);
        }
        if self.punct_norm {
            s = SP_PUNCT_RE.replace_all(&s, " ").to_string();
        }
        if self.zh_conv {
            s = zhconv::converters::ZH_TO_HANS_CONVERTER.convert(&s);
        }
        if self.lowercase {
            s = s.to_lowercase();
        }
        let cs = s.chars().collect::<Vec<_>>();
        if cs.len() < self.n_gram {
            self.minhash.sketch(&cs).unwrap();
        } else {
            let mut current = Vec::with_capacity(self.n_gram);
            for i in 0..cs.len() - self.n_gram + 1 {
                current.clear();
                current.extend_from_slice(&cs[i..i + self.n_gram]);
                self.minhash.sketch(&current).unwrap();
            }
        }
    }

    #[inline]
    fn finalize(&mut self) -> Vec<f32> {
        let s = self.minhash.get_hsketch().to_vec();
        self.minhash.reinit();
        s
    }

    #[inline]
    fn sketch_and_finalize(&mut self, s: String) -> Vec<f32> {
        self.sketch(s);
        self.finalize()
    }
}

#[pyclass]
pub struct SuperMinHasherLSH {
    lsh: LSH,
    minhasher: SuperMinHasher,
}

#[pymethods]
impl SuperMinHasherLSH {
    #[new]
    #[pyo3(
        signature = (size, n_gram = 5, lowercase = true, unicode_normalize = true, zh_conv = true, punct_norm = true),
    )]
    fn new(
        size: usize,
        n_gram: usize,
        lowercase: bool,
        unicode_normalize: bool,
        zh_conv: bool,
        punct_norm: bool,
    ) -> PyResult<Self> {
        let minhasher = SuperMinHasher::new(
            size,
            n_gram,
            lowercase,
            unicode_normalize,
            zh_conv,
            punct_norm,
        )?;
        let lsh = LSH::new();
        Ok(SuperMinHasherLSH { lsh, minhasher })
    }

    #[pyo3(
        signature = (new_id, data, threshold = 0.5, add = true, add_if_dup = false),
    )]
    #[inline]
    fn check_and_add(
        &mut self,
        new_id: &str,
        data: String,
        threshold: f64,
        add: bool,
        add_if_dup: bool,
    ) -> FnvHashMap<String, f64> {
        self.minhasher.sketch(data);
        let result = if add {
            self.lsh
                .check_and_add(new_id, self.minhasher.finalize(), threshold, add_if_dup)
        } else {
            self.lsh.check(self.minhasher.finalize(), threshold)
        };
        self.minhasher.minhash.reinit();
        result
    }
    fn keys(&self) -> Vec<String> {
        self.lsh.keys()
    }

    fn values(&self) -> Vec<Vec<f32>> {
        self.lsh.values()
    }

    fn length(&self) -> usize {
        self.lsh.length()
    }
}
