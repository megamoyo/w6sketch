mod minhash;

use pyo3::prelude::*;

#[pymodule]
fn w6r(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<minhash::SuperMinHasher>()?;
    m.add_class::<minhash::SuperMinHasherLSH>()?;
    m.add_class::<minhash::LSH>()?;
    m.add_function(wrap_pyfunction!(minhash::is_release_build, m).unwrap())?;
    Ok(())
}
