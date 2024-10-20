pub fn calculate_relative_error(obtained: f64, actual: f64) -> f64 {
    (obtained - actual).abs() / actual
}
