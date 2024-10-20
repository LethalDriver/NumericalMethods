use std::f64::consts::E;
use crate::utils;

pub fn task2() {
    println!("Approximating e^x using maclauren series...");
    let x_values = vec![0.1, 20.0, -20.0];
    for x in &x_values {
        let approximation = approximate_exponential(*x);
        let expected = E.powf(*x);
        let relative_error = utils::calculate_relative_error(approximation, expected);
        println!(
            "e^x approximation for x = {}: {}, expected: {}, relative error: {:e}",
            x, approximation, expected, relative_error
        );
    }
}

fn approximate_exponential(x: f64) -> f64 {
    if x < 0.0 {
        return 1.0 / approximate_exponential(-x);
    }

    let mut sum = 0.0;
    let mut n = 0;
    loop {
        match get_nth_maclauren_term(n, x) {
            Ok(term) => sum += term,
            Err(_e) => return sum,
        }
        n += 1
    }
}

fn get_nth_maclauren_term(n: u128, x: f64) -> Result<f64, &'static str> {
    match factorial(n) {
        Some(fact) => Ok(x.powi(n as i32) / fact as f64),
        None => Err("Overflow occurred in factorial calculation"),
    }
}

fn factorial(n: u128) -> Option<u128> {
    let mut result: u128 = 1;
    for i in 1..=n {
        result = result.checked_mul(i)?;
    }
    Some(result)
}
