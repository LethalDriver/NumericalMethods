use crate::utils::{self, calculate_relative_error};
use std::f64::consts::PI;

pub fn task1() {
    println!("Calculating backwards basel sum...");
    let (forward_sum, n_max) = basel_sum();
    let analytical_value = PI.powi(2) / 6.0;
    let relative_error = calculate_relative_error(forward_sum, analytical_value);
    println!(
        "Relative error for forwards summation: {:e}",
        relative_error
    );

    let mut multiple = 1;
    loop {
        let obtained = calculate_backwards_sum(multiple * n_max);
        let relative_error = utils::calculate_relative_error(obtained, analytical_value);
        println!(
            "Relative error for backwards summation with {}n_max: {:e}",
            multiple, relative_error
        );
        multiple = multiple * 2;
        if multiple > 16 {
            break;
        }
    }
}

pub fn basel_sum() -> (f64, i32) {
    let mut sum = 0.0;
    let mut n = 1;
    loop {
        let term = nth_term(n);
        let new_sum = sum + term;
        if new_sum == sum {
            break;
        }
        sum = new_sum;
        n += 1;
    }
    (sum, n)
}

fn calculate_backwards_sum(n_max: i32) -> f64 {
    let mut k: i32 = n_max;
    let mut sum: f64 = 0.0;

    loop {
        let n = nth_term(k);
        let prev = sum;
        sum += n;
        if sum == prev {
            break;
        }
        k -= 1;
        if k < 1 {
            break;
        }
    }

    sum
}

pub fn nth_term(n: i32) -> f64 {
    1.0 / ((n as f64).powi(2))
}
