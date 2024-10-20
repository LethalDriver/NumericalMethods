use crate::task1;
use crate::utils;
use std::f64::consts::PI;

pub fn task3() {
    println!("Comparing basic and Kahan summation algorithms for Basel sum...");

    let analytical_value = PI.powi(2) / 6.0;
    let (_, n_max) = task1::basel_sum();

    let mut multiple = 1;
    loop {
        let obtained = kahan_basel_sum(multiple * n_max);
        let relative_error = utils::calculate_relative_error(obtained, analytical_value);
        println!(
            "Relative error for kahn summation with {}n_max: {:e}",
            multiple, relative_error
        );
        multiple = multiple * 2;
        if multiple > 16 {
            break;
        }
    }
}

fn kahan_basel_sum(n_max: i32) -> f64 {
    let mut sum = 0.0;
    let mut comp = 0.0;
    let mut n = 1;
    loop {
        let number = task1::nth_term(n);
        let t = sum;
        let y = number + comp;
        sum += y;
        comp = (t - sum) + y;
        if n == n_max {
            return sum;
        }
        n += 1;
    }
}
