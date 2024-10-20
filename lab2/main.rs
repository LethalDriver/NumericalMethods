use std::time::Instant;
use std::f64::consts::PI;

fn main() {
    let start = Instant::now();

    let mut k: f64 = 1.0;
    let mut sum: f64 = 0.0;

    loop {
        let n = 1.0 / k.powi(2);
        let prev = sum;
        sum += n;
        if sum == prev {
            break;
        }
        k += 1.0;
    }

    let analytical_value = PI.powi(2) / 6.0;
    let percentage = (1.0 - (sum / analytical_value)) * 100.0;
    let duration = start.elapsed();

    println!("Sum for n max: {:.10}", sum);
    println!(
        "Relative difference from analytical value (percentage): {:.10}%",
        percentage
    );
    println!("Execution time: {:?}", duration);
}
