use libm::nextafter;
use plotly::common::Mode;
use plotly::layout::Axis;
use plotly::Plot;
use plotly::Scatter;
use std::f64::EPSILON;
fn main() {
    println!("Machine precision: {:e}", EPSILON);
    let num_points = 100;
    let x_min = 1e-300;
    let x_max = 1e+300;

    let x_values = generate_x_values(num_points, x_min, x_max);
    let delta_x_values = generate_delta_x_values(&x_values);

    plot_results(&x_values, &delta_x_values, "f(x) = Δx", "x", "Δx");

    let relative_errors = generate_relative_errors(&x_values, &delta_x_values);

    plot_results(
        &x_values,
        &relative_errors,
        "f(x) = δx",
        "x",
        "δx",
    );
}

fn generate_x_values(num_points: usize, x_min: f64, x_max: f64) -> Vec<f64> {
    let log_x_min = x_min.log10();
    let log_x_max = x_max.log10();
    let log_step = (log_x_max - log_x_min) / (num_points as f64 - 1.0);

    let mut x_values = Vec::with_capacity(num_points);

    for i in 0..num_points {
        let log_x = log_x_min + i as f64 * log_step;
        let x = 10f64.powf(log_x);
        x_values.push(x);
    }

    x_values
}

fn generate_delta_x_values(x_values: &Vec<f64>) -> Vec<f64> {
    let mut delta_x_values = Vec::with_capacity(x_values.len());

    for &x in x_values.iter() {
        let delta_x = nextafter(x, f64::INFINITY) - x;
        delta_x_values.push(delta_x);
    }

    delta_x_values
}

fn generate_relative_errors(x_values: &Vec<f64>, delta_x_values: &Vec<f64>) -> Vec<f64> {
    let mut relative_errors = Vec::with_capacity(x_values.len());

    for i in 0..x_values.len() {
        let x = x_values[i];
        let delta_x = delta_x_values[i];
        let relative_error = delta_x / x;
        relative_errors.push(relative_error);
    }

    relative_errors
}

fn plot_results(
    x_values: &Vec<f64>,
    y_values: &Vec<f64>,
    title: &str,
    x_label: &str,
    y_label: &str,
) {
    let trace = Scatter::new(x_values.clone(), y_values.clone()).mode(Mode::Lines);

    let layout = plotly::Layout::new()
        .title(plotly::common::Title::new(title))
        .width(1000)  // Set the width of the plot
        .height(600) // Set the height of the plot
        .x_axis(
            Axis::new()
                .title(plotly::common::Title::new(x_label))
                .type_(plotly::layout::AxisType::Log),
        )
        .y_axis(
            Axis::new()
                .title(plotly::common::Title::new(y_label))
                .type_(plotly::layout::AxisType::Log),
        );

    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(layout);

    plot.show();
}
