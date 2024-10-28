use plotly::common::Mode;
use plotly::layout::Axis;
use plotly::Plot;
use plotly::Scatter;

fn main() {
    let f1 = |x: f64| -0.1 * x.powi(4) - 0.15 * x.powi(3) - 0.5 * x.powi(2) - 0.25 * x + 1.2;
    let analytic1 = |x: f64| -0.4 * x.powi(3) - 0.45 * x.powi(2) - x - 0.25;
    let x = 0.5;
    process_and_plot(
        &f1,
        &analytic1,
        &forward_difference_approximation,
        x,
        "Relative error of the forward difference approximation for -0.1x^4 - 0.15x^3 - 0.5x^2 - 0.25x + 1.2",
    );
    process_and_plot(&f1, 
        &analytic1, 
        &central_difference_approximation, 
        x, 
        "Relative error of the central difference approximation for -0.1x^4 - 0.15x^3 - 0.5x^2 - 0.25x + 1.2"
    );

    let f2 = |x: f64| (2.0 * x).sin().exp();
    let analytic2 = |x: f64| 2.0 * (2.0 * x).cos() * (2.0 * x).sin().exp();
    process_and_plot(
        &f2,
        &analytic2,
        &forward_difference_approximation,
        x,
        "Relative error of the forward difference approximation for e^sin(2x)",
    );
    process_and_plot(
        &f2,
        &analytic2,
        &central_difference_approximation,
        x,
        "Relative error of the central difference approximation for e^sin(2x)",
    );
}

fn process_and_plot(
    f: &dyn Fn(f64) -> f64,
    analytic: &dyn Fn(f64) -> f64,
    approximation_method: &dyn Fn(&dyn Fn(f64) -> f64, f64, f64) -> f64,
    x: f64,
    title: &str,
) {
    let h_values = generate_h_values(1, 12);
    let mut relative_error_values = Vec::<f64>::with_capacity(h_values.len());
    for h in h_values.iter() {
        let numerical = approximation_method(f, x, *h);
        let relative_error = calculate_relative_error(numerical, analytic(x));
        relative_error_values.push(relative_error);
        println!(
            "h = {:e}, numerical = {:.10}, relative error = {:e}",
            h, numerical, relative_error
        );
    }
    plot_results(
        &h_values,
        &relative_error_values,
        title,
        "h",
        "Relative error",
    );
}

fn generate_h_values(start: i32, end: i32) -> Vec<f64> {
    let mut h_values = Vec::<f64>::with_capacity((end - start) as usize);
    for i in start..end {
        h_values.push(10.0_f64.powi(-i));
    }
    h_values
}

fn forward_difference_approximation(f: &dyn Fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (f(x + h) - f(x)) / h
}

fn central_difference_approximation(f: &dyn Fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (f(x + (h / 2.0)) - f(x - (h / 2.0))) / h
}

fn calculate_relative_error(obtained: f64, actual: f64) -> f64 {
    (obtained - actual).abs() / actual.abs()
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
        .x_axis(
            Axis::new()
                .title(plotly::common::Title::new(x_label))
                .type_(plotly::layout::AxisType::Log)
                .exponent_format("e"),
        )
        .y_axis(
            Axis::new()
                .title(plotly::common::Title::new(y_label))
                .type_(plotly::layout::AxisType::Log)
                .exponent_format("e"),

        );

    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(layout);

    plot.show();
}
