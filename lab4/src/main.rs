use plotly::common::Mode;
use plotly::layout::Axis;
use plotly::Plot;
use plotly::Scatter;

fn main() {
    let f1 = |x: f64| -0.1 * x.powi(4) - 0.15 * x.powi(3) - 0.5 * x.powi(2) - 0.25 * x + 1.2;
    let analytic1 = |x: f64| -0.4 * x.powi(3) - 0.45 * x.powi(2) - x - 0.25;
    let x = 0.5;
    let first_order_approximation_methods: Vec<(
        &dyn Fn(&dyn Fn(f64) -> f64, f64, f64) -> f64,
        &str,
    )> = vec![
        (&forward_difference_approximation, "Forward difference"),
        (&central_difference_approximation, "Central difference"),
    ];
    let second_order_approximation_methods: Vec<(
        &dyn Fn(&dyn Fn(f64) -> f64, f64, f64) -> f64,
        &str,
    )> = vec![(
        &cd_second_derivative_approximation,
        "Second-order central difference",
    )];
    process_and_plot(
        &f1,
        &analytic1,
        &first_order_approximation_methods,
        x,
        "Relative errors of the derivative approximations for -0.1x^4 - 0.15x^3 - 0.5x^2 - 0.25x + 1.2",
    );
    let f2 = |x: f64| (2.0 * x).sin().exp();
    let analytic2 = |x: f64| 2.0 * (2.0 * x).cos() * (2.0 * x).sin().exp();
    process_and_plot(
        &f2,
        &analytic2,
        &first_order_approximation_methods,
        x,
        "Relative error of the derivative approximation for e^sin(2x)",
    );

    let f3 = |x: f64| (x.cos()).ln();
    let analytic3 = |x: f64| -(1.0 / x.cos().powi(2));
    process_and_plot(
        &f3,
        &analytic3,
        &second_order_approximation_methods,
        x,
        "Relative error of the second-order central difference approximation for ln(cos(x))",
    );
}

fn process_and_plot(
    f: &dyn Fn(f64) -> f64,
    analytic: &dyn Fn(f64) -> f64,
    approximation_methods: &Vec<(&dyn Fn(&dyn Fn(f64) -> f64, f64, f64) -> f64, &str)>,
    x: f64,
    title: &str,
) {
    let h_values = generate_h_values(1, 12);
    let mut data_series = Vec::new();

    for (approximation_method, method_name) in approximation_methods {
        let relative_error_values =
            generate_relative_error_values(x, h_values.clone(), f, analytic, approximation_method);
        data_series.push((
            h_values.clone(),
            relative_error_values,
            method_name.to_string(),
        ));
    }

    plot_results(data_series, title, "h", "Relative error");
}

fn generate_relative_error_values(
    x: f64,
    h_values: Vec<f64>,
    f: &dyn Fn(f64) -> f64,
    analytic: &dyn Fn(f64) -> f64,
    approximation_method: &dyn Fn(&dyn Fn(f64) -> f64, f64, f64) -> f64,
) -> Vec<f64> {
    let mut relative_error_values = Vec::<f64>::with_capacity(h_values.len());
    for h in h_values.iter() {
        let numerical = approximation_method(f, x, *h);
        let relative_error = calculate_relative_error(numerical, analytic(x));
        relative_error_values.push(relative_error);
    }
    relative_error_values
}

fn plot_results(
    data_series: Vec<(Vec<f64>, Vec<f64>, String)>,
    title: &str,
    x_label: &str,
    y_label: &str,
) {
    let mut plot = Plot::new();

    for (x_values, y_values, name) in data_series {
        let trace = Scatter::new(x_values.clone(), y_values.clone())
            .mode(Mode::Lines)
            .name(&name);
        plot.add_trace(trace);
    }

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

    plot.set_layout(layout);
    plot.show();
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

fn cd_second_derivative_approximation(f: &dyn Fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (4.0 / h.powi(2)) * (f(x + h / 2.0) + f(x - h / 2.0) - 2.0 * f(x))
}

fn calculate_relative_error(obtained: f64, actual: f64) -> f64 {
    (obtained - actual).abs() / actual.abs()
}
