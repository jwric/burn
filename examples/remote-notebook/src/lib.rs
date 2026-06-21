//! Rich display of Burn tensors in a Rust notebook (evcxr/Jupyter).
//!
//! evcxr renders the value of a cell's final expression by calling an inherent `evcxr_display`
//! method on it, if one exists. `Tensor` is defined in Burn, so this crate provides a thin wrapper
//! that reads the tensor and prints an HTML table (values plus a heatmap background) using evcxr's
//! display protocol. A cell ending in `c.show()` then renders the matrix instead of needing
//! `println!`.

use burn::tensor::Tensor;

/// Largest matrix slice rendered as a table; larger tensors are truncated with an ellipsis.
const MAX_SHOWN: usize = 16;

/// Largest matrix slice rendered as a heatmap; larger tensors are cropped to this many rows/cols.
const MAX_HEATMAP: usize = 128;

/// A 2-D tensor wrapped for rich notebook display.
pub struct Show(pub Tensor<2>);

/// Convenience for turning a tensor into a [`Show`] in a cell's final expression.
pub trait ShowExt {
    /// Wrap `self` so a notebook cell renders it as a heatmap table.
    fn show(self) -> Show;
}

impl ShowExt for Tensor<2> {
    fn show(self) -> Show {
        Show(self)
    }
}

impl Show {
    /// Called by evcxr to render the final expression of a cell.
    pub fn evcxr_display(&self) {
        println!("{}", self.bundle());
    }

    /// The MIME bundle evcxr reads from stdout: the `text/html` table delimited by evcxr's markers.
    fn bundle(&self) -> String {
        let [rows, cols] = self.0.dims();
        let values: Vec<f32> = self.0.clone().into_data().to_vec().unwrap();
        let html = render_html(&values, rows, cols);
        format!("EVCXR_BEGIN_CONTENT text/html\n{html}\nEVCXR_END_CONTENT")
    }
}

/// A 2-D tensor wrapped for display as a heatmap image, for tensors too large to read as numbers.
pub struct Heatmap(pub Tensor<2>);

/// Convenience for turning a tensor into a [`Heatmap`] in a cell's final expression.
pub trait HeatmapExt {
    /// Wrap `self` so a notebook cell renders it as a heatmap image.
    fn heatmap(self) -> Heatmap;
}

impl HeatmapExt for Tensor<2> {
    fn heatmap(self) -> Heatmap {
        Heatmap(self)
    }
}

impl Heatmap {
    /// Called by evcxr to render the final expression of a cell.
    pub fn evcxr_display(&self) {
        println!("{}", self.bundle());
    }

    /// The MIME bundle evcxr reads from stdout: the SVG image delimited by evcxr's markers.
    fn bundle(&self) -> String {
        let [rows, cols] = self.0.dims();
        let values: Vec<f32> = self.0.clone().into_data().to_vec().unwrap();
        let svg = render_svg(&values, rows, cols);
        format!("EVCXR_BEGIN_CONTENT image/svg+xml\n{svg}\nEVCXR_END_CONTENT")
    }
}

fn min_max(values: &[f32]) -> (f32, f32) {
    values.iter().copied().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| {
        (lo.min(v), hi.max(v))
    })
}

/// Background color for a value on a white-to-blue scale over `[min, max]`.
fn heat_color(value: f32, min: f32, max: f32) -> String {
    let t = if max > min {
        ((value - min) / (max - min)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let r = (255.0 - 200.0 * t) as u8;
    let g = (255.0 - 120.0 * t) as u8;
    format!("rgb({r},{g},255)")
}

/// Render a matrix as an HTML table, truncating to [`MAX_SHOWN`] rows/columns. Pure function so it
/// can be unit-tested without a backend.
pub fn render_html(values: &[f32], rows: usize, cols: usize) -> String {
    let shown_rows = rows.min(MAX_SHOWN);
    let shown_cols = cols.min(MAX_SHOWN);
    let (min, max) = min_max(values);

    let mut html = String::new();
    html.push_str("<div style=\"font-family:monospace\">");
    html.push_str(&format!("<div>[{rows}x{cols}]</div>"));
    html.push_str("<table style=\"border-collapse:collapse\">");
    for r in 0..shown_rows {
        html.push_str("<tr>");
        for c in 0..shown_cols {
            let value = values[r * cols + c];
            let color = heat_color(value, min, max);
            html.push_str(&format!(
                "<td style=\"padding:2px 6px;text-align:right;background:{color}\">{value:.3}</td>"
            ));
        }
        if cols > shown_cols {
            html.push_str("<td>…</td>");
        }
        html.push_str("</tr>");
    }
    html.push_str("</table>");
    if rows > shown_rows {
        html.push_str("<div>…</div>");
    }
    html.push_str("</div>");
    html
}

/// Render a matrix as an SVG heatmap: one colored cell per element, no text, cropped to
/// [`MAX_HEATMAP`] rows/columns. Cell size adapts so the image stays around 512px on its long edge.
/// Pure function so it can be unit-tested without a backend.
pub fn render_svg(values: &[f32], rows: usize, cols: usize) -> String {
    let shown_rows = rows.min(MAX_HEATMAP);
    let shown_cols = cols.min(MAX_HEATMAP);
    let (min, max) = min_max(values);

    let longest = shown_rows.max(shown_cols).max(1);
    let cell = (512 / longest).clamp(2, 24);
    let width = shown_cols * cell;
    let height = shown_rows * cell;

    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" \
         shape-rendering=\"crispEdges\">"
    );
    for r in 0..shown_rows {
        for c in 0..shown_cols {
            let color = heat_color(values[r * cols + c], min, max);
            let (x, y) = (c * cell, r * cell);
            svg.push_str(&format!(
                "<rect x=\"{x}\" y=\"{y}\" width=\"{cell}\" height=\"{cell}\" fill=\"{color}\"/>"
            ));
        }
    }
    svg.push_str("</svg>");
    svg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_shape_and_every_cell() {
        let html = render_html(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        assert!(html.contains("[2x3]"));
        assert_eq!(html.matches("<td").count(), 6);
        assert!(html.contains("1.000"));
        assert!(html.contains("6.000"));
    }

    #[test]
    fn truncates_large_matrices() {
        let rows = MAX_SHOWN + 4;
        let cols = MAX_SHOWN + 4;
        let values = vec![0.0_f32; rows * cols];
        let html = render_html(&values, rows, cols);
        // One ellipsis cell per shown row, plus a trailing row ellipsis.
        assert_eq!(html.matches("…").count(), MAX_SHOWN + 1);
    }

    #[test]
    fn heat_color_handles_constant_input() {
        // No range: every cell falls back to the low end rather than dividing by zero.
        assert_eq!(heat_color(3.0, 3.0, 3.0), "rgb(255,255,255)");
    }

    #[test]
    fn bundle_wraps_html_in_evcxr_markers() {
        use burn::tensor::{Device, Tensor};

        let device = Device::default();
        let tensor = Tensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
        let bundle = Show(tensor).bundle();

        assert!(bundle.starts_with("EVCXR_BEGIN_CONTENT text/html\n"));
        assert!(bundle.trim_end().ends_with("EVCXR_END_CONTENT"));
        assert!(bundle.contains("[2x2]"));
    }

    #[test]
    fn svg_has_one_rect_per_cell() {
        let svg = render_svg(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 2, 3);
        assert!(svg.starts_with("<svg"));
        assert_eq!(svg.matches("<rect").count(), 6);
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn svg_crops_large_matrices() {
        let rows = MAX_HEATMAP + 10;
        let cols = MAX_HEATMAP + 10;
        let values = vec![0.0_f32; rows * cols];
        let svg = render_svg(&values, rows, cols);
        assert_eq!(svg.matches("<rect").count(), MAX_HEATMAP * MAX_HEATMAP);
    }

    #[test]
    fn heatmap_bundle_uses_svg_mime() {
        use burn::tensor::{Device, Tensor};

        let device = Device::default();
        let tensor = Tensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
        let bundle = Heatmap(tensor).bundle();

        assert!(bundle.starts_with("EVCXR_BEGIN_CONTENT image/svg+xml\n"));
        assert!(bundle.trim_end().ends_with("EVCXR_END_CONTENT"));
        assert!(bundle.contains("<svg"));
    }
}
