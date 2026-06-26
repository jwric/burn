//! Isolated browser WebGPU self-test. No iroh, no swarm, no remote backend — just local tensor ops
//! that expose the Dawn (Chromium WebGPU) read_write binding-aliasing bug: a tensor multiplied by
//! itself binds one buffer to two writable bindings, which Dawn rejects (the whole submit is
//! dropped, kernels don't run, the result is stale). naga (Firefox, native) accepts it.

use burn::tensor::{Device, Int, Tensor};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

type Row = (String, Option<bool>, String);

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub async fn run() {
    let mut rows: Vec<Row> = vec![("adapter".into(), None, adapter_info().await)];
    render(&rows);

    let device = Device::wgpu_async(Default::default()).await;

    macro_rules! step {
        ($name:expr, $body:expr) => {{
            let (ok, detail) = $body.await;
            rows.push(($name.into(), Some(ok), detail));
            render(&rows);
        }};
    }

    step!("zeros == 0", check_zeros(&device));
    step!("ones == 1", check_ones(&device));
    step!("x * y  (distinct buffers, control)", check_distinct(&device));
    step!("x * x  (same buffer — the bug)", check_self_mul(&device));
    step!("mandelbrot tile vs CPU reference", check_mandelbrot(&device));
}

async fn check_zeros(d: &Device) -> (bool, String) {
    match Tensor::<1>::zeros([256], d).into_data_async().await {
        Ok(data) => {
            let v: Vec<f32> = data.iter::<f32>().collect();
            let nz = v.iter().filter(|x| **x != 0.0).count();
            (nz == 0, format!("nonzero={}/{}", nz, v.len()))
        }
        Err(e) => (false, format!("readback err: {e:?}")),
    }
}

async fn check_ones(d: &Device) -> (bool, String) {
    match Tensor::<1>::ones([256], d).into_data_async().await {
        Ok(data) => {
            let v: Vec<f32> = data.iter::<f32>().collect();
            let mism = v.iter().filter(|x| (**x - 1.0).abs() > 0.5).count();
            (mism == 0, format!("mismatch={}/{}", mism, v.len()))
        }
        Err(e) => (false, format!("readback err: {e:?}")),
    }
}

async fn check_distinct(d: &Device) -> (bool, String) {
    let x = Tensor::<1, Int>::arange(0..256i64, d).float();
    let y = Tensor::<1>::ones([256], d).mul_scalar(3.0);
    match (x * y).into_data_async().await {
        Ok(data) => {
            let v: Vec<f32> = data.iter::<f32>().collect();
            let mism = v
                .iter()
                .enumerate()
                .filter(|(i, x)| (**x - *i as f32 * 3.0).abs() > 0.5)
                .count();
            (mism == 0, format!("mismatch={}/{} [10]={:.0} (exp 30)", mism, v.len(), at(&v, 10)))
        }
        Err(e) => (false, format!("readback err: {e:?}")),
    }
}

async fn check_self_mul(d: &Device) -> (bool, String) {
    let x = Tensor::<1, Int>::arange(0..256i64, d).float();
    match (x.clone() * x.clone()).into_data_async().await {
        Ok(data) => {
            let v: Vec<f32> = data.iter::<f32>().collect();
            let mism = v
                .iter()
                .enumerate()
                .filter(|(i, x)| (**x - (*i as f32).powi(2)).abs() > 0.5)
                .count();
            (mism == 0, format!("mismatch={}/{} [2]={:.0} (exp 4) [16]={:.0} (exp 256)", mism, v.len(), at(&v, 2), at(&v, 16)))
        }
        Err(e) => (false, format!("readback err: {e:?}")),
    }
}

async fn check_mandelbrot(d: &Device) -> (bool, String) {
    let (xmin, xmax, y0, y1, w, h, mi) = (-2.5f32, 1.0f32, -1.0f32, 1.0f32, 256usize, 8usize, 200usize);
    let gpu = match gpu_mandel(d, xmin, xmax, y0, y1, w, h, mi).await {
        Ok(g) => g,
        Err(e) => return (false, format!("readback err: {e}")),
    };
    let reference = ref_mandel(xmin, xmax, y0, y1, w, h, mi);
    let mism = (0..gpu.len()).filter(|&i| (gpu[i] - reference[i]).abs() > 0.5).count();
    let (gmin, gmax) = (gpu.iter().cloned().fold(f32::MAX, f32::min), gpu.iter().cloned().fold(f32::MIN, f32::max));
    let uniform = (gmax - gmin) < 0.5;
    (mism == 0, format!("mismatch={}/{} gpu[{:.0}..{:.0}]{}", mism, gpu.len(), gmin, gmax, if uniform { " UNIFORM" } else { "" }))
}

#[allow(clippy::too_many_arguments)]
async fn gpu_mandel(d: &Device, xmin: f32, xmax: f32, y0: f32, y1: f32, w: usize, h: usize, max_iter: usize) -> Result<Vec<f32>, String> {
    let step_x = (xmax - xmin) / (w as f32 - 1.0);
    let step_y = (y1 - y0) / (h as f32 - 1.0);
    let xs = Tensor::<1, Int>::arange(0..w as i64, d).float().mul_scalar(step_x).add_scalar(xmin);
    let ys = Tensor::<1, Int>::arange(0..h as i64, d).float().mul_scalar(step_y).add_scalar(y0);
    let cx = xs.reshape([1, w]).expand([h, w]);
    let cy = ys.reshape([h, 1]).expand([h, w]);
    let mut zx = Tensor::<2>::zeros([h, w], d);
    let mut zy = Tensor::<2>::zeros([h, w], d);
    let mut count = Tensor::<2>::zeros([h, w], d);
    for _ in 0..max_iter {
        let zx2 = zx.clone() * zx.clone();
        let zy2 = zy.clone() * zy.clone();
        let inside = (zx2.clone() + zy2.clone()).lower_equal_elem(4.0).float();
        count = count + inside.clone();
        let next_zx = zx2 - zy2 + cx.clone();
        let next_zy = (zx.clone() * zy.clone()).mul_scalar(2.0) + cy.clone();
        let escaped = inside.clone().mul_scalar(-1.0).add_scalar(1.0);
        zx = next_zx * inside.clone() + zx * escaped.clone();
        zy = next_zy * inside + zy * escaped;
    }
    count.into_data_async().await.map(|d| d.iter::<f32>().collect()).map_err(|e| format!("{e:?}"))
}

#[allow(clippy::too_many_arguments)]
fn ref_mandel(xmin: f32, xmax: f32, y0: f32, y1: f32, w: usize, h: usize, max_iter: usize) -> Vec<f32> {
    let step_x = (xmax - xmin) / (w as f32 - 1.0);
    let step_y = (y1 - y0) / (h as f32 - 1.0);
    let mut out = vec![0.0f32; w * h];
    for j in 0..h {
        for i in 0..w {
            let (cx, cy) = (xmin + i as f32 * step_x, y0 + j as f32 * step_y);
            let (mut zx, mut zy, mut c) = (0.0f32, 0.0f32, 0.0f32);
            for _ in 0..max_iter {
                let (zx2, zy2) = (zx * zx, zy * zy);
                let inside = if zx2 + zy2 <= 4.0 { 1.0 } else { 0.0 };
                c += inside;
                let (nzx, nzy) = (zx2 - zy2 + cx, 2.0 * zx * zy + cy);
                let esc = 1.0 - inside;
                zx = nzx * inside + zx * esc;
                zy = nzy * inside + zy * esc;
            }
            out[j * w + i] = c;
        }
    }
    out
}

fn at(v: &[f32], i: usize) -> f32 {
    v.get(i).copied().unwrap_or(-1.0)
}

fn render(rows: &[Row]) {
    let mut html = String::new();
    for (name, pass, detail) in rows {
        let (mark, color) = match pass {
            Some(true) => ("PASS", "#3cb371"),
            Some(false) => ("FAIL", "#e24b4a"),
            None => ("·", "#888"),
        };
        html.push_str(&format!(
            "<div style='margin:8px 0'><b style='color:{color}'>{mark}</b> {name}<div style='color:#999;font-size:13px'>{detail}</div></div>"
        ));
    }
    if let Some(el) = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id("results"))
    {
        el.set_inner_html(&html);
    }
}

async fn adapter_info() -> String {
    let Some(win) = web_sys::window() else {
        return "no window".into();
    };
    let nav = match js_sys::Reflect::get(&JsValue::from(win), &JsValue::from_str("navigator")) {
        Ok(n) => n,
        Err(_) => return "no navigator".into(),
    };
    let gpu = match js_sys::Reflect::get(&nav, &JsValue::from_str("gpu")) {
        Ok(g) if !g.is_undefined() && !g.is_null() => g,
        _ => return "navigator.gpu undefined (insecure context?)".into(),
    };
    let request = js_sys::Reflect::get(&gpu, &JsValue::from_str("requestAdapter"))
        .ok()
        .and_then(|f| f.dyn_into::<js_sys::Function>().ok());
    let Some(request) = request else {
        return "no requestAdapter".into();
    };
    let promise = request
        .call0(&gpu)
        .ok()
        .and_then(|p| p.dyn_into::<js_sys::Promise>().ok());
    let Some(promise) = promise else {
        return "requestAdapter failed".into();
    };
    let adapter = match wasm_bindgen_futures::JsFuture::from(promise).await {
        Ok(a) if !a.is_undefined() && !a.is_null() => a,
        _ => return "no adapter".into(),
    };
    let info = js_sys::Reflect::get(&adapter, &JsValue::from_str("info")).unwrap_or(JsValue::UNDEFINED);
    let g = |k: &str| {
        js_sys::Reflect::get(&info, &JsValue::from_str(k))
            .ok()
            .and_then(|v| v.as_string())
            .unwrap_or_default()
    };
    format!("vendor={} arch={} device={}", g("vendor"), g("architecture"), g("device"))
}
