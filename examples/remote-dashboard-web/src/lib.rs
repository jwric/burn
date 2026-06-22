//! Browser viewer for a Burn Remote peer's telemetry.
//!
//! Renders the shared egui [`Dashboard`] on a canvas, fed by a stream of [`DashboardState`]
//! snapshots the native peer publishes at `/events`. Open the peer's HTTP address in a browser and
//! this is what serves. Because the server holds the state, a refresh resumes the live picture.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use remote_compute_dashboard::{Dashboard, DashboardState, LinkStatus, StateSource};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

const STALE_MS: f64 = 2000.0;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

#[derive(Default, Clone)]
struct Shared {
    state: Rc<RefCell<DashboardState>>,
    fresh: Rc<Cell<bool>>,
    open: Rc<Cell<bool>>,
}

struct SseSource {
    shared: Shared,
    last_seen_ms: f64,
}

impl StateSource for SseSource {
    fn latest(&mut self, now_ms: f64) -> (DashboardState, LinkStatus) {
        if self.shared.fresh.replace(false) {
            self.last_seen_ms = now_ms;
        }
        let link = if !self.shared.open.get() {
            LinkStatus::Lost
        } else if self.last_seen_ms == 0.0 || now_ms - self.last_seen_ms > STALE_MS {
            LinkStatus::Stale
        } else {
            LinkStatus::Connected
        };
        (self.shared.state.borrow().clone(), link)
    }
}

/// Mount the viewer on the canvas with id `canvas_id`, streaming snapshots from `events_url`.
#[wasm_bindgen]
pub async fn run(canvas_id: String, events_url: String) -> Result<(), JsValue> {
    let canvas = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id(&canvas_id))
        .ok_or_else(|| JsValue::from_str("canvas element not found"))?
        .dyn_into::<web_sys::HtmlCanvasElement>()?;

    let shared = Shared::default();
    shared.open.set(true);
    connect(&events_url, shared.clone())?;

    let app = Dashboard::new(Box::new(SseSource {
        shared,
        last_seen_ms: 0.0,
    }));

    eframe::WebRunner::new()
        .start(
            canvas,
            eframe::WebOptions::default(),
            Box::new(|_cc| Ok(Box::new(app))),
        )
        .await
}

fn connect(url: &str, shared: Shared) -> Result<(), JsValue> {
    let source = web_sys::EventSource::new(url)?;

    let on_message = Closure::<dyn FnMut(web_sys::MessageEvent)>::new({
        let shared = shared.clone();
        move |event: web_sys::MessageEvent| {
            if let Some(text) = event.data().as_string()
                && let Ok(state) = serde_json::from_str::<DashboardState>(&text)
            {
                *shared.state.borrow_mut() = state;
                shared.fresh.set(true);
                shared.open.set(true);
            }
        }
    });
    source.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
    on_message.forget();

    let on_error = Closure::<dyn FnMut(web_sys::Event)>::new({
        let shared = shared.clone();
        move |_event: web_sys::Event| shared.open.set(false)
    });
    source.set_onerror(Some(on_error.as_ref().unchecked_ref()));
    on_error.forget();

    std::mem::forget(source);
    Ok(())
}
