//! A Burn compute peer that runs in the browser: it brings up a WebGPU device and serves tensor
//! operations to remote clients over Iroh, while rendering a live egui telemetry dashboard on a
//! canvas. The whole page is the canvas; topic entry and monitoring both live in egui.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use burn::backend::remote::{Endpoint, RemoteNode, SecretKey};
use burn::server::{BURN_REMOTE_ALPN, Router, serve_with_telemetry, telemetry::TelemetryProbe};
use burn::tensor::Device;
use eframe::egui;
use iroh::endpoint::presets;
use remote_compute_dashboard::{Dashboard, PeerHandle};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::spawn_local;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Mount the peer dashboard onto the canvas with the given element id.
#[wasm_bindgen]
pub async fn run(canvas_id: String) -> Result<(), JsValue> {
    let canvas = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id(&canvas_id))
        .ok_or_else(|| JsValue::from_str("canvas element not found"))?
        .dyn_into::<web_sys::HtmlCanvasElement>()?;

    eframe::WebRunner::new()
        .start(
            canvas,
            eframe::WebOptions::default(),
            Box::new(|_cc| Ok(Box::new(PeerApp::default()))),
        )
        .await
}

fn topic_key(topic: &str) -> SecretKey {
    let hash = blake3::hash(format!("burn-p2p:{topic}").as_bytes());
    SecretKey::from_bytes(hash.as_bytes())
}

struct Started {
    dashboard: Dashboard,
    _router: Router,
}

enum Stage {
    Idle { topic: String, error: Option<String> },
    Starting,
    Serving(Started),
}

#[derive(Default)]
struct PeerApp {
    stage: Option<Stage>,
    pending: Rc<RefCell<Option<Result<Started, String>>>>,
}

impl eframe::App for PeerApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        if let Some(result) = self.pending.borrow_mut().take() {
            self.stage = Some(match result {
                Ok(started) => Stage::Serving(started),
                Err(error) => Stage::Idle {
                    topic: "burn-web".to_string(),
                    error: Some(error),
                },
            });
        }

        let stage = self.stage.get_or_insert_with(|| Stage::Idle {
            topic: "burn-web".to_string(),
            error: None,
        });

        match stage {
            Stage::Idle { topic, error } => {
                let mut start = false;
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.add_space(48.0);
                    ui.vertical_centered(|ui| {
                        ui.heading("Burn browser compute peer");
                        ui.label("This tab serves tensor operations on WebGPU over Iroh.");
                        ui.add_space(16.0);
                        ui.horizontal(|ui| {
                            ui.label("Topic");
                            ui.text_edit_singleline(topic);
                        });
                        if let Some(error) = error {
                            ui.add_space(6.0);
                            ui.colored_label(egui::Color32::from_rgb(0xE2, 0x4B, 0x4A), error.as_str());
                        }
                        ui.add_space(12.0);
                        start = ui.button("Start serving").clicked();
                    });
                });
                if start {
                    let topic = topic.clone();
                    let pending = self.pending.clone();
                    self.stage = Some(Stage::Starting);
                    spawn_local(startup(topic, pending));
                }
            }
            Stage::Starting => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.add_space(48.0);
                    ui.vertical_centered(|ui| {
                        ui.spinner();
                        ui.label("Bringing up WebGPU and binding the endpoint…");
                    });
                });
                ctx.request_repaint();
            }
            Stage::Serving(started) => {
                started.dashboard.update(ctx, frame);
            }
        }
    }
}

async fn startup(topic: String, pending: Rc<RefCell<Option<Result<Started, String>>>>) {
    let result = build_peer(&topic).await;
    *pending.borrow_mut() = Some(result);
}

async fn build_peer(topic: &str) -> Result<Started, String> {
    let device = Device::wgpu_async(Default::default()).await;

    let endpoint = Endpoint::builder(presets::N0)
        .secret_key(topic_key(topic))
        .alpns(vec![BURN_REMOTE_ALPN.to_vec()])
        .bind()
        .await
        .map_err(|err| err.to_string())?;

    let node = RemoteNode::from_endpoint(endpoint);
    let (probe, subscription) = TelemetryProbe::channel(8192);
    let router = serve_with_telemetry(device, node.clone(), probe);

    let peers: PeerHandle = Arc::new(Mutex::new(Default::default()));
    spawn_local(poll_peers(node.clone(), peers.clone()));

    let instance = format!("{topic} · {}", node.id());
    Ok(Started {
        dashboard: Dashboard::from_in_process(instance, subscription, peers),
        _router: router,
    })
}

async fn poll_peers(node: RemoteNode, peers: PeerHandle) {
    loop {
        gloo_timers::future::TimeoutFuture::new(500).await;
        let snapshot = node.peer_snapshot().await;
        if let Ok(mut slot) = peers.lock() {
            *slot = snapshot;
        }
    }
}
