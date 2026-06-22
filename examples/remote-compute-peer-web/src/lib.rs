//! A Burn compute peer that runs in the browser and joins a gossip *swarm*: it brings up a WebGPU
//! device, serves tensor operations to remote clients over Iroh, advertises itself on a shared
//! gossip topic so clients can discover it, and renders a live egui telemetry dashboard.
//!
//! The page can be launched with a join ticket in the URL fragment (`…/#burnswarm…`, e.g. from a
//! scanned QR code); it then joins that swarm automatically. Without one, enter a topic label or
//! ticket in the UI.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use burn::backend::remote::{Endpoint, RemoteNode};
use burn::server::{
    BURN_REMOTE_ALPN, Router, serve_builder_with_telemetry, telemetry::TelemetryProbe,
};
use burn::tensor::Device;
use eframe::egui;
use iroh::EndpointAddr;
use iroh::endpoint::presets;
use remote_compute_dashboard::{Dashboard, PeerHandle};
use remote_swarm::{
    GOSSIP_ALPN, Gossip, JoinTicket, PeerAdvert, PeerCaps, RemoteTicket, Swarm, SwarmConfig,
    TopicId, topic_from_label,
};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Mount the peer dashboard onto the canvas with the given element id.
///
/// `join` is an optional join ticket (or topic label) taken from the page URL fragment; when
/// non-empty the peer starts serving and joins that swarm immediately.
#[wasm_bindgen]
pub async fn run(canvas_id: String, join: String) -> Result<(), JsValue> {
    let canvas = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id(&canvas_id))
        .ok_or_else(|| JsValue::from_str("canvas element not found"))?
        .dyn_into::<web_sys::HtmlCanvasElement>()?;

    eframe::WebRunner::new()
        .start(
            canvas,
            eframe::WebOptions::default(),
            Box::new(move |_cc| Ok(Box::new(PeerApp::new(join)))),
        )
        .await
}

fn short_id(node: &RemoteNode) -> String {
    let id = node.id().to_string();
    id[..id.len().min(8)].to_string()
}

struct Started {
    dashboard: Dashboard,
    swarm: Swarm,
    _router: Router,
}

enum Stage {
    Idle {
        topic: String,
        error: Option<String>,
    },
    Starting,
    Serving(Started),
}

struct PeerApp {
    stage: Option<Stage>,
    pending: Rc<RefCell<Option<Result<Started, String>>>>,
    // A join ticket from the URL fragment; consumed on the first frame to auto-start.
    autostart: Option<String>,
}

impl PeerApp {
    fn new(join: String) -> Self {
        let join = join.trim().to_string();
        Self {
            stage: None,
            pending: Rc::new(RefCell::new(None)),
            autostart: (!join.is_empty()).then_some(join),
        }
    }
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

        // A ticket arrived in the URL (e.g. a scanned QR) — start serving without a click.
        if let Some(input) = self.autostart.take() {
            self.stage = Some(Stage::Starting);
            spawn_local(startup(input, self.pending.clone()));
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
                        ui.label(
                            "This tab serves tensor operations on WebGPU and joins a gossip swarm.",
                        );
                        ui.add_space(16.0);
                        ui.horizontal(|ui| {
                            ui.label("Topic or join ticket");
                            ui.text_edit_singleline(topic);
                        });
                        if let Some(error) = error {
                            ui.add_space(6.0);
                            ui.colored_label(
                                egui::Color32::from_rgb(0xE2, 0x4B, 0x4A),
                                error.as_str(),
                            );
                        }
                        ui.add_space(12.0);
                        start = ui.button("Start serving").clicked();
                    });
                });
                if start {
                    let topic = topic.clone();
                    self.stage = Some(Stage::Starting);
                    spawn_local(startup(topic, self.pending.clone()));
                }
            }
            Stage::Starting => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.add_space(48.0);
                    ui.vertical_centered(|ui| {
                        ui.spinner();
                        ui.label("Bringing up WebGPU, binding the endpoint, joining the swarm…");
                    });
                });
                ctx.request_repaint();
            }
            Stage::Serving(started) => {
                let roster = started.swarm.roster();
                egui::TopBottomPanel::top("swarm").show(ctx, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.strong(format!("🐝 swarm — {} other peer(s)", roster.len()));
                        for entry in &roster {
                            if let Some(name) = &entry.advert.name {
                                ui.separator();
                                ui.label(name);
                            }
                        }
                    });
                });
                started.dashboard.update(ctx, frame);
                // Keep the roster panel fresh even when compute is idle.
                ctx.request_repaint();
            }
        }
    }
}

async fn startup(input: String, pending: Rc<RefCell<Option<Result<Started, String>>>>) {
    let result = build_peer(&input).await;
    *pending.borrow_mut() = Some(result);
}

async fn build_peer(input: &str) -> Result<Started, String> {
    let device = Device::wgpu_async(Default::default()).await;

    // The input is either a `burnswarm…` join ticket (scanned QR) or a plain topic label.
    let (topic, bootstrap): (TopicId, Vec<EndpointAddr>) = match JoinTicket::decode(input) {
        Ok(ticket) => (ticket.topic(), ticket.bootstrap().to_vec()),
        Err(_) => (topic_from_label(input), Vec::new()),
    };

    // A swarm peer keeps its own random identity (many peers share a topic) and serves both the
    // compute protocol and the gossip control protocol on one endpoint.
    let endpoint = Endpoint::builder(presets::N0)
        .alpns(vec![BURN_REMOTE_ALPN.to_vec(), GOSSIP_ALPN.to_vec()])
        .bind()
        .await
        .map_err(|err| err.to_string())?;

    let node = RemoteNode::from_endpoint(endpoint.clone());
    let (probe, subscription) = TelemetryProbe::channel(8192);

    // One endpoint, one router, two protocols: Burn Remote (tensors) + iroh-gossip (discovery).
    let gossip = Gossip::builder().spawn(endpoint.clone());
    let router = serve_builder_with_telemetry(device, node.clone(), probe)
        .accept(GOSSIP_ALPN, gossip.clone())
        .spawn();

    // Wait until a relay address is known, then learn the bootstrap peers so the gossip join
    // doesn't stall on discovery.
    endpoint.online().await;
    for addr in &bootstrap {
        let _ = endpoint.connect(addr.clone(), GOSSIP_ALPN).await;
    }

    // Advertise how to reach this tab's WebGPU device.
    let advert = PeerAdvert::new(
        RemoteTicket::new(endpoint.addr(), Vec::new()),
        Some(format!("browser · {}", short_id(&node))),
        PeerCaps {
            backend: "wgpu".to_string(),
            device: None,
            devices: 1,
            browser: true,
        },
    );
    let bootstrap_ids = bootstrap.iter().map(|addr| addr.id).collect();
    let config = SwarmConfig::new(topic)
        .bootstrap(bootstrap_ids)
        .advert(advert);
    let swarm = Swarm::join(endpoint.clone(), &gossip, config)
        .await
        .map_err(|err| err.to_string())?;

    let peers: PeerHandle = Arc::new(Mutex::new(Default::default()));
    spawn_local(poll_peers(node.clone(), peers.clone()));

    let instance = format!("swarm · {}", short_id(&node));
    Ok(Started {
        dashboard: Dashboard::from_in_process(instance, subscription, peers),
        swarm,
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
