//! A browser client for the Burn compute swarm: it joins the gossip topic, discovers the compute
//! peers, and continuously fans an animated Mandelbrot *zoom* across them (each band on a different
//! peer, re-dispatched every frame so peers stay busy), drawing it to a canvas. The roster is
//! re-read each frame, so peers joining or leaving are picked up live. Launch with a join ticket in
//! the URL fragment (`…/#burnswarm…`) or enter one in the UI.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use burn::backend::remote::{Endpoint, RemoteNode};
use burn::tensor::{Device, Int, Tensor};
use eframe::egui;
use futures_util::future::join_all;
use iroh::EndpointId;
use iroh::endpoint::presets;
use remote_swarm::{GOSSIP_ALPN, JoinTicket, Swarm, SwarmConfig, topic_from_label};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;
use web_time::Instant;

const WIDTH: usize = 256;
const HEIGHT: usize = 160;
const BAND_H: usize = 8;
const BANDS: usize = HEIGHT / BAND_H;
const FLOPS_PER_ITER: f64 = 10.0;

const CENTER: (f32, f32) = (-0.743_643_9, 0.131_825_9); // seahorse valley
const START_HALF: f32 = 1.3;
const MIN_HALF: f32 = 3e-5; // f32 detail floor; reset the zoom below this
const ZOOM_PER_FRAME: f32 = 0.94;
const BASE_ITER: f64 = 90.0;
const ITER_PER_OCTAVE: f64 = 14.0;
const ITER_CAP: f64 = 400.0;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Mount the client onto the canvas; `join` is an optional ticket/topic from the URL fragment that
/// auto-starts the render when non-empty.
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
            Box::new(move |_cc| Ok(Box::new(ClientApp::new(join)))),
        )
        .await
}

#[derive(Default)]
struct Render {
    size: [usize; 2],
    counts: Vec<f32>,
    max_iter: usize,
    bands_total: usize,
    bands_done: usize,
    attribution: Vec<String>,
    peers: Vec<String>,
    status: String,
    frame: u32,
    gflops: f64,
    error: Option<String>,
}

enum Stage {
    Idle {
        input: String,
        error: Option<String>,
    },
    Running,
}

struct ClientApp {
    stage: Stage,
    shared: Rc<RefCell<Render>>,
    autostart: Option<String>,
    texture: Option<egui::TextureHandle>,
    shown: (u32, usize),
}

impl ClientApp {
    fn new(join: String) -> Self {
        let join = join.trim().to_string();
        let input = if join.is_empty() {
            "burn-web".to_string()
        } else {
            join.clone()
        };
        Self {
            stage: Stage::Idle { input, error: None },
            shared: Rc::new(RefCell::new(Render::default())),
            autostart: (!join.is_empty()).then_some(join),
            texture: None,
            shown: (u32::MAX, usize::MAX),
        }
    }

    fn start(&mut self, input: String, ctx: &egui::Context) {
        *self.shared.borrow_mut() = Render::default();
        self.texture = None;
        self.shown = (u32::MAX, usize::MAX);
        self.stage = Stage::Running;
        spawn_local(drive(input, self.shared.clone(), ctx.clone()));
    }
}

impl eframe::App for ClientApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Some(input) = self.autostart.take() {
            self.start(input, ctx);
        }

        match &mut self.stage {
            Stage::Idle { input, error } => {
                let mut go = None;
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.add_space(48.0);
                    ui.vertical_centered(|ui| {
                        ui.heading("Burn swarm client");
                        ui.label(
                            "Continuously renders an animated Mandelbrot zoom across the swarm.",
                        );
                        ui.add_space(16.0);
                        ui.horizontal(|ui| {
                            ui.label("Topic or join ticket");
                            ui.text_edit_singleline(input);
                        });
                        if let Some(error) = error {
                            ui.add_space(6.0);
                            ui.colored_label(
                                egui::Color32::from_rgb(0xE2, 0x4B, 0x4A),
                                error.as_str(),
                            );
                        }
                        ui.add_space(12.0);
                        if ui.button("Render on the swarm").clicked() {
                            go = Some(input.clone());
                        }
                    });
                });
                if let Some(input) = go {
                    self.start(input, ctx);
                }
            }
            Stage::Running => {
                let (
                    error,
                    status,
                    size,
                    max_iter,
                    bands_done,
                    bands_total,
                    peers,
                    attribution,
                    frame,
                    gflops,
                ) = {
                    let r = self.shared.borrow();
                    (
                        r.error.clone(),
                        r.status.clone(),
                        r.size,
                        r.max_iter,
                        r.bands_done,
                        r.bands_total,
                        r.peers.clone(),
                        r.attribution.clone(),
                        r.frame,
                        r.gflops,
                    )
                };

                if let Some(error) = error {
                    self.stage = Stage::Idle {
                        input: "burn-web".to_string(),
                        error: Some(error),
                    };
                    ctx.request_repaint();
                    return;
                }

                if size[0] > 0 && (frame, bands_done) != self.shown {
                    let pixels: Vec<egui::Color32> = self
                        .shared
                        .borrow()
                        .counts
                        .iter()
                        .map(|&c| color(c, max_iter))
                        .collect();
                    let image = egui::ColorImage { size, pixels };
                    self.texture =
                        Some(ctx.load_texture("mandelbrot", image, egui::TextureOptions::NEAREST));
                    self.shown = (frame, bands_done);
                }

                egui::TopBottomPanel::top("header").show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.heading("🐝 Burn swarm");
                        ui.separator();
                        ui.label(format!("{} peer(s)", peers.len()));
                        ui.separator();
                        ui.label(format!("frame {frame}"));
                        ui.separator();
                        ui.label(format!("~{gflops:.2} GFLOP/s"));
                        ui.separator();
                        ui.label(format!("{bands_done}/{bands_total} tiles"));
                    });
                });

                egui::SidePanel::right("info").show(ctx, |ui| {
                    ui.label(&status);
                    ui.separator();
                    for peer in &peers {
                        ui.label(peer);
                    }
                    if !attribution.is_empty() {
                        ui.separator();
                        ui.label("band → peer");
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            for (band, name) in attribution.iter().enumerate() {
                                ui.monospace(format!("{band:>2}: {name}"));
                            }
                        });
                    }
                });

                egui::CentralPanel::default().show(ctx, |ui| {
                    if let Some(texture) = &self.texture {
                        let [w, h] = size;
                        let avail = ui.available_size();
                        let scale = (avail.x / w as f32).min(avail.y / h as f32).max(1.0);
                        ui.image((texture.id(), egui::vec2(w as f32 * scale, h as f32 * scale)));
                    } else {
                        ui.centered_and_justified(|ui| {
                            ui.spinner();
                        });
                    }
                });

                ctx.request_repaint();
            }
        }
    }
}

async fn drive(input: String, shared: Rc<RefCell<Render>>, ctx: egui::Context) {
    if let Err(error) = drive_inner(&input, &shared, &ctx).await {
        shared.borrow_mut().error = Some(error);
        ctx.request_repaint();
    }
}

async fn drive_inner(
    input: &str,
    shared: &Rc<RefCell<Render>>,
    ctx: &egui::Context,
) -> Result<(), String> {
    let (topic, bootstrap) = match JoinTicket::decode(input) {
        Ok(ticket) => (ticket.topic(), ticket.bootstrap().to_vec()),
        Err(_) => (topic_from_label(input), Vec::new()),
    };

    shared.borrow_mut().status = "binding endpoint…".to_string();
    ctx.request_repaint();

    let endpoint = Endpoint::builder(presets::N0)
        .alpns(vec![GOSSIP_ALPN.to_vec()])
        .bind()
        .await
        .map_err(|err| err.to_string())?;
    endpoint.online().await;
    for addr in &bootstrap {
        let _ = endpoint.connect(addr.clone(), GOSSIP_ALPN).await;
    }

    let node = RemoteNode::from_endpoint(endpoint.clone());
    let bootstrap_ids = bootstrap.iter().map(|addr| addr.id).collect();
    let (swarm, _router) = Swarm::spawn(endpoint, SwarmConfig::new(topic).bootstrap(bootstrap_ids))
        .await
        .map_err(|err| err.to_string())?;

    shared.borrow_mut().status = "discovering peers…".to_string();
    ctx.request_repaint();
    for _ in 0..40 {
        if swarm.peer_count() > 0 {
            break;
        }
        sleep_ms(300).await;
    }
    sleep_ms(1500).await;

    {
        let mut r = shared.borrow_mut();
        r.size = [WIDTH, HEIGHT];
        r.counts = vec![0.0; WIDTH * HEIGHT];
    }

    let mut devices: HashMap<EndpointId, Device> = HashMap::new();
    let mut y_half = START_HALF;
    let mut frame: u32 = 0;
    let mut ema = 0.0f64;

    loop {
        let mut roster = swarm.roster();
        roster.sort_by_key(|entry| entry.advert.caps.backend != "wgpu"); // GPU peers first
        if roster.is_empty() {
            {
                let mut r = shared.borrow_mut();
                r.status = "waiting for peers…".to_string();
                r.peers.clear();
                r.bands_total = 0;
            }
            ctx.request_repaint();
            sleep_ms(500).await;
            continue;
        }

        // Keep one pooled remote device per live peer; drop devices for peers that have left.
        let live: HashSet<EndpointId> = roster.iter().map(|e| e.advert.endpoint_id()).collect();
        devices.retain(|id, _| live.contains(id));
        let missing: Vec<(EndpointId, _)> = roster
            .iter()
            .filter(|peer| !devices.contains_key(&peer.advert.endpoint_id()))
            .map(|peer| (peer.advert.endpoint_id(), peer.advert.ticket.clone()))
            .collect();
        for (id, ticket) in missing {
            let device = Device::remote_ticket_async(&node, &ticket, 0).await;
            devices.insert(id, device);
        }

        let octaves = (START_HALF / y_half).max(1.0).log2() as f64;
        let max_iter = (BASE_ITER + octaves * ITER_PER_OCTAVE).min(ITER_CAP) as usize;
        let x_half = y_half * WIDTH as f32 / HEIGHT as f32;
        let (xmin, xmax) = (CENTER.0 - x_half, CENTER.0 + x_half);
        let (ymin, ymax) = (CENTER.1 - y_half, CENTER.1 + y_half);

        {
            let mut r = shared.borrow_mut();
            r.max_iter = max_iter;
            r.frame = frame;
            r.bands_total = BANDS;
            r.bands_done = 0;
            r.status = "rendering".to_string();
            r.peers = roster
                .iter()
                .map(|p| {
                    format!(
                        "{} [{}]",
                        p.advert.name.clone().unwrap_or_default(),
                        p.advert.caps.backend
                    )
                })
                .collect();
            r.attribution = (0..BANDS)
                .map(|b| {
                    roster[b % roster.len()]
                        .advert
                        .name
                        .clone()
                        .unwrap_or_default()
                })
                .collect();
        }

        let t0 = Instant::now();
        let tiles = (0..BANDS).map(|band| {
            let shared = shared.clone();
            let ctx = ctx.clone();
            let device = devices[&roster[band % roster.len()].advert.endpoint_id()].clone();
            async move {
                let y0 = ymin + (ymax - ymin) * band as f32 / BANDS as f32;
                let y1 = ymin + (ymax - ymin) * (band + 1) as f32 / BANDS as f32;
                // A peer that dropped mid-frame just leaves its band stale; next frame's roster drops it.
                let params = Tile {
                    xmin,
                    xmax,
                    y0,
                    y1,
                    w: WIDTH,
                    h: BAND_H,
                    max_iter,
                };
                match mandelbrot_tile(&device, params).await {
                    Ok(tile) => {
                        let mut r = shared.borrow_mut();
                        let offset = band * BAND_H * WIDTH;
                        r.counts[offset..offset + tile.len()].copy_from_slice(&tile);
                        r.bands_done += 1;
                        drop(r);
                        ctx.request_repaint();
                    }
                    Err(err) => web_sys::console::warn_1(
                        &format!("band {band} read failed: {err}").into(),
                    ),
                }
            }
        });
        join_all(tiles).await;

        let dt = t0.elapsed().as_secs_f64().max(1e-3);
        let rate = (WIDTH * HEIGHT * max_iter) as f64 * FLOPS_PER_ITER / dt / 1e9;
        ema = if frame == 0 {
            rate
        } else {
            0.6 * ema + 0.4 * rate
        };
        shared.borrow_mut().gflops = ema;
        ctx.request_repaint();

        y_half *= ZOOM_PER_FRAME;
        if y_half < MIN_HALF {
            y_half = START_HALF;
        }
        frame = frame.wrapping_add(1);
    }
}

async fn sleep_ms(ms: u32) {
    gloo_timers::future::TimeoutFuture::new(ms).await;
}

struct Tile {
    xmin: f32,
    xmax: f32,
    y0: f32,
    y1: f32,
    w: usize,
    h: usize,
    max_iter: usize,
}

async fn mandelbrot_tile(device: &Device, tile: Tile) -> Result<Vec<f32>, String> {
    let Tile {
        xmin,
        xmax,
        y0,
        y1,
        w,
        h,
        max_iter,
    } = tile;
    let step_x = (xmax - xmin) / (w as f32 - 1.0);
    let step_y = (y1 - y0) / (h as f32 - 1.0);

    let xs = Tensor::<1, Int>::arange(0..w as i64, device)
        .float()
        .mul_scalar(step_x)
        .add_scalar(xmin);
    let ys = Tensor::<1, Int>::arange(0..h as i64, device)
        .float()
        .mul_scalar(step_y)
        .add_scalar(y0);
    let cx = xs.reshape([1, w]).expand([h, w]);
    let cy = ys.reshape([h, 1]).expand([h, w]);

    let mut zx = Tensor::<2>::zeros([h, w], device);
    let mut zy = Tensor::<2>::zeros([h, w], device);
    let mut count = Tensor::<2>::zeros([h, w], device);

    for _ in 0..max_iter {
        let zx2 = zx.clone() * zx.clone();
        let zy2 = zy.clone() * zy.clone();
        // inside |z| <= 2 (|z|^2 <= 4): 1.0 while still iterating, 0.0 once escaped
        let inside = (zx2.clone() + zy2.clone()).lower_equal_elem(4.0).float();
        count = count + inside.clone();
        let next_zx = zx2 - zy2 + cx.clone();
        let next_zy = (zx.clone() * zy.clone()).mul_scalar(2.0) + cy.clone();
        // Freeze escaped points so |z| can't run away to inf/NaN. Counts are unaffected (escaped
        // points already stopped counting); it keeps `<= 4.0` well-defined on backends that mishandle
        // NaN comparisons (some mobile WebGPU drivers), which would otherwise paint every pixel black.
        let escaped = inside.clone().mul_scalar(-1.0).add_scalar(1.0);
        zx = next_zx * inside.clone() + zx * escaped.clone();
        zy = next_zy * inside + zy * escaped;
    }

    let data = count
        .into_data_async()
        .await
        .map_err(|err| format!("read tile: {err:?}"))?;
    Ok(data.iter::<f32>().collect())
}

fn color(count: f32, max_iter: usize) -> egui::Color32 {
    if count >= max_iter as f32 {
        return egui::Color32::BLACK;
    }
    let t = (count / max_iter as f32).clamp(0.0, 1.0);
    let r = 9.0 * (1.0 - t) * t * t * t;
    let g = 15.0 * (1.0 - t) * (1.0 - t) * t * t;
    let b = 8.5 * (1.0 - t) * (1.0 - t) * (1.0 - t) * t;
    egui::Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}
