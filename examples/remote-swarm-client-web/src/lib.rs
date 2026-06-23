//! A browser client for the Burn compute swarm: it joins the gossip topic, discovers the compute
//! peers, fans a Mandelbrot across them (each band on a different peer), and draws the result on a
//! canvas. Launch with a join ticket in the URL fragment (`…/#burnswarm…`) or enter one in the UI.

use std::cell::RefCell;
use std::rc::Rc;

use burn::backend::remote::{Endpoint, RemoteNode};
use burn::tensor::{Device, Int, Tensor};
use eframe::egui;
use iroh::endpoint::presets;
use remote_swarm::{GOSSIP_ALPN, JoinTicket, RosterEntry, Swarm, SwarmConfig, topic_from_label};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;

const WIDTH: usize = 120;
const BAND_H: usize = 6;
const MAX_ITER: usize = 60;
const VIEW: (f32, f32, f32, f32) = (-2.6, 1.0, -1.2, 1.2); // xmin, xmax, ymin, ymax

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
    bands_total: usize,
    bands_done: usize,
    attribution: Vec<String>,
    peers: Vec<String>,
    status: String,
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
    shown_bands: usize,
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
            shown_bands: usize::MAX,
        }
    }

    fn start(&mut self, input: String, ctx: &egui::Context) {
        *self.shared.borrow_mut() = Render::default();
        self.texture = None;
        self.shown_bands = usize::MAX;
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
                            "Fans a Mandelbrot across the swarm's compute peers and draws it.",
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
                let snapshot = {
                    let r = self.shared.borrow();
                    (
                        r.error.clone(),
                        r.status.clone(),
                        r.size,
                        r.bands_done,
                        r.bands_total,
                        r.peers.clone(),
                        r.attribution.clone(),
                    )
                };
                let (error, status, size, bands_done, bands_total, peers, attribution) = snapshot;

                if let Some(error) = error {
                    self.stage = Stage::Idle {
                        input: "burn-web".to_string(),
                        error: Some(error),
                    };
                    ctx.request_repaint();
                    return;
                }

                if size[0] > 0 && bands_done != self.shown_bands {
                    let pixels: Vec<egui::Color32> = self
                        .shared
                        .borrow()
                        .counts
                        .iter()
                        .map(|&c| color(c))
                        .collect();
                    let image = egui::ColorImage { size, pixels };
                    self.texture =
                        Some(ctx.load_texture("mandelbrot", image, egui::TextureOptions::NEAREST));
                    self.shown_bands = bands_done;
                }

                egui::SidePanel::right("info").show(ctx, |ui| {
                    ui.heading("swarm");
                    ui.label(&status);
                    if bands_total > 0 {
                        ui.label(format!("bands {bands_done}/{bands_total}"));
                    }
                    ui.separator();
                    ui.label(format!("{} compute peer(s)", peers.len()));
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

                if bands_total == 0 || bands_done < bands_total {
                    ctx.request_repaint();
                }
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

    let peers = discover(&swarm).await;
    if peers.is_empty() {
        return Err("no compute peers found in the swarm".to_string());
    }

    let bands = peers.len() * 4;
    let height = bands * BAND_H;
    {
        let mut r = shared.borrow_mut();
        r.size = [WIDTH, height];
        r.counts = vec![0.0; WIDTH * height];
        r.bands_total = bands;
        r.peers = peers
            .iter()
            .map(|p| {
                format!(
                    "{} [{}]",
                    p.advert.name.clone().unwrap_or_default(),
                    p.advert.caps.backend
                )
            })
            .collect();
        r.status = "rendering…".to_string();
    }
    ctx.request_repaint();

    let mut devices = Vec::with_capacity(peers.len());
    for peer in &peers {
        devices.push(Device::remote_ticket_async(&node, &peer.advert.ticket, 0).await);
    }

    let (xmin, xmax, ymin, ymax) = VIEW;
    for band in 0..bands {
        let pi = band % peers.len();
        let y0 = ymin + (ymax - ymin) * band as f32 / bands as f32;
        let y1 = ymin + (ymax - ymin) * (band + 1) as f32 / bands as f32;
        let tile = mandelbrot_tile(&devices[pi], xmin, xmax, y0, y1, WIDTH, BAND_H).await?;
        let name = peers[pi].advert.name.clone().unwrap_or_default();

        let mut r = shared.borrow_mut();
        let offset = band * BAND_H * WIDTH;
        r.counts[offset..offset + tile.len()].copy_from_slice(&tile);
        r.bands_done += 1;
        r.attribution.push(name);
        drop(r);
        ctx.request_repaint();
    }

    shared.borrow_mut().status = "done ✓".to_string();
    ctx.request_repaint();
    Ok(())
}

async fn discover(swarm: &Swarm) -> Vec<RosterEntry> {
    for _ in 0..50 {
        if swarm.peer_count() > 0 {
            break;
        }
        gloo_timers::future::TimeoutFuture::new(300).await;
    }
    gloo_timers::future::TimeoutFuture::new(2000).await;
    let mut roster = swarm.roster();
    roster.sort_by_key(|entry| entry.advert.caps.backend != "wgpu"); // GPU peers first
    roster
}

async fn mandelbrot_tile(
    device: &Device,
    xmin: f32,
    xmax: f32,
    y0: f32,
    y1: f32,
    w: usize,
    h: usize,
) -> Result<Vec<f32>, String> {
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

    for _ in 0..MAX_ITER {
        let zx2 = zx.clone() * zx.clone();
        let zy2 = zy.clone() * zy.clone();
        // inside |z| <= 2 (|z|^2 <= 4): 1.0 while still iterating, 0.0 once escaped
        let inside = (zx2.clone() + zy2.clone()).lower_equal_elem(4.0).float();
        count = count + inside;
        let next_zx = zx2 - zy2 + cx.clone();
        let next_zy = (zx.clone() * zy.clone()).mul_scalar(2.0) + cy.clone();
        zx = next_zx;
        zy = next_zy;
    }

    let data = count
        .into_data_async()
        .await
        .map_err(|err| format!("read tile: {err:?}"))?;
    Ok(data.iter::<f32>().collect())
}

fn color(count: f32) -> egui::Color32 {
    if count >= MAX_ITER as f32 {
        return egui::Color32::BLACK;
    }
    let t = (count / MAX_ITER as f32).clamp(0.0, 1.0);
    egui::Color32::from_rgb(
        (t * t * 255.0) as u8,
        (t * 255.0) as u8,
        (64.0 + t * 191.0) as u8,
    )
}
