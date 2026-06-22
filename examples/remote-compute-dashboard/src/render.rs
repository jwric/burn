//! Real-time egui view of a Burn Remote compute peer.
//!
//! Renders a [`DashboardState`] (current, windowed activity, not totals since boot): throughput
//! stats, an animated op-class flow graph showing how tensors transit between op categories, an
//! animated peer map, and a recent op stream. The state is produced by a [`StateSource`] -- either
//! an in-process [`Aggregator`] (the browser peer) or a remote snapshot stream (the HTTP viewer).
//! It implements [`eframe::App`] so the same code drives a native window and a browser canvas.
//!
//! Snapshots arrive in discrete steps; the [`Anim`] state interpolates between them and integrates
//! flow phase over time, so dots accelerate smoothly instead of teleporting when a rate changes.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use burn_remote::telemetry::{
    DrainStatus, OpClass, PeerSnapshot, TelemetryEvent, TelemetrySubscription,
};
use eframe::egui;

use crate::{Aggregator, DashboardState, PeerStat};

/// Shared slot a peer-snapshot poller writes and the in-process source reads.
pub type PeerHandle = Arc<Mutex<PeerSnapshot>>;

/// Connection health of the underlying source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkStatus {
    Connected,
    Stale,
    Lost,
}

/// Where the dashboard pulls its current [`DashboardState`] from.
pub trait StateSource {
    fn latest(&mut self, now_ms: f64) -> (DashboardState, LinkStatus);
}

/// A source that aggregates an in-process [`TelemetrySubscription`] locally.
pub struct InProcessSource {
    aggregator: Aggregator,
    subscription: TelemetrySubscription,
    peers: PeerHandle,
    scratch: Vec<Arc<TelemetryEvent>>,
}

impl InProcessSource {
    pub fn new(instance: String, subscription: TelemetrySubscription, peers: PeerHandle) -> Self {
        Self {
            aggregator: Aggregator::new(instance),
            subscription,
            peers,
            scratch: Vec::new(),
        }
    }
}

impl StateSource for InProcessSource {
    fn latest(&mut self, now_ms: f64) -> (DashboardState, LinkStatus) {
        self.scratch.clear();
        let status = match self.subscription.drain_into(&mut self.scratch) {
            DrainStatus::Open { .. } => LinkStatus::Connected,
            DrainStatus::Closed => LinkStatus::Lost,
        };
        for event in self.scratch.drain(..) {
            self.aggregator.apply(&event);
        }
        if let Ok(peers) = self.peers.lock() {
            self.aggregator.set_peers(&peers, now_ms);
        }
        self.aggregator.tick(now_ms);
        (self.aggregator.snapshot(now_ms), status)
    }
}

/// The dashboard application.
pub struct Dashboard {
    source: Box<dyn StateSource>,
    anim: Anim,
}

impl Dashboard {
    pub fn new(source: Box<dyn StateSource>) -> Self {
        Self {
            source,
            anim: Anim::default(),
        }
    }

    /// Build a dashboard fed by an in-process probe and peer slot.
    pub fn from_in_process(
        instance: String,
        subscription: TelemetrySubscription,
        peers: PeerHandle,
    ) -> Self {
        Self::new(Box::new(InProcessSource::new(instance, subscription, peers)))
    }
}

impl eframe::App for Dashboard {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let time = ctx.input(|i| i.time);
        let (state, link) = self.source.latest(time * 1000.0);
        let dt = ((time - self.anim.last_time) as f32).clamp(0.0, 0.1);
        self.anim.last_time = time;
        self.anim.update(&state, dt);
        self.render(ctx, &state, link);
        ctx.request_repaint();
    }
}

/// Smoothed, frame-to-frame animation state laid over the discrete snapshots.
#[derive(Default)]
struct Anim {
    last_time: f64,
    nodes: HashMap<OpClass, f32>,
    edges: HashMap<(OpClass, OpClass), Flow>,
    peers: HashMap<String, PeerAnim>,
}

#[derive(Default, Clone, Copy)]
struct Flow {
    rate: f32,
    phase: f32,
}

#[derive(Default, Clone, Copy)]
struct PeerAnim {
    send: f32,
    recv: f32,
    flow_phase: f32,
}

const SMOOTH_TAU: f32 = 0.3;
const FLOW_DOTS: usize = 5;

impl Anim {
    fn update(&mut self, state: &DashboardState, dt: f32) {
        let mut node_targets: HashMap<OpClass, f32> =
            state.flow.iter().map(|f| (f.class, f.ops_per_sec)).collect();
        for (class, rate) in self.nodes.iter_mut() {
            let target = node_targets.remove(class).unwrap_or(0.0);
            *rate = approach(*rate, target, dt);
        }
        for (class, target) in node_targets {
            self.nodes.insert(class, approach(0.0, target, dt));
        }
        self.nodes.retain(|_, rate| *rate > 0.02);

        let mut edge_targets: HashMap<(OpClass, OpClass), f32> =
            state.edges.iter().map(|e| ((e.from, e.to), e.rate)).collect();
        for (key, flow) in self.edges.iter_mut() {
            let target = edge_targets.remove(key).unwrap_or(0.0);
            flow.rate = approach(flow.rate, target, dt);
            flow.phase = (flow.phase + dt * edge_speed(flow.rate)).fract();
        }
        for (key, target) in edge_targets {
            self.edges.insert(
                key,
                Flow {
                    rate: approach(0.0, target, dt),
                    phase: 0.0,
                },
            );
        }
        self.edges.retain(|_, flow| flow.rate > 0.02);

        let present: std::collections::HashSet<&str> =
            state.peers.iter().map(|p| p.id.as_str()).collect();
        for peer in &state.peers {
            let anim = self.peers.entry(peer.id.clone()).or_default();
            anim.send = approach(anim.send, peer.send_bps, dt);
            anim.recv = approach(anim.recv, peer.recv_bps, dt);
        }
        for (id, anim) in self.peers.iter_mut() {
            if !present.contains(id.as_str()) {
                anim.send = approach(anim.send, 0.0, dt);
                anim.recv = approach(anim.recv, 0.0, dt);
            }
            anim.flow_phase = (anim.flow_phase + dt * peer_speed(anim.send + anim.recv)).fract();
        }
        self.peers
            .retain(|id, anim| present.contains(id.as_str()) || anim.send + anim.recv > 1.0);
    }
}

impl Dashboard {
    fn render(&mut self, ctx: &egui::Context, state: &DashboardState, link: LinkStatus) {
        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let (color, label) = link_badge(link);
                ui.painter()
                    .circle_filled(ui.cursor().min + egui::vec2(5.0, 10.0), 4.0, color);
                ui.add_space(14.0);
                ui.heading(short_id(&state.instance));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!("{label}  ·  up {}", fmt_duration(state.uptime_secs)));
                });
            });
            ui.add_space(4.0);
        });

        egui::TopBottomPanel::top("stats").show(ctx, |ui| {
            ui.add_space(6.0);
            ui.horizontal_wrapped(|ui| {
                stat(ui, "sessions", &state.sessions.to_string());
                stat(ui, "ops/sec", &round(state.ops_per_sec));
                stat(ui, "reads/sec", &round(state.reads_per_sec));
                stat(ui, "transfers/sec", &round(state.transfers_per_sec));
                stat(ui, "live tensors", &compact(state.live_tensors as u64));
                stat(ui, "peers", &state.peers.len().to_string());
            });
            ui.add_space(6.0);
        });

        egui::SidePanel::right("side")
            .resizable(true)
            .default_width(252.0)
            .show(ctx, |ui| {
                ui.add_space(6.0);
                ui.label(egui::RichText::new("peers").strong());
                ui.add_space(4.0);
                self.draw_peers(ui, &state.peers);
                ui.add_space(10.0);
                ui.separator();
                ui.label(egui::RichText::new("recent ops").strong());
                ui.add_space(4.0);
                for line in &state.recent {
                    ui.label(egui::RichText::new(line).monospace().size(11.0));
                }
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label(egui::RichText::new("op-class flow").strong());
            let (response, painter) =
                ui.allocate_painter(ui.available_size(), egui::Sense::hover());
            let area = response.rect;
            painter.rect_filled(area, 4.0, egui::Color32::from_black_alpha(24));

            let positions = class_positions(area);
            let hovered = response.hover_pos().and_then(|p| {
                self.anim
                    .nodes
                    .keys()
                    .copied()
                    .filter_map(|class| positions.get(&class).map(|pos| (class, pos.distance(p))))
                    .filter(|(_, distance)| *distance < 18.0)
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|(class, _)| class)
            });
            self.draw_flow(&painter, &positions, hovered);

            if let Some(class) = hovered {
                response.on_hover_ui_at_pointer(|ui| flow_tooltip(ui, &self.anim, class));
            } else if link != LinkStatus::Connected {
                painter.text(
                    area.center(),
                    egui::Align2::CENTER_CENTER,
                    match link {
                        LinkStatus::Stale => "waiting for the server…",
                        _ => "disconnected — retrying…",
                    },
                    egui::FontId::proportional(15.0),
                    egui::Color32::from_rgb(0xE2, 0x4B, 0x4A),
                );
            }
        });
    }

    fn draw_flow(
        &self,
        painter: &egui::Painter,
        positions: &HashMap<OpClass, egui::Pos2>,
        hovered: Option<OpClass>,
    ) {
        for (&(from, to), flow) in &self.anim.edges {
            let (Some(&a), Some(&b)) = (positions.get(&from), positions.get(&to)) else {
                continue;
            };
            let dim = hovered.is_some_and(|h| h != from && h != to);
            draw_flow_dots(painter, a, b, flow.rate, flow.phase, class_color(from), dim);
        }

        for (&class, &rate) in &self.anim.nodes {
            let Some(&p) = positions.get(&class) else {
                continue;
            };
            let radius = (4.0 + rate.sqrt() * 2.0).clamp(4.0, 22.0);
            let color = class_color(class);
            painter.circle_filled(p, radius, color);
            if hovered == Some(class) {
                painter.circle_stroke(p, radius + 3.0, egui::Stroke::new(1.5, color));
            }
            painter.text(
                p + egui::vec2(0.0, radius + 8.0),
                egui::Align2::CENTER_CENTER,
                class.label(),
                egui::FontId::proportional(11.0),
                egui::Color32::GRAY,
            );
        }
    }

    fn draw_peers(&self, ui: &mut egui::Ui, peers: &[PeerStat]) {
        let (response, painter) =
            ui.allocate_painter(egui::vec2(ui.available_width(), 200.0), egui::Sense::hover());
        let area = response.rect;
        let center = area.center();
        painter.circle_filled(center, 9.0, egui::Color32::from_rgb(0x37, 0x8A, 0xDD));
        painter.text(
            center + egui::vec2(0.0, 16.0),
            egui::Align2::CENTER_CENTER,
            "self",
            egui::FontId::proportional(10.0),
            egui::Color32::GRAY,
        );

        if peers.is_empty() {
            painter.text(
                egui::pos2(center.x, area.top() + 16.0),
                egui::Align2::CENTER_CENTER,
                "no peers connected",
                egui::FontId::proportional(11.0),
                egui::Color32::DARK_GRAY,
            );
            return;
        }

        let n = peers.len() as f32;
        let radius = (area.height().min(area.width()) * 0.5 - 34.0).max(28.0);
        let mut hovered = None;
        let pointer = response.hover_pos();

        for (i, peer) in peers.iter().enumerate() {
            let angle = std::f32::consts::TAU * (i as f32) / n - std::f32::consts::FRAC_PI_2;
            let p = center + egui::vec2(angle.cos(), angle.sin()) * radius;
            let color = if peer.direct {
                egui::Color32::from_rgb(0x1D, 0x9E, 0x75)
            } else {
                egui::Color32::from_rgb(0xBA, 0x75, 0x17)
            };
            let anim = self.peers_anim(&peer.id);

            painter.line_segment([center, p], egui::Stroke::new(1.0, faint(color)));
            // Throughput: dots leave on send, arrive on recv.
            draw_directional(&painter, center, p, anim.send, anim.flow_phase, color);
            draw_directional(&painter, p, center, anim.recv, anim.flow_phase, color);

            painter.circle_filled(p, 6.0, color);
            painter.text(
                p + egui::vec2(0.0, 15.0),
                egui::Align2::CENTER_CENTER,
                peer_label(peer),
                egui::FontId::proportional(10.0),
                egui::Color32::GRAY,
            );

            if pointer.is_some_and(|ptr| ptr.distance(p) < 14.0) {
                hovered = Some(peer);
            }
        }

        if let Some(peer) = hovered {
            response.on_hover_ui_at_pointer(|ui| peer_tooltip(ui, peer));
        }
    }

    fn peers_anim(&self, id: &str) -> PeerAnim {
        self.anim.peers.get(id).copied().unwrap_or_default()
    }
}

fn draw_flow_dots(
    painter: &egui::Painter,
    a: egui::Pos2,
    b: egui::Pos2,
    rate: f32,
    phase: f32,
    color: egui::Color32,
    dim: bool,
) {
    let scale = if dim { 0.2 } else { 1.0 };
    let line_alpha = (24.0 * scale) as u8;
    painter.line_segment(
        [a, b],
        egui::Stroke::new(1.0, alpha(color, line_alpha.max(6))),
    );
    let dot_alpha = ((90.0 + rate * 30.0).clamp(90.0, 235.0) * scale) as u8;
    let size = (1.4 + rate.sqrt() * 0.6).clamp(1.4, 3.6);
    for i in 0..FLOW_DOTS {
        let f = (phase + i as f32 / FLOW_DOTS as f32).fract();
        painter.circle_filled(a + (b - a) * f, size, alpha(color, dot_alpha));
    }
}

fn draw_directional(
    painter: &egui::Painter,
    from: egui::Pos2,
    to: egui::Pos2,
    bps: f32,
    phase: f32,
    color: egui::Color32,
) {
    if bps < 1.0 {
        return;
    }
    let intensity = (bps / 65_536.0).clamp(0.0, 8.0);
    let dot_alpha = (120.0 + intensity * 16.0).clamp(120.0, 235.0) as u8;
    let dots = 3;
    for i in 0..dots {
        let f = (phase + i as f32 / dots as f32).fract();
        painter.circle_filled(from + (to - from) * f, 2.2, alpha(color, dot_alpha));
    }
}

fn class_positions(area: egui::Rect) -> HashMap<OpClass, egui::Pos2> {
    let center = area.center();
    let radius = (area.height().min(area.width()) * 0.5 - 40.0).max(40.0);
    let n = OpClass::ALL.len() as f32;
    OpClass::ALL
        .iter()
        .enumerate()
        .map(|(i, class)| {
            let angle = std::f32::consts::TAU * (i as f32) / n - std::f32::consts::FRAC_PI_2;
            (*class, center + egui::vec2(angle.cos(), angle.sin()) * radius)
        })
        .collect()
}

fn flow_tooltip(ui: &mut egui::Ui, anim: &Anim, class: OpClass) {
    let rate = anim.nodes.get(&class).copied().unwrap_or(0.0);
    ui.label(egui::RichText::new(class.label()).strong());
    ui.label(format!("{} ops/sec", round(rate)));
    let mut incoming: Vec<_> = anim
        .edges
        .iter()
        .filter(|((_, to), _)| *to == class)
        .map(|((from, _), flow)| (*from, flow.rate))
        .collect();
    incoming.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (from, rate) in incoming.into_iter().take(4) {
        ui.label(format!("← {} {}", from.label(), round(rate)));
    }
}

fn peer_tooltip(ui: &mut egui::Ui, peer: &PeerStat) {
    ui.label(egui::RichText::new(&peer.id).monospace().size(11.0));
    ui.label(if peer.direct { "direct" } else { "relayed" });
    if let Some(rtt) = peer.rtt_ms {
        ui.label(format!("rtt {rtt:.0} ms"));
    }
    ui.label(format!(
        "↑ {}   ↓ {}",
        rate_or_idle(peer.send_bps),
        rate_or_idle(peer.recv_bps),
    ));
}

fn rate_or_idle(bps: f32) -> String {
    let rate = bytes_rate(bps);
    if rate.is_empty() {
        "idle".to_string()
    } else {
        rate
    }
}

fn peer_label(peer: &PeerStat) -> String {
    match peer.rtt_ms {
        Some(rtt) => format!("{} · {:.0}ms", short_id(&peer.id), rtt),
        None => short_id(&peer.id),
    }
}

fn approach(current: f32, target: f32, dt: f32) -> f32 {
    let k = 1.0 - (-dt / SMOOTH_TAU).exp();
    current + (target - current) * k
}

fn edge_speed(rate: f32) -> f32 {
    (0.12 + rate * 0.03).clamp(0.05, 1.2)
}

fn peer_speed(bps: f32) -> f32 {
    let intensity = (bps / 65_536.0).clamp(0.0, 8.0);
    (0.1 + intensity * 0.18).min(1.6)
}

fn alpha(color: egui::Color32, a: u8) -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), a)
}

fn link_badge(link: LinkStatus) -> (egui::Color32, &'static str) {
    match link {
        LinkStatus::Connected => (egui::Color32::from_rgb(0x1D, 0x9E, 0x75), "connected"),
        LinkStatus::Stale => (egui::Color32::from_rgb(0xBA, 0x75, 0x17), "stale"),
        LinkStatus::Lost => (egui::Color32::from_rgb(0xE2, 0x4B, 0x4A), "lost"),
    }
}

fn stat(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.vertical(|ui| {
        ui.label(
            egui::RichText::new(label)
                .size(11.0)
                .color(egui::Color32::GRAY),
        );
        ui.label(egui::RichText::new(value).size(20.0).strong());
    });
    ui.add_space(18.0);
}

fn class_color(class: OpClass) -> egui::Color32 {
    let (r, g, b) = match class {
        OpClass::Matmul => (0x7F, 0x77, 0xDD),
        OpClass::Conv => (0x1D, 0x9E, 0x75),
        OpClass::Linear => (0x37, 0x8A, 0xDD),
        OpClass::Activation => (0xD8, 0x5A, 0x30),
        OpClass::Reduction => (0xBA, 0x75, 0x17),
        OpClass::Elementwise => (0x63, 0x99, 0x22),
        OpClass::Compare => (0xD4, 0x53, 0x7E),
        OpClass::Cast => (0x5D, 0xCA, 0xA5),
        OpClass::Index => (0x85, 0xB7, 0xEB),
        OpClass::Reshape => (0xB4, 0xB2, 0xA9),
        OpClass::Init => (0x9F, 0xE1, 0xCB),
        OpClass::Random => (0xAF, 0xA9, 0xEC),
        OpClass::Distributed => (0xED, 0x93, 0xB1),
        OpClass::Custom => (0xF0, 0x99, 0x7B),
        OpClass::Drop | OpClass::Other => (0x88, 0x87, 0x80),
    };
    egui::Color32::from_rgb(r, g, b)
}

fn faint(color: egui::Color32) -> egui::Color32 {
    alpha(color, 40)
}

fn short_id(id: &str) -> String {
    if id.len() > 16 {
        format!("{}…", &id[..16])
    } else {
        id.to_string()
    }
}

fn round(value: f32) -> String {
    if value >= 100.0 {
        format!("{value:.0}")
    } else {
        format!("{value:.1}")
    }
}

fn compact(value: u64) -> String {
    if value >= 1_000_000 {
        format!("{:.1}M", value as f64 / 1e6)
    } else if value >= 1_000 {
        format!("{:.1}k", value as f64 / 1e3)
    } else {
        value.to_string()
    }
}

fn bytes_rate(bps: f32) -> String {
    if bps >= 1e6 {
        format!("{:.1}MB/s", bps / 1e6)
    } else if bps >= 1e3 {
        format!("{:.0}kB/s", bps / 1e3)
    } else if bps > 1.0 {
        format!("{bps:.0}B/s")
    } else {
        String::new()
    }
}

fn fmt_duration(secs: u64) -> String {
    format!("{:02}:{:02}:{:02}", secs / 3600, (secs % 3600) / 60, secs % 60)
}
