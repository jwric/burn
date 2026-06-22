//! Aggregation of raw telemetry into a current, renderable [`DashboardState`].
//!
//! This is application policy, not part of the `burn-remote` observability surface: it decides the
//! windowing, the op-class flow model, and the recent-op formatting. It carries no UI dependency,
//! so the native peer can run the [`Aggregator`] server-side and serialize [`DashboardState`]
//! without pulling in egui.

use std::collections::{HashMap, HashSet, VecDeque};

use burn_remote::telemetry::{OpClass, PeerLink, PeerSnapshot, TelemetryEvent};
use serde::{Deserialize, Serialize};

/// A node in the op-class flow graph: an op category and its current rate.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ClassFlow {
    pub class: OpClass,
    pub ops_per_sec: f32,
}

/// A directed edge in the op-class flow graph: tensors handed from one class to another.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ClassEdge {
    pub from: OpClass,
    pub to: OpClass,
    pub rate: f32,
}

/// A peer connection with current activity, ready to display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerStat {
    pub id: String,
    pub direct: bool,
    pub rtt_ms: Option<f32>,
    pub send_bps: f32,
    pub recv_bps: f32,
}

/// A renderable, current-focused view of a server's activity. Built by [`Aggregator`] and pushed
/// to a monitoring view; new viewers receive the latest one immediately, so a refresh resumes the
/// live picture instead of resetting.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DashboardState {
    pub instance: String,
    pub uptime_secs: u64,
    pub sessions: u32,
    pub ops_per_sec: f32,
    pub reads_per_sec: f32,
    pub transfers_per_sec: f32,
    pub live_tensors: u32,
    pub flow: Vec<ClassFlow>,
    pub edges: Vec<ClassEdge>,
    pub recent: Vec<String>,
    pub peers: Vec<PeerStat>,
}

const PRODUCER_CAP: usize = 50_000;
const RECENT_CAP: usize = 14;
const RATE_SMOOTHING: f32 = 0.4;
const MIN_TICK_SECS: f64 = 0.2;

/// Folds [`TelemetryEvent`]s and [`PeerSnapshot`]s into a [`DashboardState`].
///
/// Rates are smoothed per-window so the view reflects current activity rather than totals since
/// start. Run it on the server (or in-process in the browser peer) so the state survives a viewer
/// reconnect.
pub struct Aggregator {
    instance: String,
    start_ms: Option<f64>,
    last_tick_ms: f64,
    sessions: HashSet<u64>,
    live_tensors: HashSet<u64>,
    producer: HashMap<u64, OpClass>,
    producer_order: VecDeque<u64>,
    class_acc: HashMap<OpClass, u32>,
    edge_acc: HashMap<(OpClass, OpClass), u32>,
    read_acc: u32,
    transfer_acc: u32,
    class_rate: HashMap<OpClass, f32>,
    edge_rate: HashMap<(OpClass, OpClass), f32>,
    ops_rate: f32,
    reads_rate: f32,
    transfers_rate: f32,
    recent: VecDeque<(OpClass, u32, u32)>,
    peers: Vec<PeerStat>,
    prev_peer_bytes: HashMap<String, (u64, u64, f64)>,
}

impl Aggregator {
    pub fn new(instance: String) -> Self {
        Self {
            instance,
            start_ms: None,
            last_tick_ms: 0.0,
            sessions: HashSet::new(),
            live_tensors: HashSet::new(),
            producer: HashMap::new(),
            producer_order: VecDeque::new(),
            class_acc: HashMap::new(),
            edge_acc: HashMap::new(),
            read_acc: 0,
            transfer_acc: 0,
            class_rate: HashMap::new(),
            edge_rate: HashMap::new(),
            ops_rate: 0.0,
            reads_rate: 0.0,
            transfers_rate: 0.0,
            recent: VecDeque::new(),
            peers: Vec::new(),
            prev_peer_bytes: HashMap::new(),
        }
    }

    pub fn apply(&mut self, event: &TelemetryEvent) {
        match event {
            TelemetryEvent::SessionOpened { session, .. } => {
                self.sessions.insert(session.value());
            }
            TelemetryEvent::SessionClosed { session } => {
                self.sessions.remove(&session.value());
            }
            TelemetryEvent::Op {
                kind,
                inputs,
                outputs,
                ..
            } => {
                *self.class_acc.entry(*kind).or_default() += 1;
                for input in inputs {
                    if let Some(from) = self.producer.get(&input.value()).copied() {
                        *self.edge_acc.entry((from, *kind)).or_default() += 1;
                    }
                }
                for output in outputs {
                    self.set_producer(output.id.value(), *kind);
                    self.live_tensors.insert(output.id.value());
                }
                self.push_recent(*kind, inputs.len() as u32, outputs.len() as u32);
            }
            TelemetryEvent::TensorDropped { tensor, .. } => {
                self.live_tensors.remove(&tensor.value());
                self.producer.remove(&tensor.value());
            }
            TelemetryEvent::Transfer { .. } => self.transfer_acc += 1,
            TelemetryEvent::Read { .. } => self.read_acc += 1,
            TelemetryEvent::Sync { .. } => {}
        }
    }

    pub fn set_peers(&mut self, snapshot: &PeerSnapshot, now_ms: f64) {
        self.peers = snapshot
            .links
            .iter()
            .map(|link| {
                let (send_bps, recv_bps) = self.peer_rate(link, now_ms);
                PeerStat {
                    id: link.peer.clone(),
                    direct: link.direct,
                    rtt_ms: link.rtt_ms,
                    send_bps,
                    recv_bps,
                }
            })
            .collect();
        self.prev_peer_bytes
            .retain(|id, _| snapshot.links.iter().any(|link| &link.peer == id));
    }

    /// Recompute smoothed rates from the accumulators. Self-rate-limited, so it is safe to call
    /// every frame.
    pub fn tick(&mut self, now_ms: f64) {
        let start = *self.start_ms.get_or_insert(now_ms);
        if self.last_tick_ms == 0.0 {
            self.last_tick_ms = start;
        }
        let dt = (now_ms - self.last_tick_ms) / 1000.0;
        if dt < MIN_TICK_SECS {
            return;
        }
        self.last_tick_ms = now_ms;

        let dt = dt as f32;
        self.ops_rate = smooth(self.ops_rate, total(&self.class_acc) as f32 / dt);
        self.reads_rate = smooth(self.reads_rate, self.read_acc as f32 / dt);
        self.transfers_rate = smooth(self.transfers_rate, self.transfer_acc as f32 / dt);
        smooth_map(&mut self.class_rate, &self.class_acc, dt);
        smooth_map(&mut self.edge_rate, &self.edge_acc, dt);

        self.class_acc.clear();
        self.edge_acc.clear();
        self.read_acc = 0;
        self.transfer_acc = 0;
    }

    pub fn snapshot(&self, now_ms: f64) -> DashboardState {
        let uptime_secs = self
            .start_ms
            .map(|start| ((now_ms - start) / 1000.0).max(0.0) as u64)
            .unwrap_or(0);
        let flow = self
            .class_rate
            .iter()
            .filter(|(_, rate)| **rate > 0.05)
            .map(|(class, rate)| ClassFlow {
                class: *class,
                ops_per_sec: *rate,
            })
            .collect();
        let edges = self
            .edge_rate
            .iter()
            .filter(|(_, rate)| **rate > 0.05)
            .map(|((from, to), rate)| ClassEdge {
                from: *from,
                to: *to,
                rate: *rate,
            })
            .collect();
        DashboardState {
            instance: self.instance.clone(),
            uptime_secs,
            sessions: self.sessions.len() as u32,
            ops_per_sec: self.ops_rate,
            reads_per_sec: self.reads_rate,
            transfers_per_sec: self.transfers_rate,
            live_tensors: self.live_tensors.len() as u32,
            flow,
            edges,
            recent: self
                .recent
                .iter()
                .map(|(class, n_in, n_out)| format!("{} {}->{}", class.label(), n_in, n_out))
                .collect(),
            peers: self.peers.clone(),
        }
    }

    fn set_producer(&mut self, id: u64, class: OpClass) {
        if self.producer.insert(id, class).is_none() {
            self.producer_order.push_back(id);
            while self.producer_order.len() > PRODUCER_CAP {
                if let Some(old) = self.producer_order.pop_front() {
                    self.producer.remove(&old);
                }
            }
        }
    }

    fn push_recent(&mut self, class: OpClass, n_in: u32, n_out: u32) {
        self.recent.push_front((class, n_in, n_out));
        self.recent.truncate(RECENT_CAP);
    }

    fn peer_rate(&mut self, link: &PeerLink, now_ms: f64) -> (f32, f32) {
        let prev = self
            .prev_peer_bytes
            .insert(link.peer.clone(), (link.bytes_sent, link.bytes_recv, now_ms));
        match prev {
            Some((sent, recv, at)) => {
                let dt = ((now_ms - at) / 1000.0).max(0.001) as f32;
                (
                    link.bytes_sent.saturating_sub(sent) as f32 / dt,
                    link.bytes_recv.saturating_sub(recv) as f32 / dt,
                )
            }
            None => (0.0, 0.0),
        }
    }
}

fn total(map: &HashMap<OpClass, u32>) -> u32 {
    map.values().sum()
}

fn smooth(current: f32, instant: f32) -> f32 {
    current * (1.0 - RATE_SMOOTHING) + instant * RATE_SMOOTHING
}

fn smooth_map<K: Copy + Eq + std::hash::Hash>(
    rates: &mut HashMap<K, f32>,
    acc: &HashMap<K, u32>,
    dt: f32,
) {
    for rate in rates.values_mut() {
        *rate *= 1.0 - RATE_SMOOTHING;
    }
    for (key, count) in acc {
        let instant = *count as f32 / dt;
        *rates.entry(*key).or_insert(0.0) += instant * RATE_SMOOTHING;
    }
    rates.retain(|_, rate| *rate > 0.01);
}
