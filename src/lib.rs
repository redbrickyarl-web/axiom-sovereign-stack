pub mod cosmic_queue;
pub mod edge_ai;
pub mod primitives;
pub mod axiom;
pub mod crypto;
pub mod ffi;

pub use cosmic_queue::PacketFrame;
pub use edge_ai::EdgeAIPipeline;
pub use axiom::PredatorGate;
pub use crypto::{BwpMsmAccumulator, BssDecoder};
pub use ffi::{
    act_bytes, gemv, quantize_activation, AethelarchError, AethelarchMatrix, AethelarchResult,
    EncodeFailure,
};
