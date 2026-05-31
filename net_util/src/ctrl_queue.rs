// Copyright (c) 2021 Intel Corporation. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

use std::sync::Arc;
use std::sync::atomic::{AtomicU16, Ordering};

use log::{debug, error, info, warn};
use thiserror::Error;
use virtio_bindings::virtio_net::{
    VIRTIO_NET_CTRL_ANNOUNCE, VIRTIO_NET_CTRL_ANNOUNCE_ACK, VIRTIO_NET_CTRL_GUEST_OFFLOADS,
    VIRTIO_NET_CTRL_GUEST_OFFLOADS_SET, VIRTIO_NET_CTRL_MQ, VIRTIO_NET_CTRL_MQ_VQ_PAIRS_MAX,
    VIRTIO_NET_CTRL_MQ_VQ_PAIRS_MIN, VIRTIO_NET_CTRL_MQ_VQ_PAIRS_SET, VIRTIO_NET_CTRL_RX,
    VIRTIO_NET_CTRL_RX_ALLMULTI, VIRTIO_NET_CTRL_RX_ALLUNI, VIRTIO_NET_CTRL_RX_NOBCAST,
    VIRTIO_NET_CTRL_RX_NOMULTI, VIRTIO_NET_CTRL_RX_NOUNI, VIRTIO_NET_CTRL_RX_PROMISC,
    VIRTIO_NET_CTRL_VLAN, VIRTIO_NET_CTRL_VLAN_ADD, VIRTIO_NET_CTRL_VLAN_DEL, VIRTIO_NET_ERR,
    VIRTIO_NET_OK,
};
use virtio_queue::{Queue, QueueT};
use vm_memory::{ByteValued, Bytes, GuestMemoryError};
use vm_virtio::{AccessPlatform, Translatable};

use super::virtio_features_to_tap_offload;
use crate::{GuestMemoryMmap, Tap, TapError};

#[derive(Error, Debug)]
pub enum Error {
    /// Read queue failed.
    #[error("Read queue failed")]
    GuestMemory(#[source] GuestMemoryError),
    /// No control header descriptor
    #[error("No control header descriptor")]
    NoControlHeaderDescriptor,
    /// Missing the data descriptor in the chain.
    #[error("Missing the data descriptor in the chain")]
    NoDataDescriptor,
    /// No status descriptor
    #[error("No status descriptor")]
    NoStatusDescriptor,
    /// Failed adding used index
    #[error("Failed adding used index")]
    QueueAddUsed(#[source] virtio_queue::Error),
    /// Failed creating an iterator over the queue
    #[error("Failed creating an iterator over the queue")]
    QueueIterator(#[source] virtio_queue::Error),
    /// Failed enabling notification for the queue
    #[error("Failed enabling notification for the queue")]
    QueueEnableNotification(#[source] virtio_queue::Error),
}

type Result<T> = std::result::Result<T, Error>;

#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ControlHeader {
    pub class: u8,
    pub cmd: u8,
}

// SAFETY: ControlHeader only contains a series of integers
unsafe impl ByteValued for ControlHeader {}

fn is_tolerated_ctrl_command(ctrl_hdr: ControlHeader) -> bool {
    match u32::from(ctrl_hdr.class) {
        VIRTIO_NET_CTRL_RX => matches!(
            u32::from(ctrl_hdr.cmd),
            VIRTIO_NET_CTRL_RX_PROMISC
                | VIRTIO_NET_CTRL_RX_ALLMULTI
                | VIRTIO_NET_CTRL_RX_ALLUNI
                | VIRTIO_NET_CTRL_RX_NOMULTI
                | VIRTIO_NET_CTRL_RX_NOUNI
                | VIRTIO_NET_CTRL_RX_NOBCAST
        ),
        VIRTIO_NET_CTRL_VLAN => matches!(
            u32::from(ctrl_hdr.cmd),
            VIRTIO_NET_CTRL_VLAN_ADD | VIRTIO_NET_CTRL_VLAN_DEL
        ),
        VIRTIO_NET_CTRL_ANNOUNCE => u32::from(ctrl_hdr.cmd) == VIRTIO_NET_CTRL_ANNOUNCE_ACK,
        _ => false,
    }
}

/// Error returned by an [`MqBackend`].
///
/// Variants exist so net_util-internal backends (taps) can surface their
/// native error type; backends defined outside this crate (e.g. vhost-user
/// in `virtio-devices`) erase their error into a string so net_util stays
/// free of cross-crate dependencies.
#[derive(Error, Debug)]
pub enum MqBackendError {
    #[error("tap error: {0}")]
    Tap(#[from] TapError),
    #[error("{0}")]
    Other(String),
}

/// Strategy for honoring `VIRTIO_NET_CTRL_MQ_VQ_PAIRS_SET`.
///
/// Implementations translate "active queue-pair count" to whatever the
/// underlying backend needs: `TUNSETQUEUE` for kernel taps,
/// `VHOST_USER_SET_VRING_ENABLE` for vhost-user, etc. The caller (the
/// `CtrlQueue` MQ handler, or device activation) bounds `pairs` against
/// the device's advertised max before invoking.
pub trait MqBackend: Send {
    fn set_active_pairs(&mut self, pairs: u16) -> std::result::Result<(), MqBackendError>;
}

/// `MqBackend` for the local kernel-tap path. Holds a shared tracker so
/// state survives `CtrlQueue` reconstruction across reset/re-activate
/// cycles and never re-issues `TUNSETQUEUE` on a queue already in the
/// requested state (which the kernel would reject with `EINVAL`).
pub struct TapMqBackend {
    taps: Vec<Tap>,
    tracker: Arc<AtomicU16>,
}

impl TapMqBackend {
    pub fn new(taps: Vec<Tap>, tracker: Arc<AtomicU16>) -> Self {
        Self { taps, tracker }
    }
}

impl MqBackend for TapMqBackend {
    fn set_active_pairs(&mut self, pairs: u16) -> std::result::Result<(), MqBackendError> {
        Ok(align_kernel_queue_pairs(&self.taps, &self.tracker, pairs)?)
    }
}

pub struct CtrlQueue {
    pub taps: Vec<Tap>,
    /// Strategy for applying `VIRTIO_NET_CTRL_MQ_VQ_PAIRS_SET`. `None`
    /// rejects the command -- only happens for net configurations that
    /// have no working MQ backend at all, which should be rare.
    mq_backend: Option<Box<dyn MqBackend>>,
    /// Maximum queue pairs the device exposes. Bounds the requested count
    /// in addition to the spec's `VIRTIO_NET_CTRL_MQ_VQ_PAIRS_MIN/MAX`.
    max_queue_pairs: u16,
    /// Whether `VIRTIO_NET_F_MQ` was acknowledged by the driver. Captured
    /// at activation time, after feature negotiation has settled.
    mq_negotiated: bool,
}

/// Returns the ordered list of `(queue_index, attach)` ops needed to drive
/// `active` to `desired`, clamped to `max`. Detaches walk from the top down
/// so a partial failure leaves a contiguous prefix of attached queues.
fn plan_queue_pair_delta(active: u16, desired: u16, max: u16) -> Vec<(usize, bool)> {
    if max <= 1 {
        return Vec::new();
    }
    let desired = desired.min(max);
    if desired == active {
        return Vec::new();
    }
    if desired > active {
        (active..desired).map(|i| (i as usize, true)).collect()
    } else {
        (desired..active)
            .rev()
            .map(|i| (i as usize, false))
            .collect()
    }
}

/// Drive the kernel-side multi-queue attachment for `taps` to `desired`
/// pair count, updating `tracker` incrementally so a partial failure
/// leaves it in sync with kernel state.
fn align_kernel_queue_pairs(
    taps: &[Tap],
    tracker: &AtomicU16,
    desired: u16,
) -> std::result::Result<(), TapError> {
    let max = taps.len() as u16;
    let active = tracker.load(Ordering::Acquire);
    for (idx, attach) in plan_queue_pair_delta(active, desired, max) {
        taps[idx].set_queue(attach)?;
        let new_active = if attach { idx as u16 + 1 } else { idx as u16 };
        tracker.store(new_active, Ordering::Release);
    }
    Ok(())
}

impl CtrlQueue {
    pub fn new(
        taps: Vec<Tap>,
        mq_backend: Option<Box<dyn MqBackend>>,
        max_queue_pairs: u16,
        mq_negotiated: bool,
    ) -> Self {
        CtrlQueue {
            taps,
            mq_backend,
            max_queue_pairs,
            mq_negotiated,
        }
    }

    /// Drive the backend to `desired` queue-pair count, or report no
    /// backend was wired (which becomes `VIRTIO_NET_ERR` to the guest).
    fn apply_active_queue_pairs(
        &mut self,
        desired: u16,
    ) -> std::result::Result<(), MqBackendError> {
        let Some(backend) = self.mq_backend.as_mut() else {
            return Err(MqBackendError::Other(
                "no MQ backend configured for this device".into(),
            ));
        };
        backend.set_active_pairs(desired)
    }

    pub fn process(
        &mut self,
        mem: &GuestMemoryMmap,
        queue: &mut Queue,
        access_platform: Option<&dyn AccessPlatform>,
    ) -> Result<()> {
        while let Some(mut desc_chain) = queue.pop_descriptor_chain(mem) {
            let ctrl_desc = desc_chain.next().ok_or(Error::NoControlHeaderDescriptor)?;

            let ctrl_hdr: ControlHeader = desc_chain
                .memory()
                .read_obj(
                    ctrl_desc
                        .addr()
                        .translate_gva(access_platform, ctrl_desc.len() as usize)
                        .map_err(|e| Error::GuestMemory(GuestMemoryError::IOError(e)))?,
                )
                .map_err(Error::GuestMemory)?;
            let data_desc = desc_chain.next().ok_or(Error::NoDataDescriptor)?;

            let data_desc_addr = data_desc
                .addr()
                .translate_gva(access_platform, data_desc.len() as usize)
                .map_err(|e| Error::GuestMemory(GuestMemoryError::IOError(e)))?;

            let status_desc = desc_chain.next().ok_or(Error::NoStatusDescriptor)?;

            let ok = match u32::from(ctrl_hdr.class) {
                VIRTIO_NET_CTRL_MQ => {
                    let queue_pairs = desc_chain
                        .memory()
                        .read_obj::<u16>(data_desc_addr)
                        .map_err(Error::GuestMemory)?;
                    if u32::from(ctrl_hdr.cmd) != VIRTIO_NET_CTRL_MQ_VQ_PAIRS_SET {
                        warn!("Unsupported command: {}", ctrl_hdr.cmd);
                        false
                    } else if !self.mq_negotiated {
                        warn!("MQ command received without VIRTIO_NET_F_MQ negotiated");
                        false
                    } else if (queue_pairs < VIRTIO_NET_CTRL_MQ_VQ_PAIRS_MIN as u16)
                        || (queue_pairs > VIRTIO_NET_CTRL_MQ_VQ_PAIRS_MAX as u16)
                        || (queue_pairs > self.max_queue_pairs)
                    {
                        warn!(
                            "Number of MQ pairs out of range: {queue_pairs} \
                             (device max {})",
                            self.max_queue_pairs
                        );
                        false
                    } else {
                        match self.apply_active_queue_pairs(queue_pairs) {
                            Ok(()) => {
                                info!("Number of MQ pairs set: {queue_pairs}");
                                true
                            }
                            Err(e) => {
                                error!("Failed to apply MQ pairs={queue_pairs}: {e}");
                                false
                            }
                        }
                    }
                }
                VIRTIO_NET_CTRL_GUEST_OFFLOADS => {
                    let features = desc_chain
                        .memory()
                        .read_obj::<u64>(data_desc_addr)
                        .map_err(Error::GuestMemory)?;
                    if u32::from(ctrl_hdr.cmd) == VIRTIO_NET_CTRL_GUEST_OFFLOADS_SET {
                        let mut ok = true;
                        for tap in self.taps.iter_mut() {
                            info!("Reprogramming tap offload with features: {features}");
                            tap.set_offload(virtio_features_to_tap_offload(features))
                                .map_err(|e| {
                                    error!("Error programming tap offload: {e:?}");
                                    ok = false;
                                })
                                .ok();
                        }
                        ok
                    } else {
                        warn!("Unsupported command: {}", ctrl_hdr.cmd);
                        false
                    }
                }
                _ if is_tolerated_ctrl_command(ctrl_hdr) => {
                    debug!("Ignoring unsupported but tolerated control command {ctrl_hdr:?}");
                    true
                }
                _ => {
                    warn!("Unsupported command {ctrl_hdr:?}");
                    false
                }
            };

            desc_chain
                .memory()
                .write_obj(
                    if ok { VIRTIO_NET_OK } else { VIRTIO_NET_ERR } as u8,
                    status_desc
                        .addr()
                        .translate_gva(access_platform, status_desc.len() as usize)
                        .map_err(|e| Error::GuestMemory(GuestMemoryError::IOError(e)))?,
                )
                .map_err(Error::GuestMemory)?;
            // Per the virtio spec the used length is bytes the device wrote
            // to device-writable descriptors; here just the 1-byte ack.
            queue
                .add_used(desc_chain.memory(), desc_chain.head_index(), 1)
                .map_err(Error::QueueAddUsed)?;

            if !queue
                .enable_notification(mem)
                .map_err(Error::QueueEnableNotification)?
            {
                break;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn delta_single_queue_device_is_noop() {
        assert!(plan_queue_pair_delta(1, 1, 1).is_empty());
        assert!(plan_queue_pair_delta(0, 4, 1).is_empty());
    }

    #[test]
    fn delta_same_count_is_noop() {
        assert!(plan_queue_pair_delta(3, 3, 8).is_empty());
    }

    #[test]
    fn delta_grow_attaches_upper_indices() {
        assert_eq!(
            plan_queue_pair_delta(1, 4, 8),
            vec![(1, true), (2, true), (3, true)],
        );
    }

    #[test]
    fn delta_shrink_detaches_from_top_down() {
        assert_eq!(
            plan_queue_pair_delta(4, 1, 8),
            vec![(3, false), (2, false), (1, false)],
        );
    }

    #[test]
    fn delta_clamps_desired_to_device_max() {
        assert_eq!(plan_queue_pair_delta(2, 99, 4), vec![(2, true), (3, true)],);
    }

    use std::sync::Mutex;

    #[derive(Default)]
    struct MockBackendInner {
        last_pairs: Option<u16>,
        fail: bool,
    }

    #[derive(Default, Clone)]
    struct MockBackend(Arc<Mutex<MockBackendInner>>);

    impl MqBackend for MockBackend {
        fn set_active_pairs(&mut self, pairs: u16) -> std::result::Result<(), MqBackendError> {
            let mut inner = self.0.lock().unwrap();
            inner.last_pairs = Some(pairs);
            if inner.fail {
                Err(MqBackendError::Other("mock fail".into()))
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn ctrl_queue_routes_apply_to_backend() {
        let mock = MockBackend::default();
        let inner = mock.0.clone();
        let mut cq = CtrlQueue::new(Vec::new(), Some(Box::new(mock)), 4, true);
        cq.apply_active_queue_pairs(3).unwrap();
        assert_eq!(inner.lock().unwrap().last_pairs, Some(3));
    }

    #[test]
    fn ctrl_queue_propagates_backend_failure() {
        let mock = MockBackend::default();
        mock.0.lock().unwrap().fail = true;
        let mut cq = CtrlQueue::new(Vec::new(), Some(Box::new(mock)), 4, true);
        let err = cq.apply_active_queue_pairs(3).unwrap_err();
        assert!(matches!(err, MqBackendError::Other(_)));
    }

    #[test]
    fn ctrl_queue_with_no_backend_rejects_apply() {
        let mut cq = CtrlQueue::new(Vec::new(), None, 4, true);
        let err = cq.apply_active_queue_pairs(3).unwrap_err();
        assert!(matches!(err, MqBackendError::Other(_)));
    }
}
