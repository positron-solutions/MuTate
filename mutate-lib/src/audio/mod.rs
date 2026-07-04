//! # Audio
//!
//! ⚠️ This module was initially only completed far enough to get 800 samples per second dumped into
//! a buffer.  The downstream programs didn't care, and so the shape of support was not that of a
//! real client for pipewire.  The event-driven nature of pipewire and the Rust wrapper around the
//! pipewire sys crate were discovered only mid-development.  A redesign pass will be welcome so
//! long as this notice exists.
//!
//! The [`AudioContext`] is to represent a connection to the server while [`AudioConsumer`] is an
//! open streaming connection.  See CPAL APIs for other user-facing API ideas.  We are likely more
//! interested in timing data, media names, and routing multiple streams than many CPAL clients, so
//! the first implementation, the pipewire implementation, is being done by hand.  Receiving smaller
//! audio chunks is one win realized so far.
//!
//! [`AudioContext`] sets up communication threads that receive mapped buffers from an audio server
//! such as Pipewire.  The communication thread either polls or tracks available audio sources.
//! [`AudioContext`] provides [`with_choices`] and [`with_choices_blocking`] methods for displaying
//! those choices to the user.  An [`AudioChoice`] can be used to call [`connect`], which will
//! return an [`AudioConsumer`].  An `AudioConsumer`, which is backed by a ring buffer, provides
//! synchronization data, media names, and access to the sliding window for reads.
//!
//! ## Implementations
//!
//! We are usually interested in monitoring outgoing sound from other applications.  We need to find
//! valid sinks and create streams linked to their monitor ports.  The exact terminology may depend
//! on the platform, but the basic idea is to find outbound audio and tee it into our application
//! with sufficient synchronization information to align with sounds being played as closely as
//! possible.
//!
//! Use cfg gates to support future clients, likely by abstracting pipewire out first.  **Express
//! willingness to work on your preferred client if you would like this abstraction done by someone
//! running pipewire to test the abstraction.**
//!
//! ### Pipewire
//!
//! Pipewire exposes an event driven interface where we register listeners and filter the
//! information we are interested in.  The way we learn about what audio streams are available is by
//! registering a global listener, client-side filtering for our interests, and then maintaining a
//! view by watching changes published to our global listener.
//!
//! Each set of global update events has a fixed sequence number for the set.  Each set will emit a
//! done event to indicate that all messages in the set were received.  Blocking for the first set,
//! enabling callers to receive a fully populated initial view of available streams, is the reason
//! for the `with_choices_blocking` API.
//!
//! Pipewire does have some presentation data, but until Link support is expanded, tests so far read
//! zero-values for all presentation delays on sink monitors.
//!
//! ### CPAL
//!
//! This would be a welcome addition for supporting more platforms.  **Please get in touch if you
//! need the pipewire code moved and want someone else to move it while making sure that it keeps
//! working.**

// NEXT To extend the AudioContext module for other platforms, just add cfg flags wherever
// implementations and fields are platform specific.  Take a look at CPAL but consider using
// platform bindings more directly if CPAL can't give us precise timing data or control.  We might
// want to adjust the input stream latency by talking to the audio server directly, which is not an
// API expected to be found in CPAL.
// FIXME Swap in sys crate primitives directly in places where we are throwing away the safety anyway.
// NOTE Delay times from the server can be negative, so always use signed types for time offsets,
// such as i64 etc.
// NOTE The model for receiving stream data from pipewire, which might hold up when talking to other
// audio servers, is that pipewire sends us monotonic buffer chunks without skips (via padding or
// stream parameter change, the latter of which is not yet handled).  Due to audio playback being
// naturally self-pacing, the monotonic chunks without skips behavior provides implicit relative
// timing signal without use of any explicit time values.
// NEXT Absolute presentation timing data may be obtainable, but seems to require customizing our
// pipewire link to match the presentation timing of the sink monitor.
// FIXME cfg gates on Linux need features instead.
// NEXT media name events on the `AudioConsumer`!  We need to see the info events from pipewire.
// Also looks like we need DBUS for seeing Spotify title changes.  If we have it, we can render
// title changes in the middle of playback, something Milkdrop has done right for twenty years or
// so.
pub mod timing;

use std::cell::UnsafeCell;
use std::sync::atomic;
use std::time::Instant;

#[cfg(target_os = "linux")]
use pipewire::{self as pw, main_loop::MainLoopBox, spa, stream::StreamListener};
use ringbuf::traits::{Consumer, Observer, Producer, RingBuffer};

use crate::prelude::*;

/// The kinds of audio we can listen to.  Implements `Display` for an end-user meaningful string.
/// Match directly to implement custom UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioSourceKind {
    /// Listen to an input source, such as a hardware microphone.
    HardwareInput,
    /// Listen on a monitor of all output to a sink.
    SinkMonitor,
    /// Listen to an application that is playing audio.
    ApplicationStream,
}

impl AudioSourceKind {
    /// None will
    #[cfg(target_os = "linux")]
    pub fn from_media_class(class: &str) -> Option<Self> {
        match class {
            "Audio/Source" => Some(Self::HardwareInput),
            "Audio/Sink" => Some(Self::SinkMonitor),
            "Stream/Output/Audio" => Some(Self::ApplicationStream),
            _ => None,
        }
    }
}

impl std::fmt::Display for AudioSourceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::HardwareInput => "Input",
            Self::SinkMonitor => "Output",
            Self::ApplicationStream => "Application",
        })
    }
}

/// Commands for calling into the Audio thread
enum Message {
    /// Connect to a particular identifier
    Connect {
        /// The name of the connection we are creating.  Displays our application correctly
        name: String,
        choice: AudioChoice,
        tx: AudioProducer,
    },
    Terminate,
}

/// Even-driven clients will maintain a client-side view.  Polling clients do not require
/// maintaining a view of stream choices.
// XXX cfg and feature gates
struct AudioChoices {
    ready: std::sync::Condvar,
    choices: std::sync::Mutex<Vec<AudioChoice>>,
    version: atomic::AtomicUsize,
    initialized: atomic::AtomicBool,
}

impl AudioChoices {
    fn notify(&self) {
        self.initialized.store(true, atomic::Ordering::Release);
        self.ready.notify_all();
    }

    fn new() -> Self {
        Self {
            ready: std::sync::Condvar::new(),
            choices: std::sync::Mutex::new(Vec::new()),
            version: atomic::AtomicUsize::new(0),
            initialized: atomic::AtomicBool::new(false),
        }
    }
}

/// `AudioContext` represents the connection to an audio server, which usually takes care of
/// multiplexing the applications and hardware devices.  Most workflows need to look for usable
/// audio streams before obtaining connections.  The `AudioContext` provides access to usable
/// streams and provides connections to them.
///
/// ⚠️ If the context is dropped early, outstanding `AudioConsumer`s will begin returning errors as
/// the backing resources that feed them will have been torn down.
pub struct AudioContext {
    handle: Option<std::thread::JoinHandle<()>>,
    choices: *mut AudioChoices,

    #[cfg(target_os = "linux")]
    tx: pw::channel::Sender<Message>,
}

impl AudioContext {
    /// Creates initial audio server connection.
    pub fn new() -> Result<Self, MutateError> {
        // Platform binaries may use cfg flags.  For supporting different versions of the same
        // platform prefer runtime decisions.  Use features if binary weight is a concern for
        // library users.
        Self::initialize()
    }

    #[cfg(target_os = "linux")]
    fn initialize() -> Result<Self, MutateError> {
        let choices = Box::into_raw(Box::new(AudioChoices::new()));
        let choices_addr = choices as usize;
        let (pw_sender, pw_receiver) = pipewire::channel::channel();
        let handle = std::thread::spawn(move || {
            // Safety: AudioContext::drop joins this thread before freeing choices, so &AudioChoices
            // is valid for the thread's entire lifetime.
            let choices: &AudioChoices = unsafe { &*(choices_addr as *mut AudioChoices) };
            // Due to borrowed data and lack of try blocks in stable, Rust, seems like this is an
            // okay-ish way to know of issues in the terminal without forcing callers to fail.  At
            // least that's the goal.
            let mainloop = match MainLoopBox::new(None) {
                Ok(mainloop) => mainloop,
                Err(e) => {
                    eprintln!("PipeWire initialization failed: {:?}", MutateError::from(e));
                    return;
                }
            };

            let context = match pw::context::ContextBox::new(&mainloop.loop_(), None) {
                Ok(context) => context,
                Err(e) => {
                    eprintln!("PipeWire initialization failed: {:?}", MutateError::from(e));
                    return;
                }
            };

            let core = match context.connect(None) {
                Ok(core) => core,
                Err(e) => {
                    eprintln!("PipeWire initialization failed: {:?}", MutateError::from(e));
                    return;
                }
            };

            let registry = match core.get_registry() {
                Ok(registry) => registry,
                Err(e) => {
                    eprintln!("PipeWire initialization failed: {:?}", MutateError::from(e));
                    return;
                }
            };

            // NEXT add a way to destroy a single connection.
            // FIXME error without Termination may leak the connections.
            let pw_connections = Box::into_raw(Box::new(Vec::<PipewireConnection>::new()));
            let _receiver = pw_receiver.attach(mainloop.loop_(), {
                let mainloop_ptr = mainloop.as_raw_ptr();
                let core_ptr = core.as_raw_ptr();
                move |message| match message {
                    Message::Connect { choice, tx, name } => {
                        let conn_ptr = tx.conn;
                        match create_stream(core_ptr, &choice, &name, tx) {
                            Ok((listener, stream)) => {
                                unsafe { &mut *pw_connections }.push(PipewireConnection {
                                    stream: Some(stream),
                                    listener: Some(listener),
                                });
                            }
                            Err(e) => {
                                eprintln!("stream creation failed: {}", e);
                            }
                        };
                    }
                    Message::Terminate => {
                        unsafe { drop(Box::from_raw(pw_connections)) };
                        eprintln!("Terminating mainloop");
                        unsafe { pw::sys::pw_main_loop_quit(mainloop_ptr) };
                    }
                }
            });

            let _done_listener = core
                .add_listener_local()
                .done(move |_seq, _serial| choices.notify())
                .register();

            let _monitor_listener = registry
                .add_listener_local()
                .global(move |global| {
                    // NEXT this will become a big match statement in order to track a node -> ports
                    // mapping.
                    if global.type_ != pw::types::ObjectType::Node {
                        return;
                    }

                    let Some(props) = &global.props else { return };
                    let Some(media_class) = props.get("media.class") else {
                        return;
                    };
                    let Some(kind) = AudioSourceKind::from_media_class(media_class) else {
                        return;
                    };

                    match AudioChoice::try_new(kind, *props, global.id) {
                        Ok(choice) => match choices.choices.lock() {
                            Ok(mut guard) => guard.push(choice),
                            Err(e) => eprintln!(
                                "listing audio source failed.  skipping: {:?}",
                                MutateError::from(e)
                            ),
                        },
                        Err(e) => eprintln!("Skipping {}: {:?}", media_class, e),
                    }
                })
                .register();

            let _remove_listener = registry
                .add_listener_local()
                .global_remove(move |removed_id| match choices.choices.lock() {
                    Ok(mut choices) => {
                        if let Some(found) =
                            choices.iter_mut().position(|c| c.global_id == removed_id)
                        {
                            choices.remove(found);
                        }
                    }
                    Err(e) => {
                        eprintln!("removing audio source failed: {:?}", MutateError::from(e));
                    }
                })
                .register();

            match core.sync(0) {
                Err(e) => {
                    eprintln!("PipeWire initialization failed: {:?}", MutateError::from(e));
                    return;
                }
                _ => {}
            };

            mainloop.run();
        });

        Ok(AudioContext {
            handle: Some(handle),
            choices,
            tx: pw_sender,
        })
    }

    /// Connect to a stream
    pub fn connect(
        &self,
        choice: &AudioChoice,
        name: String,
    ) -> Result<AudioConsumer, MutateError> {
        let conn = AudioConnection::new();
        let msg = Message::Connect {
            choice: choice.clone(),
            tx: AudioProducer { conn: conn.clone() },
            name,
        };
        self.tx
            .send(msg)
            .map_err(|_e| MutateError::AudioConnect("connection creation failed"))?;
        Ok(AudioConsumer { conn })
    }

    pub fn choices_version(&self) -> usize {
        // Readers are deciding to do an update if one is available.  Missing one due to relaxed
        // ordering fine-grained incoherence is totally fine.
        unsafe { &*self.choices }
            .version
            .load(atomic::Ordering::Relaxed)
    }

    /// Run a function on the most recent choices.  If you need to wait on the first updates, use
    /// [`with_choices_blocking`] instead.  Your provided function should complete quickly because
    /// it uses a lock that will block the audio thread.  If you need more time, record a copy of
    /// the choices into your calling scope.
    pub fn with_choices<F>(&self, mut f: F) -> Result<(), MutateError>
    where
        F: FnMut(&[AudioChoice]),
    {
        let choices = unsafe { &*self.choices }.choices.lock()?;
        f(&choices);
        Ok(())
    }

    /// Same as with_choices, but will wait for choices to be initially populated, which may require
    /// waiting on the audio server thread when called immediately after creating the context.  The
    /// timeout is one second and will return an error.
    pub fn with_choices_blocking<F>(&self, mut f: F) -> Result<(), MutateError>
    where
        F: FnMut(&[AudioChoice]),
    {
        let ac: &AudioChoices = unsafe { &*self.choices };
        let mut choices = ac.choices.lock()?;
        while ac.initialized.load(atomic::Ordering::Relaxed) == false {
            let (guard, result) = ac
                .ready
                .wait_timeout(choices, std::time::Duration::from_millis(1000))?;
            if result.timed_out() {
                return Err(MutateError::Timeout("AudioChoices not received"));
            }
            choices = guard;
        }
        f(&choices);
        Ok(())
    }
}

impl Drop for AudioContext {
    fn drop(&mut self) {
        // send Terminate first so the thread stops touching choices
        let _ = self.tx.send(Message::Terminate);
        if self.handle.take().unwrap().join().is_err() {
            eprintln!("audio thread panicked");
        }
        unsafe { drop(Box::from_raw(self.choices)) };
    }
}

/// Exposes a platform independent interface to available streams.
#[derive(Clone, Debug)]
pub struct AudioChoice {
    // NOTE these fields are only accessed via getters to enable later interception by methods aware
    // of platform variations.
    kind: AudioSourceKind,
    name: Option<String>,
    #[cfg(target_os = "linux")]
    object_serial: u32,
    /// Integer passed to the global registry listener.  Does not correspond perfectly to any fields
    /// of any objects.  Used to support removal of previously registered audio sources.
    #[cfg(target_os = "linux")]
    global_id: u32,
}

impl AudioChoice {
    #[cfg(target_os = "linux")]
    pub fn name(&self) -> String {
        self.name
            .clone()
            .unwrap_or_else(|| self.object_serial.to_string())
    }

    #[cfg(target_os = "linux")]
    pub fn id(&self) -> String {
        format!("{}", self.object_serial)
    }

    /// Return the [`AudioSourceKind`] to differentiate nodes with the same name but different
    /// roles.
    pub fn kind(&self) -> AudioSourceKind {
        self.kind
    }

    // This was going to be a try_from implementation until I realized the global_id was needed to
    // support removals on Linux / pipewire.
    fn try_new(
        kind: AudioSourceKind,
        props: &spa::utils::dict::DictRef,
        global_id: u32,
    ) -> Result<Self, MutateError> {
        let object_serial: u32 = props
            .get("object.serial")
            .and_then(|s| s.parse::<u32>().ok())
            .ok_or(MutateError::AudioSource(
                "invalid or missing object.serial".to_owned(),
            ))?;

        // The "name" here is a rather arbitrary choice.  Different choices for different devices
        // may mean more to users.
        let name = props
            .get("node.description") // "Ryzen HD Audio Controller Analog Stereo"
            .or_else(|| props.get("media.name")) // "Loin Girding Hymns..."
            .or_else(|| props.get("application.name")) // "Firefox" (application when media.name isn't set)
            .or_else(|| props.get("node.name")) // technical but always present
            .map(ToString::to_string);

        Ok(AudioChoice {
            kind,
            object_serial,
            name,
            global_id,
        })
    }
}

/// The rendezvous point for `AudioConsumer` and `AudioProducer`.  Either side can tombstone the
/// connection to enable the other to return errors until its side drops and enables cleanup.
pub struct AudioConnection {
    // NEXT convert this to use frames?  Store the format somewhere?
    pub buffer: UnsafeCell<ringbuf::HeapRb<u8>>,

    pub ready: std::sync::Condvar,
    // XXX We may lock something else and present the AudioTiming as an untorn type (until double
    // buffering support is good).
    /// The lock payload is a u64 representing the number of chunks written.
    pub lock: std::sync::Mutex<timing::AudioTiming>,
    /// An online timing data accumulator to estimate phase and jitter to assist in accurate video
    /// tracking of audio.
    pub timing: timing::TimingFilter,

    // Tombstone for either end of the resource to finish up.
    // XXX poison if we can't drop while holding some lock?
    dropped: atomic::AtomicBool,
}

#[cfg(target_os = "linux")]
struct PipewireConnection {
    listener: Option<pw::stream::StreamListener<Box<StreamData>>>,
    stream: Option<pw::stream::StreamBox<'static>>,
}

impl AudioConnection {
    #[cfg(target_os = "linux")]
    fn new() -> *mut Self {
        let buffer = ringbuf::HeapRb::new(1024 * 256);
        Box::into_raw(Box::new(AudioConnection {
            buffer: UnsafeCell::new(buffer),

            ready: std::sync::Condvar::new(),
            lock: std::sync::Mutex::new(timing::AudioTiming::new()),
            timing: timing::TimingFilter::new(),
            // XXX make sure we can't accidentally ask a dropped object for timing data
            dropped: false.into(),
        }))
    }
}

/// The user side of a connection, obtained by calling [`connect`](AudioContext::connect) with an
/// [`AudioChoice`].  The producer side is owned by the `AudioContext`, usually inside the audio
/// server communication thread.
///
/// Dropping an `AudioConsumer` will tombstone the `AudioConnection`, enabling clean up the
/// connection after the corresponding `AudioProducer` has an opportunity to clean up.
pub struct AudioConsumer {
    pub conn: *mut AudioConnection,
}

unsafe impl Send for AudioConsumer {}

impl AudioConsumer {
    /// Wait for a buffer chunk to be written.
    pub fn wait(&self) -> Result<u64, MutateError> {
        let conn = unsafe { &(*self.conn) };
        if conn.dropped.load(atomic::Ordering::Acquire) {
            return Err(MutateError::Dropped);
        }
        let mut timing = conn.lock.lock()?;
        let initial = *timing;
        while timing.count == initial.count {
            // NEXT use a timeout wait and provide a method to wait for a specific number of bytes.
            // Possibly switch to parking instead of this silly Condvar.
            timing = conn.ready.wait(timing)?;
            if conn.dropped.load(atomic::Ordering::Acquire) {
                return Err(MutateError::Dropped);
            }
        }
        Ok(timing.count)
    }

    // The reader is doing pull-based consumption into it's own output slice, enabling us to handle
    // the ring buffer as minimally as possible.
    pub fn read(&mut self, output: &mut [u8]) -> Result<usize, MutateError> {
        let conn = unsafe { &mut (*self.conn) };
        let buf = unsafe { &mut *conn.buffer.get() };
        if conn.dropped.load(atomic::Ordering::Acquire) {
            return Err(MutateError::Dropped);
        }
        Ok(buf.pop_slice(output))
    }

    /// Return how many bytes are available for read
    pub fn occupied(&self) -> usize {
        let conn = unsafe { &(*self.conn) };
        let buf = unsafe { &mut *conn.buffer.get() };
        buf.occupied_len()
    }

    /// Remind the consumer how much capacity we requested.
    pub fn capacity(&self) -> usize {
        let conn = unsafe { &(*self.conn) };
        let buf = unsafe { &mut *conn.buffer.get() };
        usize::from(buf.capacity())
    }

    /// Get rid of some elements that are likely to cause producer slack underrun anyway.
    pub fn skip(&self, count: usize) {
        let conn = unsafe { &(*self.conn) };
        let buf = unsafe { &mut *conn.buffer.get() };
        buf.skip(count);
    }

    /// Get most recent phase data.
    pub fn timing(&self) -> Result<timing::AudioTiming, MutateError> {
        let conn = unsafe { &(*self.conn) };
        Ok(conn.lock.lock().map(|t| t.clone())?)
    }
}

impl Drop for AudioConsumer {
    fn drop(&mut self) {
        let was_dropped = unsafe { (*self.conn).dropped.swap(true, atomic::Ordering::AcqRel) };
        if was_dropped {
            unsafe { drop(Box::from_raw(self.conn)) };
        }
    }
}

/// The Tx side of creating a connection to the audio server.   This structure is handed off to the
/// audio thread.
struct AudioProducer {
    conn: *mut AudioConnection,
}

unsafe impl Send for AudioProducer {}

impl AudioProducer {
    fn write(
        &mut self,
        datas: &mut [spa::buffer::Data],
        arrived: Instant,
    ) -> Result<usize, MutateError> {
        let conn = unsafe { &mut *self.conn };
        let buf = unsafe { &mut *conn.buffer.get() };
        if conn.dropped.load(atomic::Ordering::Acquire) {
            return Err(MutateError::Dropped);
        }

        let input_len = datas.iter().fold(0, |accum, d| accum + d.chunk().size()) as usize;
        let capacity: usize = buf.capacity().into();
        if input_len > capacity {
            eprintln!(
                "total input len {} exceeds ring capacity {}",
                input_len, capacity
            );
            return Err(MutateError::AudioSource("ring too small".to_owned()));
        }
        let vacant_len = buf.vacant_len();
        if input_len > vacant_len {
            eprintln!("audio consumer falling behind");
        }
        let mut written = 0;
        datas.iter_mut().for_each(|d| {
            let offset = d.chunk().offset() as usize;
            let size = d.chunk().size() as usize;
            if let Some(input) = d.data() {
                written += buf.push_slice(&input[offset..offset + size]);
            }
        });

        let snapshot = conn.timing.observe(arrived, written);
        let mut audio_timing = conn.lock.lock()?;
        *audio_timing = snapshot;
        audio_timing.count += 1;
        audio_timing.last = arrived;

        conn.ready.notify_all();
        Ok(written)
    }
}

impl Drop for AudioProducer {
    fn drop(&mut self) {
        let was_dropped = unsafe { (*self.conn).dropped.swap(true, atomic::Ordering::AcqRel) };
        unsafe { (*self.conn).ready.notify_all() }; // wake any waiting consumer
        if was_dropped {
            unsafe { drop(Box::from_raw(self.conn)) };
        }
    }
}

#[cfg(target_os = "linux")]
struct StreamData {
    format: spa::param::audio::AudioInfoRaw,
    tx: AudioProducer,
}

#[cfg(target_os = "linux")]
fn create_stream<'c>(
    core: *mut pw::sys::pw_core,
    choice: &AudioChoice,
    name: &str,
    tx: AudioProducer,
) -> Result<
    (
        StreamListener<Box<StreamData>>,
        pw::stream::StreamBox<'static>,
    ),
    MutateError,
> {
    let props = pw::properties::properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_ROLE => "Music",
        *pw::keys::STREAM_CAPTURE_SINK => "true",
        // NODE_LATENCY values control the chunk sizes sent to the process callback.  Pipewire will
        // use rounded up PoT values.  Latency less than the frame size (800 for 60FPS) will result
        // in a better approximation of continuous feed, making it easier to achieve just-in-time
        // processing without underruns.  However, low values can also cause crackling.  Switching
        // the scheduling configuration for the pipewire process or other changes may help avoid
        // this, but for now it was easier just to choose a slightly relaxed value to be on the safe
        // side.
        // NEXT run-time changes to the latency?
        *pw::keys::NODE_LATENCY => "512/48000",
        *pw::keys::TARGET_OBJECT => choice.object_serial.to_string(),
    };

    // 🤠 Whatever breauxseph, just let me use a pointer like a pointer!
    let core_raw = std::ptr::NonNull::new(core).unwrap();
    let core = unsafe { core_raw.cast::<pw::core::Core>().as_ref() };

    let stream = pw::stream::StreamBox::new(core, name, props)?;

    let data = Box::new(StreamData {
        format: Default::default(), // XXX format is not exposed to receiver
        tx,
    });

    // This is the minimum
    let listener = stream
        .add_local_listener_with_user_data(data)
        .state_changed(|_stream, _user_data, _old_state, new_state| {
            eprintln!("state changed!: {:?}", new_state);
        })
        .param_changed(|stream, user_data, id, param| {
            let Some(param) = param else {
                return;
            };
            if id != pw::spa::param::ParamType::Format.as_raw() {
                return;
            }

            let (media_type, media_subtype) = match spa::param::format_utils::parse_format(param) {
                Ok(v) => v,
                Err(_) => return,
            };
            if media_type != spa::param::format::MediaType::Audio
                || media_subtype != spa::param::format::MediaSubtype::Raw
            {
                return;
            }

            // NEXT actually do the negotiation and propagate format information downstream.
            user_data
                .format
                .parse(param)
                .expect("Failed to parse param changed to AudioInfoRaw");

            if let Some(object_serial) = stream.properties().get("object.serial") {
                println!("new stream object serial: {}", object_serial);
            }
            if let Some(target_id) = stream.properties().get("target.object") {
                println!("connected to target: {}", target_id);
            }
            println!(
                "capturing rate:{} channels:{}",
                user_data.format.rate(),
                user_data.format.channels()
            );
        })
        .process(|stream, user_data| {
            let arrived = Instant::now();
            match stream.dequeue_buffer() {
                Some(mut buffer) => {
                    let datas = buffer.datas_mut(); // drop implicitly dequeues
                    match user_data.tx.write(datas, arrived) {
                        Ok(_written) => {}
                        Err(e) => {
                            eprintln!("Stream write error: {:?}", e)
                        }
                    };
                }
                None => {
                    eprintln!("no buffer dequeued");
                }
            }
        })
        .register()?;

    let pod_object = spa::pod::object! {
        spa::utils::SpaTypes::ObjectParamFormat,
        spa::param::ParamType::EnumFormat,
        spa::pod::property!(
            spa::param::format::FormatProperties::MediaType,
            Id,
            spa::param::format::MediaType::Audio
        ),
        spa::pod::property!(
            spa::param::format::FormatProperties::MediaSubtype,
            Id,
            spa::param::format::MediaSubtype::Raw
        ),
        spa::pod::property!(
            spa::param::format::FormatProperties::AudioFormat,
            Id,
            // spa::param::audio::AudioFormat::F32P
            spa::param::audio::AudioFormat::F32LE
        ),
    };

    let mut buf = Vec::new();
    let _ = pw::spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(&mut buf),
        &pw::spa::pod::Value::Object(pod_object),
    )
    .map_err(|e| {
        eprintln!("serializing pod failed: {}", e);
        return;
    });
    let pod = pw::spa::pod::Pod::from_bytes(&buf).unwrap();

    // NOTE Unless we pass AUTOCONNECT, an explicit link must be created between a compatible output
    // port and input port.
    stream.connect(
        spa::utils::Direction::Input,
        None, // read docs.  use PW_KEY_TARGET_OBJECT.  This argument is deprecated
        pw::stream::StreamFlags::MAP_BUFFERS
            | pw::stream::StreamFlags::AUTOCONNECT
            | pw::stream::StreamFlags::RT_PROCESS,
        &mut [pod],
    )?;

    // NEXT configure node delay.  Pipewire might allow it, but so far this is doubtful.

    Ok((listener, stream))
}
