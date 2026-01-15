// Copyright 2025 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Core µTate audio recognition & transformation capabilities.
//!
//! Alternative frontends and applications may be interested in obtaining raw inputs to drive
//! behaviors besides visualization.  This crate is kept separate so that µTate behaviors can be
//! embedded directly into 3rd party applications without the need to run a separate daemon.

//! AudioContext sets up communication threads that receive mapped buffers from an audio server such
//! as Pipewire.  It tracks available audio sources and provides with_choices and
//! with_choices_blocking methods for displaying choices to the user.  An AudioChoice can be used to
//! call connect, which will return an AudioConsumer.  An AudioConsumer, which is backed by a ring
//! buffer, provides  enough synchronization information to precisely obtain sliding windows of
//! audio data that can be used to develop whatever visual representations the user wants.

// NEXT To extend the AudioContext module for other platforms, just add cfg flags wherever
// implementations and fields are platform specific.  Take a look at CPAL but consider using
// platform bindings more directly if CPAL can't give us precise timing data or control.  We might
// want to adjust the input stream latency by talking to the audio server directly, which is not an
// API expected to be found in CPAL.

// NOTE remember, delay times from the server can be negative, so always use signed types, such as
// i64 etc.

// The model for working with pipewire, which might hold up when talking to other audio servers, is
// that pipewire sends us monotonic buffer chunks without skips (via padding or stream parameter
// change, the latter of which is not yet handled).
use pipewire as pw;
#[cfg(target_os = "linux")]
use pipewire::stream::StreamListener;
use pw::{main_loop::MainLoopBox, spa};
use ringbuf::traits::{Consumer, Observer, Producer};

// NEXT Audio will be its own kind of error that must fit into the MutateError hierarchy.
#[derive(thiserror::Error, Debug)]
pub enum MutateError {
    #[cfg(target_os = "linux")]
    #[error("Pipewire: {0}")]
    Pipewire(#[from] pw::Error),
    #[error("thread poisoned")]
    Poison,

    #[error("audio source error: {0}")]
    AudioSource(String),
    #[error("cannot use dropped audio connection")]
    Dropped,

    #[error("audio connection error: {0}")]
    AudioConnect(&'static str),
    #[error("audio thread termination error")]
    AudioTerminate,

    #[error("Timeout: {0}")]
    Timeout(&'static str),
}

impl<T> From<std::sync::PoisonError<T>> for MutateError {
    fn from(_: std::sync::PoisonError<T>) -> Self {
        MutateError::Poison
    }
}

/// Commands for calling into the Audio thread
enum Message {
    /// Connect to a particular identifier
    Connect {
        name: String,
        choice: AudioChoice,
        tx: AudioProducer,
    },
    Terminate,
}

// interior sync safe
struct AudioChoicesInner {
    ready: std::sync::Condvar,
    choices: std::sync::Mutex<Vec<AudioChoice>>,
    version: std::sync::atomic::AtomicUsize,
    initialized: std::sync::atomic::AtomicBool,
}

// Sharing an Arc.
#[derive(Clone)]
struct AudioChoices {
    inner: std::sync::Arc<AudioChoicesInner>,
}

impl AudioChoices {
    fn new() -> Self {
        AudioChoices {
            inner: std::sync::Arc::new(AudioChoicesInner {
                ready: std::sync::Condvar::new(),
                choices: std::sync::Mutex::new(Vec::new()),
                version: std::sync::atomic::AtomicUsize::new(0),
                initialized: std::sync::atomic::AtomicBool::new(false),
            }),
        }
    }

    fn notify(&self) {
        // Only one writer.  We mainly care that readers see the version when awoken
        self.inner
            .initialized
            .store(true, std::sync::atomic::Ordering::Release);
        self.inner.ready.notify_all();
    }
}

/// The AudioContext indirection will be the basis of the public API for acquiring sound input. Even
/// though only pipewire is supported so far, we are only interested in exposing and using a
/// narrow set of capabilities that is not expected to depend at all on the host platform.
///
/// We are usually interested in monitoring outgoing sound from other applications.  We need to find
/// valid sinks and create streams linked to their monitor ports.  The exact terminology may depend
/// on the platform, but the basic idea is to find outbound audio and tee it into our application
/// with sufficient synchronization information to align with sounds being played as closely as
/// possible.
// Don't make anything too public on this struct!
pub struct AudioContext {
    handle: std::thread::JoinHandle<()>,
    choices: AudioChoices,

    #[cfg(target_os = "linux")]
    tx: pw::channel::Sender<Message>,
}

impl AudioContext {
    /// Creates initial platform resources.  This will create a thread handle and begin tracking
    /// available useful sinks.
    pub fn new() -> Result<Self, MutateError> {
        // Platform binaries may use cfg flags.  For supporting different versions of the same
        // platform prefer runtime decisions.  Use features if binary weight is a concern for
        // library users.
        Self::initialize()
    }

    #[cfg(target_os = "linux")]
    fn initialize() -> Result<Self, MutateError> {
        let choices = AudioChoices::new();
        let choices_clone = choices.clone();
        let (pw_sender, pw_receiver) = pipewire::channel::channel();

        let handle = std::thread::spawn(move || {
            // Due to borrowed data and lack of try blocks in stable, Rust, seems like this is an
            // okay-ish way to know of issues in the terminal without forcing callers to fail.  At
            // least that's the goal.

            let choices_done = choices_clone.clone();
            let choices_remove = choices_clone.clone();
            let choices_add = choices_clone;

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

            let _receiver = pw_receiver.attach(mainloop.loop_(), {
                let mainloop_ptr = mainloop.as_raw_ptr();
                // NOTE The crate is basically begging us to use the provided Rc wrapper here.  We
                // always know the callback is outlived by the referents, but the "safe high-level"
                // API doesn't really anticipate our style of usage (dynamic stream creation?) and
                // so we're left with a dilemma of doing meaningless reference counting (a moral
                // hazard) or hoping our unsafe code is sound.
                //
                // I could be wrong, but since we don't have some guard or way to tell the compiler
                // where this callback may be ran, nobody will easily know.  The point is that a
                // different memory soundness guarantee for same-thread callbacks seems necessary
                // here.  I don't know of a way to create this guarantee or if a good tool already
                // exists.
                let core_ptr = core.as_raw_ptr();
                move |message| match message {
                    Message::Connect { choice, tx, name } => {
                        let mut conn =
                            std::mem::ManuallyDrop::new(unsafe { Box::from_raw(tx.conn) });
                        match create_stream(core_ptr, &choice, &name, tx) {
                            Ok((listener, stream)) => {
                                conn.stream.replace(stream);
                                conn.listener.replace(listener);
                            }
                            Err(e) => {
                                eprintln!("stream creation failed: {}", e);
                                return;
                            }
                        };
                    }
                    Message::Terminate => {
                        eprintln!("Terminating mainloop");
                        unsafe { pw::sys::pw_main_loop_quit(mainloop_ptr) };
                    }
                }
            });

            let _done_listener = core
                .add_listener_local()
                .done(move |_seq, _serial| choices_done.notify())
                .register();

            let _monitor_listener = registry
                .add_listener_local()
                .global(move |global| {
                    if global.type_ != pw::types::ObjectType::Node {
                        return;
                    }

                    if let Some(props) = &global.props {
                        if props.get("media.class").map(|c| c.starts_with("Audio/")) == Some(true) {
                            match AudioChoice::try_new(*props, global.id) {
                                Ok(choice) => match choices_add.inner.choices.lock() {
                                    Ok(mut choices) => {
                                        choices.push(choice);
                                    }
                                    Err(e) => {
                                        eprintln!(
                                            "adding audio source failed: {:?}",
                                            MutateError::from(e)
                                        );
                                    }
                                },
                                Err(e) => {
                                    eprintln!("Skipping Audio/Source: {:?}", e);
                                }
                            }
                        }
                    }
                })
                .register();

            let _remove_listener = registry
                .add_listener_local()
                .global_remove(
                    move |removed_id| match choices_remove.inner.choices.lock() {
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
                    },
                )
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
            handle,
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

    /// Disconnect to a stream
    pub fn destroy(&self) -> Result<(), MutateError> {
        self.tx
            .send(Message::Terminate)
            .map_err(|_e| MutateError::AudioTerminate)
    }

    pub fn choices_version(&self) -> usize {
        // Readers are deciding to do an update if one is available.  Missing one due to relaxed
        // ordering fine-grained incoherence is totally fine.
        self.choices
            .inner
            .version
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Run a function on the most recent choices.  If you need to wait on the first updates, use
    /// wait_read instead.  Your provided function should complete quickly because it uses a lock
    /// that will block the audio thread.  If you need more time, record a copy of the choices into
    /// your calling scope.
    pub fn with_choices<F>(&self, mut f: F) -> Result<(), MutateError>
    where
        F: FnMut(&[AudioChoice]),
    {
        let choices = self.choices.inner.choices.lock()?;
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
        let mut choices = self.choices.inner.choices.lock()?;
        while self
            .choices
            .inner
            .initialized
            .load(std::sync::atomic::Ordering::Relaxed)
            == false
        {
            let (guard, result) = self
                .choices
                .inner
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

#[derive(Clone, Debug)]
/// Platform independent Choice to enable building cross-platform UIs
pub struct AudioChoice {
    #[cfg(target_os = "linux")]
    object_serial: u32,
    name: Option<String>,
    description: Option<String>,
    #[cfg(target_os = "linux")]
    /// Integer passed to the global registry listener.  Does not correspond perfectly to any fields
    /// of any objects.  Used to support removal of previously registered audio sources.
    global_id: u32,
}

impl AudioChoice {
    #[cfg(target_os = "linux")]
    pub fn name(&self) -> String {
        self.name
            .clone()
            .or(self.description.clone())
            .unwrap_or_else(|| self.object_serial.to_string())
    }

    #[cfg(target_os = "linux")]
    pub fn id(&self) -> String {
        format!("{}", self.object_serial)
    }

    // This was going to be a try_from implementation until I realized the global_id was needed to
    // support removals on Linux / pipewire.
    fn try_new(props: &spa::utils::dict::DictRef, global_id: u32) -> Result<Self, MutateError> {
        let object_serial: u32 = props
            .get("object.serial")
            .and_then(|s| s.parse::<u32>().ok())
            .ok_or(MutateError::AudioSource(
                "invalid or missing object.serial".to_owned(),
            ))?;

        // The "name" here is a rather arbitrary choice.  Different choices for different devices
        // may mean more for users.
        let name = props
            .get("device.description")
            .or_else(|| props.get("device.nick"))
            .or_else(|| props.get("device.name"))
            .or_else(|| props.get("object.path"))
            .map(ToString::to_string);

        let description = props.get("node.description").map(ToString::to_string);

        Ok(AudioChoice {
            object_serial,
            name,
            description,
            global_id,
        })
    }
}

/// AudioConnection is never directly handled.  It is created by calling connect.  Dropping the
/// returned AudioConsumer will clean up the connection after the corresponding AudioProducer has an
/// opportunity to clean up.
pub struct AudioConnection {
    // NEXT convert this to use frames?  Store the format somewhere?
    pub buffer: ringbuf::HeapRb<u8>,

    pub ready: std::sync::Condvar,
    /// The lock payload is a u64 representing the number of chunks written.
    pub lock: std::sync::Mutex<u64>,

    // Tombstone for either end of the resource to finish up.
    // XXX instead, we want one-sided drop behavior,
    // the last drop being the producer
    // and drop flags that detect the poisoning of the producer?
    dropped: std::sync::atomic::AtomicBool,

    #[cfg(target_os = "linux")]
    user_data: Option<Box<StreamData>>,
    #[cfg(target_os = "linux")]
    stream: Option<pw::stream::StreamBox<'static>>,
    #[cfg(target_os = "linux")]
    listener: Option<pw::stream::StreamListener<Box<StreamData>>>,
}

impl AudioConnection {
    #[cfg(target_os = "linux")]
    fn new() -> *mut Self {
        let buffer = ringbuf::HeapRb::new(1024 * 256);
        Box::into_raw(Box::new(AudioConnection {
            buffer,

            ready: std::sync::Condvar::new(),
            lock: std::sync::Mutex::new(0),
            dropped: false.into(),

            user_data: None,
            stream: None,
            listener: None,
        }))
    }
}

/// The Rx side of creating a connection to the audio server.  Dropping the consumer will tombstone
/// the connection and the backing resources will be cleaned up by the audio server communication
/// thread.
pub struct AudioConsumer {
    pub conn: *mut AudioConnection,
}

unsafe impl Send for AudioConsumer {}

impl AudioConsumer {
    /// Wait for a buffer chunk to be written.
    pub fn wait(&self) -> Result<u64, MutateError> {
        let conn = unsafe { &(*self.conn) };
        if conn.dropped.load(std::sync::atomic::Ordering::Acquire) {
            return Err(MutateError::Dropped);
        }
        let mut count = conn.lock.lock()?;
        let initial = *count;
        while *count == initial {
            count = conn.ready.wait(count)?;
        }
        Ok(*count)
    }
}

impl Drop for AudioConsumer {
    fn drop(&mut self) {
        unsafe {
            (*self.conn)
                .dropped
                .store(false, std::sync::atomic::Ordering::Release)
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
    fn write(&mut self, datas: &mut [spa::buffer::Data]) -> Result<usize, MutateError> {
        let mut conn = std::mem::ManuallyDrop::new(unsafe { Box::from_raw(self.conn) });
        // Check if the receiver died before putting data into the ring
        if conn.dropped.load(std::sync::atomic::Ordering::Acquire) {
            return Err(MutateError::Dropped);
        }

        let input_len = datas.iter().fold(0, |accum, d| accum + d.chunk().size()) as usize;
        let capacity: usize = conn.buffer.capacity().into();
        if input_len > capacity {
            eprintln!(
                "total input len {} exceeds ring capacity {}",
                input_len, capacity
            );
            return Err(MutateError::AudioSource("ring too small".to_owned()));
        }
        let vacant_len = conn.buffer.vacant_len();
        if input_len > vacant_len {
            eprintln!("audio consumer falling behind");
            conn.buffer.skip(input_len - vacant_len);
        }
        let mut written = 0;
        datas.iter_mut().for_each(|d| {
            let offset = d.chunk().offset() as usize;
            let size = d.chunk().size() as usize;
            if let Some(input) = d.data() {
                written += conn.buffer.push_slice(&input[offset..offset + size]);
            }
        });
        *conn.lock.lock()? += 1;
        conn.ready.notify_all();
        Ok(written)
    }
}

impl Drop for AudioProducer {
    fn drop(&mut self) {
        unsafe {
            if (*self.conn)
                .dropped
                .swap(true, std::sync::atomic::Ordering::AcqRel)
                == false
            {
                drop(Box::from_raw(self.conn));
            }
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
        // FIXME this is not yet respected 😠
        *pw::keys::TARGET_OBJECT => choice.global_id.to_string(),
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
            match stream.dequeue_buffer() {
                Some(mut buffer) => {
                    let datas = buffer.datas_mut(); // drop implicitly dequeues

                    match user_data.tx.write(datas) {
                        _ => {}
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
        None, // read docs.  use PW_KEY_TARGET_OBJECT.  This argument is deprectated
        pw::stream::StreamFlags::MAP_BUFFERS
            | pw::stream::StreamFlags::AUTOCONNECT
            | pw::stream::StreamFlags::RT_PROCESS,
        &mut [pod],
    )?;

    // NEXT configure node delay.  Pipewire might allow it, but so far this is doubtful.

    Ok((listener, stream))
}
