use crate::config::Config;
use crate::remote_session::RemoteSessionState;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct ApiState {
    pub config: Arc<Mutex<Config>>,
    pub sessions: Arc<RemoteSessionState>,
}
