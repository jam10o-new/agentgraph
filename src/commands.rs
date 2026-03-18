//! Command parsing for agentgraph.
//!
//! Parses command markers like `【EXEC ... CEXE】` from model output.

use crate::types::{CmdOpenType, CommandType, MAX_OPENER_LEN};
use crate::{CMD_CLOSE_EXEC, CMD_CLOSE_KILL, CMD_CLOSE_READ, CMD_CLOSE_WRIT};

/// Helper struct to manage command state within the loop
pub struct CommandParser {
    buffer: String,
    active: bool,
    cmd_type: Option<CmdOpenType>,
}

impl CommandParser {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            active: false,
            cmd_type: None,
        }
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.active = false;
        self.cmd_type = None;
    }

    /// Check if combined (buffer + new content) contains a command opener
    /// Returns (bytes_before_opener, opener_type, bytes_after_opener) if found
    fn find_opener(&self, combined: &str) -> Option<(usize, CmdOpenType, usize)> {
        // Check for each opener type, find earliest occurrence
        let mut earliest: Option<(usize, CmdOpenType, usize)> = None;

        if let Some(pos) = combined.find(crate::CMD_OPEN_EXEC) {
            let after_pos = pos + crate::CMD_OPEN_EXEC.len();
            if earliest.is_none() || pos < earliest.unwrap().0 {
                earliest = Some((pos, CmdOpenType::Exec, after_pos));
            }
        }
        if let Some(pos) = combined.find(crate::CMD_OPEN_KILL) {
            let after_pos = pos + crate::CMD_OPEN_KILL.len();
            if earliest.is_none() || pos < earliest.unwrap().0 {
                earliest = Some((pos, CmdOpenType::Kill, after_pos));
            }
        }
        if let Some(pos) = combined.find(crate::CMD_OPEN_READ) {
            let after_pos = pos + crate::CMD_OPEN_READ.len();
            if earliest.is_none() || pos < earliest.unwrap().0 {
                earliest = Some((pos, CmdOpenType::Read, after_pos));
            }
        }
        if let Some(pos) = combined.find(crate::CMD_OPEN_WRIT) {
            let after_pos = pos + crate::CMD_OPEN_WRIT.len();
            if earliest.is_none() || pos < earliest.unwrap().0 {
                earliest = Some((pos, CmdOpenType::Writ, after_pos));
            }
        }

        earliest
    }

    /// Check if buffer contains a closer
    /// Returns (cmd_content, remaining_after_closer) if found
    fn find_closer(&self, cmd_type: CmdOpenType) -> Option<(String, String)> {
        let closer = match cmd_type {
            CmdOpenType::Exec => CMD_CLOSE_EXEC,
            CmdOpenType::Kill => CMD_CLOSE_KILL,
            CmdOpenType::Read => CMD_CLOSE_READ,
            CmdOpenType::Writ => CMD_CLOSE_WRIT,
        };

        self.buffer.find(closer).map(|pos| {
            let content = self.buffer[..pos].to_string();
            let remaining = self.buffer[pos + closer.len()..].to_string();
            (content, remaining)
        })
    }

    /// Process new content with sliding window handling
    /// Returns (content_to_output, completed_command, leftover_for_next_iteration)
    pub fn process(&mut self, content: &str) -> (String, Option<CommandType>, String) {
        if !self.active {
            // PRE-ACTIVE MODE: Check if opener appeared across buffer+content boundary

            let combined = format!("{}{}", self.buffer, content);
            if let Some((before_pos, cmd_type, after_pos)) = self.find_opener(&combined) {
                // Found opener! Transition to active mode

                // Content before opener goes to output
                let before_opener = &combined[..before_pos];

                // Content after opener goes into command buffer
                let after_opener = &combined[after_pos..];

                self.active = true;
                self.cmd_type = Some(cmd_type);
                self.buffer.clear();
                self.buffer.push_str(after_opener);

                // Check if closer is already in the remaining content
                if let Some((cmd_content, remaining)) = self.find_closer(cmd_type) {
                    let cmd = self.parse_command_content(cmd_type, &cmd_content);
                    self.reset();
                    return (before_opener.to_string(), cmd, remaining);
                }

                return (before_opener.to_string(), None, String::new());
            }

            // No opener found - update sliding window buffer
            let combined = format!("{}{}", self.buffer, content);
            if combined.len() <= MAX_OPENER_LEN {
                // Still building up to max opener length
                self.buffer = combined;
                return (String::new(), None, String::new());
            } else {
                // Find a valid char boundary to split at, rounding UP buffer size if needed
                let split_point = combined.floor_char_boundary(combined.len() - MAX_OPENER_LEN);

                // Everything before split_point goes to output
                let output = combined[..split_point].to_string();
                // Everything from split_point onward stays in buffer
                self.buffer = combined[split_point..].to_string();

                return (output, None, String::new());
            }
        }

        // ACTIVE MODE: Accumulating command content
        self.buffer.push_str(content);

        // Check for closer in the accumulated buffer
        if let Some(cmd_type) = self.cmd_type {
            if let Some((cmd_content, remaining)) = self.find_closer(cmd_type) {
                let cmd = self.parse_command_content(cmd_type, &cmd_content);
                self.reset();
                return (String::new(), cmd, remaining);
            }
        }

        // Still accumulating - no output, no command, no leftover
        (String::new(), None, String::new())
    }

    /// Parse command content based on type
    fn parse_command_content(&self, cmd_type: CmdOpenType, content: &str) -> Option<CommandType> {
        match cmd_type {
            CmdOpenType::Exec => Some(CommandType::Exec(content.to_string())),
            CmdOpenType::Kill => content.trim().parse::<usize>().ok().map(CommandType::Kill),
            CmdOpenType::Read => content.trim().parse::<usize>().ok().map(CommandType::Read),
            CmdOpenType::Writ => {
                let trimmed = content.trim();
                if let Some(space_idx) = trimmed.find(' ') {
                    let idx = trimmed[..space_idx].parse::<usize>().ok()?;
                    let input = trimmed[space_idx + 1..].to_string();
                    Some(CommandType::Writ(idx, input))
                } else {
                    None
                }
            }
        }
    }

    /// Flush any characters held in the sliding-window buffer at end-of-stream.
    pub fn flush(&mut self) -> String {
        if self.active {
            self.reset();
            String::new()
        } else {
            std::mem::take(&mut self.buffer)
        }
    }
}

impl Default for CommandParser {
    fn default() -> Self {
        Self::new()
    }
}
