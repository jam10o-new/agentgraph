//! Utility functions for agentgraph.

use crate::Args;

/// Check if any input is configured
pub fn has_any_input(args: &Args) -> bool {
    !args.input_final.is_empty()
        || !args.input_cat.is_empty()
        || !args.system_final.is_empty()
        || !args.system_cat.is_empty()
        || !args.assistant_final.is_empty()
        || !args.assistant_cat.is_empty()
}

/// Check if any output is configured
pub fn has_any_output(args: &Args) -> bool {
    args.output_new.is_some() || args.output_overwrite.is_some()
}
